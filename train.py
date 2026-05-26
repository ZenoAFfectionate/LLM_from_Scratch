import os
import json
import time
import argparse
from pathlib import Path
from itertools import cycle
from typing import Dict, Any, Iterator

import numpy as np
import torch
import torch.nn as nn
import wandb

from torch.optim import AdamW
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from model.config import Config
from model.transformer import TransformerLM
from model.mHC_transformer import mHCTransformerLM
from model.tokenizer.bpe_tokenizer import Tokenizer
from model.optimizer.Muon import MuonAdamWOptimizer
from data.lm_dataset import PretrainDataset
from model.utils import (
    save_checkpoint, load_checkpoint,
    cos_learning_rate_schedule_with_warmup
)


def load_data_memmap(data_path: str, dtype=np.int32):
    """Load training data using memory mapping for efficient access"""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    # load data using memmap for memory efficiency
    data = np.memmap(data_path, dtype=dtype, mode='r')
    print(f"Loaded {len(data):,} tokens from {data_path}")
    return data


def prepare_data(config: Dict[str, Any]):
    """Prepare training and validation data using memory-mapped files"""
    data_dir = Path(config['data_dir'])

    train_bin = data_dir / "tokens_train.bin"
    valid_bin = data_dir / "tokens_valid.bin"

    train_data = load_data_memmap(str(train_bin))
    valid_data = load_data_memmap(str(valid_bin))

    return train_data, valid_data


def train(model: nn.Module, optimizer: torch.optim.Optimizer,
          train_loader_iter: Iterator, config: Dict[str, Any], device: torch.device,
          gradient_accumulation_steps: int = 1, accumulation_step: int = 0):
    """
    Perform a single training step with BF16 mixed precision and gradient accumulation

    Args:
        model: The model to train
        optimizer: The optimizer
        train_loader_iter: Iterator from the training DataLoader
        config: Configuration dictionary
        device: Device to run on
        gradient_accumulation_steps: Number of steps to accumulate gradients over
        accumulation_step: Current accumulation step (0 to gradient_accumulation_steps-1)

    Returns:
        Tuple of (loss_tensor, grad_norm_tensor) - detached tensors for async logging
        Call .item() only when actually logging to avoid GPU-CPU sync overhead
    """
    model.train()

    # get next batch from DataLoader iterator and move to device
    inputs, targets = next(train_loader_iter)
    inputs = inputs.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)

    # configure mixed precision training
    use_amp = config.get('use_amp', False)
    amp_dtype = torch.bfloat16 if use_amp else torch.float32

    # Forward pass with autocast for BF16
    with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
        logits = model(inputs)
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = targets.view(-1)
        # compute loss and scale for gradient accumulation
        loss = F.cross_entropy(logits_flat, targets_flat)

        # z-loss (PaLM §5.1): keeps logsumexp(logits) near 0 to prevent softmax
        # entropy collapse under BF16. No-op when z_loss_alpha == 0.
        z_loss_alpha = config.get('z_loss_alpha', 0.0)
        if z_loss_alpha > 0:
            log_z = torch.logsumexp(logits_flat.float(), dim=-1)
            loss = loss + z_loss_alpha * (log_z ** 2).mean()

        loss = loss / gradient_accumulation_steps

    # backward pass with gradient accumulation:
    if accumulation_step == 0: optimizer.zero_grad(set_to_none=True)

    loss.backward()  # accumulate gradients without optimize

    # only update weights and clip gradients on the last accumulation step
    grad_norm = torch.tensor(0.0, device=device)
    if accumulation_step == gradient_accumulation_steps - 1:
        # ── Defensive grad scrub for GDA hybrids ──
        # fla 0.4.2's `chunk_gated_delta_rule` Triton kernel (used by
        # GatedDeltaAttention) allocates its forward output and backward
        # gradient buffers via `torch.empty()` and on some input shapes does
        # not write to every position. The unwritten positions inherit
        # whatever the CUDA allocator handed out (often NaN/Inf), which then
        # propagates through the rest of the backward graph. We already
        # scrub the forward output inside the GDA module; here we scrub the
        # remaining parameter grads so a single bad chunk does not poison
        # the whole optimizer state. No-op for clean grads, so safe to
        # always run.
        if config.get('gda_ratio', 'none') != 'none':
            for p in model.parameters():
                if p.grad is not None:
                    torch.nan_to_num_(p.grad, nan=0.0, posinf=0.0, neginf=0.0)

        grad_norm = clip_grad_norm_(model.parameters(), config['max_grad_norm'])
        optimizer.step()

        # update expert biases for load balance
        if hasattr(model, 'update_moe_biases'):
            model.update_moe_biases()

    # return detached tensors - no .item() call here to avoid GPU-CPU sync
    return (loss.detach() * gradient_accumulation_steps, grad_norm.detach())


def valid(model: nn.Module, val_loader: DataLoader, config: Dict[str, Any], device: torch.device):
    """Evaluate model on validation data with BF16 autocasting"""
    # use ._orig_mod to access the original uncompiled model for validation
    eval_model = model._orig_mod if hasattr(model, '_orig_mod') else model
    eval_model.eval()

    total_loss = 0.0
    num_batches = config['eval_batches']

    use_amp = config.get('use_amp', False)
    amp_dtype = torch.bfloat16 if use_amp else torch.float32

    with torch.no_grad():
        val_iter = cycle(val_loader)
        # evaluate for a fixed number of batches
        for _ in range(num_batches):
            inputs, targets = next(val_iter)
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
                logits = eval_model(inputs)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            total_loss += loss.item()

    avg_loss = total_loss / num_batches
    perplexity = np.exp(avg_loss)
    return avg_loss, perplexity


def main():
    parser = argparse.ArgumentParser(description='Train Transformer Language Model')

    parser.add_argument('--config', type=str, required=True, help='Path to config JSON file')
    parser.add_argument('--resume', type=str, default=None,  help='Path to checkpoint to resume from')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='Number of gradient accumulation steps to simulate larger batch size')

    parser.add_argument('--optimizer', type=str, default=None, choices=['muon', 'adamw'])
    # Muon-specific arguments (can override config file values)
    parser.add_argument('--muon_lr', type=float, default=None, help='Learning rate for Muon optimizer ')
    parser.add_argument('--muon_momentum', type=float, default=None, help='Momentum for Muon optimizer')
    parser.add_argument('--muon_ns_steps', type=int, default=None, help='Newton-Schulz iters for Muon')
    args = parser.parse_args()

    # Load configuration using Config class
    config = Config.from_json(args.config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Set performance optimizations for PyTorch
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')  # enable TF32
        torch.backends.cudnn.benchmark = True       # enable cuDNN benchmark
        print("Performance optimizations enabled: TF32 matmul, cuDNN benchmark")

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)

    if args.optimizer is not None: config.optimizer = args.optimizer.lower()

    if args.muon_lr is not None:
        config.muon_lr = args.muon_lr
    if args.muon_momentum is not None:
        config.muon_momentum = args.muon_momentum
    if args.muon_ns_steps is not None:
        config.muon_ns_steps = args.muon_ns_steps

    # === Early checkpoint gate: skip if done, auto-resume if partially trained ===
    # Resolve the checkpoint directory from config alone (no model/data needed yet),
    # so we can short-circuit BEFORE any expensive init (wandb, tokenizer, model compile).
    attention_type = config.attention_type
    use_moe = config.use_moe
    ffn_type = 'MoE' if use_moe else 'FFN'
    module_config = f"{attention_type}+{ffn_type}"
    # Ablation suffixes: keep folder names backward-compatible (no suffix when
    # gda_ratio is "none" and residual_type is "resscale"), otherwise append
    # `_gda{ratio}` and/or `_res-{type}` so different ablations don't collide.
    suffix = ""
    if getattr(config, "gda_ratio", "none") != "none":
        suffix += f"_gda{config.gda_ratio.replace(':', '-')}"
    if getattr(config, "residual_type", "resscale") != "resscale":
        suffix += f"_res-{config.residual_type}"
    dataset_name = config.dataset
    checkpoint_folder_name = f"{dataset_name}_{module_config}{suffix}"
    checkpoint_dir = Path(config.checkpoint_dir) / checkpoint_folder_name
    final_model_path = checkpoint_dir / "final_model.pt"

    if final_model_path.is_file():
        print(f"[SKIP] {checkpoint_folder_name}: final_model.pt already exists at {final_model_path}")
        print(f"       Training already complete. Delete the directory to retrain from scratch.")
        return

    auto_resumed = False
    if args.resume is None and checkpoint_dir.is_dir():
        iter_ckpts = sorted(checkpoint_dir.glob("checkpoint_iter_*.pt"))
        if iter_ckpts:
            args.resume = str(iter_ckpts[-1])
            auto_resumed = True
            print(f"[AUTO-RESUME] {checkpoint_folder_name}: found {len(iter_ckpts)} checkpoint(s)")
            print(f"              Resuming from latest: {args.resume}")

    # Create extended config dict for wandb (config is still Config object)
    wandb_config = config.to_dict()
    wandb_config['gradient_accumulation_steps'] = args.gradient_accumulation_steps
    wandb_config['effective_batch_size'] = config.batch_size * args.gradient_accumulation_steps

    wandb.init(
        project="Transformer_LLM",
        entity="scut_zeno",
        name=config.run_name,
        config=wandb_config
    )

    # Load tokenizer
    tokenizer = Tokenizer.from_files(
        vocab_filepath=config.vocab_file,
        merges_filepath=config.merges_file,
        special_tokens=config.special_tokens
    )

    vocab_size = len(tokenizer.decoder_vocab)
    print(f"Vocabulary size: {vocab_size:,}")

    config.vocab_size = vocab_size  # Update config with actual vocab_size

    # Prepare data and cache config dict for later use
    config_dict = config.to_dict()
    train_data, valid_data = prepare_data(config_dict)

    train_dataset = PretrainDataset(data=train_data, context_length=config.context_length)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True,
        num_workers=4, 
        pin_memory=True,
        persistent_workers=True, 
        prefetch_factor=4,
        drop_last=True,
    )

    valid_dataset = PretrainDataset(data=valid_data, context_length=config.context_length)
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=config.batch_size, 
        shuffle=False,
        num_workers=4, 
        pin_memory=True,
        persistent_workers=True, 
        prefetch_factor=4,
        drop_last=False,
    )

    print(f"Training dataset: {len(train_dataset):,} samples")
    print(f"Validate dataset: {len(valid_dataset):,} samples")
    print(f"DataLoaders initialized successfully!\n")

    # Initialize model with BF16 weights for native mixed-precision training.
    # This eliminates costly FP32→BF16 weight casting on every forward pass.
    # The optimizer (fused AdamW) maintains FP32 master weights internally.
    # Residual dispatch: mHC needs the parallel-streams TransformerLM variant;
    # vanilla / resscale share the standard TransformerLM (Block toggles internally).
    if getattr(config, "residual_type", "resscale") == "mhc":
        model = mHCTransformerLM(config=config, device=device, dtype=torch.bfloat16).to(device)
    else:
        model = TransformerLM(config=config, device=device, dtype=torch.bfloat16).to(device)

    # count trainable parameters of the model
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Compile the model with torch.compile for better performance.
    # GDA layers use fla's `causal_conv1d` whose decorator contexts trip
    # `fullgraph=True` Dynamo tracing — fall back to graph-breaking compile
    # when any GDA layer is present.
    has_gda = getattr(config, "gda_ratio", "none") != "none"
    if has_gda:
        print("Compiling model with torch.compile (fullgraph=False due to GDA fla kernels)...")
        model = torch.compile(model, mode="default", fullgraph=False)
    else:
        print("Compiling model with torch.compile...")
        model = torch.compile(model, mode="default", fullgraph=True)
    print("Model compiled successfully")

    # Initialize optimizer based on config
    use_muon = config.optimizer == "muon"
    if use_muon:
        # Use MuonAdamW combined optimizer (Muon for 2D weights, AdamW for rest)
        # Access the uncompiled model for parameter separation
        base_model = model._orig_mod if hasattr(model, '_orig_mod') else model
        optimizer = MuonAdamWOptimizer(
            model=base_model,
            muon_lr=config.muon_lr,
            adamw_lr=config.max_lr,
            muon_momentum=config.muon_momentum,
            muon_nesterov=config.muon_nesterov,
            muon_ns_steps=config.muon_ns_steps,
            muon_weight_decay=config.muon_weight_decay,
            adamw_betas=(config.beta1, config.beta2),
            adamw_eps=config.eps,
            adamw_weight_decay=config.weight_decay,
            support_engram=config.use_engram,
        )
        print(f"Using Muon + AdamW combined optimizer")
    else:
        # Use standard AdamW optimizer
        optimizer = AdamW(
            model.parameters(),
            lr=config.max_lr,
            betas=(config.beta1, config.beta2),
            eps=config.eps,
            weight_decay=config.weight_decay,
            fused=True
        )
        print(f"Using AdamW optimizer")

    start_iteration = 0  # initialize training state

    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        start_iteration = load_checkpoint(args.resume, model, optimizer)
        print(f"Resumed from iteration {start_iteration}")
    # checkpoint_dir, module_config, etc. were resolved earlier (see early gate).
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"Checkpoint directory: {checkpoint_dir}")

    # get gradient accumulation steps from command line argument
    gradient_accumulation_steps = args.gradient_accumulation_steps
    effective_batch_size = config.batch_size * gradient_accumulation_steps
    print(f"Gradient Accumulation: {gradient_accumulation_steps} steps")
    print(f"Micro Batch Size: {config.batch_size}")
    print(f"Effective Batch Size: {effective_batch_size}")

    record_file_path = checkpoint_dir / "record.txt"  # create record file path

    # Append on resume (preserves prior training log); 'w' on fresh runs.
    is_resume = auto_resumed or args.resume is not None
    record_mode = 'a' if is_resume else 'w'
    with open(record_file_path, record_mode) as record_file:
        if is_resume:
            record_file.write("\n" + "=" * 80 + "\n")
            record_file.write(f"[RESUMED] {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            record_file.write(f"[RESUMED] from checkpoint: {args.resume}\n")
            record_file.write(f"[RESUMED] at iteration: {start_iteration}\n")
            record_file.write("=" * 80 + "\n\n")
        else:
            record_file.write(f"Training Record for {config.dataset}\n")
            record_file.write("=" * 80 + "\n")
            record_file.write(f"Model: {config.run_name}\n")
            record_file.write(f"Optimizer: {config.optimizer.upper()}\n")
            record_file.write(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            record_file.write(f"Config file: {args.config}\n")
            record_file.write("=" * 80 + "\n\n")

        # Header sections below only on a fresh run (already in record.txt on resume).
        if not is_resume:
            # Write optimizer configuration
            if use_muon:
                record_file.write("OPTIMIZER CONFIGURATION (Muon + AdamW):\n")
                record_file.write("-" * 80 + "\n")
                record_file.write(f"Muon LR (max): {config.muon_lr}\n")
                record_file.write(f"Muon LR (min): {config.muon_min_lr}\n")
                record_file.write(f"Muon Momentum: {config.muon_momentum}\n")
                record_file.write(f"Muon Nesterov: {config.muon_nesterov}\n")
                record_file.write(f"Muon NS Steps: {config.muon_ns_steps}\n")
                record_file.write(f"Muon Weight Decay: {config.muon_weight_decay}\n")
                record_file.write(f"AdamW LR (max): {config.max_lr}\n")
                record_file.write(f"AdamW LR (min): {config.min_lr}\n")
                record_file.write(f"AdamW Weight Decay: {config.weight_decay}\n")
                record_file.write("-" * 80 + "\n\n")

            # Write full configuration
            record_file.write("FULL CONFIGURATION:\n")
            record_file.write("-" * 80 + "\n")
            record_file.write(json.dumps(config.to_dict(), indent=2))
            record_file.write("\n" + "-" * 80 + "\n\n")

            # Write model architecture summary
            record_file.write("MODEL ARCHITECTURE:\n")
            record_file.write("-" * 80 + "\n")
            record_file.write(
                f"ATT Type: [{attention_type}]   MLP Type: [{ffn_type}]\n")
            record_file.write(f"Total parameters: {total_params:,}\n")
            record_file.write(f"Trainable parameters: {trainable_params:,}\n")
            record_file.write(f"Gradient Accumulation Steps: {gradient_accumulation_steps}\n")
            record_file.write(f"Micro Batch Size: {config.batch_size}\n")
            record_file.write(f"Effective Batch Size: {effective_batch_size}\n")
            record_file.write("-" * 80 + "\n\n")

    print("Starting training...")
    print("-" * 60)

    # Training loop
    model.train()
    running_loss = 0.0
    best_val_loss = float('+inf')
    best_val_ppl = float('+inf')

    # create an infinite iterator from the training DataLoader
    # to ensures we never run out of data during training
    train_loader_iter = iter(train_loader)

    for iteration in range(start_iteration, config.max_iterations):
        start_time = time.time()

        # update learning rate with cosine schedule
        if use_muon:
            # For Muon, compute separate schedules for Muon and AdamW
            current_muon_lr = cos_learning_rate_schedule_with_warmup(
                iteration,
                max_lr=config.muon_lr,
                min_lr=config.muon_min_lr,
                warmup_iter=config.warmup_iterations,
                cos_iter=config.max_iterations
            )
            current_adamw_lr = cos_learning_rate_schedule_with_warmup(
                iteration,
                max_lr=config.max_lr,
                min_lr=config.min_lr,
                warmup_iter=config.warmup_iterations,
                cos_iter=config.max_iterations
            )
            optimizer.set_lr_absolute(current_muon_lr, current_adamw_lr)
            lr = current_adamw_lr  # For logging (primary LR)
        else:
            # For standard AdamW, single learning rate schedule
            lr = cos_learning_rate_schedule_with_warmup(
                iteration,
                max_lr=config.max_lr,
                min_lr=config.min_lr,
                warmup_iter=config.warmup_iterations,
                cos_iter=config.max_iterations
            )
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        # Gradient accumulation: perform multiple forward/backward passes
        accumulated_loss_tensor = torch.tensor(0.0, device=device)
        grad_norm_tensor = torch.tensor(0.0, device=device)

        for accum_step in range(gradient_accumulation_steps):
            try:
                loss_t, grad_norm_t = train(
                    model, optimizer, train_loader_iter, config_dict, device,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    accumulation_step=accum_step
                )
            except StopIteration:
                train_loader_iter = iter(train_loader)
                loss_t, grad_norm_t = train(
                    model, optimizer, train_loader_iter, config_dict, device,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    accumulation_step=accum_step
                )
            accumulated_loss_tensor = accumulated_loss_tensor + loss_t
            grad_norm_tensor = grad_norm_t  # Last step's grad_norm

        # Keep running_loss as tensor on GPU to avoid sync
        avg_accum_loss_tensor = accumulated_loss_tensor / gradient_accumulation_steps
        running_loss += avg_accum_loss_tensor.item()

        step_time = time.time() - start_time

        # Log training metrics - only call .item() when actually logging
        if (iteration + 1) % config.log_interval == 0:
            avg_loss = running_loss / config.log_interval
            perplexity = np.exp(avg_loss)
            # only sync grad_norm when logging
            grad_norm_value = grad_norm_tensor.item()

            if use_muon:
                content = f"Iter {iteration + 1:6d} | Loss: {avg_loss:.4f} | PPL: {perplexity:.2f} | " \
                          f"Muon_LR: {current_muon_lr:.6f} | AdamW_LR: {current_adamw_lr:.6f} | " \
                          f"Grad Norm: {grad_norm_value:.4f} | Time: {step_time:.3f}s"
            else:
                content = f"Iter {iteration + 1:6d} | Loss: {avg_loss:.4f} | PPL: {perplexity:.2f} | " \
                          f"LR: {lr:.6f} | Grad Norm: {grad_norm_value:.4f} | Time: {step_time:.3f}s"
            print(content)

            # Save training content to record file
            with open(record_file_path, 'a') as record_file:
                record_file.write(f"[TRAIN] {content}\n")

            # Log to wandb
            log_dict = {
                'train/loss': avg_loss,
                'train/perplexity': perplexity,
                'train/grad_norm': grad_norm_value,
                'train/step_time': step_time
            }
            if use_muon:
                log_dict['train/muon_lr'] = current_muon_lr
                log_dict['train/adamw_lr'] = current_adamw_lr
            else:
                log_dict['train/learning_rate'] = lr
            wandb.log(log_dict, step=iteration + 1)

            running_loss = 0.0

        # Validation and checkpointing
        if (iteration + 1) % config.eval_interval == 0:
            print("Running validation...")
            val_loss, val_perplexity = valid(model, valid_loader, config_dict, device)

            val_content = f"Validation | Loss: {val_loss:.4f} | PPL: {val_perplexity:.2f}"
            print(val_content)

            with open(record_file_path, 'a') as record_file:
                record_file.write(f"[VALID] {val_content}\n")

            wandb.log({
                'val/loss': val_loss,
                'val/perplexity': val_perplexity
            }, step=iteration + 1)

            # save checkpoint
            checkpoint_path = checkpoint_dir / f"checkpoint_iter_{iteration + 1:06d}.pt"
            save_checkpoint(model, optimizer, iteration + 1, str(checkpoint_path))
            print(f"Saved checkpoint: {checkpoint_path}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_ppl = val_perplexity
                best_checkpoint_path = checkpoint_dir / "best_model.pt"
                save_checkpoint(model, optimizer, iteration + 1, str(best_checkpoint_path))
                best_model_content = f"New best model saved: {best_checkpoint_path} (val_loss: {val_loss:.4f}, PPL: {val_perplexity:.2f})"
                print(best_model_content)

                # Save best model info to record file
                with open(record_file_path, 'a') as record_file:
                    record_file.write(f"[BEST] {best_model_content}\n")

                wandb.log({
                    'val/best_loss': best_val_loss,
                    'val/best_ppl': best_val_ppl
                }, step=iteration + 1)

            model.train()  # Switch back to training mode

    print("Training completed!")

    # Save training completion info to record file
    with open(record_file_path, 'a') as record_file:
        record_file.write(f"\n{'='*50}\n")
        record_file.write(f"Training completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        record_file.write(f"Total iterations: {config.max_iterations}\n")
        record_file.write(f"Final validation loss: {best_val_loss:.4f}   Final PPL: {best_val_ppl:.2f}\n")
        record_file.write(f"{'='*50}\n")

    print(f"\nFinal validation loss: {best_val_loss:.4f}   Final PPL: {best_val_ppl:.2f}")

    final_checkpoint_path = checkpoint_dir / "final_model.pt"
    save_checkpoint(model, optimizer, config.max_iterations, str(final_checkpoint_path))
    final_checkpoint_content = f"Final checkpoint saved: {final_checkpoint_path}"
    print(f"Final checkpoint saved: {final_checkpoint_path}")

    with open(record_file_path, 'a') as record_file:
        record_file.write(f"[FINAL] {final_checkpoint_content}\n")

    wandb.finish()


if __name__ == "__main__":
    main()
