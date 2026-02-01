import os
import json
import time
import wandb
import argparse
from pathlib import Path
from typing import Dict, Any, Iterator

import numpy as np
import torch
import torch.nn as nn

import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from model.config import Config
from model.transformer import TransformerLM
from model.tokenizer.bpe_tokenizer import Tokenizer
from data.lm_dataset import PretrainDataset
from model.optimizer.Muon import MuonAdamWOptimizer
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


def train(model: nn.Module, optimizer: MuonAdamWOptimizer,
          train_loader_iter: Iterator, config: Dict[str, Any], device: torch.device,
          gradient_accumulation_steps: int = 1, accumulation_step: int = 0):
    """Perform a single training step with BF16 mixed precision and gradient accumulation

    Args:
        model: The model to train
        optimizer: The MuonAdamW optimizer
        train_loader_iter: Iterator from the training DataLoader
        config: Configuration dictionary
        device: Device to run on
        gradient_accumulation_steps: Number of steps to accumulate gradients over
        accumulation_step: Current accumulation step (0 to gradient_accumulation_steps-1)

    Returns:
        Tuple of (loss, grad_norm) where grad_norm is only valid on the last accumulation step
    """
    model.train()

    # get next batch from DataLoader iterator and move to device
    inputs, targets = next(train_loader_iter)
    inputs  = inputs.to(device,  non_blocking=True)
    targets = targets.to(device, non_blocking=True)

    # configure mixed precision training, which means:
    # FP32 weights + BF16 forward pass + FP32 optimizer
    use_amp = config.get('use_amp', False)
    amp_dtype = torch.bfloat16 if use_amp else torch.float32

    # Forward pass with autocast for BF16
    with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
        logits = model(inputs)
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = targets.view(-1)
        # compute loss and scale for gradient accumulation
        loss = F.cross_entropy(logits_flat, targets_flat)
        loss = loss / gradient_accumulation_steps

    # backward pass with gradient accumulation:
    if accumulation_step == 0: optimizer.zero_grad(set_to_none=True)

    loss.backward()  # accumulate gradients without optimize

    # only update weights and clip gradients on the last accumulation step
    grad_norm = torch.tensor(0.0)
    if accumulation_step == gradient_accumulation_steps - 1:
        grad_norm = clip_grad_norm_(model.parameters(), config['max_grad_norm'])
        optimizer.step()

        # update expert biases for load balance
        if hasattr(model, 'update_moe_biases'):
            model.update_moe_biases()

    return loss.item() * gradient_accumulation_steps, grad_norm.item()


def valid(model: nn.Module, val_loader: DataLoader, config: Dict[str, Any], device: torch.device):
    """Evaluate model on validation data with BF16 autocasting"""
    # use ._orig_mod to access the original uncompiled model for validation
    eval_model = model._orig_mod if hasattr(model, '_orig_mod') else model
    eval_model.eval()

    total_loss = 0.0
    num_batches = config['eval_batches']

    # configure mixed precision training, which means:
    # FP32 weights + BF16 forward pass + FP32 optimizer
    use_amp = config.get('use_amp', False)
    amp_dtype = torch.bfloat16 if use_amp else torch.float32

    with torch.no_grad():
        val_loader_iter = iter(val_loader)
        for _ in range(num_batches):
            try:
                inputs, targets = next(val_loader_iter)
            except StopIteration:
                val_loader_iter = iter(val_loader)
                inputs, targets = next(val_loader_iter)

            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
                logits = eval_model(inputs)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            total_loss += loss.item()

    avg_loss = total_loss / num_batches
    perplexity = np.exp(avg_loss)
    return avg_loss, perplexity


def muon_lr_schedule(
    iteration: int,
    max_muon_lr: float,
    min_muon_lr: float,
    warmup_iter: int,
    cos_iter: int
) -> float:
    """
    Cosine learning rate schedule with warmup for Muon optimizer.
    
    Muon typically uses higher learning rates (0.01-0.05) compared to AdamW.
    """
    return cos_learning_rate_schedule_with_warmup(
        iteration, max_muon_lr, min_muon_lr, warmup_iter, cos_iter
    )


def main():
    parser = argparse.ArgumentParser(description='Train Transformer LM with Muon + AdamW')
    parser.add_argument('--config', type=str, required=True, help='Path to config JSON file')
    parser.add_argument('--resume', type=str, default=None,  help='Path to checkpoint to resume from')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='Number of gradient accumulation steps to simulate larger batch size')
    # Muon-specific arguments (can be overridden from config file)
    parser.add_argument('--muon_lr', type=float, default=None,
                        help='Learning rate for Muon optimizer (default: from config or 0.02)')
    parser.add_argument('--muon_momentum', type=float, default=None,
                        help='Momentum for Muon optimizer (default: from config or 0.95)')
    parser.add_argument('--muon_ns_steps', type=int, default=None,
                        help='Newton-Schulz iterations for Muon (default: from config or 5)')
    args = parser.parse_args()

    # Load configuration using Config class
    config = Config.from_json(args.config)
    
    # Get Muon parameters from config file or command line (command line takes precedence)
    muon_lr = args.muon_lr if args.muon_lr is not None else getattr(config, 'muon_lr', 0.02)
    muon_min_lr = getattr(config, 'muon_min_lr', muon_lr * 0.1)  # Default: 10% of max
    muon_momentum = args.muon_momentum if args.muon_momentum is not None else getattr(config, 'muon_momentum', 0.95)
    muon_nesterov = getattr(config, 'muon_nesterov', True)
    muon_ns_steps = args.muon_ns_steps if args.muon_ns_steps is not None else getattr(config, 'muon_ns_steps', 5)
    muon_weight_decay = getattr(config, 'muon_weight_decay', 0.0)

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

    # Create extended config dict for wandb (config is still Config object)
    wandb_config = config.to_dict()
    wandb_config['gradient_accumulation_steps'] = args.gradient_accumulation_steps
    wandb_config['effective_batch_size'] = config.batch_size * args.gradient_accumulation_steps
    wandb_config['optimizer'] = 'Muon+AdamW'
    wandb_config['muon_lr'] = muon_lr
    wandb_config['muon_min_lr'] = muon_min_lr
    wandb_config['muon_momentum'] = muon_momentum
    wandb_config['muon_nesterov'] = muon_nesterov
    wandb_config['muon_ns_steps'] = muon_ns_steps
    wandb_config['muon_weight_decay'] = muon_weight_decay

    wandb.init(
        project="Transformer_LLM",
        entity="scut_zeno",
        name=f"{config.run_name}_Muon",
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
    config.vocab_size = vocab_size

    # Prepare data and cache config dict for later use
    config_dict = config.to_dict()
    train_data, valid_data = prepare_data(config_dict)

    num_workers = config.num_workers
    use_workers = num_workers > 0

    # Create training dataset and loader
    train_dataset = PretrainDataset(data=train_data, context_length=config.context_length)
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=use_workers,
        persistent_workers=use_workers, prefetch_factor=4 if use_workers else None,
        drop_last=True,
    )

    # Create validation dataset and loader
    valid_dataset = PretrainDataset(data=valid_data, context_length=config.context_length)
    valid_loader = DataLoader(
        valid_dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=use_workers,
        persistent_workers=use_workers, prefetch_factor=4 if use_workers else None,
        drop_last=False,
    )

    print(f"Training dataset: {len(train_dataset):,} samples")
    print(f"Validate dataset: {len(valid_dataset):,} samples")
    print(f"DataLoaders initialized successfully!\n")

    # Initialize model with FP32 weights (AMP handles BF16 forward pass)
    model = TransformerLM(config=config, device=device, dtype=torch.float32).to(device)
    
    # count trainable parameters of the model
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Compile the model with torch.compile for better performance
    # - mode="default": Balanced optimization between compilation time and performance
    # - fullgraph=False: Allow graph breaks for dynamic ops (bincount in MoE, etc.)
    # - dynamic=True: Support dynamic tensor shapes
    print("Compiling model with torch.compile...")
    model = torch.compile(model, mode="default", fullgraph=False, dynamic=True)
    print("Model compiled successfully")

    # Initialize Muon + AdamW optimizer
    original_model = model._orig_mod if hasattr(model, '_orig_mod') else model
    
    optimizer = MuonAdamWOptimizer(
        model=original_model,
        muon_lr=muon_lr,
        adamw_lr=config.max_lr,
        muon_momentum=muon_momentum,
        muon_nesterov=muon_nesterov,
        muon_ns_steps=muon_ns_steps,
        muon_weight_decay=muon_weight_decay,
        adamw_betas=(config.beta1, config.beta2),
        adamw_eps=config.eps,
        adamw_weight_decay=config.weight_decay,
    )
    
    # Store min learning rates for schedule
    adamw_min_lr = config.min_lr

    start_iteration = 0  # initialize training state

    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except Exception as e:
                print(f"Warning: Could not load optimizer state: {e}")
                print("Starting with fresh optimizer state.")
        start_iteration = checkpoint.get('iteration', 0)
        print(f"Resumed from iteration {start_iteration}")

    # config model type
    attention_type = config.attention_type
    use_moe = config.use_moe
    ffn_type = 'MoE' if use_moe else 'FFN'
    module_config = f"{attention_type}+{ffn_type}"
    
    # config ckpt directory
    dataset_name = config.dataset
    checkpoint_folder_name = f"{dataset_name}_{module_config}[Muon]"
    checkpoint_dir = Path(config.checkpoint_dir) / checkpoint_folder_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"Checkpoint directory: {checkpoint_dir}")

    # get gradient accumulation steps from command line argument
    gradient_accumulation_steps = args.gradient_accumulation_steps
    effective_batch_size = config.batch_size * gradient_accumulation_steps
    print(f"Gradient Accumulation: {gradient_accumulation_steps} steps")
    print(f"Micro Batch Size: {config.batch_size}")
    print(f"Effective Batch Size: {effective_batch_size}")

    record_file_path = checkpoint_dir / "record.txt" # create record file path

    # initialize record file with header and config
    with open(record_file_path, 'w') as record_file:
        record_file.write(f"Training Record for {config.dataset} (Muon + AdamW)\n")
        record_file.write("=" * 80 + "\n")
        record_file.write(f"Model: {config.run_name}\n")
        record_file.write(f"Optimizer: Muon + AdamW\n")
        record_file.write(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        record_file.write(f"Config file: {args.config}\n")
        record_file.write("=" * 80 + "\n\n")

        # Write Muon configuration
        record_file.write("MUON CONFIGURATION:\n")
        record_file.write("-" * 80 + "\n")
        record_file.write(f"Muon LR (max): {muon_lr}\n")
        record_file.write(f"Muon LR (min): {muon_min_lr}\n")
        record_file.write(f"Muon Momentum: {muon_momentum}\n")
        record_file.write(f"Muon Nesterov: {muon_nesterov}\n")
        record_file.write(f"Muon NS Steps: {muon_ns_steps}\n")
        record_file.write(f"Muon Weight Decay: {muon_weight_decay}\n")
        record_file.write(f"AdamW LR (max): {config.max_lr}\n")
        record_file.write(f"AdamW LR (min): {adamw_min_lr}\n")
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
        record_file.write(f"ATT Type: [{attention_type}]   MLP Type: [{ffn_type}]\n")
        record_file.write(f"Total parameters: {total_params:,}\n")
        record_file.write(f"Trainable parameters: {trainable_params:,}\n")
        record_file.write(f"Gradient Accumulation Steps: {gradient_accumulation_steps}\n")
        record_file.write(f"Micro Batch Size: {config.batch_size}\n")
        record_file.write(f"Effective Batch Size: {effective_batch_size}\n")
        record_file.write("-" * 80 + "\n\n")

    print("Starting training with Muon + AdamW...")
    print("-" * 60)

    # Training loop
    model.train()
    running_loss = 0.0
    best_val_loss = float('+inf')
    best_val_ppl  = float('+inf')

    # create an infinite iterator from the training DataLoader
    # to ensures we never run out of data during training
    train_loader_iter = iter(train_loader)

    for iteration in range(start_iteration, config.max_iterations):
        start_time = time.time()

        # update learning rate with cosine schedule
        # Compute separate schedules for Muon and AdamW
        current_muon_lr = cos_learning_rate_schedule_with_warmup(
            iteration,
            max_lr=muon_lr,
            min_lr=muon_min_lr,
            warmup_iter=config.warmup_iterations,
            cos_iter=config.max_iterations
        )
        current_adamw_lr = cos_learning_rate_schedule_with_warmup(
            iteration,
            max_lr=config.max_lr,
            min_lr=adamw_min_lr,
            warmup_iter=config.warmup_iterations,
            cos_iter=config.max_iterations
        )
        optimizer.set_lr_absolute(current_muon_lr, current_adamw_lr)

        # Gradient accumulation: perform multiple forward/backward passes
        accumulated_loss = 0.0
        for accum_step in range(gradient_accumulation_steps):
            try:
                loss, grad_norm = train(
                    model, optimizer, train_loader_iter, config_dict, device,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    accumulation_step=accum_step
                )
            except StopIteration:
                train_loader_iter = iter(train_loader)
                loss, grad_norm = train(
                    model, optimizer, train_loader_iter, config_dict, device,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    accumulation_step=accum_step
                )
            accumulated_loss += loss

        # Average loss over accumulation steps
        avg_accum_loss = accumulated_loss / gradient_accumulation_steps
        running_loss += avg_accum_loss

        step_time = time.time() - start_time

        # Log training metrics
        if (iteration + 1) % config.log_interval == 0:
            avg_loss = running_loss / config.log_interval
            perplexity = np.exp(avg_loss)

            content = f"Iter {iteration + 1:6d} | Loss: {avg_loss:.4f} | PPL: {perplexity:.2f} | " \
                      f"Muon_LR: {current_muon_lr:.6f} | AdamW_LR: {current_adamw_lr:.6f} | Grad Norm: {grad_norm:.4f} | Time: {step_time:.3f}s"
            print(content)

            # Save training content to record file
            with open(record_file_path, 'a') as record_file:
                record_file.write(f"[TRAIN] {content}\n")
            
            wandb.log({
                'train/loss': avg_loss,
                'train/perplexity': perplexity,
                'train/muon_lr': current_muon_lr,
                'train/adamw_lr': current_adamw_lr,
                'train/grad_norm': grad_norm,
                'train/step_time': step_time
            }, step=iteration + 1)

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
            # Custom save for Muon + AdamW optimizer
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'iteration': iteration + 1,
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_ppl = val_perplexity
                best_checkpoint_path = checkpoint_dir / "best_model.pt"
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'iteration': iteration + 1,
                }, best_checkpoint_path)
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

    # Print final results
    print(f"\nFinal validation loss: {best_val_loss:.4f}   Final PPL: {best_val_ppl:.2f}")

    # Save final checkpoint
    final_checkpoint_path = checkpoint_dir / "final_model.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': config.max_iterations,
    }, final_checkpoint_path)
    final_checkpoint_content = f"Final checkpoint saved: {final_checkpoint_path}"
    print(final_checkpoint_content)

    # Save final checkpoint info to record file
    with open(record_file_path, 'a') as record_file:
        record_file.write(f"[FINAL] {final_checkpoint_content}\n")

    wandb.finish()


if __name__ == "__main__":
    main()
