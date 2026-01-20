import os
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any, Iterator

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from model.transformer import TransformerLM
from model.tokenizer.bpe_tokenizer import Tokenizer
from data.lm_dataset import PretrainDataset
from model.mtp import MultiTokenPredictor
from model.utils import (
    save_checkpoint, load_checkpoint,
    cos_learning_rate_schedule_with_warmup
)


def load_data_memmap(data_path: str, dtype=np.int32):
    """Load training data using memory mapping for efficient access"""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    # Load data using memmap for memory efficiency
    data = np.memmap(data_path, dtype=dtype, mode='r')
    print(f"Loaded {len(data):,} tokens from {data_path}")
    return data


def read_text_chunks(file_path: str, chunk_size: int = 1024*1024) -> Iterator[str]:
    """Generator that yields text chunks for memory-efficient processing"""
    with open(file_path, 'r', encoding='utf-8') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk: break
            yield chunk


def tokenize_and_save_data(text_file: str, tokenizer: Tokenizer, output_file: str, chunk_size: int = 1024*1024):
    """Memory-efficient tokenization using encode_iterable"""
    if os.path.exists(output_file):
        print(f"Tokenized data already exists: {output_file}")
        return

    print(f"Tokenizing {text_file} using memory-efficient streaming...")

    # Get file size for progress tracking
    file_size = os.path.getsize(text_file)
    print(f"File size: {file_size:,} bytes")

    # Use encode_iterable for memory-efficient tokenization
    text_chunks = read_text_chunks(text_file, chunk_size)

    # Process tokens in batches to avoid memory issues
    token_batch_size = 10_000_000  # Process 10M tokens at a time
    token_buffer = []
    total_tokens = 0

    with open(output_file, 'wb') as out_f:
        for token_id in tokenizer.encode_iterable(text_chunks):
            token_buffer.append(token_id)

            # Write batch when buffer is full
            if len(token_buffer) >= token_batch_size:
                token_array = np.array(token_buffer, dtype=np.int32)
                token_array.tofile(out_f)
                total_tokens += len(token_buffer)
                print(f"Processed {total_tokens:,} tokens...")
                token_buffer = []

        # Write remaining tokens
        if token_buffer:
            token_array = np.array(token_buffer, dtype=np.int32)
            token_array.tofile(out_f)
            total_tokens += len(token_buffer)

    print(f"Saved {total_tokens:,} tokens to {output_file}")


def prepare_data(config: Dict[str, Any], tokenizer: Tokenizer):
    """Prepare training and validation data with automatic method selection"""
    data_dir = Path(config['data_dir'])

    # Paths for text and tokenized data
    train_text = data_dir / config['train_file']
    valid_text = data_dir / config['valid_file']
    train_bin = data_dir / f"tokens_train.bin"  # tokenized train data
    valid_bin = data_dir / f"tokens_valid.bin"  # tokenized valid data

    # Process training data
    print("Using memory-efficient tokenization for train data...")
    tokenize_and_save_data(str(train_text), tokenizer, str(train_bin))

    # Process validation data
    print("Using memory-efficient tokenization for valid data...")
    tokenize_and_save_data(str(valid_text), tokenizer, str(valid_bin))

    # Load tokenized data with memmap
    train_data = load_data_memmap(str(train_bin))
    valid_data = load_data_memmap(str(valid_bin))

    return train_data, valid_data


def valid(model: nn.Module, val_loader: DataLoader, config: Dict[str, Any], device: torch.device):
    """Evaluate model on validation data with BF16 autocasting"""
    # Use ._orig_mod to access the original uncompiled model for validation
    # This avoids CUDA graph capture issues during eval mode
    eval_model = model._orig_mod if hasattr(model, '_orig_mod') else model
    eval_model.eval()

    total_loss = 0.0
    num_batches = config['eval_batches']

    # Configure mixed precision
    use_amp = config.get('use_amp', False)
    amp_dtype = torch.bfloat16 if use_amp else torch.float32

    with torch.no_grad():
        # Create an iterator from the DataLoader
        val_loader_iter = iter(val_loader)
        for _ in range(num_batches):
            try:
                inputs, targets = next(val_loader_iter)
                # Move to device with non_blocking for async transfer
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
            except StopIteration:
                # If we run out of validation data, restart the iterator
                val_loader_iter = iter(val_loader)
                inputs, targets = next(val_loader_iter)
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

            # Use autocast for BF16 inference
            with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
                logits = eval_model(inputs)
                logits_flat = logits.view(-1, logits.size(-1))
                targets_flat = targets.view(-1)
                # compute cross entropy loss
                loss = F.cross_entropy(logits_flat, targets_flat)
            total_loss += loss.item()

    avg_loss = total_loss / num_batches
    perplexity = np.exp(avg_loss)
    return avg_loss, perplexity


def compute_mtp_loss(
    depth_representations: list,
    token_ids: torch.Tensor,
    lm_head: nn.Module,
    final_norm: nn.Module,
    mtp_lambda: float = 0.1
) -> torch.Tensor:
    """
    Compute Multi-Token Prediction (MTP) loss across all depths.

    Args:
        depth_representations: List of representations at each depth [h_1, h_2, ..., h_D]
        token_ids: Input token IDs, shape (batch, seq_len)
        lm_head: Shared output head (language model head)
        final_norm: Final RMSNorm layer (shared with main model)
        mtp_lambda: Weighting factor for MTP loss (λ in paper)

    Returns:
        Total MTP loss: λ * (1/D) * Σ L_k_MTP
    """
    if not depth_representations:
        return torch.tensor(0.0, device=token_ids.device)

    num_depths = len(depth_representations)
    total_mtp_loss = 0.0

    for k, h_k in enumerate(depth_representations):
        # At depth k, we predict tokens at positions [2+k, 3+k, ..., T+k]
        # h_k has shape (batch, seq_len - k - 1, d_model)
        # We need target tokens at positions [2+k, 3+k, ..., T] in original sequence

        # Get sequence length at this depth
        batch_size, depth_seq_len, d_model = h_k.shape

        # Apply final normalization and output head
        h_k_norm = final_norm(h_k)  # (batch, depth_seq_len, d_model)
        logits_k = lm_head(h_k_norm)  # (batch, depth_seq_len, vocab_size)

        # Get target tokens: tokens at positions [2+k : T+1] (0-indexed: [k+2 : seq_len+1])
        # For k=0: targets are tokens[2:], for k=1: targets are tokens[3:], etc.
        target_start_idx = k + 2
        targets_k = token_ids[:, target_start_idx : target_start_idx + depth_seq_len]

        # Flatten for cross-entropy
        logits_k_flat = logits_k.reshape(-1, logits_k.size(-1))  # (batch * depth_seq_len, vocab_size)
        targets_k_flat = targets_k.reshape(-1)  # (batch * depth_seq_len)

        # Compute cross-entropy loss for this depth
        loss_k = F.cross_entropy(logits_k_flat, targets_k_flat)
        total_mtp_loss += loss_k

    # Average across depths and multiply by lambda
    avg_mtp_loss = total_mtp_loss / num_depths
    weighted_mtp_loss = mtp_lambda * avg_mtp_loss

    return weighted_mtp_loss


def train(model: nn.Module, mtp_predictor: nn.Module, optimizer: torch.optim.Optimizer,
          train_loader_iter: Iterator, config: Dict[str, Any], device: torch.device,
          gradient_accumulation_steps: int = 1, accumulation_step: int = 0):
    """Perform a single training step with MTP, BF16 mixed precision and gradient accumulation

    Args:
        model: The model to train
        mtp_predictor: Multi-token predictor module (or None)
        optimizer: The optimizer
        train_loader_iter: Iterator from the training DataLoader
        config: Configuration dictionary
        device: Device to run on
        gradient_accumulation_steps: Number of steps to accumulate gradients over
        accumulation_step: Current accumulation step (0 to gradient_accumulation_steps-1)

    Returns:
        Tuple of (main_loss, mtp_loss, total_loss, grad_norm)
    """
    model.train()
    if mtp_predictor is not None:
        mtp_predictor.train()

    # Get next batch from DataLoader iterator
    inputs, targets = next(train_loader_iter)
    # Move to device with non_blocking for async transfer
    inputs = inputs.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)

    # Configure mixed precision training
    use_amp = config.get('use_amp', False)
    amp_dtype = torch.bfloat16 if use_amp else torch.float32

    # ====================================
    # Forward pass with autocast for BF16
    # ====================================
    with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
        # main model forward and compute loss
        logits = model(inputs)
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = targets.view(-1)
        main_loss = F.cross_entropy(logits_flat, targets_flat)

        # mtp forward and compute loss
        mtp_loss = torch.tensor(0.0, device=device)

        if mtp_predictor is not None and config.get('use_mtp', False):
            # Get main model representations (before final LM head)
            # We need to recompute to get intermediate representations
            # Access the original uncompiled model if compiled
            original_model = model._orig_mod if hasattr(model, '_orig_mod') else model

            x = original_model.token_embeddings(inputs)
            residual = None
            for block in original_model.layers:
                x, residual = block(x, residual, start_pos=0, mask=None)
            # Apply final norm
            x, _ = original_model.final_norm(x, residual)
            h_main = x  # (batch, seq_len, d_model)

            # Forward through MTP predictor
            depth_representations = mtp_predictor(
                h_main=h_main,
                token_ids=inputs,
                embedding_layer=original_model.token_embeddings
            )

            # Compute MTP loss
            mtp_loss = compute_mtp_loss(
                depth_representations=depth_representations,
                token_ids=inputs,
                lm_head=original_model.lm_head,
                final_norm=original_model.final_norm,
                mtp_lambda=config.get('mtp_lambda', 0.1)
            )

        total_loss = main_loss + mtp_loss
        # Scale for gradient accumulation
        total_loss = total_loss / gradient_accumulation_steps

    # =========================================
    # Backward pass with gradient accumulation
    # =========================================
    if accumulation_step == 0: optimizer.zero_grad(set_to_none=True)

    total_loss.backward()

    # Only update weights and clip gradients on the last accumulation step
    grad_norm = torch.tensor(0.0)
    if accumulation_step == gradient_accumulation_steps - 1:
        # Clip gradients for both model and MTP predictor
        params_to_clip = list(model.parameters())
        if mtp_predictor is not None:
            params_to_clip += list(mtp_predictor.parameters())
        grad_norm = clip_grad_norm_(params_to_clip, config['max_grad_norm'])
        optimizer.step()

        # Update expert biases for load balance
        if hasattr(model, 'update_moe_biases'):
            model.update_moe_biases()

    return main_loss.item(), mtp_loss.item(), total_loss.item() * gradient_accumulation_steps, grad_norm.item()


def main():
    parser = argparse.ArgumentParser(description='Train Transformer Language Model with MTP')
    parser.add_argument('--config', type=str, required=True, help='Path to config JSON file')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='Number of gradient accumulation steps to simulate larger batch size')
    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Set performance optimizations for PyTorch
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')  # enable TF32
        torch.backends.cudnn.benchmark = True       # enable cuDNN benchmark
        print("Performance optimizations enabled: TF32 matmul, cuDNN benchmark")

    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['seed'])

    # Create extended config with gradient accumulation info for wandb
    wandb_config = config.copy()
    wandb_config['gradient_accumulation_steps'] = args.gradient_accumulation_steps
    wandb_config['effective_batch_size'] = config['batch_size'] * args.gradient_accumulation_steps

    wandb.init(
        project="Transformer_LLM",
        entity="scut_zeno",
        name=config.get('run_name', 'transformer_training'),
        config=wandb_config
    )

    # Load tokenizer
    tokenizer = Tokenizer.from_files(
        vocab_filepath=config['vocab_file'],
        merges_filepath=config['merges_file'],
        special_tokens=config.get('special_tokens', ['<|endoftext|>'])
    )

    vocab_size = len(tokenizer.decoder_vocab)
    print(f"Vocabulary size: {vocab_size:,}")

    train_data, valid_data = prepare_data(config, tokenizer)

    num_workers = config.get('num_workers', 8)

    # Create training dataset and loader
    train_dataset = PretrainDataset(
        data=train_data,
        context_length=config['context_length']
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if num_workers > 0 else False,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=4 if num_workers > 0 else None,
        drop_last=True,
    )

    # Create validation dataset and loader
    valid_dataset = PretrainDataset(
        data=valid_data,
        context_length=config['context_length']
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if num_workers > 0 else False,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=4 if num_workers > 0 else None,
        drop_last=False,
    )

    print(f"Training dataset: {len(train_dataset):,} samples")
    print(f"Validation dataset: {len(valid_dataset):,} samples")
    print(f"DataLoaders initialized successfully!\n")

    model_dtype = torch.float32  # Always use FP32 for model weights

    # Initialize model with explicit dtype
    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=config['context_length'],
        d_model=config['d_model'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        d_ff=config['d_ff'],
        rope_theta=config['rope_theta'],
        drop_p=config.get('dropout', 0.0),
        use_moe=config.get('use_moe', False),
        moe_layers=config.get('moe_layers', None),
        n_routed_experts=config.get('n_routed_experts', 8),
        num_experts_per_tok=config.get('num_experts_per_tok', 2),
        n_shared_experts=config.get('n_shared_experts', 0),
        aux_seq_loss_alpha=config.get('aux_loss_alpha', 0.01),
        num_kv_heads=config.get('num_kv_heads', config['num_heads']),
        attention_type=config.get('attention_type', 'GQA'),
        d_rope=config.get('d_rope', None),
        kv_lora_rank=config.get('kv_lora_rank', None),
        q_lora_rank=config.get('q_lora_rank', None),
        device=device,
        dtype=model_dtype  # Pass dtype to model initialization
    ).to(device)

    # Print model configuration (simplified)
    attention_type = config.get('attention_type', 'GQA')
    mlp_type = 'MoE' if config.get('use_moe', False) else 'FFN'
    print(f"ATT Type: [{attention_type}]   MLP Type: [{mlp_type}]")

    # count the trainable parameters of current model
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Initialize MTP predictor if enabled
    mtp_predictor = None
    if config.get('use_mtp', False):
        print("\nInitializing Multi-Token Prediction (MTP) modules...")
        mtp_predictor = MultiTokenPredictor(
            num_depths=config.get('mtp_num_depths', 2),
            d_model=config['d_model'],
            num_heads=config['num_heads'],
            d_ff=config['d_ff'],
            rope=model.rope,
            drop_p=config.get('dropout', 0.0),
            use_moe=config.get('use_moe', False),
            moe_layers=config.get('moe_layers', None),
            n_routed_experts=config.get('n_routed_experts', 8),
            num_experts_per_tok=config.get('num_experts_per_tok', 2),
            n_shared_experts=config.get('n_shared_experts', 0),
            aux_seq_loss_alpha=config.get('aux_loss_alpha', 0.01),
            num_kv_heads=config.get('num_kv_heads', config['num_heads']),
            attention_type=config.get('attention_type', 'GQA'),
            d_rope=config.get('d_rope', None),
            kv_lora_rank=config.get('kv_lora_rank', None),
            q_lora_rank=config.get('q_lora_rank', None),
            device=device
        ).to(device)

        mtp_params = sum(p.numel() for p in mtp_predictor.parameters())
        print(f"MTP parameters: {mtp_params:,}")
        print(f"Total model + MTP parameters: {total_params + mtp_params:,}")
        print(f"MTP lambda: {config.get('mtp_lambda', 0.1)}")
        print(f"MTP depths: {config.get('mtp_num_depths', 2)}")

    # ============================================================================
    # OPTIMIZED: torch.compile configuration for MoE models
    # ============================================================================
    print("Compiling model with torch.compile...")
    model = torch.compile(model, mode='default', dynamic=True, fullgraph=False)
    print("Model compiled successfully")

    # Initialize optimizer (include MTP parameters if enabled)
    params_to_optimize = list(model.parameters())
    if mtp_predictor is not None:
        params_to_optimize += list(mtp_predictor.parameters())

    optimizer = AdamW(
        params_to_optimize,
        lr=config['max_lr'],
        betas=(config['beta1'], config['beta2']),
        eps=config['eps'],
        weight_decay=config['weight_decay'],
        fused=True
    )

    # Initialize training state
    start_iteration = 0

    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        start_iteration = load_checkpoint(args.resume, model, optimizer)
        print(f"Resumed from iteration {start_iteration}")

    # Create checkpoint directory
    checkpoint_dir = Path(config['checkpoint_dir']) / config['dataset']
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Get gradient accumulation steps from command line argument
    gradient_accumulation_steps = args.gradient_accumulation_steps
    effective_batch_size = config['batch_size'] * gradient_accumulation_steps
    print(f"Gradient Accumulation: {gradient_accumulation_steps} steps")
    print(f"Micro Batch Size: {config['batch_size']}")
    print(f"Effective Batch Size: {effective_batch_size}")

    # Create record file path for logging training and validation content
    record_file_path = checkpoint_dir / "record.txt"

    # Initialize record file with header and config
    with open(record_file_path, 'w') as record_file:
        record_file.write(f"Training Record for {config['dataset']}\n")
        record_file.write("=" * 80 + "\n")
        record_file.write(f"Model: {config.get('run_name', 'transformer_training')}\n")
        record_file.write(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        record_file.write(f"Config file: {args.config}\n")
        record_file.write("=" * 80 + "\n\n")

        # Write full configuration
        record_file.write("CONFIGURATION:\n")
        record_file.write("-" * 80 + "\n")
        record_file.write(json.dumps(config, indent=2))
        record_file.write("\n" + "-" * 80 + "\n\n")

        # Write model architecture summary
        record_file.write("MODEL ARCHITECTURE:\n")
        record_file.write("-" * 80 + "\n")
        record_file.write(f"ATT Type: [{attention_type}]   MLP Type: [{mlp_type}]\n")
        record_file.write(f"Total parameters: {total_params:,}\n")
        record_file.write(f"Trainable parameters: {trainable_params:,}\n")
        record_file.write(f"Gradient Accumulation Steps: {gradient_accumulation_steps}\n")
        record_file.write(f"Micro Batch Size: {config['batch_size']}\n")
        record_file.write(f"Effective Batch Size: {effective_batch_size}\n")
        record_file.write("-" * 80 + "\n\n")

    print("Starting training...")
    print("-" * 60)

    # Training loop
    model.train()
    running_loss = 0.0
    running_main_loss = 0.0
    running_mtp_loss = 0.0
    best_val_loss = float('inf')
    best_val_ppl = float('inf')

    # Create an infinite iterator from the training DataLoader
    # This ensures we never run out of data during training
    train_loader_iter = iter(train_loader)

    for iteration in range(start_iteration, config['max_iterations']):
        start_time = time.time()

        # update learning rate
        lr = cos_learning_rate_schedule_with_warmup(
            iteration,
            max_lr=config['max_lr'],
            min_lr=config['min_lr'],
            warmup_iter=config['warmup_iterations'],
            cos_iter=config['max_iterations']
        )
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Gradient accumulation: perform multiple forward/backward passes
        accumulated_loss = 0.0
        accumulated_main_loss = 0.0
        accumulated_mtp_loss = 0.0
        for accum_step in range(gradient_accumulation_steps):
            try:
                main_loss, mtp_loss, total_loss, grad_norm = train(
                    model, mtp_predictor, optimizer, train_loader_iter, config, device,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    accumulation_step=accum_step
                )
            except StopIteration:
                # If we exhaust the DataLoader, create a new iterator
                train_loader_iter = iter(train_loader)
                main_loss, mtp_loss, total_loss, grad_norm = train(
                    model, mtp_predictor, optimizer, train_loader_iter, config, device,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    accumulation_step=accum_step
                )
            accumulated_loss += total_loss
            accumulated_main_loss += main_loss
            accumulated_mtp_loss += mtp_loss

        # Average loss over accumulation steps
        avg_accum_loss = accumulated_loss / gradient_accumulation_steps
        avg_accum_main_loss = accumulated_main_loss / gradient_accumulation_steps
        avg_accum_mtp_loss = accumulated_mtp_loss / gradient_accumulation_steps

        running_loss += avg_accum_loss
        running_main_loss += avg_accum_main_loss
        running_mtp_loss += avg_accum_mtp_loss

        step_time = time.time() - start_time

        # Log training metrics
        if (iteration + 1) % config['log_interval'] == 0:
            avg_loss = running_loss / config['log_interval']
            avg_main_loss = running_main_loss / config['log_interval']
            avg_mtp_loss = running_mtp_loss / config['log_interval']
            perplexity = np.exp(avg_main_loss)

            if config.get('use_mtp', False):
                content = f"Iter {iteration + 1:6d} | Loss: {avg_loss:.4f} (Main: {avg_main_loss:.4f}, MTP: {avg_mtp_loss:.4f}) | " \
                          f"PPL: {perplexity:.2f} | LR: {lr:.6f} | Grad Norm: {grad_norm:.4f} | Time: {step_time:.3f}s"
            else:
                content = f"Iter {iteration + 1:6d} | Loss: {avg_loss:.4f} | PPL: {perplexity:.2f} | " \
                          f"LR: {lr:.6f} | Grad Norm: {grad_norm:.4f} | Time: {step_time:.3f}s"
            print(content)

            # Save training content to record file
            with open(record_file_path, 'a') as record_file:
                record_file.write(f"[TRAIN] {content}\n")


            # Log to wandb
            log_dict = {
                'train/loss': avg_loss,
                'train/main_loss': avg_main_loss,
                'train/perplexity': perplexity,
                'train/learning_rate': lr,
                'train/grad_norm': grad_norm,
                'train/step_time': step_time
            }
            if config.get('use_mtp', False):
                log_dict['train/mtp_loss'] = avg_mtp_loss

            wandb.log(log_dict, step=iteration + 1)

            running_loss = 0.0
            running_main_loss = 0.0
            running_mtp_loss = 0.0

        # Validation and checkpointing
        if (iteration + 1) % config['eval_interval'] == 0:
            print("Running validation...")
            val_loss, val_perplexity = valid(model, valid_loader, config, device)

            val_content = f"Validation | Loss: {val_loss:.4f} | PPL: {val_perplexity:.2f}"
            print(val_content)

            # Save validation content to record file
            with open(record_file_path, 'a') as record_file:
                record_file.write(f"[VALID] {val_content}\n")

            # Log validation metrics to wandb
            wandb.log({
                'val/loss': val_loss,
                'val/perplexity': val_perplexity
            }, step=iteration + 1)

            # Save checkpoint
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

                # Log best model to wandb
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
        record_file.write(f"Total iterations: {config['max_iterations']}\n")
        record_file.write(f"Final validation loss: {best_val_loss:.4f}   Final PPL: {best_val_ppl:.2f}\n")
        record_file.write(f"{'='*50}\n")

    # Print final results
    print(f"\nFinal validation loss: {best_val_loss:.4f}   Final PPL: {best_val_ppl:.2f}")

    # Save final checkpoint
    final_checkpoint_path = checkpoint_dir / "final_model.pt"
    save_checkpoint(model, optimizer, config['max_iterations'], str(final_checkpoint_path))
    final_checkpoint_content = f"Final checkpoint saved: {final_checkpoint_path}"
    print(final_checkpoint_content)

    # Save final checkpoint info to record file
    with open(record_file_path, 'a') as record_file:
        record_file.write(f"[FINAL] {final_checkpoint_content}\n")

    wandb.finish()


if __name__ == "__main__":
    main()
