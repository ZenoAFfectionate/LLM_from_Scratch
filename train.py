import os
import sys
import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any, Iterator

import numpy as np
import torch
import torch.nn as nn
import wandb

from cs336_basics.transformer import TransformerLM
from cs336_basics.bpe_tokenizer import Tokenizer
from cs336_basics.optimizer import AdamW, cos_learning_rate_schedule_with_warmup
from cs336_basics.utils import (
    data_loading, cross_entropy, gradient_clipping,
    save_checkpoint, load_checkpoint, data_loading
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


def valid(model: nn.Module, val_data: np.ndarray, config: Dict[str, Any], device: torch.device):
    """Evaluate model on validation data"""
    model.eval()
    total_loss = 0.0
    num_batches = config['eval_batches']

    with torch.no_grad():
        for _ in range(num_batches):
            inputs, targets = data_loading(
                val_data,
                config['batch_size'],
                config['context_length'],
                device
            )
            # 
            logits = model(inputs)
            logits_flat = logits.view(-1, logits.size(-1))
            targets_flat = targets.view(-1)
            # 
            loss = cross_entropy(logits_flat, targets_flat)
            total_loss += loss.item()

    avg_loss = total_loss / num_batches
    perplexity = np.exp(avg_loss)
    return avg_loss, perplexity


def train(model: nn.Module, optimizer: torch.optim.Optimizer,
          train_data: np.ndarray, config: Dict[str, Any], device: torch.device):
    """Perform a single training step"""
    model.train()

    inputs, targets = data_loading(
        train_data,
        config['batch_size'],
        config['context_length'],
        device
    )

    # =============
    # Forward pass
    # =============
    logits = model(inputs)
    logits_flat = logits.view(-1, logits.size(-1))
    targets_flat = targets.view(-1)
    loss = cross_entropy(logits_flat, targets_flat)

    # ==============
    # Backward pass
    # ==============
    optimizer.zero_grad()
    loss.backward()
    grad_norm = gradient_clipping(model.parameters(), config['max_grad_norm'])
    optimizer.step()

    return loss.item(), grad_norm.item()


def main():
    parser = argparse.ArgumentParser(description='Train Transformer Language Model')
    parser.add_argument('--config', type=str, required=True, help='Path to config JSON file')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['seed'])

    wandb.init(
        project="Transformer_LLM",
        entity="scut_zeno",
        name=config.get('run_name', 'transformer_training'),
        config=config
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

    # Initialize model
    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=config['context_length'],
        d_model=config['d_model'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        d_ff=config['d_ff'],
        rope_theta=config['rope_theta'],
        drop_p=config.get('dropout', None),
        device=device
    ).to(device)

    # count the trainable parameters of current model
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Initialize optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config['max_lr'],
        betas=(config['beta1'], config['beta2']),
        eps=config['eps'],
        weight_decay=config['weight_decay']
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

    # Create record file path for logging training and validation content
    record_file_path = checkpoint_dir / "record.txt"

    # Initialize record file with header
    with open(record_file_path, 'w') as record_file:
        record_file.write(f"Training Record for {config['dataset']}\n")
        record_file.write("=" * 50 + "\n")
        record_file.write(f"Model: {config.get('run_name', 'transformer_training')}\n")
        record_file.write(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        record_file.write("=" * 50 + "\n\n")

    print("Starting training...")
    print("-" * 60)

    # Training loop
    model.train()
    running_loss = 0.0
    best_val_loss = float('inf')

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

        # Training step
        loss, grad_norm = train(model, optimizer, train_data, config, device)
        running_loss += loss

        step_time = time.time() - start_time

        # Log training metrics
        if (iteration + 1) % config['log_interval'] == 0:
            avg_loss = running_loss / config['log_interval']
            perplexity = np.exp(avg_loss)

            content = f"Iter {iteration + 1:6d} | Loss: {avg_loss:.4f} | PPL: {perplexity:.2f} | " \
                      f"LR: {lr:.6f} | Grad Norm: {grad_norm:.4f} | Time: {step_time:.3f}s"
            print(content)

            # Save training content to record file
            with open(record_file_path, 'a') as record_file:
                record_file.write(f"[TRAIN] {content}\n")
            

            # Log to wandb
            wandb.log({
                'train/loss': avg_loss,
                'train/perplexity': perplexity,
                'train/learning_rate': lr,
                'train/grad_norm': grad_norm,
                'train/step_time': step_time
            }, step=iteration + 1)

            running_loss = 0.0

        # Validation and checkpointing
        if (iteration + 1) % config['eval_interval'] == 0:
            print("Running validation...")
            val_loss, val_perplexity = valid(model, valid_data, config, device)

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
                best_checkpoint_path = checkpoint_dir / "best_model.pt"
                save_checkpoint(model, optimizer, iteration + 1, str(best_checkpoint_path))
                best_model_content = f"New best model saved: {best_checkpoint_path} (val_loss: {val_loss:.4f})"
                print(best_model_content)

                # Save best model info to record file
                with open(record_file_path, 'a') as record_file:
                    record_file.write(f"[BEST] {best_model_content}\n")

                # Log best model to wandb
                wandb.log({'val/best_loss': best_val_loss}, step=iteration + 1)

            model.train()  # Switch back to training mode

    print("Training completed!")

    # Save training completion info to record file
    with open(record_file_path, 'a') as record_file:
        record_file.write(f"\n{'='*50}\n")
        record_file.write(f"Training completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        record_file.write(f"Total iterations: {config['max_iterations']}\n")
        record_file.write(f"Final validation loss: {best_val_loss:.4f}\n")
        record_file.write(f"{'='*50}\n")

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
