"""
Fine-tuning Script for Transformer Language Model

Supports two fine-tuning modes:
1. SFT (Supervised Fine-Tuning): Full parameter fine-tuning
2. LoRA (Low-Rank Adaptation): Parameter-efficient fine-tuning

Use --mode argument to switch between modes:
  --mode sft: Full parameter fine-tuning (default)
  --mode lora: LoRA fine-tuning with frozen base model
"""

import json
import time
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import wandb


from model.config import Config
from model.transformer import TransformerLM
from model.tokenizer.bpe_tokenizer import Tokenizer
from torch.optim import AdamW
# from model.optimizer.AdamW import AdamW
from data.lm_dataset import SFTDataset
from torch.nn.utils import clip_grad_norm_
from model.utils import (
    save_checkpoint, load_checkpoint,
    cos_learning_rate_schedule_with_warmup
)
from Courses.STF_LLM.Assignment_1.utils.lora import (
    apply_lora,
    freeze_non_lora_params,
    get_lora_params,
    save_lora_weights,
)


def load_sft_data(data_path: str, max_samples: int = None):
    """
    Load raw conversation data from JSONL format (without tokenization).
    Tokenization will happen on-the-fly during training for memory efficiency.

    Args:
        data_path: Path to the JSONL file containing conversations
        max_samples: Optional limit on number of conversations to load (for testing)

    Returns:
        List of raw conversation data (list of dicts with 'role' and 'content')
    """
    print(f"Loading raw SFT conversation data from {data_path}...")
    print("NOTE: Using online tokenization - no preprocessing or caching")

    # Count total lines for progress tracking
    print("Counting total conversations...")
    with open(data_path, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
    print(f"Total conversations in file: {total_lines:,}")

    if max_samples is not None:
        print(f"Will load only first {max_samples:,} conversations")
        total_lines = min(total_lines, max_samples)

    raw_conversations = []
    start_time = time.time()

    with open(data_path, 'r', encoding='utf-8') as f:
        for line_idx, line in enumerate(f):
            # Stop if max_samples reached
            if max_samples is not None and line_idx >= max_samples:
                break

            # Progress update every 5000 conversations (faster than before since no tokenization)
            if line_idx % 5000 == 0 and line_idx > 0:
                elapsed = time.time() - start_time
                rate = line_idx / elapsed
                remaining = (total_lines - line_idx) / rate if rate > 0 else 0
                print(f"Progress: {line_idx:,}/{total_lines:,} ({100*line_idx/total_lines:.1f}%) | "
                      f"Rate: {rate:.1f} conv/s | ETA: {remaining/60:.1f} min")

            try:
                data = json.loads(line.strip())
                conversations = data.get('conversations', [])

                # Only store raw conversation data - no tokenization yet
                if conversations:  # Skip empty conversations
                    raw_conversations.append(conversations)

            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line {line_idx}: {e}")
                continue

    elapsed = time.time() - start_time
    print(f"\nLoading completed in {elapsed:.2f} seconds")
    print(f"Loaded {len(raw_conversations)} conversations (raw, not tokenized)")

    return raw_conversations


def train(model: nn.Module, optimizer: torch.optim.Optimizer,
          dataset: SFTDataset, batch_size: int,
          device: torch.device, max_grad_norm: float,
          gradient_accumulation_steps: int = 1, accumulation_step: int = 0,
          use_amp: bool = False, trainable_params=None):
    """
    Perform a single training step with gradient accumulation and loss masking.

    Args:
        model: The model to train
        optimizer: The optimizer
        dataset: SFTDataset instance
        batch_size: Batch size
        device: Device to run on
        max_grad_norm: Maximum gradient norm for clipping
        gradient_accumulation_steps: Number of steps to accumulate gradients over
        accumulation_step: Current accumulation step
        use_amp: Whether to use automatic mixed precision
        trainable_params: Specific parameters to clip (for LoRA mode), None = all params

    Returns:
        Tuple of (loss, grad_norm)
    """
    model.train()

    # Get a batch from the dataset with loss masks
    inputs, targets, loss_masks = dataset.get_batch(batch_size, device)

    # Configure mixed precision settings
    amp_dtype = torch.bfloat16 if use_amp else torch.float32

    # Forward pass with autocast for BF16
    with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
        logits = model(inputs)
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = targets.view(-1)

        # Compute per-token losses
        loss_per_token = nn.functional.cross_entropy(
            logits_flat, targets_flat, reduction='none'
        )

        # Apply loss mask: only compute loss on assistant responses
        loss_mask_flat = loss_masks.view(-1)
        masked_loss = loss_per_token * loss_mask_flat

        # Average over non-masked tokens
        num_non_masked = loss_mask_flat.sum()
        if num_non_masked > 0:
            loss = masked_loss.sum() / num_non_masked
        else:
            loss = masked_loss.sum()  # Fallback if all tokens masked

        # Scale for gradient accumulation
        loss = loss / gradient_accumulation_steps

    # Backward pass with gradient accumulation
    if accumulation_step == 0:
        optimizer.zero_grad(set_to_none=True)

    loss.backward()

    # Only update weights and clip gradients on the last accumulation step
    grad_norm = torch.tensor(0.0)
    if accumulation_step == gradient_accumulation_steps - 1:
        # Clip gradients (use trainable_params for LoRA, otherwise all params)
        params_to_clip = trainable_params if trainable_params is not None else model.parameters()
        grad_norm = clip_grad_norm_(params_to_clip, max_grad_norm)
        optimizer.step()

        # Update expert biases for load balance if using MoE
        if hasattr(model, 'update_moe_biases'):
            model.update_moe_biases()

    return loss.item() * gradient_accumulation_steps, grad_norm.item()


def save_checkpoint_with_mode(model: nn.Module, optimizer: torch.optim.Optimizer,
                               iteration: int, checkpoint_path: str, mode: str):
    """
    Save checkpoint based on fine-tuning mode.

    For SFT mode: Saves full model checkpoint
    For LoRA mode: Saves only LoRA weights

    Args:
        model: The model
        optimizer: The optimizer
        iteration: Current iteration
        checkpoint_path: Path to save checkpoint
        mode: 'sft' or 'lora'
    """
    if mode == 'lora':
        # Save only LoRA weights
        save_lora_weights(model, checkpoint_path)
    else:
        # Save full checkpoint (SFT mode)
        save_checkpoint(model, optimizer, iteration, checkpoint_path)


def main():
    parser = argparse.ArgumentParser(description='Fine-tune Transformer Language Model')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to pretrained model config JSON file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to pretrained model checkpoint')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to fine-tuning data JSONL file (lora_512.jsonl, lora_2048.jsonl, sft_512.jsonl, or sft_2048.jsonl)')
    parser.add_argument('--mode', type=str, choices=['sft', 'lora'], default='sft',
                        help='Fine-tuning mode: "sft" for full fine-tuning, "lora" for LoRA')
    parser.add_argument('--lora_rank', type=int, default=8,
                        help='LoRA rank (only used in lora mode)')
    parser.add_argument('--epochs', type=int, default=4,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=5e-7,
                        help='Learning rate for fine-tuning')
    parser.add_argument('--accumulation_steps', type=int, default=1,
                        help='Gradient accumulation steps')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Gradient clipping value')
    parser.add_argument('--warmup_iters', type=int, default=0,
                        help='Number of warmup iterations')
    parser.add_argument('--log_interval', type=int, default=100,
                        help='Logging interval')
    parser.add_argument('--save_interval', type=int, default=100,
                        help='Checkpoint saving interval')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Limit number of conversations to load (for testing). None = load all')

    args = parser.parse_args()

    # Load configuration using Config class
    config = Config.from_json(args.config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Set performance optimizations
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')
        torch.backends.cudnn.benchmark = True
        print("Performance optimizations enabled: TF32 matmul, cuDNN benchmark")

    # Set random seeds
    seed = config.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Determine dataset type from config
    dataset_name = config.dataset
    print(f"Dataset type: {dataset_name}")
    print(f"Fine-tuning mode: {args.mode.upper()}")

    # Load appropriate tokenizer based on dataset
    tokenizer = Tokenizer.from_files(
        vocab_filepath=config.vocab_file,
        merges_filepath=config.merges_file,
        special_tokens=config.special_tokens
    )

    vocab_size = len(tokenizer.decoder_vocab)
    print(f"Vocabulary size: {vocab_size:,}")

    # Update config with actual vocab_size
    config.vocab_size = vocab_size

    # Load and prepare fine-tuning data (online tokenization - no preprocessing)
    raw_conversations = load_sft_data(args.data_path, args.max_samples)
    dataset = SFTDataset(raw_conversations, tokenizer, config.context_length)

    # Initialize model
    use_amp = config.use_amp
    model_dtype = torch.float32

    model = TransformerLM(
        config=config,
        device=device,
        dtype=model_dtype
    ).to(device)

    # Print model configuration
    attention_type = config.attention_type
    mlp_type = 'MoE' if config.use_moe else 'FFN'
    print(f"ATT Type: [{attention_type}]   MLP Type: [{mlp_type}]")

    # Load pretrained checkpoint
    print(f"Loading pretrained checkpoint from: {args.checkpoint}")
    load_checkpoint(args.checkpoint, model, None)  # Don't load optimizer state
    print("Pretrained checkpoint loaded successfully")

    # Apply LoRA if in lora mode (BEFORE compiling)
    trainable_params = None
    if args.mode == 'lora':
        print(f"\nApplying LoRA with rank={args.lora_rank}...")
        apply_lora(model, rank=args.lora_rank)
        freeze_non_lora_params(model)
        trainable_params = get_lora_params(model)
        print(f"LoRA applied successfully")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params_count:,}")
    print(f"Trainable ratio: {100 * trainable_params_count / total_params:.2f}%")

    # Initialize optimizer (only trainable parameters)
    optimizer_params = trainable_params if trainable_params is not None else model.parameters()
    optimizer = AdamW(
        optimizer_params,
        lr=args.learning_rate,
        betas=(config.beta1, config.beta2),
        eps=config.eps,
        weight_decay=config.weight_decay,
        fused=True
    )

    # Compile the model (AFTER applying LoRA)
    print("\nCompiling model with torch.compile...")
    model = torch.compile(model, mode='default')
    print("Model compiled successfully")

    # Extract model information from checkpoint path
    # e.g., checkpoints/TinyStories_GQA+MoE/best_model.pt -> TinyStories_GQA+MoE
    checkpoint_path = Path(args.checkpoint)
    model_info = checkpoint_path.parent.name  # Get the parent directory name

    # Create output directory with proper naming convention
    # [FT] for full fine-tuning (SFT), [LR] for LoRA
    mode_prefix_short = "[LR]" if args.mode == 'lora' else "[FT]"
    output_dir_name = f"{mode_prefix_short}{model_info}"

    # Add LoRA rank to directory name if using LoRA
    if args.mode == 'lora':
        output_dir_name += f"_r{args.lora_rank}"

    # Create output directory under checkpoints/
    output_dir = Path("checkpoints") / output_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}")

    # Create run name for wandb
    mode_prefix_long = "LoRA" if args.mode == 'lora' else "SFT"
    run_name = f"{mode_prefix_long}-{dataset_name}_{attention_type}+{mlp_type}"
    if args.mode == 'lora':
        run_name += f"_r{args.lora_rank}"

    # Initialize wandb
    wandb_config = {
        'mode': args.mode,
        'dataset': dataset_name,
        'data_path': args.data_path,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'epochs': args.epochs,
        'accumulation_steps': args.accumulation_steps,
        'effective_batch_size': args.batch_size * args.accumulation_steps,
        'warmup_iters': args.warmup_iters,
        'context_length': config.context_length,
        'total_params': total_params,
        'trainable_params': trainable_params_count,
        **config.to_dict()
    }

    if args.mode == 'lora':
        wandb_config['lora_rank'] = args.lora_rank

    wandb.init(
        project="Transformer_LLM",
        entity="scut_zeno",
        name=run_name,
        config=wandb_config
    )

    # Calculate total iterations based on epochs
    samples_per_epoch = len(dataset)
    iters_per_epoch = samples_per_epoch // (args.batch_size * args.accumulation_steps)
    total_iterations = iters_per_epoch * args.epochs

    print(f"\nConversations in dataset: {samples_per_epoch:,}")
    print(f"Iterations per epoch: {iters_per_epoch}")
    print(f"Total iterations: {total_iterations}")
    print(f"Effective batch size: {args.batch_size * args.accumulation_steps}")

    # Create record file
    record_file_path = output_dir / f"{args.mode}_record.txt"
    with open(record_file_path, 'w') as record_file:
        record_file.write(f"{mode_prefix_long} Fine-tuning Record for {dataset_name}\n")
        record_file.write("=" * 80 + "\n")
        record_file.write(f"Run: {run_name}\n")
        record_file.write(f"Mode: {args.mode.upper()}\n")
        if args.mode == 'lora':
            record_file.write(f"LoRA Rank: {args.lora_rank}\n")
        record_file.write(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        record_file.write(f"Data: {args.data_path}\n")
        record_file.write(f"Total params: {total_params:,}\n")
        record_file.write(f"Trainable params: {trainable_params_count:,}\n")
        record_file.write("=" * 80 + "\n\n")

    # Training loop
    print("\nStarting fine-tuning...")
    print("-" * 60)

    model.train()
    running_loss = 0.0

    for iteration in range(total_iterations):
        start_time = time.time()

        # Update learning rate with warmup and cosine schedule
        lr = cos_learning_rate_schedule_with_warmup(
            iteration,
            max_lr=args.learning_rate,
            min_lr=args.learning_rate * 0.1,
            warmup_iter=args.warmup_iters,
            cos_iter=total_iterations
        )
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Gradient accumulation
        accumulated_loss = 0.0
        for accum_step in range(args.accumulation_steps):
            loss, grad_norm = train(
                model, optimizer, dataset,
                args.batch_size,
                device, args.grad_clip,
                gradient_accumulation_steps=args.accumulation_steps,
                accumulation_step=accum_step,
                use_amp=use_amp,
                trainable_params=trainable_params
            )
            accumulated_loss += loss

        avg_accum_loss = accumulated_loss / args.accumulation_steps
        running_loss += avg_accum_loss
        step_time = time.time() - start_time

        # Logging
        if (iteration + 1) % args.log_interval == 0:
            avg_loss = running_loss / args.log_interval
            perplexity = np.exp(avg_loss)

            content = f"Iter {iteration + 1:6d} | Loss: {avg_loss:.4f} | PPL: {perplexity:.2f} | " \
                      f"LR: {lr:.6e} | Grad Norm: {grad_norm:.4f} | Time: {step_time:.3f}s"
            print(content)

            with open(record_file_path, 'a') as record_file:
                record_file.write(f"[TRAIN] {content}\n")

            wandb.log({
                'train/loss': avg_loss,
                'train/perplexity': perplexity,
                'train/learning_rate': lr,
                'train/grad_norm': grad_norm,
                'train/step_time': step_time
            }, step=iteration + 1)

            running_loss = 0.0

        # Checkpointing
        if (iteration + 1) % args.save_interval == 0:
            if args.mode == 'lora':
                checkpoint_path = output_dir / f"lora_rank{args.lora_rank}_iter_{iteration + 1:06d}.pt"
            else:
                checkpoint_path = output_dir / f"checkpoint_iter_{iteration + 1:06d}.pt"

            save_checkpoint_with_mode(model, optimizer, iteration + 1, str(checkpoint_path), args.mode)
            print(f"Saved checkpoint: {checkpoint_path}")

    print("Fine-tuning completed!")

    # Save final results
    with open(record_file_path, 'a') as record_file:
        record_file.write(f"\n{'='*50}\n")
        record_file.write(f"Fine-tuning completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        record_file.write(f"Total iterations: {total_iterations}\n")
        record_file.write(f"{'='*50}\n")

    # Save final checkpoint
    if args.mode == 'lora':
        final_checkpoint_path = output_dir / f"lora_rank{args.lora_rank}_final.pt"
    else:
        final_checkpoint_path = output_dir / "final_model.pt"

    save_checkpoint_with_mode(model, optimizer, total_iterations, str(final_checkpoint_path), args.mode)
    print(f"Final checkpoint saved: {final_checkpoint_path}")

    wandb.finish()


if __name__ == "__main__":
    main()
