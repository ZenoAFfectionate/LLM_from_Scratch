"""
Profiling script to identify performance bottlenecks in training.

This script uses PyTorch Profiler to analyze:
1. Time spent in each operation
2. GPU vs CPU time breakdown
3. Memory usage patterns
4. Kernel launch overhead

Usage:
    CUDA_VISIBLE_DEVICES=0 python profile_train.py --gradient_accumulation_steps 32 --config config/[GQA+MoE]train_openwebtext.json
"""

import os
import json
import time
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW

# Import profiler
from torch.profiler import profile, record_function, ProfilerActivity

from model.transformer import TransformerLM
from model.tokenizer.bpe_tokenizer import Tokenizer
from data.lm_dataset import PretrainDataset
from model.utils import (
    save_checkpoint, load_checkpoint,
    cos_learning_rate_schedule_with_warmup
)


def load_data_memmap(data_path: str, dtype=np.int32):
    """Load training data using memory mapping for efficient access"""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    data = np.memmap(data_path, dtype=dtype, mode='r')
    print(f"Loaded {len(data):,} tokens from {data_path}")
    return data


def prepare_data(config, tokenizer):
    """Prepare training and validation data"""
    data_dir = Path(config['data_dir'])
    train_bin = data_dir / f"tokens_train.bin"
    valid_bin = data_dir / f"tokens_valid.bin"
    train_data = load_data_memmap(str(train_bin))
    valid_data = load_data_memmap(str(valid_bin))
    return train_data, valid_data


def train_step(model, optimizer, train_loader_iter, config, device,
               gradient_accumulation_steps=1, accumulation_step=0):
    """Single training step with detailed timing"""
    model.train()

    # Get next batch
    with record_function("data_loading"):
        inputs, targets = next(train_loader_iter)
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

    use_amp = config.get('use_amp', False)
    amp_dtype = torch.bfloat16 if use_amp else torch.float32

    # Forward pass
    with record_function("forward_pass"):
        with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
            logits = model(inputs)
            logits_flat = logits.view(-1, logits.size(-1))
            targets_flat = targets.view(-1)
            loss = F.cross_entropy(logits_flat, targets_flat)
            loss = loss / gradient_accumulation_steps

    # Backward pass
    if accumulation_step == 0:
        optimizer.zero_grad()

    with record_function("backward_pass"):
        loss.backward()

    # Optimizer step and gradient clipping
    grad_norm = torch.tensor(0.0)
    if accumulation_step == gradient_accumulation_steps - 1:
        with record_function("gradient_clipping"):
            grad_norm = clip_grad_norm_(model.parameters(), config['max_grad_norm'])

        with record_function("optimizer_step"):
            optimizer.step()

        with record_function("moe_bias_update"):
            if hasattr(model, 'update_moe_biases'):
                model.update_moe_biases()

    return loss.item() * gradient_accumulation_steps, grad_norm.item()


def main():
    parser = argparse.ArgumentParser(description='Profile Transformer Training')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--profile_steps', type=int, default=10,
                        help='Number of steps to profile')
    parser.add_argument('--warmup_steps', type=int, default=3,
                        help='Number of warmup steps before profiling')
    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Set performance optimizations
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')
        torch.backends.cudnn.benchmark = True

    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])

    # Load tokenizer
    tokenizer = Tokenizer.from_files(
        vocab_filepath=config['vocab_file'],
        merges_filepath=config['merges_file'],
        special_tokens=config.get('special_tokens', ['<|endoftext|>'])
    )
    vocab_size = len(tokenizer.decoder_vocab)
    print(f"Vocabulary size: {vocab_size:,}")

    train_data, valid_data = prepare_data(config, tokenizer)

    # Create DataLoader
    num_workers = config.get('num_workers', 4)
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
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=True
    )

    model_dtype = torch.float32

    # Initialize model
    print("\nInitializing model...")
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
        aux_seq_loss_alpha=config.get('aux_seq_loss_alpha', 0.0),
        bias_update_speed=config.get('bias_update_speed', 0.01),
        num_kv_heads=config.get('num_kv_heads', config['num_heads']),
        attention_type=config.get('attention_type', 'GQA'),
        d_rope=config.get('d_rope', None),
        kv_lora_rank=config.get('kv_lora_rank', None),
        q_lora_rank=config.get('q_lora_rank', None),
        device=device,
        dtype=model_dtype
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Compile model (if not already causing issues)
    print("Compiling model with torch.compile...")
    model = torch.compile(model, mode='reduce-overhead')
    print("Model compiled successfully")

    # Initialize optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config['max_lr'],
        betas=(config['beta1'], config['beta2']),
        eps=config['eps'],
        weight_decay=config['weight_decay'],
        fused=True
    )

    gradient_accumulation_steps = args.gradient_accumulation_steps
    print(f"\nGradient Accumulation: {gradient_accumulation_steps} steps")
    print(f"Micro Batch Size: {config['batch_size']}")
    print(f"Effective Batch Size: {config['batch_size'] * gradient_accumulation_steps}")

    # Create iterator
    train_loader_iter = iter(train_loader)

    print(f"\n{'='*80}")
    print("Starting profiling...")
    print(f"Warmup steps: {args.warmup_steps}")
    print(f"Profile steps: {args.profile_steps}")
    print(f"{'='*80}\n")

    # Warmup steps (not profiled)
    print("Running warmup steps...")
    for i in range(args.warmup_steps):
        for accum_step in range(gradient_accumulation_steps):
            try:
                loss, grad_norm = train_step(
                    model, optimizer, train_loader_iter, config, device,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    accumulation_step=accum_step
                )
            except StopIteration:
                train_loader_iter = iter(train_loader)
                loss, grad_norm = train_step(
                    model, optimizer, train_loader_iter, config, device,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    accumulation_step=accum_step
                )
        print(f"  Warmup step {i+1}/{args.warmup_steps} completed")

    # Synchronize before profiling
    torch.cuda.synchronize()

    # Profile training steps
    print(f"\nProfiling {args.profile_steps} steps...")

    output_dir = Path("./profiler_output")
    output_dir.mkdir(exist_ok=True)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(str(output_dir)),
    ) as prof:
        for iteration in range(args.profile_steps):
            step_start = time.time()

            for accum_step in range(gradient_accumulation_steps):
                try:
                    loss, grad_norm = train_step(
                        model, optimizer, train_loader_iter, config, device,
                        gradient_accumulation_steps=gradient_accumulation_steps,
                        accumulation_step=accum_step
                    )
                except StopIteration:
                    train_loader_iter = iter(train_loader)
                    loss, grad_norm = train_step(
                        model, optimizer, train_loader_iter, config, device,
                        gradient_accumulation_steps=gradient_accumulation_steps,
                        accumulation_step=accum_step
                    )

            prof.step()

            step_time = time.time() - step_start
            print(f"  Step {iteration+1}/{args.profile_steps}: {step_time:.3f}s, Loss: {loss:.4f}")

    print(f"\n{'='*80}")
    print("Profiling completed!")
    print(f"{'='*80}\n")

    # Print profiling summary
    print("=" * 80)
    print("PROFILING SUMMARY - Sorted by CUDA Time")
    print("=" * 80)
    print(prof.key_averages().table(
        sort_by="cuda_time_total",
        row_limit=30
    ))

    print("\n" + "=" * 80)
    print("PROFILING SUMMARY - Sorted by CPU Time")
    print("=" * 80)
    print(prof.key_averages().table(
        sort_by="cpu_time_total",
        row_limit=30
    ))

    print("\n" + "=" * 80)
    print("PROFILING SUMMARY - Memory Usage")
    print("=" * 80)
    print(prof.key_averages().table(
        sort_by="cuda_memory_usage",
        row_limit=20
    ))

    # Save detailed profiling data
    trace_file = output_dir / "trace.json"
    print(f"\nDetailed trace saved to: {trace_file}")
    print(f"View in Chrome: chrome://tracing or use TensorBoard:")
    print(f"  tensorboard --logdir={output_dir}")

    # Additional timing breakdown
    print("\n" + "=" * 80)
    print("TIMING BREAKDOWN BY OPERATION")
    print("=" * 80)

    key_averages = prof.key_averages()

    timing_categories = {
        'Data Loading': ['data_loading'],
        'Forward Pass': ['forward_pass'],
        'Backward Pass': ['backward_pass'],
        'Gradient Clipping': ['gradient_clipping'],
        'Optimizer Step': ['optimizer_step'],
        'MoE Bias Update': ['moe_bias_update'],
        'Attention': ['scaled_dot_product_attention', 'matmul', 'bmm'],
        'MoE Operations': ['gather', 'scatter', 'topk', 'bincount'],
        'Normalization': ['rms_norm', 'layer_norm'],
    }

    for category, keywords in timing_categories.items():
        total_time = 0
        for event in key_averages:
            if any(keyword.lower() in event.key.lower() for keyword in keywords):
                total_time += event.cuda_time_total if event.cuda_time_total > 0 else event.cpu_time_total
        if total_time > 0:
            print(f"{category:.<40} {total_time/1000:.2f} ms")

    print("=" * 80)


if __name__ == "__main__":
    main()
