"""
Direct Preference Optimization (DPO) Alignment Script

This script performs preference alignment using DPO on a pretrained model.
It loads preference pairs (prompt, chosen, rejected) and trains the model
to prefer chosen responses over rejected ones.
"""

import json
import time
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from model.transformer import TransformerLM
from model.tokenizer.bpe_tokenizer import Tokenizer
from torch.optim import AdamW
from data.lm_dataset import DPODataset
from torch.nn.utils import clip_grad_norm_
from model.utils import (
    save_checkpoint, load_checkpoint,
    cos_learning_rate_schedule_with_warmup
)


def load_dpo_data(data_path: str, max_samples: int = None):
    """
    Load DPO preference data from JSONL format.

    Args:
        data_path: Path to the JSONL file containing preference pairs
        max_samples: Optional limit on number of samples to load

    Returns:
        List of preference pairs with 'prompt', 'chosen', 'rejected'
    """
    print(f"Loading DPO preference data from {data_path}...")

    # Count total lines
    with open(data_path, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
    print(f"Total preference pairs in file: {total_lines:,}")

    if max_samples is not None:
        print(f"Will load only first {max_samples:,} samples")
        total_lines = min(total_lines, max_samples)

    preference_data = []
    start_time = time.time()

    with open(data_path, 'r', encoding='utf-8') as f:
        for line_idx, line in enumerate(f):
            if max_samples is not None and line_idx >= max_samples:
                break

            try:
                data = json.loads(line.strip())
                # Ensure required fields exist
                if 'prompt' in data and 'chosen' in data and 'rejected' in data:
                    preference_data.append({
                        'prompt': str(data['prompt']),
                        'chosen': str(data['chosen']),
                        'rejected': str(data['rejected'])
                    })
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line {line_idx}: {e}")
                continue

    elapsed = time.time() - start_time
    print(f"Loading completed in {elapsed:.2f} seconds")
    print(f"Loaded {len(preference_data):,} preference pairs")

    return preference_data


def logits_to_log_probs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Convert logits to log probabilities for the given labels.

    Args:
        logits: (batch_size, seq_len, vocab_size)
        labels: (batch_size, seq_len)

    Returns:
        log_probs: (batch_size, seq_len)
    """
    log_probs = F.log_softmax(logits, dim=-1)
    # Gather log probs at label indices
    per_token_log_probs = torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(-1)
    return per_token_log_probs


def dpo_loss(policy_chosen_log_probs: torch.Tensor,
             policy_rejected_log_probs: torch.Tensor,
             reference_chosen_log_probs: torch.Tensor,
             reference_rejected_log_probs: torch.Tensor,
             mask_chosen: torch.Tensor,
             mask_rejected: torch.Tensor,
             beta: float = 0.1) -> torch.Tensor:
    """
    Compute DPO loss.

    Args:
        policy_chosen_log_probs: Log probs from policy model for chosen (batch_size, seq_len)
        policy_rejected_log_probs: Log probs from policy model for rejected (batch_size, seq_len)
        reference_chosen_log_probs: Log probs from reference model for chosen (batch_size, seq_len)
        reference_rejected_log_probs: Log probs from reference model for rejected (batch_size, seq_len)
        mask_chosen: Mask for chosen responses (batch_size, seq_len)
        mask_rejected: Mask for rejected responses (batch_size, seq_len)
        beta: Temperature parameter for DPO

    Returns:
        loss: Scalar DPO loss
    """
    # Compute average log probability per sequence (only over non-masked tokens)
    chosen_lengths = mask_chosen.sum(dim=1, keepdim=True).clamp(min=1)  # (batch_size, 1)
    rejected_lengths = mask_rejected.sum(dim=1, keepdim=True).clamp(min=1)  # (batch_size, 1)

    # Average log probs over sequence length
    policy_chosen_avg = (policy_chosen_log_probs * mask_chosen).sum(dim=1) / chosen_lengths.squeeze()
    policy_rejected_avg = (policy_rejected_log_probs * mask_rejected).sum(dim=1) / rejected_lengths.squeeze()

    reference_chosen_avg = (reference_chosen_log_probs * mask_chosen).sum(dim=1) / chosen_lengths.squeeze()
    reference_rejected_avg = (reference_rejected_log_probs * mask_rejected).sum(dim=1) / rejected_lengths.squeeze()

    # Compute log ratios
    policy_log_ratios = policy_chosen_avg - policy_rejected_avg
    reference_log_ratios = reference_chosen_avg - reference_rejected_avg

    # DPO loss: -log sigmoid(beta * (policy_log_ratio - reference_log_ratio))
    logits = beta * (policy_log_ratios - reference_log_ratios)
    loss = -F.logsigmoid(logits).mean()

    return loss


def train(policy_model: nn.Module, reference_model: nn.Module,
          optimizer: torch.optim.Optimizer,
          dataset: DPODataset, batch_size: int,
          device: torch.device, max_grad_norm: float,
          beta: float = 0.1,
          gradient_accumulation_steps: int = 1,
          accumulation_step: int = 0,
          use_amp: bool = False):
    """
    Perform a single DPO training step.

    Args:
        policy_model: The model being trained
        reference_model: The frozen reference model
        optimizer: The optimizer
        dataset: DPODataset instance
        batch_size: Batch size
        device: Device to run on
        max_grad_norm: Maximum gradient norm for clipping
        beta: DPO temperature parameter
        gradient_accumulation_steps: Number of steps to accumulate gradients
        accumulation_step: Current accumulation step
        use_amp: Whether to use automatic mixed precision

    Returns:
        Tuple of (loss, grad_norm)
    """
    policy_model.train()
    reference_model.eval()

    # Get a batch of preference pairs
    batch = dataset.get_batch(batch_size, device)

    x_chosen = batch['x_chosen']
    y_chosen = batch['y_chosen']
    mask_chosen = batch['mask_chosen']
    x_rejected = batch['x_rejected']
    y_rejected = batch['y_rejected']
    mask_rejected = batch['mask_rejected']

    # Configure mixed precision
    amp_dtype = torch.bfloat16 if use_amp else torch.float32

    with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
        # Get reference model log probs (no gradient)
        with torch.no_grad():
            ref_logits_chosen = reference_model(x_chosen)
            ref_log_probs_chosen = logits_to_log_probs(ref_logits_chosen, y_chosen)

            ref_logits_rejected = reference_model(x_rejected)
            ref_log_probs_rejected = logits_to_log_probs(ref_logits_rejected, y_rejected)

        # Get policy model log probs (with gradient)
        policy_logits_chosen = policy_model(x_chosen)
        policy_log_probs_chosen = logits_to_log_probs(policy_logits_chosen, y_chosen)

        policy_logits_rejected = policy_model(x_rejected)
        policy_log_probs_rejected = logits_to_log_probs(policy_logits_rejected, y_rejected)

        # Compute DPO loss
        loss = dpo_loss(
            policy_log_probs_chosen,
            policy_log_probs_rejected,
            ref_log_probs_chosen,
            ref_log_probs_rejected,
            mask_chosen,
            mask_rejected,
            beta=beta
        )

        # Scale for gradient accumulation
        loss = loss / gradient_accumulation_steps

    # Backward pass
    if accumulation_step == 0:
        optimizer.zero_grad(set_to_none=True)

    loss.backward()

    # Update weights on last accumulation step
    grad_norm = torch.tensor(0.0)
    if accumulation_step == gradient_accumulation_steps - 1:
        grad_norm = clip_grad_norm_(policy_model.parameters(), max_grad_norm)
        optimizer.step()

        # Update expert biases if using MoE
        if hasattr(policy_model, 'update_moe_biases'):
            policy_model.update_moe_biases()

    return loss.item() * gradient_accumulation_steps, grad_norm.item()


def main():
    parser = argparse.ArgumentParser(description='DPO Alignment for Transformer Language Model')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to pretrained model config JSON file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to pretrained/finetuned model checkpoint')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to DPO data JSONL file (dpo_512.jsonl or dpo_2048.jsonl)')
    parser.add_argument('--epochs', type=int, default=2,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-8,
                        help='Learning rate for DPO (use very small LR to avoid forgetting)')
    parser.add_argument('--beta', type=float, default=0.1,
                        help='DPO temperature parameter')
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
                        help='Limit number of preference pairs to load (for testing)')

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
        print("Performance optimizations enabled: TF32 matmul, cuDNN benchmark")

    # Set random seeds
    seed = config.get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Determine dataset type from config
    dataset_name = config.get('dataset', 'TinyStories')
    print(f"Dataset type: {dataset_name}")

    # Load tokenizer
    tokenizer = Tokenizer.from_files(
        vocab_filepath=config['vocab_file'],
        merges_filepath=config['merges_file'],
        special_tokens=config.get('special_tokens', ['<|endoftext|>'])
    )

    vocab_size = len(tokenizer.decoder_vocab)
    print(f"Vocabulary size: {vocab_size:,}")

    # Load DPO preference data
    preference_data = load_dpo_data(args.data_path, args.max_samples)
    dataset = DPODataset(preference_data, tokenizer, config['context_length'])

    # Initialize models
    use_amp = config.get('use_amp', False)
    model_dtype = torch.float32

    # Create policy model (will be trained)
    policy_model = TransformerLM(
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

    # Create reference model (frozen copy)
    reference_model = TransformerLM(
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

    # Print model configuration
    attention_type = config.get('attention_type', 'GQA')
    mlp_type = 'MoE' if config.get('use_moe', False) else 'FFN'
    print(f"ATT Type: [{attention_type}]   MLP Type: [{mlp_type}]")

    # Count parameters
    total_params = sum(p.numel() for p in policy_model.parameters())
    trainable_params = sum(p.numel() for p in policy_model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Initialize optimizer
    optimizer = AdamW(
        policy_model.parameters(),
        lr=args.learning_rate,
        betas=(config['beta1'], config['beta2']),
        eps=config['eps'],
        weight_decay=config.get('weight_decay', 0.1),
        fused=True
    )

    # Load checkpoint into both models
    print(f"Loading checkpoint from: {args.checkpoint}")
    load_checkpoint(args.checkpoint, policy_model, optimizer)
    load_checkpoint(args.checkpoint, reference_model, None)  # No optimizer for reference
    print("Checkpoint loaded for both policy and reference models")

    # Freeze reference model
    reference_model.eval()
    for param in reference_model.parameters():
        param.requires_grad = False
    print("Reference model frozen")

    # Reset optimizer learning rate for DPO
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.learning_rate

    # Compile the policy model
    print("Compiling policy model with torch.compile...")
    policy_model = torch.compile(policy_model, mode='default')
    print("Model compiled successfully")

    # Extract model information from checkpoint path
    # e.g., checkpoints/TinyStories_GQA+MoE/best_model.pt -> TinyStories_GQA+MoE
    checkpoint_path = Path(args.checkpoint)
    model_info = checkpoint_path.parent.name  # Get the parent directory name

    # Create output directory with [PO] prefix for Preference Optimization
    output_dir_name = f"[PO]{model_info}"

    # Create output directory under checkpoints/
    output_dir = Path("checkpoints") / output_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}")

    # Create run name for wandb
    run_name = f"DPO-{dataset_name}_{attention_type}+{mlp_type}"

    # Initialize wandb
    wandb_config = {
        'dataset': dataset_name,
        'dpo_data': args.data_path,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'epochs': args.epochs,
        'beta': args.beta,
        'accumulation_steps': args.accumulation_steps,
        'effective_batch_size': args.batch_size * args.accumulation_steps,
        'warmup_iters': args.warmup_iters,
        'context_length': config['context_length'],
        **config
    }

    wandb.init(
        project="Transformer_LLM",
        entity="scut_zeno",
        name=run_name,
        config=wandb_config
    )

    # Calculate total iterations
    samples_per_epoch = len(dataset)
    iters_per_epoch = samples_per_epoch // (args.batch_size * args.accumulation_steps)
    total_iterations = iters_per_epoch * args.epochs

    print(f"Preference pairs in dataset: {samples_per_epoch:,}")
    print(f"Iterations per epoch: {iters_per_epoch}")
    print(f"Total iterations: {total_iterations}")
    print(f"Effective batch size: {args.batch_size * args.accumulation_steps}")

    # Create record file
    record_file_path = output_dir / "dpo_record.txt"
    with open(record_file_path, 'w') as record_file:
        record_file.write(f"DPO Alignment Record for {dataset_name}\n")
        record_file.write("=" * 80 + "\n")
        record_file.write(f"Run: {run_name}\n")
        record_file.write(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        record_file.write(f"DPO Data: {args.data_path}\n")
        record_file.write(f"Beta: {args.beta}\n")
        record_file.write("=" * 80 + "\n\n")

    # Training loop
    print("Starting DPO alignment...")
    print("-" * 60)

    policy_model.train()
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
                policy_model, reference_model, optimizer, dataset,
                args.batch_size,
                device, args.grad_clip,
                beta=args.beta,
                gradient_accumulation_steps=args.accumulation_steps,
                accumulation_step=accum_step,
                use_amp=use_amp
            )
            accumulated_loss += loss

        avg_accum_loss = accumulated_loss / args.accumulation_steps
        running_loss += avg_accum_loss
        step_time = time.time() - start_time

        # Logging
        if (iteration + 1) % args.log_interval == 0:
            avg_loss = running_loss / args.log_interval

            content = f"Iter {iteration + 1:6d} | Loss: {avg_loss:.4f} | " \
                      f"LR: {lr:.6e} | Grad Norm: {grad_norm:.4f} | Time: {step_time:.3f}s"
            print(content)

            with open(record_file_path, 'a') as record_file:
                record_file.write(f"[TRAIN] {content}\n")

            wandb.log({
                'train/loss': avg_loss,
                'train/learning_rate': lr,
                'train/grad_norm': grad_norm,
                'train/step_time': step_time
            }, step=iteration + 1)

            running_loss = 0.0

        # Checkpointing
        if (iteration + 1) % args.save_interval == 0:
            checkpoint_path = output_dir / f"checkpoint_iter_{iteration + 1:06d}.pt"
            save_checkpoint(policy_model, optimizer, iteration + 1, str(checkpoint_path))
            print(f"Saved checkpoint: {checkpoint_path}")

    print("DPO alignment completed!")

    # Save final results
    with open(record_file_path, 'a') as record_file:
        record_file.write(f"\n{'='*50}\n")
        record_file.write(f"DPO alignment completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        record_file.write(f"Total iterations: {total_iterations}\n")
        record_file.write(f"{'='*50}\n")

    # Save final checkpoint
    final_checkpoint_path = output_dir / "final_model.pt"
    save_checkpoint(policy_model, optimizer, total_iterations, str(final_checkpoint_path))
    print(f"Final checkpoint saved: {final_checkpoint_path}")

    wandb.finish()


if __name__ == "__main__":
    main()
