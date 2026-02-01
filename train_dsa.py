"""
Training script for DeepSeek Sparse Attention (DSA).

This script implements the two-stage training procedure described in 
"DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models":

Stage 1 - Dense Warm-up:
    - Train only the lightning indexer
    - Use KL-divergence loss to align indexer with main attention distribution
    - Keep dense attention (no token selection)
    - Freeze all model parameters except indexer

Stage 2 - Sparse Training:
    - Train both main model and indexer
    - Use sparse attention with fine-grained token selection
    - Separate optimization: indexer trained with KL loss, model with LM loss
    - Detach indexer input from model's computational graph

Usage:
    python train_dsa.py --config config/[DSA]train_openwebtext.json --stage warmup
    python train_dsa.py --config config/[DSA]train_openwebtext.json --stage sparse --resume checkpoints/warmup/final.pt
"""

import os
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any, Iterator, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from model.config import Config
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


def prepare_data(config: Dict[str, Any]):
    """Prepare training and validation data using memory-mapped files"""
    data_dir = Path(config['data_dir'])
    train_bin = data_dir / "tokens_train.bin"
    valid_bin = data_dir / "tokens_valid.bin"
    train_data = load_data_memmap(str(train_bin))
    valid_data = load_data_memmap(str(valid_bin))
    return train_data, valid_data


class DSAModelHelper:
    """
    [OPT] Cache DSA layer information to avoid repeated model traversal.
    This helper is created once and reused throughout training.
    """
    def __init__(self, model: nn.Module):
        self.model = model
        # [OPT] Cache DSA layers once
        self._dsa_layers = None
        self._indexer_params = None
        self._indexer_param_ids = None
        self._main_params = None
        self._refresh_caches()
    
    def _refresh_caches(self):
        """Compute and cache all parameter lists."""
        # Find DSA layers
        self._dsa_layers = []
        for idx, block in enumerate(self.model.layers):
            if hasattr(block, 'att') and hasattr(block.att, 'indexer'):
                self._dsa_layers.append((idx, block.att))
        
        # Cache indexer parameters
        self._indexer_params = []
        for _, att in self._dsa_layers:
            self._indexer_params.extend(list(att.indexer.parameters()))
        
        # Cache indexer param IDs
        self._indexer_param_ids = set(id(p) for p in self._indexer_params)
        
        # Cache main model parameters
        self._main_params = [p for p in self.model.parameters() 
                            if id(p) not in self._indexer_param_ids]
    
    @property
    def dsa_layers(self):
        return self._dsa_layers
    
    @property
    def indexer_params(self):
        return self._indexer_params
    
    @property
    def main_params(self):
        return self._main_params
    
    @property
    def num_dsa_layers(self):
        return len(self._dsa_layers)
    
    def freeze_except_indexer(self):
        """Freeze all model parameters except the lightning indexer."""
        for param in self.model.parameters():
            param.requires_grad = False
        
        for layer_idx, att in self._dsa_layers:
            for name, param in att.indexer.named_parameters():
                param.requires_grad = True
                print(f"  Layer {layer_idx} indexer.{name}: requires_grad=True")
        
        return self._indexer_params
    
    def unfreeze_all(self):
        """Unfreeze all model parameters."""
        for param in self.model.parameters():
            param.requires_grad = True


# Legacy functions for compatibility
def get_dsa_attention_layers(model: nn.Module):
    """Extract all DSA attention layers from the model."""
    dsa_layers = []
    for idx, block in enumerate(model.layers):
        if hasattr(block, 'att') and hasattr(block.att, 'indexer'):
            dsa_layers.append((idx, block.att))
    return dsa_layers


def freeze_model_except_indexer(model: nn.Module):
    """Freeze all model parameters except the lightning indexer."""
    for param in model.parameters():
        param.requires_grad = False
    
    indexer_params = []
    for layer_idx, att in get_dsa_attention_layers(model):
        for name, param in att.indexer.named_parameters():
            param.requires_grad = True
            indexer_params.append(param)
            print(f"  Layer {layer_idx} indexer.{name}: requires_grad=True")
    
    return indexer_params


def unfreeze_all_parameters(model: nn.Module):
    """Unfreeze all model parameters for Sparse Training Stage."""
    for param in model.parameters():
        param.requires_grad = True


def get_indexer_parameters(model: nn.Module):
    """Get all indexer parameters for separate optimization."""
    indexer_params = []
    for _, att in get_dsa_attention_layers(model):
        indexer_params.extend(att.indexer.parameters())
    return indexer_params


def get_main_model_parameters(model: nn.Module):
    """Get all non-indexer parameters for main model optimization."""
    indexer_param_ids = set(id(p) for p in get_indexer_parameters(model))
    return [p for p in model.parameters() if id(p) not in indexer_param_ids]


# =============================================================================
# Stage 1: Dense Warm-up Training
# =============================================================================

def train_warmup_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader_iter: Iterator,
    config: Dict[str, Any],
    device: torch.device,
    gradient_accumulation_steps: int = 1,
    accumulation_step: int = 0,
    helper: Optional['DSAModelHelper'] = None,  # [OPT] Reuse helper
    trainable_params: Optional[list] = None,    # [OPT] Cache trainable params
):
    """
    Single training step for Dense Warm-up Stage.
    [OPT] Accepts cached helper and trainable_params to avoid repeated computation.
    """
    model.train()
    
    inputs, targets = next(train_loader_iter)
    inputs = inputs.to(device, non_blocking=True)
    
    use_amp = config.get('use_amp', False)
    amp_dtype = torch.bfloat16 if use_amp else torch.float32
    
    # [OPT] Get num_dsa_layers from helper or compute once
    if helper is not None:
        num_dsa_layers = helper.num_dsa_layers
    else:
        num_dsa_layers = len(get_dsa_attention_layers(model))
    
    with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
        batch_size, seq_len = inputs.shape
        x = model.token_embeddings(inputs)
        
        # [OPT] Create causal mask once (could be cached but seq_len may vary)
        mask = torch.tril(torch.ones((seq_len, seq_len), device=x.device, dtype=torch.bool)) if seq_len > 1 else None
        
        # [OPT] Initialize loss as tensor on device directly
        total_kl_loss = x.new_zeros(())
        
        # Process through layers
        residual = None
        for idx, block in enumerate(model.layers):
            if residual is None:
                x_norm, residual = block.att_norm(x), x
            else:
                x_norm, residual = block.att_norm(x, residual)
            
            if hasattr(block.att, 'indexer'):
                kl_loss = block.att.compute_indexer_loss_dense(x_norm, start_pos=0, mask=mask)
                total_kl_loss = total_kl_loss + kl_loss
                attn_out, _ = block.att(x_norm, start_pos=0, mask=mask, use_sparse=False)
            else:
                attn_out = block.att(x_norm, start_pos=0, mask=mask)
            
            x = block.dropout(attn_out)
            x, residual = block.ffn_norm(x, residual)
            x = block.ffn(x)
            x = block.dropout(x)
        
        # Average loss across DSA layers
        loss = total_kl_loss / max(num_dsa_layers, 1)
        loss = loss / gradient_accumulation_steps
    
    if accumulation_step == 0:
        optimizer.zero_grad(set_to_none=True)
    
    loss.backward()
    
    grad_norm = 0.0
    if accumulation_step == gradient_accumulation_steps - 1:
        # [OPT] Use cached trainable_params if provided
        if trainable_params is not None:
            grad_norm = clip_grad_norm_(trainable_params, config.get('max_grad_norm', 1.0)).item()
        else:
            grad_norm = clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                config.get('max_grad_norm', 1.0)
            ).item()
        optimizer.step()
    
    return loss.item() * gradient_accumulation_steps, grad_norm


def run_warmup_stage(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    config: Config,
    config_dict: Dict[str, Any],
    device: torch.device,
    checkpoint_dir: Path,
    gradient_accumulation_steps: int = 1
):
    """
    Run Dense Warm-up Stage.
    [OPT] Uses DSAModelHelper for efficient parameter caching.
    """
    print("\n" + "=" * 60)
    print("STAGE 1: Dense Warm-up")
    print("=" * 60)
    
    # [OPT] Create helper for efficient parameter caching
    helper = DSAModelHelper(model)
    
    # Freeze model except indexer
    print("\nFreezing model parameters except indexer...")
    indexer_params = helper.freeze_except_indexer()
    print(f"Trainable parameters: {sum(p.numel() for p in indexer_params):,}")
    
    # Warmup-specific hyperparameters
    warmup_lr = config_dict.get('warmup_stage_lr', 1e-3)
    warmup_steps = config_dict.get('warmup_stage_steps', 1000)
    
    print(f"Learning rate: {warmup_lr}")
    print(f"Total steps: {warmup_steps}")
    
    # Initialize optimizer for indexer only
    optimizer = AdamW(
        indexer_params,
        lr=warmup_lr,
        betas=(config.beta1, config.beta2),
        eps=config.eps,
        weight_decay=0.0,
        fused=torch.cuda.is_available()  # [OPT] Use fused optimizer if available
    )
    
    # Initialize wandb
    wandb.init(
        project="DSA_Training",
        entity="scut_zeno",
        name=f"{config.run_name}_warmup",
        config={**config_dict, 'stage': 'warmup'}
    )
    
    # Training loop
    model.train()
    running_loss = 0.0
    train_loader_iter = iter(train_loader)
    log_interval = config_dict.get('log_interval', 100)
    
    for step in range(warmup_steps):
        start_time = time.time()
        
        # Gradient accumulation
        accumulated_loss = 0.0
        for accum_step in range(gradient_accumulation_steps):
            try:
                loss, grad_norm = train_warmup_step(
                    model, optimizer, train_loader_iter, config_dict, device,
                    gradient_accumulation_steps, accum_step,
                    helper=helper,  # [OPT] Pass helper
                    trainable_params=indexer_params  # [OPT] Pass cached params
                )
            except StopIteration:
                train_loader_iter = iter(train_loader)
                loss, grad_norm = train_warmup_step(
                    model, optimizer, train_loader_iter, config_dict, device,
                    gradient_accumulation_steps, accum_step,
                    helper=helper,
                    trainable_params=indexer_params
                )
            accumulated_loss += loss
        
        avg_loss = accumulated_loss / gradient_accumulation_steps
        running_loss += avg_loss
        step_time = time.time() - start_time
        
        # Logging
        if (step + 1) % log_interval == 0:
            avg_running_loss = running_loss / log_interval
            print(f"[Warmup] Step {step + 1:5d}/{warmup_steps} | "
                  f"KL Loss: {avg_running_loss:.4f} | "
                  f"Grad Norm: {grad_norm:.4f} | "
                  f"Time: {step_time:.3f}s")
            
            wandb.log({
                'warmup/kl_loss': avg_running_loss,
                'warmup/grad_norm': grad_norm,
                'warmup/step_time': step_time
            }, step=step + 1)
            
            running_loss = 0.0
    
    # Save warmup checkpoint
    warmup_checkpoint_path = checkpoint_dir / "warmup_final.pt"
    save_checkpoint(model, optimizer, warmup_steps, str(warmup_checkpoint_path))
    print(f"\nWarmup complete! Saved checkpoint: {warmup_checkpoint_path}")
    
    wandb.finish()
    return warmup_steps


# =============================================================================
# Stage 2: Sparse Training
# =============================================================================

def train_sparse_step(
    model: nn.Module,
    main_optimizer: torch.optim.Optimizer,
    indexer_optimizer: torch.optim.Optimizer,
    train_loader_iter: Iterator,
    config: Dict[str, Any],
    device: torch.device,
    gradient_accumulation_steps: int = 1,
    accumulation_step: int = 0,
    helper: Optional['DSAModelHelper'] = None,  # [OPT] Reuse cached info
):
    """
    Single training step for Sparse Training Stage.
    [OPT] Uses DSAModelHelper to cache parameter lists and avoid repeated computation.
    """
    model.train()
    
    inputs, targets = next(train_loader_iter)
    inputs = inputs.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)
    
    use_amp = config.get('use_amp', False)
    amp_dtype = torch.bfloat16 if use_amp else torch.float32
    
    # [OPT] Get cached parameter lists
    if helper is not None:
        main_params = helper.main_params
        indexer_params = helper.indexer_params
        num_dsa_layers = helper.num_dsa_layers
    else:
        main_params = get_main_model_parameters(model)
        indexer_params = get_indexer_parameters(model)
        num_dsa_layers = len(get_dsa_attention_layers(model))
    
    # ========================================
    # Part 1: Main Model Training (LM Loss)
    # ========================================
    with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
        logits = model(inputs)
        lm_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        lm_loss = lm_loss / gradient_accumulation_steps
    
    if accumulation_step == 0:
        main_optimizer.zero_grad(set_to_none=True)
    
    lm_loss.backward()
    
    lm_grad_norm = 0.0
    if accumulation_step == gradient_accumulation_steps - 1:
        lm_grad_norm = clip_grad_norm_(main_params, config.get('max_grad_norm', 1.0)).item()
        main_optimizer.step()
        
        if hasattr(model, 'update_moe_biases'):
            model.update_moe_biases()
    
    # ========================================
    # Part 2: Indexer Training (KL Loss)
    # ========================================
    with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
        batch_size, seq_len = inputs.shape
        
        # [OPT] Get embeddings detached for indexer training
        x = model.token_embeddings(inputs).detach()
        
        # [OPT] Create mask once
        mask = torch.tril(torch.ones((seq_len, seq_len), device=x.device, dtype=torch.bool)) if seq_len > 1 else None
        
        # [OPT] Initialize loss on device
        total_kl_loss = x.new_zeros(())
        
        # Process through layers
        residual = None
        for block in model.layers:
            if residual is None:
                x_norm, residual = block.att_norm(x), x
            else:
                x_norm, residual = block.att_norm(x, residual)
            
            if hasattr(block.att, 'indexer'):
                # Compute KL loss with detached input
                kl_loss = block.att.compute_indexer_loss_sparse(
                    x_norm.detach(), start_pos=0, mask=mask
                )
                total_kl_loss = total_kl_loss + kl_loss
            
            # [OPT] Continue forward with no_grad for non-loss computation
            with torch.no_grad():
                if hasattr(block.att, 'indexer'):
                    attn_out, _ = block.att(x_norm, start_pos=0, mask=mask, use_sparse=True)
                else:
                    attn_out = block.att(x_norm, start_pos=0, mask=mask)
                x = block.dropout(attn_out)
                x, residual = block.ffn_norm(x, residual)
                x = block.ffn(x)
                x = block.dropout(x)
        
        kl_loss_avg = total_kl_loss / max(num_dsa_layers, 1)
        kl_loss_avg = kl_loss_avg / gradient_accumulation_steps
    
    if accumulation_step == 0:
        indexer_optimizer.zero_grad(set_to_none=True)
    
    kl_loss_avg.backward()
    
    kl_grad_norm = 0.0
    if accumulation_step == gradient_accumulation_steps - 1:
        kl_grad_norm = clip_grad_norm_(indexer_params, config.get('max_grad_norm', 1.0)).item()
        indexer_optimizer.step()
    
    return (
        lm_loss.item() * gradient_accumulation_steps,
        kl_loss_avg.item() * gradient_accumulation_steps,
        lm_grad_norm,
        kl_grad_norm
    )


def valid_sparse(
    model: nn.Module,
    val_loader: DataLoader,
    config: Dict[str, Any],
    device: torch.device
):
    """Evaluate model with sparse attention on validation data."""
    eval_model = model._orig_mod if hasattr(model, '_orig_mod') else model
    eval_model.eval()
    
    total_loss = 0.0
    num_batches = config.get('eval_batches', 100)
    
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
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1)
                )
            total_loss += loss.item()
    
    avg_loss = total_loss / num_batches
    perplexity = np.exp(avg_loss)
    return avg_loss, perplexity


def run_sparse_stage(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    config: Config,
    config_dict: Dict[str, Any],
    device: torch.device,
    checkpoint_dir: Path,
    start_step: int = 0,
    gradient_accumulation_steps: int = 1
):
    """
    Run Sparse Training Stage.
    [OPT] Uses DSAModelHelper for efficient parameter caching.
    """
    print("\n" + "=" * 60)
    print("STAGE 2: Sparse Training")
    print("=" * 60)
    
    # [OPT] Create helper for efficient parameter caching
    helper = DSAModelHelper(model)
    
    # Unfreeze all parameters
    print("\nUnfreezing all parameters...")
    helper.unfreeze_all()
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Sparse-specific hyperparameters
    sparse_lr = config_dict.get('sparse_stage_lr', 7.3e-6)
    indexer_lr = config_dict.get('sparse_indexer_lr', 1e-4)
    sparse_steps = config_dict.get('sparse_stage_steps', 15000)
    
    print(f"Main model learning rate: {sparse_lr}")
    print(f"Indexer learning rate: {indexer_lr}")
    print(f"Total steps: {sparse_steps}")
    
    # [OPT] Use cached parameter lists from helper
    main_params = helper.main_params
    indexer_params = helper.indexer_params
    
    # [OPT] Use fused optimizer if available
    use_fused = torch.cuda.is_available()
    
    main_optimizer = AdamW(
        main_params,
        lr=sparse_lr,
        betas=(config.beta1, config.beta2),
        eps=config.eps,
        weight_decay=config.weight_decay,
        fused=use_fused
    )
    
    indexer_optimizer = AdamW(
        indexer_params,
        lr=indexer_lr,
        betas=(config.beta1, config.beta2),
        eps=config.eps,
        weight_decay=0.0,
        fused=use_fused
    )
    
    # Initialize wandb
    wandb.init(
        project="DSA_Training",
        entity="scut_zeno",
        name=f"{config.run_name}_sparse",
        config={**config_dict, 'stage': 'sparse'}
    )
    
    # Training loop
    model.train()
    running_lm_loss = 0.0
    running_kl_loss = 0.0
    best_val_loss = float('inf')
    train_loader_iter = iter(train_loader)
    
    log_interval = config_dict.get('log_interval', 100)
    eval_interval = config_dict.get('eval_interval', 1000)
    
    # [OPT] Pre-compute warmup iterations
    warmup_iter = min(1000, sparse_steps // 10)
    
    for step in range(start_step, sparse_steps):
        start_time = time.time()
        
        # Learning rate schedule for main model
        lr = cos_learning_rate_schedule_with_warmup(
            step,
            max_lr=sparse_lr,
            min_lr=sparse_lr * 0.1,
            warmup_iter=warmup_iter,
            cos_iter=sparse_steps
        )
        for param_group in main_optimizer.param_groups:
            param_group['lr'] = lr
        
        # Gradient accumulation
        accumulated_lm_loss = 0.0
        accumulated_kl_loss = 0.0
        for accum_step in range(gradient_accumulation_steps):
            try:
                lm_loss, kl_loss, lm_grad, kl_grad = train_sparse_step(
                    model, main_optimizer, indexer_optimizer,
                    train_loader_iter, config_dict, device,
                    gradient_accumulation_steps, accum_step,
                    helper=helper  # [OPT] Pass helper
                )
            except StopIteration:
                train_loader_iter = iter(train_loader)
                lm_loss, kl_loss, lm_grad, kl_grad = train_sparse_step(
                    model, main_optimizer, indexer_optimizer,
                    train_loader_iter, config_dict, device,
                    gradient_accumulation_steps, accum_step,
                    helper=helper
                )
            accumulated_lm_loss += lm_loss
            accumulated_kl_loss += kl_loss
        
        avg_lm_loss = accumulated_lm_loss / gradient_accumulation_steps
        avg_kl_loss = accumulated_kl_loss / gradient_accumulation_steps
        running_lm_loss += avg_lm_loss
        running_kl_loss += avg_kl_loss
        step_time = time.time() - start_time
        
        # Logging
        if (step + 1) % log_interval == 0:
            avg_running_lm = running_lm_loss / log_interval
            avg_running_kl = running_kl_loss / log_interval
            ppl = np.exp(avg_running_lm)
            
            print(f"[Sparse] Step {step + 1:5d}/{sparse_steps} | "
                  f"LM Loss: {avg_running_lm:.4f} | PPL: {ppl:.2f} | "
                  f"KL Loss: {avg_running_kl:.4f} | "
                  f"LR: {lr:.2e} | Time: {step_time:.3f}s")
            
            wandb.log({
                'sparse/lm_loss': avg_running_lm,
                'sparse/perplexity': ppl,
                'sparse/kl_loss': avg_running_kl,
                'sparse/learning_rate': lr,
                'sparse/lm_grad_norm': lm_grad,
                'sparse/kl_grad_norm': kl_grad,
                'sparse/step_time': step_time
            }, step=step + 1)
            
            running_lm_loss = 0.0
            running_kl_loss = 0.0
        
        # Validation
        if (step + 1) % eval_interval == 0:
            print("Running validation...")
            val_loss, val_ppl = valid_sparse(model, valid_loader, config_dict, device)
            print(f"Validation | Loss: {val_loss:.4f} | PPL: {val_ppl:.2f}")
            
            wandb.log({
                'sparse/val_loss': val_loss,
                'sparse/val_perplexity': val_ppl
            }, step=step + 1)
            
            # Save checkpoint
            ckpt_path = checkpoint_dir / f"sparse_step_{step + 1:06d}.pt"
            save_checkpoint(model, main_optimizer, step + 1, str(ckpt_path))
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = checkpoint_dir / "best_model.pt"
                save_checkpoint(model, main_optimizer, step + 1, str(best_path))
                print(f"New best model! Loss: {val_loss:.4f}")
            
            model.train()
    
    # Save final checkpoint
    final_path = checkpoint_dir / "sparse_final.pt"
    save_checkpoint(model, main_optimizer, sparse_steps, str(final_path))
    print(f"\nSparse training complete! Saved checkpoint: {final_path}")
    
    wandb.finish()


def main():
    parser = argparse.ArgumentParser(description='Train DSA (DeepSeek Sparse Attention)')
    parser.add_argument('--config', type=str, required=True, help='Path to config JSON file')
    parser.add_argument('--stage', type=str, choices=['warmup', 'sparse', 'both'], default='both',
                        help='Training stage: warmup, sparse, or both')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='Number of gradient accumulation steps')
    args = parser.parse_args()
    
    # Load configuration
    config = Config.from_json(args.config)
    config_dict = config.to_dict()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Performance optimizations
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')
        torch.backends.cudnn.benchmark = True
        print("Performance optimizations enabled: TF32 matmul, cuDNN benchmark")
    
    # Set random seeds
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
    
    # Load tokenizer
    tokenizer = Tokenizer.from_files(
        vocab_filepath=config.vocab_file,
        merges_filepath=config.merges_file,
        special_tokens=config.special_tokens
    )
    config.vocab_size = len(tokenizer.decoder_vocab)
    print(f"Vocabulary size: {config.vocab_size:,}")
    
    # Prepare data
    train_data, valid_data = prepare_data(config_dict)
    
    num_workers = config.num_workers
    use_workers = num_workers > 0
    
    train_dataset = PretrainDataset(data=train_data, context_length=config.context_length)
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=use_workers,
        persistent_workers=use_workers, prefetch_factor=4 if use_workers else None,
        drop_last=True
    )
    
    valid_dataset = PretrainDataset(data=valid_data, context_length=config.context_length)
    valid_loader = DataLoader(
        valid_dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=use_workers,
        persistent_workers=use_workers, prefetch_factor=4 if use_workers else None,
        drop_last=False
    )
    
    print(f"Training dataset: {len(train_dataset):,} samples")
    print(f"Validation dataset: {len(valid_dataset):,} samples")
    
    # Initialize model
    # NOTE: Make sure attention_type is set to "DSA" in config
    if config.attention_type != "DSA":
        print(f"Warning: attention_type is '{config.attention_type}', expected 'DSA'")
        print("DSA training requires DSA attention layers in the model.")
    
    model = TransformerLM(config=config, device=device, dtype=torch.float32).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Setup checkpoint directory
    checkpoint_dir = Path(config.checkpoint_dir) / f"{config.dataset}_DSA"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoint directory: {checkpoint_dir}")
    
    # Resume from checkpoint if specified
    start_step = 0
    if args.resume:
        print(f"Loading checkpoint: {args.resume}")
        # Create a dummy optimizer for loading
        dummy_optimizer = AdamW(model.parameters(), lr=1e-4)
        start_step = load_checkpoint(args.resume, model, dummy_optimizer)
        print(f"Resumed from step {start_step}")
    
    gradient_accumulation_steps = args.gradient_accumulation_steps
    
    # Run training stages
    if args.stage in ['warmup', 'both']:
        run_warmup_stage(
            model, train_loader, valid_loader,
            config, config_dict, device, checkpoint_dir,
            gradient_accumulation_steps
        )
    
    if args.stage in ['sparse', 'both']:
        run_sparse_stage(
            model, train_loader, valid_loader,
            config, config_dict, device, checkpoint_dir,
            start_step if args.stage == 'sparse' else 0,
            gradient_accumulation_steps
        )
    
    print("\nDSA Training Complete!")


if __name__ == "__main__":
    main()
