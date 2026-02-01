import torch
import torch.nn as nn
from torch.optim import AdamW
from typing import Dict, Tuple, List

# define coefficients here:
a = +3.4445
b = -4.7750
c = +2.0315


def newtonschulz5_orthogonalization(G, steps: int):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval.
    """
    assert G.ndim >= 2, "Input tensor must be at least 2D"
    X = G.to(torch.bfloat16)

    if G.size(-2) > G.size(-1): X = X.mT  # transpose for tall matrices

    # ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A  # quintic computation strategy
        X = a * X + B @ X

    if G.size(-2) > G.size(-1): X = X.mT  # transpose for tall matrices

    return X


def muon_update(grad, momentum, scaling_factor, beta=0.95, ns_steps=5, nesterov=True):
    """
    Compute Muon update with momentum and orthogonalization.

    Args:
        grad: Gradient tensor
        momentum: Momentum buffer
        beta: Momentum coefficient (default: 0.95)
        ns_steps: Number of Newton-Schulz iterations (default: 5)
        nesterov: Whether to use Nesterov momentum (default: True)

    Returns:
        Orthogonalized update tensor
    """
    # update momentum with EMA
    momentum.lerp_(grad, 1 - beta)
    # Use non-in-place lerp to avoid modifying the gradient tensor
    update = grad.lerp(momentum, beta) if nesterov else momentum
    # for the case of conv filters
    if update.ndim == 4: 
        update = update.view(len(update), -1)
    # perform Newton-Schulz orthogonalization
    update = newtonschulz5_orthogonalization(update, steps=ns_steps)
    return update * scaling_factor


class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz (This implementation contain aux Adam)

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. For efficient orthogonalization we use a Newton-Schulz iteration, which has the
    advantage that it can be stably run in bfloat16 on the GPU.

    Muon should only be used for hidden weight layers. The input embedding, final output layer,
    and any internal gains or biases should be optimized using a standard method such as AdamW.

    Arguments:
        params: Iterable of parameters to optimize or dicts defining parameter groups
        lr: Learning rate (default: 0.02) - in units of spectral norm per update
        weight_decay: Weight decay coefficient (default: 0) - AdamW-style weight decay
        momentum: Momentum coefficient (default: 0.95)
        nesterov: Whether to use Nesterov momentum (default: True)
        ns_steps: Number of Newton-Schulz iterations (default: 5)
    """
    def __init__(self, params, lr=0.02, weight_decay=0, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """ Performs a single optimization step. """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                if len(state) == 0:
                    # initialize momentum buffer
                    state["momentum_buffer"] = torch.zeros_like(p)
                    # compute and cache scaling factor
                    if p.ndim >= 2:
                        m, n = p.shape[-2], p.shape[-1]
                        scale_value = max(1.0, m / n) ** 0.5
                        state["scaling_factor"] = scale_value
                    else:
                        state["scaling_factor"] = 1.0

                # compute Muon update with orthogonalization
                update = muon_update(
                    p.grad,
                    state["momentum_buffer"],  # 
                    state["scaling_factor"],   # 
                    beta=group["momentum"],
                    ns_steps=group["ns_steps"],
                    nesterov=group["nesterov"]
                )

                # apply AdamW-style weight decay and update parameters
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update.reshape(p.shape), alpha=-group["lr"])

        return loss


def separate_params_for_muon(
    model: nn.Module,
    muon_lr: float = 0.02,
    adamw_lr: float = 3e-4,
    muon_weight_decay: float = 0.0,
    adamw_weight_decay: float = 0.1,
    support_engram: bool = False,
) -> Tuple[List[Dict], List[Dict], Dict[str, int]]:
    """
    Separate model parameters into Muon and AdamW groups.
    
    Muon is used for 2D hidden weight matrices (attention and FFN layers).
    AdamW is used for:
    - Token embeddings
    - lm_head
    - Engram multi-head embeddings (when support_engram=True)
    - Biases and 1D parameters (norms, etc.)
    
    Args:
        model: The transformer model (with or without Engram)
        muon_lr: Learning rate for Muon optimizer
        adamw_lr: Learning rate for AdamW optimizer
        muon_weight_decay: Weight decay for Muon parameters
        adamw_weight_decay: Weight decay for AdamW parameters
        support_engram: Whether to handle Engram-specific parameters
        
    Returns:
        Tuple of (muon_param_groups, adamw_param_groups, stats_dict)
        stats_dict contains parameter counts for logging
    """
    muon_params = []
    adamw_params = []
    adamw_params_no_decay = []  # For biases and 1D params
    
    # Track parameter counts for logging
    muon_param_count = 0
    adamw_param_count = 0
    adamw_no_decay_count = 0
    engram_param_count = 0
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        # Determine parameter type based on name and shape
        is_embedding = 'token_embeddings' in name or 'embedding' in name.lower()
        is_lm_head = 'lm_head' in name
        is_bias = 'bias' in name or name.endswith('.b')
        is_norm = 'norm' in name.lower() or 'ln' in name.lower() or 'layernorm' in name.lower()
        is_scale = 'scale' in name or 'gamma' in name
        
        # Check if this is an Engram parameter (only when support_engram is enabled)
        is_engram = support_engram and ('engram' in name.lower())
        is_engram_embedding = is_engram and ('multi_head_embedding' in name or 'embedding' in name.lower())
        
        # Check if parameter is 2D and suitable for Muon
        is_2d_weight = param.ndim >= 2
        
        # Muon criteria: 2D weight that is NOT embedding, lm_head, Engram embedding, or special params
        use_muon = (
            is_2d_weight and
            not is_embedding and
            not is_lm_head and
            not is_bias and
            not is_norm and
            not is_scale and
            not is_engram_embedding  # Engram embeddings should use AdamW
        )
        
        if use_muon:
            muon_params.append(param)
            muon_param_count += param.numel()
        elif is_bias or is_norm or is_scale or param.ndim == 1:
            # No weight decay for biases and 1D params
            adamw_params_no_decay.append(param)
            adamw_no_decay_count += param.numel()
        else:
            # Embeddings, lm_head, and Engram embeddings with weight decay
            adamw_params.append(param)
            adamw_param_count += param.numel()
            if is_engram:
                engram_param_count += param.numel()
    
    # Print parameter distribution
    total_params = muon_param_count + adamw_param_count + adamw_no_decay_count
    print(f"\n{'='*60}")
    if support_engram:
        print("Parameter Distribution for Muon + AdamW Optimization (Engram):")
    else:
        print("Parameter Distribution for Muon + AdamW Optimization:")
    print(f"{'='*60}")
    print(f"  Muon parameters (2D hidden weights):    {muon_param_count:>12,} ({100*muon_param_count/total_params:.1f}%)")
    print(f"  AdamW parameters (with decay):          {adamw_param_count:>12,} ({100*adamw_param_count/total_params:.1f}%)")
    if support_engram and engram_param_count > 0:
        print(f"    - Engram parameters:                  {engram_param_count:>12,} ({100*engram_param_count/total_params:.1f}%)")
    print(f"  AdamW parameters (no decay):            {adamw_no_decay_count:>12,} ({100*adamw_no_decay_count/total_params:.1f}%)")
    print(f"  Total trainable parameters:             {total_params:>12,}")
    print(f"{'='*60}\n")
    
    # Build param groups
    muon_param_groups = [
        {'params': muon_params, 'lr': muon_lr, 'weight_decay': muon_weight_decay}
    ]
    
    adamw_param_groups = [
        {'params': adamw_params, 'lr': adamw_lr, 'weight_decay': adamw_weight_decay},
        {'params': adamw_params_no_decay, 'lr': adamw_lr, 'weight_decay': 0.0}
    ]
    
    # Statistics for external use
    stats = {
        'muon_params': muon_param_count,
        'adamw_params': adamw_param_count,
        'adamw_no_decay_params': adamw_no_decay_count,
        'engram_params': engram_param_count,
        'total_params': total_params,
    }
    
    return muon_param_groups, adamw_param_groups, stats


class MuonAdamWOptimizer:
    """
    Combined optimizer that uses Muon for 2D hidden weights and AdamW for the rest.
    
    This class wraps both optimizers and provides a unified interface for:
    - zero_grad()
    - step()
    - state_dict() / load_state_dict()
    - Learning rate scheduling
    
    Supports both standard Transformer models and models with Engram modules.
    When support_engram=True, Engram embedding tables are optimized with AdamW
    while Engram projection layers use Muon.
    """

    def __init__(
        self,
        model: nn.Module,
        muon_lr: float = 0.02,
        adamw_lr: float = 3e-4,
        muon_momentum: float = 0.95,
        muon_nesterov: bool = True,
        muon_ns_steps: int = 5,
        muon_weight_decay: float = 0.0,
        adamw_betas: Tuple[float, float] = (0.9, 0.999),
        adamw_eps: float = 1e-8,
        adamw_weight_decay: float = 0.1,
        support_engram: bool = False,
    ):
        """
        Initialize combined Muon + AdamW optimizer.
        
        Args:
            model: The transformer model to optimize (with or without Engram)
            muon_lr: Learning rate for Muon (default: 0.02, in spectral norm units)
            adamw_lr: Learning rate for AdamW (default: 3e-4)
            muon_momentum: Momentum coefficient for Muon (default: 0.95)
            muon_nesterov: Whether to use Nesterov momentum in Muon (default: True)
            muon_ns_steps: Number of Newton-Schulz iterations in Muon (default: 5)
            muon_weight_decay: Weight decay for Muon parameters (default: 0.0)
            adamw_betas: Adam beta coefficients (default: (0.9, 0.999))
            adamw_eps: Adam epsilon (default: 1e-8)
            adamw_weight_decay: Weight decay for AdamW parameters (default: 0.1)
            support_engram: Whether to handle Engram-specific parameters (default: False)
        """
        self.support_engram = support_engram
        
        # Separate parameters (with or without Engram support)
        muon_param_groups, adamw_param_groups, self.param_stats = separate_params_for_muon(
            model,
            muon_lr=muon_lr,
            adamw_lr=adamw_lr,
            muon_weight_decay=muon_weight_decay,
            adamw_weight_decay=adamw_weight_decay,
            support_engram=support_engram,
        )

        # Store base learning rates for scheduling
        self.muon_base_lr = muon_lr
        self.adamw_base_lr = adamw_lr

        # Initialize Muon optimizer for 2D hidden weights
        self.muon_optimizer = Muon(
            muon_param_groups,
            lr=muon_lr,
            momentum=muon_momentum,
            nesterov=muon_nesterov,
            ns_steps=muon_ns_steps,
            weight_decay=muon_weight_decay,
        )

        # Initialize AdamW optimizer for embeddings, lm_head, Engram embeddings, biases, etc.
        self.adamw_optimizer = AdamW(
            adamw_param_groups,
            lr=adamw_lr,
            betas=adamw_betas,
            eps=adamw_eps,
            weight_decay=adamw_weight_decay,
            fused=True,  # Use fused AdamW for better performance
        )

        print(f"Muon optimizer: lr={muon_lr}, momentum={muon_momentum}, ns_steps={muon_ns_steps}")
        print(f"AdamW optimizer: lr={adamw_lr}, betas={adamw_betas}, weight_decay={adamw_weight_decay}")

    def zero_grad(self, set_to_none: bool = True):
        """Zero gradients for both optimizers."""
        self.muon_optimizer.zero_grad(set_to_none=set_to_none)
        self.adamw_optimizer.zero_grad(set_to_none=set_to_none)

    def step(self):
        """Perform optimization step for both optimizers."""
        self.muon_optimizer.step()
        self.adamw_optimizer.step()

    def set_lr(self, lr_scale: float):
        """
        Set learning rate as a fraction of base learning rates.
        
        Args:
            lr_scale: Scale factor to apply to base learning rates (0.0 to 1.0)
        """
        muon_lr = self.muon_base_lr * lr_scale
        adamw_lr = self.adamw_base_lr * lr_scale

        for param_group in self.muon_optimizer.param_groups:
            param_group['lr'] = muon_lr

        for param_group in self.adamw_optimizer.param_groups:
            param_group['lr'] = adamw_lr

    def set_lr_absolute(self, muon_lr: float, adamw_lr: float):
        """
        Set absolute learning rates for both optimizers.
        
        Args:
            muon_lr: Learning rate for Muon optimizer
            adamw_lr: Learning rate for AdamW optimizer
        """
        for param_group in self.muon_optimizer.param_groups:
            param_group['lr'] = muon_lr

        for param_group in self.adamw_optimizer.param_groups:
            param_group['lr'] = adamw_lr

    def state_dict(self) -> Dict:
        """Return combined state dict."""
        return {
            'muon': self.muon_optimizer.state_dict(),
            'adamw': self.adamw_optimizer.state_dict(),
            'muon_base_lr': self.muon_base_lr,
            'adamw_base_lr': self.adamw_base_lr,
        }

    def load_state_dict(self, state_dict: Dict):
        """Load combined state dict."""
        self.muon_optimizer.load_state_dict(state_dict['muon'])
        self.adamw_optimizer.load_state_dict(state_dict['adamw'])
        self.muon_base_lr = state_dict.get('muon_base_lr', self.muon_base_lr)
        self.adamw_base_lr = state_dict.get('adamw_base_lr', self.adamw_base_lr)

    @property
    def param_groups(self):
        """Return all parameter groups (for compatibility)."""
        return self.muon_optimizer.param_groups + self.adamw_optimizer.param_groups
