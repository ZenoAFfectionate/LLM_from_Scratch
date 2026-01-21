"""
LoRA (Low-Rank Adaptation) Implementation

This module implements LoRA for parameter-efficient fine-tuning of language models.
LoRA adds trainable low-rank decomposition matrices to existing weights while keeping
the original model frozen.
"""

import torch
import torch.nn as nn
from model.architecture.mlp import Linear as CustomLinear


class LoRA(nn.Module):
    """
    LoRA layer that applies low-rank adaptation to a linear layer.

    Instead of fine-tuning W directly, LoRA learns delta_W = BA where:
    - B: (out_features, rank)
    - A: (rank, in_features)
    - rank << min(in_features, out_features)

    The forward pass computes: output = Wx + BAx
    """
    def __init__(self, in_features: int, out_features: int, rank: int = 8):
        """
        Args:
            in_features: Input dimension
            out_features: Output dimension
            rank: Rank of the low-rank decomposition (smaller = fewer parameters)
        """
        super().__init__()
        self.rank = rank

        # Low-rank matrices A and B
        self.A = nn.Linear(in_features, rank, bias=False)
        self.B = nn.Linear(rank, out_features, bias=False)

        # Initialize A with small Gaussian noise
        nn.init.normal_(self.A.weight, mean=0.0, std=0.02)

        # Initialize B to zero so initial delta_W = BA = 0
        nn.init.zeros_(self.B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through LoRA layer.

        Args:
            x: Input tensor (..., in_features)

        Returns:
            delta_W * x where delta_W = BA (..., out_features)
        """
        return self.B(self.A(x))


def apply_lora_to_linear(module: nn.Module, rank: int, device: torch.device) -> nn.Module:
    """
    Apply LoRA to a single linear layer (works with both nn.Linear and custom Linear).

    This creates a LoRA layer and modifies the forward pass to include it.

    Args:
        module: The linear layer to apply LoRA to
        rank: LoRA rank
        device: Device to place LoRA weights on

    Returns:
        Modified module with LoRA applied
    """
    in_features = module.in_features
    out_features = module.out_features

    # Create LoRA layer
    lora = LoRA(in_features, out_features, rank=rank).to(device)

    # Register LoRA as a submodule (this is important for PyTorch's module tracking)
    module.add_module("lora", lora)

    # Save original forward method
    _original_forward = module.forward

    # Create new forward that includes LoRA
    def forward_with_lora(self, x):
        """Modified forward: output = W * x + delta_W * x"""
        return _original_forward(x) + self.lora(x)

    # Bind the method to the instance
    import types
    module.forward = types.MethodType(forward_with_lora, module)

    return module


def apply_lora(model: nn.Module, rank: int = 8, target_modules: list = None) -> nn.Module:
    """
    Apply LoRA to specified modules in the model.

    By default, applies LoRA to all linear layers in attention and MLP blocks.

    Args:
        model: The model to apply LoRA to
        rank: LoRA rank (default: 8)
        target_modules: List of module name patterns to apply LoRA to.
                       If None, applies to attention and MLP linear layers.

    Returns:
        Model with LoRA applied
    """
    device = next(model.parameters()).device

    # Default target modules: attention Q,K,V,O and MLP projections
    if target_modules is None:
        target_modules = ['q_proj', 'k_proj', 'v_proj', 'output_proj', 'w1', 'w2', 'w3']

    # Collect all modules first (before modifying the model structure)
    modules_to_modify = []
    for name, module in model.named_modules():
        # Check if this is a linear layer (either nn.Linear or custom Linear)
        is_linear = isinstance(module, (nn.Linear, CustomLinear))

        if is_linear:
            # Check if name matches any target pattern
            should_apply = any(target in name for target in target_modules)

            if should_apply:
                modules_to_modify.append((name, module))

    # Now apply LoRA to collected modules
    lora_count = 0
    for name, module in modules_to_modify:
        apply_lora_to_linear(module, rank, device)
        lora_count += 1

    print(f"Applied LoRA to {lora_count} linear layers with rank={rank}")
    return model


def freeze_non_lora_params(model: nn.Module):
    """
    Freeze all parameters except LoRA parameters.

    This ensures only LoRA weights are trained during fine-tuning.

    Args:
        model: Model with LoRA applied
    """
    frozen_count = 0
    lora_count = 0

    for name, param in model.named_parameters():
        if 'lora' in name:
            param.requires_grad = True
            lora_count += 1
        else:
            param.requires_grad = False
            frozen_count += 1

    print(f"Frozen {frozen_count} parameters, kept {lora_count} LoRA parameters trainable")


def get_lora_params(model: nn.Module) -> list:
    """
    Get all LoRA parameters from the model.

    Args:
        model: Model with LoRA applied

    Returns:
        List of LoRA parameters
    """
    lora_params = []
    for name, param in model.named_parameters():
        if 'lora' in name and param.requires_grad:
            lora_params.append(param)
    return lora_params


def save_lora_weights(model: nn.Module, path: str):
    """
    Save only LoRA weights to a checkpoint file.

    This saves a much smaller checkpoint containing only the LoRA adapter weights.

    Args:
        model: Model with LoRA applied
        path: Path to save LoRA weights
    """
    lora_state_dict = {}

    for name, module in model.named_modules():
        if hasattr(module, 'lora'):
            # Save LoRA weights with module name as prefix
            lora_weights = {
                f'{name}.lora.{k}': v
                for k, v in module.lora.state_dict().items()
            }
            lora_state_dict.update(lora_weights)

    torch.save(lora_state_dict, path)

    # Calculate size
    total_params = sum(p.numel() for p in lora_state_dict.values())
    print(f"Saved {len(lora_state_dict)} LoRA tensors ({total_params:,} parameters) to {path}")


def load_lora_weights(model: nn.Module, path: str):
    """
    Load LoRA weights from a checkpoint file.

    Args:
        model: Model with LoRA applied (must have same architecture)
        path: Path to load LoRA weights from
    """
    device = next(model.parameters()).device
    lora_state_dict = torch.load(path, map_location=device)

    loaded_count = 0
    for name, module in model.named_modules():
        if hasattr(module, 'lora'):
            # Extract LoRA weights for this module
            module_lora_state = {
                k.replace(f'{name}.lora.', ''): v
                for k, v in lora_state_dict.items()
                if k.startswith(f'{name}.lora.')
            }

            if module_lora_state:
                module.lora.load_state_dict(module_lora_state)
                loaded_count += 1

    print(f"Loaded LoRA weights for {loaded_count} modules from {path}")


def merge_lora_weights(model: nn.Module) -> nn.Module:
    """
    Merge LoRA weights into the base model weights.

    After merging, LoRA adapters are removed and the model can be used
    as a standard model without LoRA overhead.

    This computes: W_new = W_original + BA

    Args:
        model: Model with LoRA applied

    Returns:
        Model with merged weights (LoRA removed)
    """
    merged_count = 0

    for name, module in model.named_modules():
        is_linear = isinstance(module, (nn.Linear, CustomLinear))

        if is_linear and hasattr(module, 'lora'):
            # Get LoRA matrices
            lora = module.lora

            # Compute delta_W = B @ A
            with torch.no_grad():
                delta_w = lora.B.weight @ lora.A.weight  # (out_features, in_features)

                # Merge: W_new = W + delta_W
                module.weight.data += delta_w

            # Remove LoRA from this module
            delattr(module, 'lora')
            merged_count += 1

    print(f"Merged LoRA weights into {merged_count} layers")
    return model
