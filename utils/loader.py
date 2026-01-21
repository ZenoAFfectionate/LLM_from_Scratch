import os
import torch
from torch import nn
from glob import glob
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    """Default weight loader that directly copies weights to parameter."""
    param.data.copy_(loaded_weight)


def load_model_safetensors(model: nn.Module, path: str):
    """
    Load model weights from safetensors files (original nano-vllm format).

    Args:
        model: The model instance to load weights into
        path: Directory containing .safetensors files
    """
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))


def load_model_checkpoint(model: nn.Module, path: str):
    """
    Load model weights from a PyTorch checkpoint file (.pt format).

    This loader is compatible with checkpoints saved by the training code using
    save_checkpoint() from model.utils, which saves model_state_dict, optimizer_state_dict,
    and iteration number.
    """
    # Handle both file path and directory
    checkpoint_path = path
    if os.path.isdir(path):
        # Look for best_model.pt, final_model.pt, or latest checkpoint
        candidates = [
            os.path.join(path, "best_model.pt"),
            os.path.join(path, "final_model.pt"),
        ]
        # Also look for numbered checkpoints
        checkpoint_files = sorted(glob(os.path.join(path, "checkpoint_iter_*.pt")))
        if checkpoint_files:
            candidates.append(checkpoint_files[-1])  # Latest checkpoint

        for candidate in candidates:
            if os.path.exists(candidate):
                checkpoint_path = candidate
                break
        else:
            raise FileNotFoundError(f"No checkpoint file found in {path}")

    # Ensure .pt extension
    if not checkpoint_path.endswith('.pt'):
        checkpoint_path = f"{checkpoint_path}.pt"

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading checkpoint from: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Get state dict (handle both direct state dict and nested format)
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            iteration = checkpoint.get('iteration', 0)
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            iteration = checkpoint.get('iteration', 0)
        else:
            # Assume it's a direct state dict
            state_dict = checkpoint
            iteration = 0
    else:
        state_dict = checkpoint
        iteration = 0

    # Handle _orig_mod. prefix from torch.compile
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

    # Load state dict with strict=False to handle any missing/extra keys
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    if missing_keys:
        print(f"Warning: Missing keys in checkpoint: {missing_keys}")
    if unexpected_keys:
        print(f"Warning: Unexpected keys in checkpoint: {unexpected_keys}")

    print(f"Loaded checkpoint from iteration {iteration}")
    return iteration


def load_model(model: nn.Module, path: str):
    """
    Unified model loader that automatically detects checkpoint format.

    Supports:
    - .pt checkpoint files (from training code)
    - .safetensors files (from HuggingFace/nano-vllm)

    Args:
        model: The model instance to load weights into
        path: Path to checkpoint file or directory
    """
    if os.path.isfile(path) and path.endswith('.pt'):
        return load_model_checkpoint(model, path)
    elif os.path.isdir(path):
        pt_files = glob(os.path.join(path, "*.pt"))
        safetensor_files = glob(os.path.join(path, "*.safetensors"))
        # prioritize .pt files if both exist
        if pt_files:
            return load_model_checkpoint(model, path)
        elif safetensor_files:
            return load_model_safetensors(model, path)
        else:
            raise FileNotFoundError(f"No checkpoint files found in {path}")
    else:
        if os.path.exists(path + '.pt'):
            return load_model_checkpoint(model, path + '.pt')
        raise FileNotFoundError(f"Checkpoint not found: {path}")
