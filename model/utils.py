import os
import math
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from typing import Union, BinaryIO, IO, Optional


# ------------------------------------
#  Problem 2: Implement Linear Module
# ------------------------------------
class Embedding(nn.Module):
    """ PyTorch implementation of Embedding, function as nn.Embedding

    For mixed precision training, embeddings should use FP32 for numerical stability.
    The autocast context will handle dtype conversion in forward pass if needed.
    """

    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings  # the size of vocab
        self.embedding_dim = embedding_dim    # dim of emb vector
        if dtype is None:
            dtype = torch.float32  # default to FP32
        # create an uninitialized tensor on requested device and dtype
        weight = torch.empty((num_embeddings, embedding_dim),
                             device=device, dtype=dtype)
        # initialize the weight matrix and warp in Parameter
        nn.init.trunc_normal_(weight, mean=0.0, std=1.0, a=-3, b=+3)
        self.weight = nn.Parameter(weight)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        ''' select the embedding vector of each token ID by indexing into an embedding matrix
            of shape (vocab_size, d_model) using token IDs with shape (batch_size, seq_len) '''
        return self.weight[token_ids.long()]  # cast to torch.int64


# -------------------------------------
#  Problem 3: Implement RMSNorm Module
# -------------------------------------
class RMSNorm(nn.Module):
    """
    PyTorch implementation of Root Mean Square Normalization with optional Fused Add & Norm.
    """

    def __init__(self, d_model: int, eps: float = 1e-5, device=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, dtype=torch.float32, device=device))

    def forward(self, x: torch.Tensor, residual: Optional[torch.Tensor] = None):
        """Forward pass with optional fused residual addition"""
        dtype = x.dtype
        if residual is None:
            x = x.float()
            # [OPT] Use rsqrt of mean(x^2) directly, avoid storing intermediate pow result
            rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
            return (self.weight * x * rms).to(dtype)
        else:
            # [OPT] Use add_ for in-place addition on new tensor (safe for autograd)
            x = x.float().add_(residual.float())
            residual = x  # share memory, no copy needed
            rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
            return (self.weight * x * rms).to(dtype), residual.to(dtype)


# ---------------------------------------------------
#  Problem 11: Implement Cross-Entropy Loss Function
# ---------------------------------------------------
def cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """ Compute cross-entropy loss using PyTorch's native F.cross_entropy

    Args:
        logits (torch.Tensor): The input logits of shape (batch_size, num_classes).
        target (torch.Tensor): The target labels of shape (batch_size,) with class indices.

    Returns:
        torch.Tensor: The computed cross-entropy loss as a scalar tensor.

    Note:
        Using F.log_softmax for full torch.compile + AMP compatibility.
        Native implementation handles mixed precision and compile optimization automatically.
    """
    # [OPT] Use F.log_softmax to fuse log + softmax, avoiding temporary tensors
    # This replaces: logits - logits.amax() -> exp() -> sum() -> log() -> subtract
    # with a single fused kernel that is numerically stable
    import torch.nn.functional as F
    log_probs = F.log_softmax(logits, dim=-1)
    # Step 2: Gather the log probabilities corresponding to the target labels
    target_probs = log_probs.gather(dim=-1, index=target.unsqueeze(-1))
    # Step 3: Compute the negative log likelihood loss
    return -target_probs.squeeze(-1).mean()


# ------------------------------------------------------------------
#  Problem 14: Implement Cosine learning rate scheduler with warmup
# ------------------------------------------------------------------
def cos_learning_rate_schedule_with_warmup(t: int, max_lr: float, min_lr: float, warmup_iter: int, cos_iter: int):
    """  """
    # step 1: linear warmup phase
    if t < warmup_iter:
        return max_lr * (t / warmup_iter)
    # step 2: cosine annealing
    elif t < cos_iter:
        factor = (t - warmup_iter) / (cos_iter - warmup_iter)
        return min_lr + 0.5 * (1 + math.cos(factor * 3.1415926)) * (max_lr - min_lr)
    # step 3: post annealing
    else:
        return min_lr


# --------------------------------------------------
#  Problem 16: Implement data loading function
# ---------------------------------------------
def data_loading(x: np.ndarray, batch_size: int, context_length: int, device=None):
    """ Generates a random batch of training data from a 1D sequence of token IDs """
    # choose random start indices for each sequence in the batch
    start_indices = np.random.randint(
        0, len(x) - context_length, size=batch_size)

    # create input and target sequences based on start indices
    input_sequences = [x[i: i + context_length] for i in start_indices]
    target_sequences = [x[i + 1: i + 1 + context_length]
                        for i in start_indices]

    inputs_np, targets_np = np.array(
        input_sequences), np.array(target_sequences)

    # convert to PyTorch tensors and move to the specified device
    inputs = torch.from_numpy(inputs_np).to(torch.long).to(device)
    targets = torch.from_numpy(targets_np).to(torch.long).to(device)

    return inputs, targets


class ShortConv(nn.Module):
    """ 
    A lightweight layer that combines Grouped RMSNorm, Causal Depthwise Convolution, and
    activation. It is designed to capture local, short-term dependencies in sequence data.

    Input  Shape: (Batch, Time, Groups, Channels)
    Output Shape: (Batch, Time, Groups, Channels)
    """

    def __init__(
        self,
        hidden_size: int,
        kernel_size: int = 4,
        dilation: int = 1,
        norm_eps: float = 1e-5,
        hc_mult: int = 4,
        activation: bool = True,
    ):
        super().__init__()
        self.hc_mult = hc_mult
        self.activation = activation

        total_channels = hidden_size * hc_mult

        self.conv = nn.Conv1d(
            in_channels=total_channels,
            out_channels=total_channels,
            kernel_size=kernel_size,
            # each channel is convolved independently
            groups=total_channels,
            bias=False,
            # prepare tensor for causal slicing
            padding=(kernel_size - 1) * dilation,
            dilation=dilation,
        )
        # apply separaete RMSNorm for each group
        self.norms = nn.ModuleList([
            nn.RMSNorm(hidden_size, eps=norm_eps)
            for _ in range(hc_mult)
        ])

        if self.activation:
            self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward pass through the ShortConv layer """
        B, T, G, C = x.shape
        assert G == self.hc_mult, f"Input groups {G} != hc_mult {self.hc_mult}"

        # group-wise normalization and concatenation
        normed_chunks = []
        for i in range(G):
            chunk = x[:, :, i, :]
            normed_chunks.append(self.norms[i](chunk))
        x_norm = torch.cat(normed_chunks, dim=-1)

        x_bct = x_norm.transpose(1, 2)  # prepare tensor for convolution
        y_bct = self.conv(x_bct)        # apply 1D convolution
        y_bct = y_bct[..., :T]          # causal slicing to remove future info

        if self.activation:
            y_bct = self.act_fn(y_bct)

        return y_bct.transpose(1, 2).view(B, T, G, C).contiguous()


# ---------------------------------------------------------
#  Problem 17: Implement checkpoint save and load function
# ---------------------------------------------------------
def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    iteration: int,
    out: Union[str, os.PathLike, BinaryIO, IO[bytes]]
):
    """
    Saves the model, optimizer state, and current iteration to a single checkpoint file.

    Args:
        model (nn.Module): The PyTorch model to save.
        optimizer (optim.Optimizer): The optimizer whose state needs to be saved.
        iteration (int): The current training iteration number.
        out (Union[str, os.PathLike, BinaryIO, IO[bytes]]): The path for the checkpoint file.
            Will save everything to a single .pt file.
    """
    # Convert path to string and ensure .pt extension
    checkpoint_path = str(out)
    if not checkpoint_path.endswith('.pt'):
        checkpoint_path = f"{checkpoint_path}.pt"

    # Save everything in a single .pt file
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration,
    }
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(
    src: Union[str, os.PathLike, BinaryIO, IO[bytes]],
    model: nn.Module,
    optimizer: optim.Optimizer
) -> int:
    """
    Loads a checkpoint from a .pt file and restores the model and optimizer states.

    Args:
        src (Union[str, os.PathLike, BinaryIO, IO[bytes]]): The path to the checkpoint file.
            Will load from a .pt file containing model, optimizer, and iteration.
        model (nn.Module): The model instance to load the state into.
        optimizer (optim.Optimizer): The optimizer instance to load the state into.

    Returns:
        int: The iteration number restored from the checkpoint, allowing training to resume.
    """
    # Convert path to string and ensure .pt extension
    checkpoint_path = str(src)
    if not checkpoint_path.endswith('.pt'):
        checkpoint_path = f"{checkpoint_path}.pt"

    # Determine the device the model is currently on
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device('cpu')

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get the model state dict from checkpoint
    state_dict = checkpoint['model_state_dict']

    # If the checkpoint has _orig_mod. prefix, remove it
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)  # Restore model state

    # Restore optimizer state if optimizer is provided
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Return the iteration number
    return checkpoint['iteration']
