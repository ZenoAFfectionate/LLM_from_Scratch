import os
import math
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from collections.abc import Iterable

from typing import Union, BinaryIO, IO

# ------------------------------------
#  Problem 1: Implement Linear Module
# ------------------------------------
class Linear(nn.Module):
    """ A minimal Linear-like module without bias.
        weight shape: (out_features, in_features)  """
    def __init__(self, in_features, out_features, device=None, dtype=None):
        '''  '''
        super().__init__()
        self.in_features  = in_features   # input  feature dimension
        self.out_features = out_features  # output feature dimension
        # create an uninitialized tensor on requested device and dtype
        weight = torch.empty((out_features, in_features), device=device, dtype=dtype)
        # initialize the weight matrix and warp in Parameter
        std = math.sqrt(2.0 / (in_features + out_features))
        nn.init.trunc_normal_(weight, mean=0.0, std=std, a=-3.0*std, b=+3.0*std)
        self.weight = nn.Parameter(weight)

    def forward(self, x):
        if x.shape[-1] != self.in_features:
            raise RuntimeError(f"Expected input last dim {self.in_features}, got {x.shape[-1]}")
        return torch.matmul(x, self.weight.t())


# ------------------------------------
#  Problem 2: Implement Linear Module
# ------------------------------------
class Embedding(nn.Module):
    """ PyTorch implementation of Embedding, function as nn.Embedding """

    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        '''  '''
        super().__init__()
        self.num_embeddings = num_embeddings  # the size of vocab
        self.embedding_dim = embedding_dim    # dim of emb vector
        # create an uninitialized tensor on requested device and dtype
        weight = torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)
        # initialize the weight matrix and warp in Parameter
        nn.init.trunc_normal_(weight, mean=0.0, std=1.0, a=-3, b=+3)
        self.weight = nn.Parameter(weight)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        ''' select the embedding vector of each token ID by indexing into an embedding matrix
            of shape (vocab_size, d_model) using token IDs with shape (batch_size, seq_len) '''
        return self.weight[token_ids.long()]  # remember cast the tensor to torch.int64 type


# -------------------------------------
#  Problem 3: Implement RMSNorm Module
# -------------------------------------
class RMSNorm(nn.Module):
    """ PyTorch implementation of Root Mean Square Normalization """
    def __init__(self,  d_model: int, eps: float = 1e-5, device=None, dtype=None):
        ''' '''
        super().__init__()
        self.eps = eps  # 
        # initialize learnable 'gain' parameter
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ''' Process an input tensor of shape (batch_size, seq_len, d_model) 
            and return a tensor of the same shape. ''' 
        def _norm(x): return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * _norm(x.float()).type_as(x)


# --------------------------------------------------------
#  Problem 4: Implement SwiGLU FeedForward Network Module
# --------------------------------------------------------
def silu(x: torch.Tensor) -> torch.Tensor:
    """ SiLU activation function, also known as the swish function """
    return x * torch.sigmoid(x)

class SwiGLU_FFN(nn.Module):
    """ Special SwiGLU feed-forward network implementation """
    def __init__(self, d_model: int, d_ff: int = None, device=None, dtype=None):
        '''  '''
        super().__init__()
        if d_ff is None: d_ff = 64 * ((int(d_model * 8 / 3) + 64 - 1) // 64)
        # initialize three linear projection for SwiGLU:
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)  # shape: (d_ff, d_model)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)  # shape: (d_ff, d_model)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)  # shape: (d_model, d_ff)
        # self.dropout = nn.Dropout(dropout=0.1)  # no need Dropout here actually ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(silu(self.w1(x)) * self.w3(x))


# ---------------------------------------
#  Problem 5: Implement Softmax Function
# ---------------------------------------
def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """ PyTorch implementation of numerically stable softmax function """
    x_max = x.amax(dim=dim, keepdim=True)  # find the max
    x_exp = torch.exp(x - x_max)           # subract this
    return x_exp / x_exp.sum(dim=dim, keepdim=True)  # broadcast


# ---------------------------------------------------
#  Problem 11: Implement Cross-Entropy Loss Function
# ---------------------------------------------------
def cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """ Compute the cross-entropy loss between logits and target labels.

    Args:
        logits (torch.Tensor): The input logits of shape (batch_size, num_classes).
        target (torch.Tensor): The target labels of shape (batch_size,) with class indices.

    Returns:
        torch.Tensor: The computed cross-entropy loss as a scalar tensor.
    """
    logits = logits - logits.amax(dim=-1, keepdim=True)
    # Step 1: Compute log-softmax of logits for numerical stability
    predic_probs = logits - torch.log(torch.exp(logits).sum(dim=-1, keepdim=True))
    # Step 2: Gather the log probabilities corresponding to the target labels
    target_probs = predic_probs.gather(dim=-1, index=target.unsqueeze(-1))
    # Step 3: Compute the negative log likelihood loss
    return -target_probs.squeeze(-1).mean()


# --------------------------------------------------
#  Problem 15: Implement Gradient Clipping Function
# --------------------------------------------------
def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> torch.Tensor:
    """Clips gradient norm of an iterable of parameters.

    Args:
        parameters (Iterable[torch.nn.Parameter]): An iterable of model parameters.
        max_l2_norm (float): The maximum allowed norm of the gradients.

    Returns:
        torch.Tensor: The total norm of the parameters (viewed as a single vector).
    """
    # consider parameters whose requires_grad=True 
    grads = [p.grad for p in parameters if p.grad is not None]

    if len(grads) == 0: return torch.tensor(0.)

    # 1. calculate the l2-norm of each gradient
    # 2. calculate the l2-norm of gradient norm
    total_norm = torch.norm(torch.stack([torch.norm(g.detach(), 2.0) for g in grads]), 2.0)

    clip_coef = max_l2_norm / (total_norm + 1e-6)
    # modify each parameter gradient in place
    if clip_coef < 1:
        for g in grads:
            g.detach().mul_(clip_coef)

    return total_norm


# ---------------------------------------------
#  Problem 16: Implement data loading function
# ---------------------------------------------
def data_loading(x: np.ndarray, batch_size: int, context_length: int, device=None):
    """ Generates a random batch of training data from a 1D sequence of token IDs """
    # choose random start indices for each sequence in the batch
    start_indices = np.random.randint(0, len(x) - context_length, size=batch_size)

    # create input and target sequences based on start indices
    input_sequences  = [x[i : i + context_length] for i in start_indices]
    target_sequences = [x[i + 1 : i + 1 + context_length] for i in start_indices]

    inputs_np, targets_np = np.array(input_sequences), np.array(target_sequences)

    # convert to PyTorch tensors and move to the specified device
    inputs  = torch.from_numpy(inputs_np).to(torch.long).to(device)
    targets = torch.from_numpy(targets_np).to(torch.long).to(device)

    return inputs, targets

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
    Saves the model, optimizer state, and current iteration to a checkpoint file or buffer.

    Args:
        model (nn.Module): The PyTorch model to save.
        optimizer (optim.Optimizer): The optimizer whose state needs to be saved.
        iteration (int): The current training iteration number.
        out (Union[str, os.PathLike, BinaryIO, IO[bytes]]): The path or file-like object
            to which the checkpoint will be written.
    """
    # create a dictionary to hold all the state we need to save.
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration,
    }

    torch.save(checkpoint, out)  # serialize and save the dictionary


def load_checkpoint(
    src: Union[str, os.PathLike, BinaryIO, IO[bytes]], 
    model: nn.Module, 
    optimizer: optim.Optimizer
) -> int:
    """
    Loads a checkpoint from a file or buffer and restores the model and optimizer states.

    Args:
        src (Union[str, os.PathLike, BinaryIO, IO[bytes]]): The path or file-like object
            from which to read the checkpoint.
        model (nn.Module): The model instance to load the state into.
        optimizer (optim.Optimizer): The optimizer instance to load the state into.

    Returns:
        int: The iteration number restored from the checkpoint, allowing training to resume.
    """
    # determine the device the model is currently on.
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device('cpu')

    # load the checkpoint dictionary from the source.
    checkpoint = torch.load(src, map_location=device)

    # restore the states of the model and optimizer.
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # extract and return the iteration number.
    iteration = checkpoint['iteration']
    
    return iteration
    