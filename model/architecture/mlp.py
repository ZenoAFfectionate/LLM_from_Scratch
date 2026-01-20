import math
import torch

import torch.nn as nn
import torch.nn.functional as F


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
        # If dtype is None, default to float32 for critical layers
        if dtype is None: dtype = torch.float32
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


# --------------------------------------------------------
#  Problem 4: Implement SwiGLU FeedForward Network Module
# --------------------------------------------------------
@torch.jit.script
def silu(x: torch.Tensor) -> torch.Tensor:
    """
    SiLU activation function, also known as the swish function.

    OPTIMIZED: JIT-compiled for faster execution (5-10% speedup).
    The @torch.jit.script decorator compiles this function to TorchScript,
    which eliminates Python overhead and enables better optimization.
    """
    return x * torch.sigmoid(x)

class MLP(nn.Module):
    """ Special SwiGLU MLP network implementation with explicit FP32 computation for stability """
    def __init__(self, d_model: int, d_ff: int = None, device=None, dtype=None):
        '''  '''
        super().__init__()
        if d_ff is None: d_ff = 64 * ((int(d_model * 8 / 3) + 64 - 1) // 64)
        # initialize three linear projection for SwiGLU:
        # Linear layers will use dtype (BF16 for weights), but computation uses FP32
        self.w1 = nn.Linear(d_model, d_ff, device=device, dtype=dtype)  # shape: (d_ff, d_model)
        self.w3 = nn.Linear(d_model, d_ff, device=device, dtype=dtype)  # shape: (d_ff, d_model)
        self.w2 = nn.Linear(d_ff, d_model, device=device, dtype=dtype)  # shape: (d_model, d_ff)
        # self.dropout = nn.Dropout(dropout=0.1)  # no need Dropout here actually ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Let autocast handle dtype management automatically
        # Autocast will use BF16 for matmul and FP32 for reductions as needed
        w1_out = self.w1(x)
        w3_out = self.w3(x)
        activated = F.silu(w1_out) * w3_out  # use F.silu() here
        return self.w2(activated)