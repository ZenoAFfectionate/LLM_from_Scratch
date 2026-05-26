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
    """SiLU activation function"""
    # return x / (1 + torch.exp(-x))
    return x * torch.sigmoid(x)


# class MLP(nn.Module):
#     """ Special SwiGLU MLP network implementation with explicit FP32 computation for stability """
#     def __init__(self, d_model: int, d_ff: int = None, device=None, dtype=None):
#         super().__init__()
#         if d_ff is None: d_ff = 64 * ((int(d_model * 8 / 3) + 64 - 1) // 64)
#         # initialize three linear projection for SwiGLU:
#         # use BF16 for weights but computation uses FP32
#         self.w1 = nn.Linear(d_model, d_ff, device=device, dtype=dtype)  # shape: (d_ff, d_model)
#         self.w3 = nn.Linear(d_model, d_ff, device=device, dtype=dtype)  # shape: (d_ff, d_model)
#         self.w2 = nn.Linear(d_ff, d_model, device=device, dtype=dtype)  # shape: (d_model, d_ff)
#         # self.dropout = nn.Dropout(dropout=0.1)  # no need Dropout inside

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # Let autocast handle dtype management automatically
#         w1_out = self.w1(x)
#         w3_out = self.w3(x)
#         activated = F.silu(w1_out) * w3_out  # use F.silu() here
#         return self.w2(activated)


class MLP(nn.Module):
    """SwiGLU MLP with fused gate+up projection.

    Combines two engineering knobs from the DeepSeek-V3.1/V4 expert design with a
    fused gate/up matmul (Llama / Mistral style) for performance:

      * ``swiglu_limit``: optional symmetric clamp on the up projection and an
        upper clamp on the gate, used by DeepSeek-V3.1/V4 to keep SwiGLU
        activations numerically stable under low-precision training.
      * ``weights`` (forward kwarg): optional per-token routing weight that the
        layer multiplies into the activations before the down projection. Lets
        the module double as an MoE expert without an external scatter step.
        When ``None`` the module behaves exactly like the previous fused MLP.
    """

    def __init__(self, d_model: int, d_ff: int = None,
                 swiglu_limit: float = 0.0, device=None, dtype=None):
        super().__init__()
        if d_ff is None: d_ff = 64 * ((int(d_model * 8 / 3) + 64 - 1) // 64)
        # Fused gate+up projection: (2*d_ff, d_model), split via chunk in forward.
        self.w1 = nn.Linear(d_model, d_ff * 2, device=device, dtype=dtype)
        self.w2 = nn.Linear(d_ff, d_model, device=device, dtype=dtype)
        self.swiglu_limit = float(swiglu_limit)

    def forward(self, x: torch.Tensor, weights: torch.Tensor = None) -> torch.Tensor:
        gate, up = self.w1(x).chunk(2, dim=-1)
        if self.swiglu_limit > 0:
            # See DeepSeek-V3.1/V4: symmetric clamp on up, upper clamp on gate.
            up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
            gate = torch.clamp(gate, max=self.swiglu_limit)
        h = F.silu(gate) * up
        if weights is not None:
            h = h * weights
        return self.w2(h)