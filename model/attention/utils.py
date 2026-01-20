import torch
import torch.nn as nn



# ---------------------------------------
#  Problem 5: Implement Softmax Function
# ---------------------------------------
def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """ PyTorch implementation of Softmax function """
    x_max = x.amax(dim=dim, keepdim=True)  # find the max
    x_exp = torch.exp(x - x_max)           # subract this
    return x_exp / x_exp.sum(dim=dim, keepdim=True)


# -------------------------------------------------------
#  Problem 6: Implement Rotary Position Embedding Module
# -------------------------------------------------------
def _apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Helper function to apply rotary embeddings with optimized chunking"""
    # split x into two halves in FP32 for stability
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    y1 = x1 * cos - x2 * sin  # rotate 1st half
    y2 = x2 * cos + x1 * sin  # rotate 2nd half
    # concatenate and cast back to original dtype
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class RotaryPositionalEmbedding(nn.Module):
    """PyTorch implementation of Rotary Position Embedding module"""
    
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        # generate inverse frequency using einsum for clarity and optimization
        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device).float() / d_k))
        position = torch.arange(max_seq_len, device=device).float()
        angles = torch.einsum('i, j -> ij', position, inv_freq)
        # compute cos and sin and concatenate along last dimension
        cos = torch.cos(angles)  # (max_seq_len, d_k//2)
        sin = torch.sin(angles)  # (max_seq_len, d_k//2)
        cos_sin = torch.cat([cos, sin], dim=-1)  # (max_seq_len, d_k)
        self.register_buffer('cos_sin_cached', cos_sin, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """Apply rotary position embeddings using token_positions to slice tensors"""
        d_x = x.shape[-1]
        # select positions and truncate to d_x (actual input dimension)
        cos_sin = self.cos_sin_cached[token_positions, :d_x]  # (seq_len, d_x)
        cos_sin = cos_sin.view(*([1] * (x.ndim - 2)), *cos_sin.shape)
        cos, sin = torch.chunk(cos_sin, 2, dim=-1)  # split back to sin and cos
        return _apply_rotary(x, cos, sin)


# ---------------------------------------------------
#  Problem 7: Implement Scaled Dot-Product Attention
# ---------------------------------------------------
def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask=None):
    """ Implement scaled dot-product attention mechanism. """
    assert k.size(-2) == v.size(-2), "k and v must have the same seq len"
    d_k = q.size(-1)

    attn_scores = (q @ k.transpose(-2, -1)) / (d_k ** 0.5)  # (batch_size, seq_len, seq_len)

    if mask is not None: 
        attn_scores = attn_scores.masked_fill(~mask, float('-inf'))

    attn_weights = softmax(attn_scores, dim=-1)             # (batch_size, seq_len, seq_len)

    return attn_weights @ v                                 # (batch_size, seq_len, d_v)


# import tilelang
# import tilelang.language as T
# from typing import Tuple, Optional

# pass_configs = {
#     tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
#     tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
#     tilelang.PassConfigKey.TL_DISABLE_FAST_MATH: True,
# }

# FP8 = "float8_e4m3"
# BF16 = "bfloat16"
# FP32 = "float32"


# def fast_log2_ceil(x):
#     '''  '''
#     bits_x = T.reinterpret("uint32", x)
#     exp_x = (bits_x >> 23) & 0xFF
#     man_bits = bits_x & ((1 << 23) - 1)
#     return T.Cast("int32", exp_x - 127 + T.if_then_else(man_bits != 0, 1, 0))

# def fast_pow2(x):
#     """  """
#     return T.reinterpret("float32", (x + 127) << 23)

# def fast_round_scale(amax, fp8_max_inv):
#     """ Computes the rounded scale for FP8 quantization """
#     return fast_pow2(fast_log2_ceil(amax * fp8_max_inv))


# @tilelang.jit(pass_configs=pass_configs)
# def act_quant_kernel(
#     N, in_dtype=BF16, out_dtype=FP8, scale_dtype=FP32, round_scale=False
# ):
#     M = T.symbolic("M")
#     fp8_min = -448.0
#     fp8_max = 448.0
#     fp8_max_inv = 1 / fp8_max
#     num_stages = 0 if round_scale else 2
#     blk_m = 32
#     group_size = 128

#     @T.prim_func
#     def act_quant_kernel_(
#         X: T.Tensor[(M, N), in_dtype],
#         Y: T.Tensor[(M, N), out_dtype],
#         S: T.Tensor[(M, T.ceildiv(N, group_size)), scale_dtype],
#     ):
#         with T.Kernel(T.ceildiv(M, blk_m), T.ceildiv(N, group_size), threads=128) as (
#             pid_m,
#             pid_n,
#         ):
#             x_shared = T.alloc_shared((blk_m, group_size), in_dtype)
#             x_local = T.alloc_fragment((blk_m, group_size), in_dtype)
#             amax_local = T.alloc_fragment((blk_m,), scale_dtype)
#             s_local = T.alloc_fragment((blk_m,), scale_dtype)
#             y_local = T.alloc_fragment((blk_m, group_size), out_dtype)
#             y_shared = T.alloc_shared((blk_m, group_size), out_dtype)

#             for _ in T.Pipelined(1, num_stages=num_stages):
#                 T.copy(X[pid_m * blk_m, pid_n * group_size], x_shared)
#                 T.copy(x_shared, x_local)
#                 T.reduce_absmax(x_local, amax_local, dim=1)
#                 for i in T.Parallel(blk_m):
#                     amax_local[i] = T.max(amax_local[i], 1e-4)
#                     if round_scale:
#                         s_local[i] = fast_round_scale(amax_local[i], fp8_max_inv)
#                     else:
#                         s_local[i] = amax_local[i] * fp8_max_inv
#                 for i, j in T.Parallel(blk_m, group_size):
#                     y_local[i, j] = T.clamp(
#                         x_local[i, j] / s_local[i], fp8_min, fp8_max
#                     )
#                 for i in T.Parallel(blk_m):
#                     S[pid_m * blk_m + i, pid_n] = s_local[i]
#                 T.copy(y_local, y_shared)
#                 T.copy(y_shared, Y[pid_m * blk_m, pid_n * group_size])

#     return act_quant_kernel_


# def act_quant(
#     x: torch.Tensor, block_size: int = 128, scale_fmt: Optional[str] = None
# ) -> Tuple[torch.Tensor, torch.Tensor]:
#     """
#     Quantizes the input tensor `x` using block-wise quantization.

#     Args:
#         x (torch.Tensor): The input tensor to be quantized. Must be contiguous and its last dimension size must be divisible by `block_size`.
#         block_size (int, optional): The size of the blocks to be used for quantization. Default is 128.
#         scale_fmt (Optional[str], optional): The format of the scale. Default is None.
#     Returns:
#         Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
#             - The quantized tensor with dtype `torch.float8_e4m3fn`.
#             - A tensor of scaling factors with dtype `torch.float32`.
#     """
#     assert x.is_contiguous(), "Input tensor must be contiguous"
#     assert x.size(-1) % block_size == 0, (
#         f"Last dimension size must be divisible by block_size (block_size={block_size})"
#     )
#     N = x.size(-1)
#     y = torch.empty_like(x, dtype=torch.float8_e4m3fn)
#     s = x.new_empty(*x.size()[:-1], N // block_size, dtype=torch.float32)
#     kernel = act_quant_kernel(N, round_scale=scale_fmt is not None)
#     kernel(x.view(-1, N), y.view(-1, N), s.view(-1, N // block_size))
#     return y, s


# @tilelang.jit(out_idx=[4], pass_configs=pass_configs)
# def fp8_index_kernel(h: int, d: int):
#     b = T.symbolic("b")
#     m = T.symbolic("m")
#     n = T.symbolic("n")

#     blk_n1 = 512
#     blk_n2 = 128

#     @T.prim_func
#     def fp8_index_kernel_(
#         q: T.Tensor[(b, m, h, d), FP8],
#         q_s: T.Tensor[(b, m, h), FP32],
#         k: T.Tensor[(b, n, d), FP8],
#         k_s: T.Tensor[(b, n), FP32],
#         o: T.Tensor[(b, m, n), FP32],
#     ) -> None:
#         with T.Kernel(b, m, T.ceildiv(n, blk_n1)) as (i_b, i_m, i1_n):
#             q_smem = T.alloc_shared((h, d), FP8)
#             T.copy(q[i_b, i_m, 0, 0], q_smem)

#             q_s_frag = T.alloc_fragment(h, FP32)
#             T.copy(q_s[i_b, i_m, 0], q_s_frag)

#             for i2_n in T.Pipelined(blk_n1 // blk_n2, num_stages=2):
#                 k_smem = T.alloc_shared((blk_n2, d), FP8)
#                 T.copy(k[i_b, i1_n * blk_n1 + i2_n * blk_n2, 0], k_smem)

#                 k_s_frag = T.alloc_fragment(blk_n2, FP32)
#                 T.copy(k_s[i_b, i1_n * blk_n1 + i2_n * blk_n2], k_s_frag)

#                 logits = T.alloc_fragment((blk_n2, h), FP32)
#                 T.gemm(
#                     k_smem,
#                     q_smem,
#                     logits,
#                     transpose_A=False,
#                     transpose_B=True,
#                     clear_accum=True,
#                 )

#                 for i_h, i3_n in T.Parallel(h, blk_n2):
#                     logits[i3_n, i_h] = T.max(logits[i3_n, i_h], 0) * q_s_frag[i_h]

#                 logits_sum = T.alloc_fragment(blk_n2, FP32)
#                 T.reduce_sum(logits, logits_sum, dim=1)

#                 for i3_n in T.Parallel(blk_n2):
#                     logits_sum[i3_n] *= k_s_frag[i3_n]

#                 T.copy(logits_sum, o[i_b, i_m, i1_n * blk_n1 + i2_n * blk_n2])

#     return fp8_index_kernel_


# def fp8_index(
#     q: torch.Tensor,
#     q_s: torch.Tensor,
#     k: torch.Tensor,
#     k_s: torch.Tensor,
# ) -> torch.Tensor:
#     """
#     Perform index score using FP8 precision.

#     Args:
#         q (torch.Tensor): The Q tensor, must be contiguous.
#         q_s (torch.Tensor): The scaling factor for Q (float), must be contiguous.
#         k (torch.Tensor): The K tensor, must be contiguous.
#         k_s (torch.Tensor): The scaling factor for K (e8m0 here), must be contiguous.

#         fp8 q @ fp8 k -> fp32 logits
#         relu(fp32 logits) * q_s (weights) -> fp32 logits
#         fp32 logits -> fp32 logits_sum
#         fp32 logits_sum * k_s (e8m0) -> fp32 index_score
#     """
#     return fp8_index_kernel(q.shape[2], q.shape[3])(q, q_s, k, k_s)