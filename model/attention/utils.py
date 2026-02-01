"""
Attention Utilities and Kernels

This module contains:
1. Softmax, RoPE, and Scaled Dot-Product Attention utilities
2. Triton kernels for FP8 quantization (DSA)
3. Paged Attention kernels for efficient inference with vLLM-style engine
"""

import torch
import torch.nn as nn
import triton
import triton.language as tl
from typing import Tuple, Optional


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
        # peform rotary positional embedding for x
        x1, x2 = torch.chunk(x, 2, dim=-1)
        y1 = torch.addcmul(x1 * cos, x2, sin, value=-1)
        y2 = torch.addcmul(x2 * cos, x1, sin, value=+1)
        return torch.cat((y1, y2), dim=-1)


# ---------------------------------------------------
#  Problem 7: Implement Scaled Dot-Product Attention
# ---------------------------------------------------
def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask=None):
    """ Implement scaled dot-product attention mechanism. """
    assert k.size(-2) == v.size(-2), "k and v must have the same seq len"
    d_k = q.size(-1)

    attn_scores = (q @ k.transpose(-2, -1)) / \
        (d_k ** 0.5)  # (batch_size, seq_len, seq_len)

    if mask is not None:
        attn_scores = attn_scores.masked_fill(~mask, float('-inf'))

    # (batch_size, seq_len, seq_len)
    attn_weights = softmax(attn_scores, dim=-1)

    # (batch_size, seq_len, d_v)
    return attn_weights @ v


# =============================================================================
# Triton Kernels for DeepSeek Sparse Attention (DSA) - FP8 Quantization
# =============================================================================

@triton.jit
def _act_quant_kernel(
    # Pointers to tensors
    X_ptr,          # Input tensor: (M, N) in BF16/FP16/FP32
    Y_ptr,          # Output tensor: (M, N) in FP8
    S_ptr,          # Scale tensor: (M, N // group_size) in FP32
    # Tensor dimensions
    M,              # Number of rows
    N,              # Number of columns
    # Strides
    stride_x_m,     # Stride of X along M dimension
    stride_x_n,     # Stride of X along N dimension
    stride_y_m,     # Stride of Y along M dimension
    stride_y_n,     # Stride of Y along N dimension
    stride_s_m,     # Stride of S along M dimension
    stride_s_n,     # Stride of S along N (groups) dimension
    # Block sizes (constexpr for compile-time optimization)
    BLOCK_M: tl.constexpr,      # Block size for M dimension (e.g., 32)
    GROUP_SIZE: tl.constexpr,   # Quantization group size (e.g., 128)
    ROUND_SCALE: tl.constexpr,  # Whether to round scale to power of 2
):
    """
    Block-wise FP8 activation quantization kernel.

    For each block of (BLOCK_M, GROUP_SIZE):
    1. Compute absolute maximum for each row within the group
    2. Compute scale = amax / fp8_max (optionally rounded to power of 2)
    3. Quantize: y = clamp(x / scale, -fp8_max, fp8_max)

    This kernel mirrors the tilelang implementation from DeepSeek-V3.
    """
    # FP8 E4M3 constants
    FP8_MAX: tl.constexpr = 448.0
    FP8_MIN: tl.constexpr = -448.0
    FP8_MAX_INV: tl.constexpr = 1.0 / 448.0

    # Program IDs
    pid_m = tl.program_id(0)  # Block ID along M
    pid_n = tl.program_id(1)  # Block ID along N (groups)

    # Compute offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * GROUP_SIZE + tl.arange(0, GROUP_SIZE)

    # Masks for boundary checking
    mask_m = offs_m < M
    mask_n = offs_n < N
    mask = mask_m[:, None] & mask_n[None, :]

    # Load input block
    x_ptrs = X_ptr + offs_m[:, None] * \
        stride_x_m + offs_n[None, :] * stride_x_n
    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

    # Compute absolute maximum for each row within this group
    abs_x = tl.abs(x)
    amax = tl.max(abs_x, axis=1)  # (BLOCK_M,)

    # Ensure minimum scale for numerical stability
    amax = tl.maximum(amax, 1e-4)

    # Compute scale
    if ROUND_SCALE:
        # Round scale to power of 2 for better hardware efficiency
        # scale = 2^ceil(log2(amax * fp8_max_inv))
        log2_scale = tl.ceil(tl.log2(amax * FP8_MAX_INV))
        scale = tl.exp2(log2_scale)
    else:
        scale = amax * FP8_MAX_INV

    # Quantize: y = clamp(x / scale, fp8_min, fp8_max)
    scale_broadcast = scale[:, None]
    y = x / scale_broadcast
    y = tl.minimum(tl.maximum(y, FP8_MIN), FP8_MAX)

    # Store quantized output
    y_ptrs = Y_ptr + offs_m[:, None] * \
        stride_y_m + offs_n[None, :] * stride_y_n
    tl.store(y_ptrs, y.to(tl.float8e4nv), mask=mask)

    # Store scale for each row in this group
    s_ptrs = S_ptr + offs_m * stride_s_m + pid_n * stride_s_n
    tl.store(s_ptrs, scale, mask=mask_m)


def act_quant(
    x: torch.Tensor,
    block_size: int = 128,
    scale_fmt: Optional[str] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantizes the input tensor `x` using block-wise FP8 quantization.

    This is a Triton implementation of the tilelang act_quant kernel from DeepSeek-V3.

    Args:
        x (torch.Tensor): Input tensor to be quantized. Must be contiguous and 
                          last dimension size must be divisible by `block_size`.
        block_size (int): Size of the quantization groups. Default is 128.
        scale_fmt (Optional[str]): If not None, rounds scale to power of 2 for 
                                   better hardware efficiency (e8m0 format).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: 
            - Quantized tensor with dtype `torch.float8_e4m3fn`
            - Scale tensor with dtype `torch.float32`

    Example:
        >>> x = torch.randn(32, 256, dtype=torch.bfloat16, device='cuda')
        >>> x_fp8, scale = act_quant(x, block_size=128)
        >>> # x_fp8.shape == (32, 256), scale.shape == (32, 2)
    """
    assert x.is_contiguous(), "Input tensor must be contiguous"
    assert x.size(-1) % block_size == 0, (
        f"Last dimension size must be divisible by block_size (block_size={block_size})"
    )

    # Flatten to 2D for kernel processing
    original_shape = x.shape
    N = x.size(-1)
    x_2d = x.view(-1, N)
    M = x_2d.size(0)

    # Allocate outputs
    y = torch.empty_like(x_2d, dtype=torch.float8_e4m3fn)
    num_groups = N // block_size
    s = torch.empty(M, num_groups, dtype=torch.float32, device=x.device)

    # Launch kernel
    BLOCK_M = 32  # Process 32 rows at a time
    grid = (triton.cdiv(M, BLOCK_M), num_groups)

    _act_quant_kernel[grid](
        x_2d, y, s,
        M, N,
        x_2d.stride(0), x_2d.stride(1),
        y.stride(0), y.stride(1),
        s.stride(0), s.stride(1),
        BLOCK_M=BLOCK_M,
        GROUP_SIZE=block_size,
        ROUND_SCALE=(scale_fmt is not None),
    )

    # Reshape outputs to match input shape
    y = y.view(*original_shape)
    s = s.view(*original_shape[:-1], num_groups)

    return y, s


# =============================================================================
# Triton Kernels for DeepSeek Sparse Attention (DSA) - FP8 Index Scoring
# =============================================================================

@triton.jit
def _fp8_index_kernel(
    # Input pointers
    Q_ptr,          # Query tensor: (B, M, H, D) in FP8
    Q_scale_ptr,    # Query scale: (B, M, H, num_groups) in FP32
    K_ptr,          # Key tensor: (B, N, D) in FP8
    K_scale_ptr,    # Key scale: (B, N, num_groups) in FP32
    # Output pointer
    O_ptr,          # Output scores: (B, M, N) in FP32
    # Dimensions
    B,              # Batch size
    M,              # Number of query tokens
    N,              # Number of key tokens
    H,              # Number of heads
    D,              # Head dimension
    num_groups,     # Number of scale groups (D // group_size)
    # Strides for Q (B, M, H, D)
    stride_qb, stride_qm, stride_qh, stride_qd,
    # Strides for Q_scale (B, M, H, num_groups)
    stride_qsb, stride_qsm, stride_qsh, stride_qsg,
    # Strides for K (B, N, D)
    stride_kb, stride_kn, stride_kd,
    # Strides for K_scale (B, N, num_groups)
    stride_ksb, stride_ksn, stride_ksg,
    # Strides for O (B, M, N)
    stride_ob, stride_om, stride_on,
    # Block sizes
    BLOCK_N: tl.constexpr,      # Block size for N dimension
    BLOCK_D: tl.constexpr,      # Block size for D dimension
    NUM_HEADS: tl.constexpr,    # Number of heads (constexpr for unrolling)
    # Number of scale groups (constexpr for unrolling)
    NUM_GROUPS: tl.constexpr,
):
    """
    Optimized FP8 index scoring kernel for DSA.

    Computes: O[b, m, n] = sum_h(max(0, Q[b,m,h,:] @ K[b,n,:]^T) * Q_scale[b,m,h]) * K_scale[b,n]

    Optimizations:
    1. NUM_HEADS and NUM_GROUPS as constexpr for compile-time loop unrolling
    2. Pre-load all Q scales for the query token once
    3. Vectorized K scale loading
    """
    # Program IDs
    pid_bm = tl.program_id(0)  # Combined batch and query token
    pid_n = tl.program_id(1)   # Key token block

    # Extract batch and query token indices
    pid_b = pid_bm // M
    pid_m = pid_bm % M

    # Initialize accumulator for this key block
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # [OPT] Pre-compute base pointers to avoid redundant address computation
    q_base = Q_ptr + pid_b * stride_qb + pid_m * stride_qm
    qs_base = Q_scale_ptr + pid_b * stride_qsb + pid_m * stride_qsm
    k_base = K_ptr + pid_b * stride_kb
    ks_base = K_scale_ptr + pid_b * stride_ksb

    # [OPT] Load K_scale once, compute product across groups
    k_scale = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for g in tl.static_range(NUM_GROUPS):
        k_scale_g = tl.load(ks_base + offs_n * stride_ksn +
                            g * stride_ksg, mask=mask_n, other=1.0)
        k_scale = k_scale + k_scale_g
    # Multiply by reciprocal is faster than divide
    k_scale = k_scale * (1.0 / NUM_GROUPS)

    # [OPT] Process heads with constexpr loop for unrolling
    for h in tl.static_range(NUM_HEADS):
        # [OPT] Load Q_scale for this head - use static_range for unrolling
        q_scale = tl.zeros((), dtype=tl.float32)
        for g in tl.static_range(NUM_GROUPS):
            q_scale = q_scale + \
                tl.load(qs_base + h * stride_qsh + g * stride_qsg)
        q_scale = q_scale * (1.0 / NUM_GROUPS)

        # Compute dot product Q[b,m,h,:] @ K[b,n,:]^T for each key in block
        logits = tl.zeros((BLOCK_N,), dtype=tl.float32)

        # Process D dimension in blocks
        for d_start in range(0, D, BLOCK_D):
            offs_d = d_start + tl.arange(0, BLOCK_D)
            mask_d = offs_d < D

            # Load Q block: (BLOCK_D,)
            q = tl.load(q_base + h * stride_qh + offs_d * stride_qd,
                        mask=mask_d, other=0.0).to(tl.float32)

            # Load K block: (BLOCK_N, BLOCK_D)
            k_ptrs = k_base + offs_n[:, None] * \
                stride_kn + offs_d[None, :] * stride_kd
            k = tl.load(k_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(
                tl.float32)

            # Accumulate dot product
            logits = logits + tl.sum(k * q[None, :], axis=1)

        # Apply ReLU and accumulate with query scale
        logits = tl.maximum(logits, 0.0) * q_scale
        acc = acc + logits

    # Multiply by key scale
    acc = acc * k_scale

    # Store result
    o_ptrs = O_ptr + pid_b * stride_ob + pid_m * stride_om + offs_n * stride_on
    tl.store(o_ptrs, acc, mask=mask_n)


def fp8_index(
    q: torch.Tensor,
    q_s: torch.Tensor,
    k: torch.Tensor,
    k_s: torch.Tensor,
) -> torch.Tensor:
    """
    Perform FP8 index scoring for DSA (DeepSeek Sparse Attention).

    This computes the index scores used by the lightning indexer to select
    which key-value tokens are most relevant for each query.

    Args:
        q (torch.Tensor): Query tensor in FP8, shape (B, M, H, D)
        q_s (torch.Tensor): Query scales, shape (B, M, H, num_groups) or (B, M, H)
        k (torch.Tensor): Key tensor in FP8, shape (B, N, D)  
        k_s (torch.Tensor): Key scales, shape (B, N, num_groups) or (B, N)

    Returns:
        torch.Tensor: Index scores, shape (B, M, N) in FP32

    The computation is:
        score[b,m,n] = sum_h(ReLU(Q[b,m,h,:] @ K[b,n,:]^T) * Q_scale[b,m,h]) * K_scale[b,n]
    """
    assert q.is_contiguous() and k.is_contiguous()
    assert q_s.is_contiguous() and k_s.is_contiguous()

    B, M, H, D = q.shape
    _, N, _ = k.shape

    # Handle different scale tensor shapes
    if q_s.ndim == 3:  # (B, M, H) - single scale per head
        q_s = q_s.unsqueeze(-1)  # (B, M, H, 1)
    if k_s.ndim == 2:  # (B, N) - single scale per key
        k_s = k_s.unsqueeze(-1)  # (B, N, 1)

    num_groups = q_s.size(-1)

    # Allocate output
    o = torch.empty(B, M, N, dtype=torch.float32, device=q.device)

    # [OPT] Choose block sizes based on tensor dimensions
    BLOCK_N = min(128, triton.next_power_of_2(N))
    BLOCK_D = min(64, triton.next_power_of_2(D))

    # Launch kernel with constexpr parameters for better optimization
    grid = (B * M, triton.cdiv(N, BLOCK_N))

    _fp8_index_kernel[grid](
        q, q_s, k, k_s, o,
        B, M, N, H, D, num_groups,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        q_s.stride(0), q_s.stride(1), q_s.stride(
            2), q_s.stride(3) if q_s.ndim == 4 else 0,
        k.stride(0), k.stride(1), k.stride(2),
        k_s.stride(0), k_s.stride(1), k_s.stride(2) if k_s.ndim == 3 else 0,
        o.stride(0), o.stride(1), o.stride(2),
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        NUM_HEADS=H,
        NUM_GROUPS=num_groups,
    )

    return o


# =============================================================================
# Paged Attention Kernels for efficient inference with vLLM-style engine
# =============================================================================

# =============================================================================
# Kernel: Store KV Cache - Write new K/V values to paged cache
# =============================================================================

@triton.jit
def _store_kvcache_kernel(
    key_ptr,          # Input key tensor pointer
    value_ptr,        # Input value tensor pointer
    k_cache_ptr,      # K cache pointer (paged)
    v_cache_ptr,      # V cache pointer (paged)
    slot_mapping_ptr,  # Maps each token to a cache slot
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
):
    """
    Store keys and values into paged KV cache.

    Each token is mapped to a slot via slot_mapping.
    Grid layout: (num_tokens, num_kv_heads)

    Input shapes:
        key/value: (num_tokens, num_kv_heads, head_dim)
    Cache shapes:
        k_cache/v_cache: (num_blocks, block_size, num_kv_heads, head_dim)
    """
    # Thread indices
    token_idx = tl.program_id(0)  # Which token
    head_idx = tl.program_id(1)   # Which KV head

    # Load slot index for this token
    slot_idx = tl.load(slot_mapping_ptr + token_idx)

    # Skip if slot is invalid (-1 means no write)
    if slot_idx == -1:
        return

    # Calculate which block and position within block
    block_idx = slot_idx // block_size
    block_offset = slot_idx % block_size

    # Compute offsets for loading from input
    # Input layout: (num_tokens, num_kv_heads, head_dim)
    head_offsets = tl.arange(0, head_dim)
    input_offset = (token_idx * num_kv_heads * head_dim +
                    head_idx * head_dim +
                    head_offsets)

    # Compute offsets for storing to cache
    # Cache layout: (num_blocks, block_size, num_kv_heads, head_dim)
    cache_offset = (block_idx * block_size * num_kv_heads * head_dim +
                    block_offset * num_kv_heads * head_dim +
                    head_idx * head_dim +
                    head_offsets)

    # Load key and value from input
    key = tl.load(key_ptr + input_offset)
    value = tl.load(value_ptr + input_offset)

    # Store into cache
    tl.store(k_cache_ptr + cache_offset, key)
    tl.store(v_cache_ptr + cache_offset, value)


def store_kvcache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int,
):
    """
    Store key-value pairs into paged cache.

    Args:
        key: (num_tokens, num_kv_heads, head_dim)
        value: (num_tokens, num_kv_heads, head_dim)
        k_cache: (num_blocks, block_size, num_kv_heads, head_dim)
        v_cache: (num_blocks, block_size, num_kv_heads, head_dim)
        slot_mapping: (num_tokens,) - maps each token to a cache slot
        block_size: number of tokens per block
    """
    num_tokens, num_kv_heads, head_dim = key.shape

    # Make contiguous if needed
    if not key.is_contiguous():
        key = key.contiguous()
    if not value.is_contiguous():
        value = value.contiguous()

    # Launch kernel
    grid = (num_tokens, num_kv_heads)
    _store_kvcache_kernel[grid](
        key, value,
        k_cache, v_cache,
        slot_mapping,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=block_size,
    )


# =============================================================================
# Kernel: Flash Attention Prefill - Variable-length attention for prefill
# =============================================================================

@triton.jit
def _flash_attention_varlen_kernel(
    Q, K, V, O,
    cu_seqlens_q_ptr,
    scale,
    num_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Flash Attention kernel for variable-length sequences (prefill phase).

    Each program processes one block of queries for one head in one sequence.
    Uses online softmax for numerical stability and memory efficiency.

    Input shapes (varlen mode):
        Q: (total_tokens, num_heads, head_dim)
        K: (total_tokens, num_kv_heads, head_dim)
        V: (total_tokens, num_kv_heads, head_dim)
        cu_seqlens: (num_seqs + 1,)
    """
    # Program IDs
    start_m = tl.program_id(0)  # Query block index
    off_h = tl.program_id(1)    # Head index
    seq_idx = tl.program_id(2)  # Sequence index

    # Determine which KV head to use (for GQA: multiple Q heads share one KV head)
    kv_head_idx = off_h // (num_heads // num_kv_heads)

    # Load sequence boundaries from cumulative lengths
    seq_start = tl.load(cu_seqlens_q_ptr + seq_idx)
    seq_end = tl.load(cu_seqlens_q_ptr + seq_idx + 1)
    seq_len = seq_end - seq_start

    # Early exit if this block is beyond sequence length
    if start_m * BLOCK_M >= seq_len:
        return

    # Offset for this block of queries
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, head_dim)

    # Query pointers: Q has shape (total_tokens, num_heads, head_dim)
    q_ptrs = Q + (seq_start + offs_m[:, None]) * num_heads * \
        head_dim + off_h * head_dim + offs_d[None, :]

    # Load Q block - shape (BLOCK_M, head_dim)
    mask_m = offs_m < seq_len
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)

    # Initialize online softmax accumulators
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)       # Sum of exp
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - 1e10  # Max score
    acc = tl.zeros([BLOCK_M, head_dim], dtype=tl.float32)

    # Number of K/V blocks to process
    num_blocks = tl.cdiv(seq_len, BLOCK_N)

    # Loop over K, V blocks
    for block_n in range(num_blocks):
        start_n = block_n * BLOCK_N
        offs_n = start_n + tl.arange(0, BLOCK_N)

        # Mask for valid K positions
        mask_n = offs_n < seq_len

        # K pointers: K has shape (total_tokens, num_kv_heads, head_dim)
        k_ptrs = K + (seq_start + offs_n[None, :]) * num_kv_heads * \
            head_dim + kv_head_idx * head_dim + offs_d[:, None]

        # Load K block - shape (head_dim, BLOCK_N)
        k = tl.load(k_ptrs, mask=mask_n[None, :], other=0.0)

        # Compute QK^T - shape (BLOCK_M, BLOCK_N)
        qk = tl.dot(q, k)
        qk = qk * scale

        # Apply causal mask: only attend to positions <= current position
        mask_causal = offs_m[:, None] >= offs_n[None, :]
        qk = tl.where(mask_causal & mask_n[None, :], qk, -1e10)

        # Online softmax update (Flash Attention algorithm)
        m_ij = tl.max(qk, axis=1)
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        p = tl.exp(qk - m_i_new[:, None])

        # Rescale previous accumulator
        acc = acc * alpha[:, None]

        # Load V block - shape (BLOCK_N, head_dim)
        v_ptrs = V + (seq_start + offs_n[:, None]) * num_kv_heads * \
            head_dim + kv_head_idx * head_dim + offs_d[None, :]
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)

        # Accumulate weighted values
        acc = acc + tl.dot(p.to(v.dtype), v)

        # Update normalizer
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_i_new

    # Final normalization
    acc = acc / l_i[:, None]

    # Store output: O has shape (total_tokens, num_heads, head_dim)
    o_ptrs = O + (seq_start + offs_m[:, None]) * num_heads * \
        head_dim + off_h * head_dim + offs_d[None, :]
    tl.store(o_ptrs, acc.to(O.dtype.element_ty), mask=mask_m[:, None])


def flash_attention_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    scale: float,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """
    Flash Attention for prefill phase with variable-length sequences.

    Args:
        q: (total_tokens, num_heads, head_dim)
        k: (total_tokens, num_kv_heads, head_dim)
        v: (total_tokens, num_kv_heads, head_dim)
        cu_seqlens: (num_seqs + 1,) cumulative sequence lengths
        scale: attention scale factor (typically 1/sqrt(head_dim))
        num_heads: number of query heads
        num_kv_heads: number of key/value heads
        head_dim: dimension per head

    Returns:
        output: (total_tokens, num_heads, head_dim)
    """
    # Make tensors contiguous
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    # Allocate output
    output = torch.empty_like(q)

    # Choose block sizes based on head_dim to avoid shared memory overflow
    if head_dim <= 64:
        BLOCK_M, BLOCK_N = 64, 64
    elif head_dim <= 128:
        BLOCK_M, BLOCK_N = 32, 32
    else:
        BLOCK_M, BLOCK_N = 16, 16

    # Number of sequences
    num_seqs = cu_seqlens.shape[0] - 1

    # Find max sequence length to determine grid size
    cu_seqlens_cpu = cu_seqlens.cpu()
    max_seq_len = (cu_seqlens_cpu[1:] - cu_seqlens_cpu[:-1]).max().item()

    # Launch kernel
    grid = (triton.cdiv(max_seq_len, BLOCK_M), num_heads, num_seqs)

    _flash_attention_varlen_kernel[grid](
        q, k, v, output,
        cu_seqlens,
        scale,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )

    return output


# =============================================================================
# Kernel: Paged Attention Decode - Read from paged cache during decode
# =============================================================================

@triton.jit
def _paged_attention_decode_kernel(
    output_ptr,
    query_ptr,
    k_cache_ptr,
    v_cache_ptr,
    block_tables_ptr,
    context_lens_ptr,
    scale: tl.constexpr,
    num_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    max_num_blocks: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Paged attention kernel for decode phase.

    For each batch element and head:
    1. Load the single query token
    2. Read K/V from paged cache using block_tables
    3. Compute attention with online softmax
    4. Store output

    Input shapes:
        query: (batch_size, num_heads, head_dim)
        k_cache/v_cache: (num_blocks, block_size, num_kv_heads, head_dim)
        block_tables: (batch_size, max_num_blocks)
        context_lens: (batch_size,)
    """
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    # Determine which KV head this query head uses (for GQA)
    kv_head_idx = head_idx // (num_heads // num_kv_heads)

    # Load context length for this sequence
    context_len = tl.load(context_lens_ptr + batch_idx)

    # Load query: shape (batch_size, num_heads, head_dim)
    offs_d = tl.arange(0, head_dim)
    q_offset = batch_idx * num_heads * head_dim + head_idx * head_dim + offs_d
    q = tl.load(query_ptr + q_offset)

    # Initialize online softmax accumulators
    acc = tl.zeros([head_dim], dtype=tl.float32)
    l_i = 0.0   # Sum of exp
    m_i = -1e10  # Max score

    # Calculate total number of token chunks to process
    max_chunks = tl.cdiv(max_num_blocks * block_size, BLOCK_N)

    # Process all tokens in chunks
    for chunk_idx in range(max_chunks):
        token_start = chunk_idx * BLOCK_N

        # Only process if within valid range
        if token_start < context_len:
            offs_n = token_start + tl.arange(0, BLOCK_N)
            mask_n = offs_n < context_len

            # Compute attention scores for this chunk
            qk = tl.zeros([BLOCK_N], dtype=tl.float32) - 1e10

            # Load K for each valid position and compute scores
            for i in range(BLOCK_N):
                token_idx = token_start + i
                if token_idx < context_len:
                    block_num = token_idx // block_size
                    block_offset = token_idx % block_size

                    if block_num < max_num_blocks:
                        # Look up physical block from block table
                        block_table_offset = batch_idx * max_num_blocks + block_num
                        physical_block_idx = tl.load(
                            block_tables_ptr + block_table_offset)

                        if physical_block_idx != -1:
                            # Load K from cache
                            k_offset = (physical_block_idx * block_size * num_kv_heads * head_dim +
                                        block_offset * num_kv_heads * head_dim +
                                        kv_head_idx * head_dim + offs_d)
                            k_vec = tl.load(k_cache_ptr + k_offset)

                            # Compute attention score
                            score = tl.sum(q * k_vec) * scale

                            # Update qk array at position i
                            mask_i = tl.arange(0, BLOCK_N) == i
                            qk = tl.where(mask_i, score, qk)

            # Apply mask to invalid positions
            qk = tl.where(mask_n, qk, -1e10)

            # Online softmax update
            m_ij = tl.max(qk)
            m_i_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - m_i_new)
            p = tl.exp(qk - m_i_new)

            # Rescale accumulator
            acc = acc * alpha
            l_i = l_i * alpha

            # Load V and accumulate weighted values
            for i in range(BLOCK_N):
                token_idx = token_start + i
                if token_idx < context_len:
                    block_num = token_idx // block_size
                    block_offset = token_idx % block_size

                    if block_num < max_num_blocks:
                        block_table_offset = batch_idx * max_num_blocks + block_num
                        physical_block_idx = tl.load(
                            block_tables_ptr + block_table_offset)

                        if physical_block_idx != -1:
                            # Load V from cache
                            v_offset = (physical_block_idx * block_size * num_kv_heads * head_dim +
                                        block_offset * num_kv_heads * head_dim +
                                        kv_head_idx * head_dim + offs_d)
                            v_vec = tl.load(v_cache_ptr + v_offset)

                            # Get weight for this token
                            mask_i = tl.arange(0, BLOCK_N) == i
                            weight = tl.sum(tl.where(mask_i, p, 0.0))

                            acc = acc + weight * v_vec
                            l_i = l_i + weight

            m_i = m_i_new

    # Final normalization
    output = acc / l_i

    # Store output
    output_offset = batch_idx * num_heads * head_dim + head_idx * head_dim + offs_d
    tl.store(output_ptr + output_offset, output)


def paged_attention_decode(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    scale: float,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
) -> torch.Tensor:
    """
    Compute attention in decode mode using paged KV cache.

    Args:
        query: (batch_size, num_heads, head_dim)
        k_cache: (num_blocks, block_size, num_kv_heads, head_dim)
        v_cache: (num_blocks, block_size, num_kv_heads, head_dim)
        block_tables: (batch_size, max_num_blocks) - logical to physical block mapping
        context_lens: (batch_size,) - number of tokens per sequence
        scale: attention scale factor
        num_heads: number of query heads
        num_kv_heads: number of KV heads
        head_dim: dimension per head
        block_size: tokens per block

    Returns:
        output: (batch_size, num_heads, head_dim)
    """
    batch_size = query.shape[0]
    max_num_blocks = block_tables.shape[1]

    # Make contiguous
    query = query.contiguous()

    output = torch.empty_like(query)

    # Chunk size for processing KV tokens
    BLOCK_N = 64 if head_dim <= 128 else 32

    grid = (batch_size, num_heads)

    _paged_attention_decode_kernel[grid](
        output,
        query,
        k_cache,
        v_cache,
        block_tables,
        context_lens,
        scale=scale,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=block_size,
        max_num_blocks=max_num_blocks,
        BLOCK_N=BLOCK_N,
    )

    return output


# =============================================================================
# Kernel: MLA Store Cache - Store compressed KV and RoPE to paged cache
# =============================================================================

@triton.jit
def _store_mla_cache_kernel(
    kv_ptr,           # Input kv_compressed tensor pointer
    pe_ptr,           # Input k_rope (position encoding) tensor pointer
    kv_cache_ptr,     # KV cache pointer (paged) for kv_compressed
    pe_cache_ptr,     # PE cache pointer (paged) for k_rope
    slot_mapping_ptr,  # Maps each token to a cache slot
    kv_dim: tl.constexpr,    # kv_lora_rank
    pe_dim: tl.constexpr,    # rope_dim
    block_size: tl.constexpr,
):
    """
    Store MLA's compressed KV and position encoding to paged cache.

    Input shapes:
        kv: (num_tokens, kv_lora_rank)
        pe: (num_tokens, rope_dim)
    Cache shapes:
        kv_cache: (num_blocks, block_size, kv_lora_rank)
        pe_cache: (num_blocks, block_size, rope_dim)
    """
    token_idx = tl.program_id(0)

    # Load slot index for this token
    slot_idx = tl.load(slot_mapping_ptr + token_idx)

    if slot_idx == -1:
        return

    # Calculate block and offset
    block_idx = slot_idx // block_size
    block_offset = slot_idx % block_size

    # Store kv_compressed
    kv_offsets = tl.arange(0, kv_dim)
    kv_input_offset = token_idx * kv_dim + kv_offsets
    kv_cache_offset = block_idx * block_size * \
        kv_dim + block_offset * kv_dim + kv_offsets
    kv_data = tl.load(kv_ptr + kv_input_offset)
    tl.store(kv_cache_ptr + kv_cache_offset, kv_data)

    # Store k_rope (position encoding)
    pe_offsets = tl.arange(0, pe_dim)
    pe_input_offset = token_idx * pe_dim + pe_offsets
    pe_cache_offset = block_idx * block_size * \
        pe_dim + block_offset * pe_dim + pe_offsets
    pe_data = tl.load(pe_ptr + pe_input_offset)
    tl.store(pe_cache_ptr + pe_cache_offset, pe_data)


def store_mla_cache(
    kv_compressed: torch.Tensor,
    k_rope: torch.Tensor,
    kv_cache: torch.Tensor,
    pe_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int,
):
    """
    Store MLA's compressed KV and position encoding to paged cache.

    Args:
        kv_compressed: (num_tokens, kv_lora_rank) - compressed KV representation
        k_rope: (num_tokens, rope_dim) - position encoding after RoPE
        kv_cache: (num_blocks, block_size, kv_lora_rank) - paged cache for KV
        pe_cache: (num_blocks, block_size, rope_dim) - paged cache for PE
        slot_mapping: (num_tokens,) - maps each token to a cache slot
        block_size: number of tokens per block
    """
    num_tokens = kv_compressed.shape[0]
    kv_dim = kv_compressed.shape[1]
    pe_dim = k_rope.shape[1]

    # Make contiguous
    if not kv_compressed.is_contiguous():
        kv_compressed = kv_compressed.contiguous()
    if not k_rope.is_contiguous():
        k_rope = k_rope.contiguous()

    grid = (num_tokens,)
    _store_mla_cache_kernel[grid](
        kv_compressed, k_rope,
        kv_cache, pe_cache,
        slot_mapping,
        kv_dim=kv_dim,
        pe_dim=pe_dim,
        block_size=block_size,
    )
