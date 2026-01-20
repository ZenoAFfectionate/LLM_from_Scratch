import torch
import torch.nn as nn
from typing import Optional

from .utils  import *

from torch.nn import functional as F


class GroupedQueryAttention(nn.Module):
    """
    Grouped-Query Attention (GQA).
    - num_query_heads: number of query heads (H_q)
    - num_kv_heads: number of key/value heads (G groups). Must divide num_query_heads
    - head_dim = d_model // num_query_heads
    """
    def __init__(self, d_model: int, num_query_heads: int, num_kv_heads: int,
                 rope: RotaryPositionalEmbedding = None, use_gate: bool = True,
                 cache_enabled: bool = False, max_batch_size: int = 4,
                 max_seq_len: int = 2048, device=None, dtype=None):
        super().__init__()
        assert d_model % num_query_heads == 0, "d_model must be divisible by num_query_heads"
        assert num_query_heads % num_kv_heads == 0, "num_query_heads must be divisible by num_kv_heads"
        self.d_model = d_model
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_query_heads
        self.group_size = num_query_heads // num_kv_heads
        self.use_gate = use_gate
        self.cache_enabled = cache_enabled
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        # initialize the projection layers with explicit dtype (BF16 for weights)
        self.q_proj = nn.Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = nn.Linear(d_model, num_kv_heads*self.head_dim, device=device, dtype=dtype)
        self.v_proj = nn.Linear(d_model, num_kv_heads*self.head_dim, device=device, dtype=dtype)
        self.output_proj = nn.Linear(d_model, d_model, device=device, dtype=dtype)
        # initalize the normalization layers (RMSNorm uses FP32 internally)
        self.q_norm = nn.RMSNorm(d_model, device=device)
        self.k_norm = nn.RMSNorm(num_kv_heads*self.head_dim, device=device)
        self.rope = rope

        # A single scalar per head. We keep it simple and learn logits.
        # Applying sigmoid to these logits gives a head-wise scalar in (0,1).
        # Gates should be in FP32 for stability
        if self.use_gate:
            self.head_gates_logits = nn.Parameter(torch.zeros(self.num_query_heads, device=device, dtype=torch.float32))

        # KV cache buffers: For GQA, we cache num_kv_heads (not num_query_heads)
        cache_dtype = dtype if dtype is not None else torch.float32
        if self.cache_enabled:
            self.register_buffer(
                "k_cache",
                torch.zeros(max_batch_size, num_kv_heads, max_seq_len, self.head_dim, device=device, dtype=cache_dtype),
                persistent=False
            )
            self.register_buffer(
                "v_cache",
                torch.zeros(max_batch_size, num_kv_heads, max_seq_len, self.head_dim, device=device, dtype=cache_dtype),
                persistent=False
            )
            # Track the current position in the cache
            self.register_buffer("cache_position", torch.zeros(1, dtype=torch.long), persistent=False)

    def forward(self, x: torch.Tensor, start_pos: int, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with KV caching support for efficient inference.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            start_pos: Starting position for RoPE and KV cache updates
            mask: Optional attention mask

        Returns:
            - output: Output tensor with same shape as input
        """
        bsz, seq_len, d_model = x.shape
        # generate token positions based on start_pos
        token_positions = torch.arange(start_pos, start_pos + seq_len, device=x.device)

        q = self.q_proj(x); q = self.q_norm(q)  # [OPT] add RMSNorm for q
        k = self.k_proj(x); k = self.k_norm(k)  # [OPT] add RMSNorm for k
        v = self.v_proj(x)

        # (bsz, seq_len, num_q_heads, head_dim)  -> (bsz, num_q_heads, seq_len, head_dim)
        q = q.view(bsz, seq_len, self.num_query_heads, self.head_dim).transpose(-2, -3)
        # (bsz, seq_len, num_kv_heads, head_dim) -> (bsz, num_kv_heads, seq_len, head_dim)
        k = k.view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(-2, -3)
        v = v.view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(-2, -3)

        # apply RoPE to q and k using token_positions
        if self.rope is not None:
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        # ==========================
        # Inference: with KV caching
        # ==========================
        if self.cache_enabled:
            # store new k, v in cache at start_pos (bsz, num_kv_heads, seq_len, head_dim)
            self.k_cache[:bsz, :, start_pos:start_pos + seq_len, :] = k
            self.v_cache[:bsz, :, start_pos:start_pos + seq_len, :] = v
            # use all cached k, v up to current position
            total_seq_len = start_pos + seq_len
            k = self.k_cache[:bsz, :, :total_seq_len, :]
            v = self.v_cache[:bsz, :, :total_seq_len, :]

            # handle mask differently for prefill vs decode
            if mask is not None:
                # prefill stage: need to expand mask to account for cached positions
                expanded_mask = torch.ones((seq_len, total_seq_len), device=x.device, dtype=torch.bool)
                expanded_mask[:, :seq_len] = mask  # apply causal mask to current tokens
                mask = expanded_mask
        else:
            # Training mode: total_seq_len is just seq_len
            total_seq_len = seq_len

        if self.group_size > 1:
            # OPTIMIZED: Use expand + reshape instead of repeat_interleave (no memory copy)
            k = k.unsqueeze(2).expand(
                bsz, self.num_kv_heads, self.group_size, total_seq_len, self.head_dim
            ).reshape(bsz, self.num_query_heads, total_seq_len, self.head_dim)
            v = v.unsqueeze(2).expand(
                bsz, self.num_kv_heads, self.group_size, total_seq_len, self.head_dim
            ).reshape(bsz, self.num_query_heads, total_seq_len, self.head_dim)

        # ============================
        # Training: standard attention
        # ============================
        attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

        if self.use_gate:
            # apply head-wise sigmoid gates
            head_gates = torch.sigmoid(self.head_gates_logits)          # (num_query_heads,)
            gate_view = head_gates.view(1, self.num_query_heads, 1, 1)  # (1, num_query_heads, 1, 1)
            attn_output = attn_output * gate_view  # multiply elementwise

        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, d_model)
        return self.output_proj(attn_output)
