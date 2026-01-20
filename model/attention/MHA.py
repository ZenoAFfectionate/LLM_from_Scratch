import torch
import torch.nn as nn
from typing import Optional

from .utils  import *

from torch.nn import functional as F


# -------------------------------------------------------
#  Problem 8: Implement Multi-Head Self-Attention Module
# -------------------------------------------------------
class MultiHeadSelfAttention(nn.Module):
    """ Causal Multi-Head Self-Attention module with RoPE """

    def __init__(self, d_model: int, num_heads: int, rope: RotaryPositionalEmbedding,
                 use_gate: bool = True, cache_enabled: bool = False,
                 max_batch_size: int = 4, max_seq_len: int = 2048, device=None, dtype=None):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.use_gate = use_gate
        self.cache_enabled = cache_enabled
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len

        # initialize the projection layers with explicit dtype
        self.q_proj = nn.Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = nn.Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = nn.Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = nn.Linear(d_model, d_model, device=device, dtype=dtype)
        # initalize the normalization layers (uses FP32 internally)
        self.q_norm = nn.RMSNorm(d_model, device=device)
        self.k_norm = nn.RMSNorm(d_model, device=device)
        self.rope = rope

        # A single scalar per head. We keep it simple and learn logits.
        # Applying sigmoid to these logits gives a head-wise scalar in (0,1).
        # Gates should be FP32 for stability
        if self.use_gate:
            self.head_gates_logits = nn.Parameter(torch.zeros(self.num_heads, device=device, dtype=torch.float32))

        # KV cache buffers (only used during inference when cache_enabled=True)
        cache_dtype = dtype if dtype is not None else torch.float32
        if self.cache_enabled:
            self.register_buffer(
                "k_cache",
                torch.zeros(max_batch_size, num_heads, max_seq_len, self.head_dim, device=device, dtype=cache_dtype),
                persistent=False
            )
            self.register_buffer(
                "v_cache",
                torch.zeros(max_batch_size, num_heads, max_seq_len, self.head_dim, device=device, dtype=cache_dtype),
                persistent=False
            )
            # Track the current position in the cache
            self.register_buffer("cache_position", torch.zeros(1, dtype=torch.long), persistent=False)

    def forward(self, x: torch.Tensor, start_pos: int, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with KV caching support for efficient inference."""
        bsz, seq_len, d_model = x.shape
        # Generate token positions based on start_pos
        token_positions = torch.arange(start_pos, start_pos + seq_len, device=x.device)

        # project input to q, k, v (batch, seq_len, d_model)
        q = self.q_proj(x); q = self.q_norm(q)  # [OPT] add RMSNorm for q
        k = self.k_proj(x); k = self.k_norm(k)  # [OPT] add RMSNorm for k
        v = self.v_proj(x)

        # reshape q, k, v for multi-head attention: (batch, seq_len, num_heads, head_dim)
        q = q.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # apply RoPE to q and k using token_positions
        if self.rope is not None:
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        # ==========================
        # Inference: with KV caching
        # ==========================
        if self.cache_enabled:
            # store new k, v in cache at start_pos (batch_size, num_heads, seq_len, head_dim)
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

        # ============================
        # Training: standard attention
        # ============================
        attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

        if self.use_gate:
            # apply head-wise sigmoid gates
            head_gates = torch.sigmoid(self.head_gates_logits)    # (num_heads,)
            gate_view = head_gates.view(1, self.num_heads, 1, 1)  # (1, num_heads, 1, 1)
            attn_output = attn_output * gate_view  # multiply elementwise

        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, d_model)
        return self.output_proj(attn_output)
