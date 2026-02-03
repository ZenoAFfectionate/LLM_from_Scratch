import torch
import torch.nn as nn
from typing import Optional

from .utils import (
    RotaryPositionalEmbedding,
    store_kvcache, flash_attention_prefill, paged_attention_decode
)

from torch.nn import functional as F


# -------------------------------------------------------
#  Problem 8: Implement Multi-Head Self-Attention Module
# -------------------------------------------------------
class MultiHeadSelfAttention(nn.Module):
    """
    Causal Multi-Head Self-Attention module with RoPE.

    Supports two modes:
    1. Training mode: Use forward() for standard attention with optional causal mask
    2. Inference mode: Use inference() for paged attention with vLLM-style engine
       - Set paged_attention=True and assign k_cache/v_cache from ModelRunner
    """

    def __init__(self, d_model: int, num_heads: int, rope: RotaryPositionalEmbedding,
                 device=None, dtype=None):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Paged attention mode (for inference with engine)
        self.paged_attention = False
        self.k_cache = None  # Will be set by ModelRunner
        self.v_cache = None  # Will be set by ModelRunner
        self.block_size = 256  # Default, will be updated by ModelRunner

        # Pre-compute attention scale factor
        self.scale = 1.0 / (self.head_dim ** 0.5)

        # initialize the projection layers with explicit dtype
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, device=device, dtype=dtype)
        self.output_proj = nn.Linear(d_model, d_model, device=device, dtype=dtype)
        # initalize the normalization layers (uses FP32 internally)
        self.q_norm = nn.RMSNorm(d_model, device=device)
        self.k_norm = nn.RMSNorm(d_model, device=device)
        self.rope = rope

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for training mode.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional attention mask

        Returns:
            output: Output tensor with same shape as input
        """
        bsz, seq_len, d_model = x.shape
        # Generate token positions (always starting from 0 for training)
        token_positions = torch.arange(seq_len, device=x.device)

        # project input to q, k, v using fused projection
        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)
        q = self.q_norm(q)
        k = self.k_norm(k)

        # reshape q, k, v for multi-head attention: (batch, num_heads, seq_len, head_dim)
        q = q.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # apply RoPE to q and k using token_positions
        if self.rope is not None:
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        # Training: standard attention
        attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, d_model)
        return self.output_proj(attn_output)

    def inference(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inference pass using paged attention for efficient inference.
        Uses context from utils.context to get slot_mapping, block_tables, etc.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            output: Output tensor with same shape as input
        """
        from utils.context import get_context
        context = get_context()

        bsz, seq_len, _ = x.shape
        total_tokens = bsz * seq_len
        x_flat = x.view(total_tokens, self.d_model)  # Flatten for processing

        # Compute token positions based on context
        if context.is_prefill and context.cu_seqlens_q is not None:
            # For varlen prefill: positions restart at 0 for each sequence
            positions = []
            cu_seqlens = context.cu_seqlens_q.cpu().tolist()
            for i in range(len(cu_seqlens) - 1):
                seq_len_i = cu_seqlens[i+1] - cu_seqlens[i]
                positions.extend(range(seq_len_i))
            token_positions = torch.tensor(
                positions, dtype=torch.long, device=x.device)
        elif context.is_prefill:
            token_positions = torch.arange(total_tokens, device=x.device)
        else:
            # For decode: context_lens - 1 gives the current position for each sequence
            token_positions = context.context_lens - 1

        # Project Q, K, V using fused projection
        q, k, v = self.qkv_proj(x_flat).chunk(3, dim=-1)
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Reshape to (total_tokens, num_heads, head_dim)
        q = q.view(total_tokens, self.num_heads, self.head_dim)
        k = k.view(total_tokens, self.num_heads, self.head_dim)
        v = v.view(total_tokens, self.num_heads, self.head_dim)

        # Apply RoPE
        if self.rope is not None:
            # RoPE expects (..., seq_len, head_dim), so we need positions at dim -2
            # Current shape: (tokens, num_heads, head_dim) -> transpose to (num_heads, tokens, head_dim)
            q = self.rope(q.transpose(0, 1), token_positions).transpose(0, 1)
            k = self.rope(k.transpose(0, 1), token_positions).transpose(0, 1)

        # Store K, V to paged cache
        if context.slot_mapping is not None:
            store_kvcache(k, v, self.k_cache, self.v_cache, context.slot_mapping, self.block_size)

        if context.is_prefill:
            # Prefill: use flash attention with variable-length support
            cu_seqlens = context.cu_seqlens_q
            if cu_seqlens is None:
                cu_seqlens = torch.tensor(
                    [0, total_tokens], dtype=torch.int32, device=x.device)

            attn_output = flash_attention_prefill(
                q, k, v, cu_seqlens, self.scale,
                self.num_heads, self.num_heads, self.head_dim  # MHA: num_kv_heads = num_heads
            )
            # Output shape: (total_tokens, num_heads, head_dim)
        else:
            # Decode: use paged attention to read from cache
            # Query shape for decode: (batch_size, num_heads, head_dim)
            q_decode = q.view(bsz, self.num_heads, self.head_dim)

            attn_output = paged_attention_decode(
                q_decode,
                self.k_cache, self.v_cache,
                context.block_tables, context.context_lens,
                self.scale, self.num_heads, self.num_heads,  # MHA: num_kv_heads = num_heads
                self.head_dim, self.block_size
            )
            # (batch_size, num_heads, head_dim) -> (total_tokens, num_heads, head_dim)
            attn_output = attn_output.view(total_tokens, self.num_heads, self.head_dim)

        # Reshape and project output
        output = self.output_proj(attn_output.view(total_tokens, self.d_model))
        return output.view(bsz, seq_len, self.d_model)
