import torch
import torch.nn as nn
from typing import Optional

from .utils import (
    RotaryPositionalEmbedding,
    store_kvcache, flash_attention_prefill, paged_attention_decode,
    compute_varlen_positions
)

from torch.nn import functional as F


class CCAConvMixing(nn.Module):
    """
    Compressed Convolutional Preconditioner for Q and K.

    Paper §C "Sequence-Mixing Convolutions":
      A short convolution and grouped head-wise convolution act as
      lightweight preconditioners before attention.

    Two-stage mixing per head:
      1. Depthwise causal conv along sequence (mixes nearby tokens per channel)
      2. Pointwise conv (mixes channels within the compressed head)

    Reference: Figliolia et al. 2025 — Compressed Convolutional Attention
    """

    def __init__(self, head_dim, kernel_size=3):
        super().__init__()

        # Depthwise causal conv — short convolution along sequence
        self.seq_conv = nn.Conv1d(
            in_channels=head_dim,
            out_channels=head_dim,
            kernel_size=kernel_size,
            padding=kernel_size - 1,
            groups=head_dim,
        )

        # Pointwise conv — grouped head-wise channel mixing
        self.channel_conv = nn.Conv1d(
            in_channels=head_dim,
            out_channels=head_dim,
            kernel_size=1,
        )

    def _mix(self, x):
        """
        Args:
            x: (batch, seq, heads, head_dim)
        Returns:
            x: (batch, seq, heads, head_dim) — mixed representation
        """
        batch, seq, heads, head_dim = x.shape

        # Conv1d expects [batch, channels, seq]
        x = x.permute(0, 2, 3, 1)  # (batch, heads, head_dim, seq)
        x = x.reshape(batch * heads, head_dim, seq)

        # Causal sequence convolution
        x = self.seq_conv(x)
        x = x[..., :seq]  # remove right padding (causal)

        # Channel mixing within each head
        x = self.channel_conv(x)

        # Back to (batch, seq, heads, head_dim)
        x = x.reshape(batch, heads, head_dim, seq)
        x = x.permute(0, 3, 1, 2)

        return x

    def forward(self, q, k):
        """
        Args:
            q: (batch, seq, heads, c_dim)
            k: (batch, seq, heads, c_dim)
        Returns:
            q, k after convolutional mixing
        """
        q = self._mix(q)
        k = self._mix(k)
        return q, k


class CompressedConvAttention(nn.Module):
    """
    Compressed Convolutional Attention (CCA) — CCGQA variant.

    Based on ZAYA1-8B / Figliolia et al. 2025:

      Paper §C — CCA has several core components:
        1. Low-Rank Projections: compress Q/K/V to latent space (c_dim)
        2. Sequence-Mixing Convolutions: CCAConvMixing on Q and K
        3. Value Head Time-Delay: 1-token shift on half the V heads
        4. Skip Connections: Q/K mean skip for representational similarity
        5. Key Temperature: learned linear T (not exp(T)) for stability
        6. 50% RoPE: RoPE applied to half the compressed head dimension
        7. GQA head sharing: num_kv_heads ≤ num_query_heads
        8. Attention in compressed space → up-projection to d_model

    ZAYA1-8B config (Table I):
      d_model=2048, query_heads=8, kv_heads=2, head_dim=128
      Query compression 2×, KV-cache compression 8× vs full MHA
      Position embeddings: 50% RoPE on each head

    Supports training (forward) and inference (paged attention) modes.
    """

    def __init__(
        self,
        d_model: int,
        num_query_heads: int,
        num_kv_heads: int,
        c_dim: int,
        rope: RotaryPositionalEmbedding = None,
        conv_kernel_size: int = 3,
        device=None,
        dtype=None
    ):
        super().__init__()
        assert d_model % num_query_heads == 0, "d_model must be divisible by num_query_heads"
        assert num_query_heads % num_kv_heads == 0, "num_query_heads must be divisible by num_kv_heads"

        self.d_model = d_model
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_query_heads   # original full head dim
        self.c_dim = c_dim                            # compressed dim per head
        self.group_size = num_query_heads // num_kv_heads

        # 50% RoPE: only half the compressed dim gets RoPE (paper Table I)
        self.rope_dim = c_dim // 2
        self.non_rope_dim = c_dim - self.rope_dim

        total_q_dim = num_query_heads * c_dim
        total_kv_dim = num_kv_heads * c_dim

        # Paged attention
        self.paged_attention = False
        self.k_cache = None
        self.v_cache = None
        self.block_size = 256

        # Attention scale based on compressed dimension
        self.scale = 1.0 / (c_dim ** 0.5)

        # ── 1. Low-Rank Projections (paper §C) ──
        self.q_down_proj = nn.Linear(d_model, total_q_dim, device=device, dtype=dtype)
        self.k_down_proj = nn.Linear(d_model, total_kv_dim, device=device, dtype=dtype)
        self.v_down_proj = nn.Linear(d_model, total_kv_dim, device=device, dtype=dtype)

        # RMSNorm on compressed representations
        self.q_norm = nn.RMSNorm(total_q_dim, device=device)
        self.k_norm = nn.RMSNorm(total_kv_dim, device=device)

        # ── 2. Sequence-Mixing Convolutions (paper §C) ──
        self.conv_mix = CCAConvMixing(c_dim, kernel_size=conv_kernel_size)

        # ── 5. Learned key temperature T (linear, not exp) for stability ──
        self.key_temp = nn.Parameter(torch.ones(1))

        # ── RoPE (applied to rope_dim = c_dim // 2 only) ──
        self.rope = rope

        # ── Output up-projection ──
        self.output_proj = nn.Linear(total_q_dim, d_model, device=device, dtype=dtype)

    # ====================================================================
    #  Training forward pass
    # ====================================================================

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for training mode.

        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            output: (batch_size, seq_len, d_model)
        """
        bsz, seq_len, _ = x.shape
        token_positions = torch.arange(seq_len, device=x.device)

        # ── 1. Low-rank down-projections ──
        q = self.q_norm(self.q_down_proj(x))  # (bsz, seq, num_q_heads * c_dim)
        k = self.k_norm(self.k_down_proj(x))  # (bsz, seq, num_kv_heads * c_dim)
        v = self.v_down_proj(x)               # (bsz, seq, num_kv_heads * c_dim)

        # Reshape for per-head processing: (bsz, seq, heads, c_dim)
        q = q.view(bsz, seq_len, self.num_query_heads, self.c_dim)
        k = k.view(bsz, seq_len, self.num_kv_heads, self.c_dim)
        v = v.view(bsz, seq_len, self.num_kv_heads, self.c_dim)

        # ── 4. Q/K Mean Skip Connections (paper §C) ──
        # Save pre-conv means to add back after mixing
        q_mean = q.mean(dim=-1, keepdim=True)
        k_mean = k.mean(dim=-1, keepdim=True)

        # ── 2. Convolutional preconditioner on Q and K ──
        q_mixed, k_mixed = self.conv_mix(q, k)

        # Apply mean skip: Q = Q_conv + Q_pre_mean
        q = q_mixed + q_mean
        k = k_mixed + k_mean

        # ── 5. Key temperature (linear T, not exp(T)) ──
        k = k * self.key_temp

        # ── 6. 50% RoPE: split into RoPE and non-RoPE parts ──
        # Split along last dim: first half gets RoPE, second half passed through
        q_rope = q[..., :self.rope_dim]
        q_nope = q[..., self.rope_dim:]
        k_rope = k[..., :self.rope_dim]
        k_nope = k[..., self.rope_dim:]

        # Transpose for RoPE: (bsz, heads, seq, rope_dim)
        q_rope = q_rope.transpose(1, 2)
        k_rope = k_rope.transpose(1, 2)

        if self.rope is not None:
            q_rope = self.rope(q_rope, token_positions)
            k_rope = self.rope(k_rope, token_positions)

        # Concatenate back: (bsz, heads, seq, c_dim)
        q_nope = q_nope.transpose(1, 2)
        k_nope = k_nope.transpose(1, 2)
        q = torch.cat([q_rope, q_nope], dim=-1)
        k = torch.cat([k_rope, k_nope], dim=-1)

        # Transpose V and apply GQA expansion
        v = v.transpose(1, 2)  # (bsz, num_kv_heads, seq, c_dim)

        # ── 3. Value Head Time-Delay (paper §C) ──
        # Apply 1-token delay to half the value heads
        v = self._apply_value_time_delay(v)

        # ── 7. GQA: expand KV heads to match Q heads ──
        if self.group_size > 1:
            k = k.unsqueeze(2).expand(
                bsz, self.num_kv_heads, self.group_size, seq_len, self.c_dim
            ).reshape(bsz, self.num_query_heads, seq_len, self.c_dim)
            v = v.unsqueeze(2).expand(
                bsz, self.num_kv_heads, self.group_size, seq_len, self.c_dim
            ).reshape(bsz, self.num_query_heads, seq_len, self.c_dim)

        # ── 8. Scaled dot-product attention in compressed space ──
        attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # (bsz, num_q_heads, seq, c_dim) → (bsz, seq, total_q_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            bsz, seq_len, self.num_query_heads * self.c_dim)

        return self.output_proj(attn_output)

    def _apply_value_time_delay(self, v: torch.Tensor) -> torch.Tensor:
        """
        Paper §C "Value Head Time-Delay":
          A time delay of one token is applied to half of the value heads.

        Shifts half the V heads forward by 1 position (pad with zero at pos 0).
        This gives those heads access to the previous token's value,
        providing temporal context in the compressed space.

        Args:
            v: (bsz, num_kv_heads, seq, c_dim)
        Returns:
            v with time delay applied to heads [0 : num_heads//2]
        """
        num_delay_heads = self.num_kv_heads // 2
        if num_delay_heads == 0:
            return v

        # Split into delayed and non-delayed heads
        v_delay = v[:, :num_delay_heads]    # first half: delay by 1
        v_pass = v[:, num_delay_heads:]     # second half: no delay

        # Shift forward by 1 (delay) — drop last token, prepend zero at position 0
        v_delay = F.pad(v_delay[..., :-1, :], (0, 0, 1, 0))

        return torch.cat([v_delay, v_pass], dim=1)

    # ====================================================================
    #  Inference pass
    # ====================================================================

    def inference(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inference using paged attention.  CCA stores compressed K/V in cache
        (c_dim < head_dim → reduced memory).
        """
        from utils.context import get_context
        context = get_context()

        bsz, seq_len, _ = x.shape
        total_tokens = bsz * seq_len
        x_flat = x.view(total_tokens, self.d_model)

        if context.is_prefill and context.cu_seqlens_q is not None:
            token_positions = compute_varlen_positions(
                context.cu_seqlens_q, total_tokens, x.device)
        elif context.is_prefill:
            token_positions = torch.arange(total_tokens, device=x.device)
        else:
            token_positions = context.context_lens - 1

        # Low-rank projections
        q = self.q_norm(self.q_down_proj(x_flat))
        k = self.k_norm(self.k_down_proj(x_flat))
        v = self.v_down_proj(x_flat)

        q = q.view(total_tokens, self.num_query_heads, self.c_dim)
        k = k.view(total_tokens, self.num_kv_heads, self.c_dim)
        v = v.view(total_tokens, self.num_kv_heads, self.c_dim)

        # Key temperature
        k = k * self.key_temp

        # 50% RoPE
        if self.rope is not None:
            q_rope = q[..., :self.rope_dim]
            q_nope = q[..., self.rope_dim:]
            k_rope = k[..., :self.rope_dim]
            k_nope = k[..., self.rope_dim:]

            q_rope = self.rope(q_rope.transpose(0, 1), token_positions).transpose(0, 1)
            k_rope = self.rope(k_rope.transpose(0, 1), token_positions).transpose(0, 1)

            q = torch.cat([q_rope, q_nope], dim=-1)
            k = torch.cat([k_rope, k_nope], dim=-1)

        # Store compressed K, V to paged cache
        if context.slot_mapping is not None:
            store_kvcache(k, v, self.k_cache, self.v_cache,
                          context.slot_mapping, self.block_size)

        if context.is_prefill:
            cu_seqlens = context.cu_seqlens_q
            if cu_seqlens is None:
                cu_seqlens = torch.tensor(
                    [0, total_tokens], dtype=torch.int32, device=x.device)

            attn_output = flash_attention_prefill(
                q, k, v, cu_seqlens, self.scale,
                self.num_query_heads, self.num_kv_heads, self.c_dim
            )
        else:
            q_decode = q.view(bsz, self.num_query_heads, self.c_dim)
            attn_output = paged_attention_decode(
                q_decode,
                self.k_cache, self.v_cache,
                context.block_tables, context.context_lens,
                self.scale, self.num_query_heads, self.num_kv_heads,
                self.c_dim, self.block_size
            )
            attn_output = attn_output.view(total_tokens, self.num_query_heads, self.c_dim)

        output = self.output_proj(attn_output.view(
            total_tokens, self.num_query_heads * self.c_dim))
        return output.view(bsz, seq_len, self.d_model)
