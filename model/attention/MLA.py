import torch
import torch.nn as nn
from typing import Optional

from .utils  import *
from ..utils import *

from torch.nn import functional as F


class MultiHeadLatentAttention(nn.Module):
    """
    Multi-Head Latent Attention (MLA) as used in DeepSeek-v3.

    Key innovations:
    1. Low-rank compression of KV cache to reduce memory usage
    2. Decoupled RoPE: Split into RoPE part and non-RoPE part, concatenate after RoPE
    3. Gating mechanism for multi-head attention output

    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        rope: Rotary Position Embedding module
        d_rope: Dimension for RoPE (default: 8)
        kv_lora_rank: Rank for KV compression (default: d_model // 2)
        q_lora_rank:  Rank for non-RoPE query compression (default: kv_lora_rank)
        use_gate: Whether to use head-wise gating mechanism (default: True)
        cache_enabled: Whether to enable KV caching for inference (default: False)
        max_batch_size: Maximum batch size for KV cache (default: 4)
        max_seq_len: Maximum sequence length for KV cache (default: 2048)
        device: Device for parameters
    """

    def __init__(
        self,
        d_model: int,
        head_num: int,
        rope: RotaryPositionalEmbedding = None,
        rope_dim: int = None,
        q_lora_rank: int = None,
        kv_lora_rank: int = None,
        use_gate: bool = True,
        cache_enabled: bool = False,
        max_batch_size: int = 4,
        max_seq_len: int = 2048,
        device=None,
        dtype=None  # Add dtype parameter
    ):
        super().__init__()
        assert d_model % head_num == 0, "d_model must be divisible by head_num"

        self.d_model = d_model
        self.num_heads = head_num
        self.head_dim = d_model // head_num
        self.use_gate = use_gate
        self.cache_enabled = cache_enabled
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len

        # parameter for low-rank compression
        self.rope_dim = rope_dim if rope_dim is not None else 8
        self.kv_lora_rank = kv_lora_rank if kv_lora_rank is not None else (d_model // 2)
        self.q_lora_rank  = q_lora_rank  if q_lora_rank  is not None else self.kv_lora_rank

        # q projection path with explicit dtype:
        self.q_down_proj = nn.Linear(d_model, self.q_lora_rank, device=device, dtype=dtype)
        self.q_nope_up_proj = nn.Linear(self.q_lora_rank, d_model, device=device, dtype=dtype)
        self.q_rope_up_proj = nn.Linear(self.q_lora_rank, head_num*self.rope_dim, device=device, dtype=dtype)

        # kv projection path with explicit dtype:
        self.kv_down_proj = nn.Linear(d_model, self.kv_lora_rank, device=device, dtype=dtype)
        self.k_up_proj = nn.Linear(self.kv_lora_rank, d_model, device=device, dtype=dtype)
        self.v_up_proj = nn.Linear(self.kv_lora_rank, d_model, device=device, dtype=dtype)
        self.k_rope_proj = nn.Linear(d_model, self.rope_dim, device=device, dtype=dtype)

        # initalize normalization and output projection (RMSNorm uses FP32 internally)
        self.q_norm  = nn.RMSNorm(self.q_lora_rank,  device=device)
        self.kv_norm = nn.RMSNorm(self.kv_lora_rank, device=device)
        self.rope = rope

        # A single scalar per head. We keep it simple and learn logits.
        # Applying sigmoid to these logits gives a head-wise scalar in (0,1).
        if self.use_gate:
            self.head_gates_logits = nn.Parameter(torch.zeros(self.num_heads, device=device, dtype=torch.float32))

        self.output_proj = nn.Linear(head_num*self.head_dim, d_model, device=device, dtype=dtype)

        # KV cache buffers (only used during inference when cache_enabled=True)
        # In MLA, we cache the COMPRESSED representations, not the full K and V
        cache_dtype = dtype if dtype is not None else torch.float32
        if self.cache_enabled:
            self.register_buffer(
                "kv_cache",
                torch.zeros(max_batch_size, max_seq_len, self.kv_lora_rank, device=device, dtype=cache_dtype),
                persistent=False
            )
            self.register_buffer(
                "pe_cache",
                torch.zeros(max_batch_size, max_seq_len, self.rope_dim, device=device, dtype=cache_dtype),
                persistent=False
            )
            # track actual valid sequence length for each sequence in the batch
            # to proper masking in batch inference with variable-length sequences
            self.register_buffer(
                "cache_seqlens",
                torch.zeros(max_batch_size, device=device, dtype=torch.long),
                persistent=False
            )

    def forward(self, x: torch.Tensor, start_pos: int, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """ Forward pass with KV caching support for efficient inference. """
        batch, seq_len, _ = x.shape
        # generate token positions based on start_pos
        token_positions = torch.arange(start_pos, start_pos + seq_len, device=x.device)

        # =========
        # Process Q
        # =========
        q_compressed = self.q_norm(self.q_down_proj(x))  # (bsz, seq_len, q_lora_rank)
        q_nope = self.q_nope_up_proj(q_compressed).view(batch, seq_len, self.num_heads, self.head_dim)
        q_rope = self.q_rope_up_proj(q_compressed).view(batch, seq_len, self.num_heads, self.rope_dim)

        q_rope = q_rope.transpose(1, 2)  # (bsz, num_heads, seq_len, d_rope)
        q_rope = self.rope(q_rope, token_positions)  # apply RoPE
        q_rope = q_rope.transpose(1, 2)  # (bsz, seq_len, num_heads, d_rope)

        # ===============
        # Process K and V
        # ===============
        kv_compressed = self.kv_norm(self.kv_down_proj(x))  # (bsz, seq_len, kv_lora_rank)
        k_rope = self.k_rope_proj(x)                        # (bsz, seq_len, d_rope)

        k_rope = k_rope.unsqueeze(1)  # (bsz, 1, seq_len, d_rope)
        k_rope = self.rope(k_rope, token_positions)  # apply RoPE
        k_rope = k_rope.squeeze(1)    # (bsz, seq_len, d_rope)

        # ============================
        # Inference: matrix absorption
        # ============================
        if self.cache_enabled:
            self.kv_cache[:batch, start_pos:start_pos + seq_len, :] = kv_compressed
            self.pe_cache[:batch, start_pos:start_pos + seq_len, :] = k_rope

            self.cache_seqlens[:batch] = start_pos + seq_len          # update valid lengths
            max_cached_len = self.cache_seqlens[:batch].max().item()  # maximum across batch

            cached_kv = self.kv_cache[:batch, :max_cached_len, :]
            cached_pe = self.pe_cache[:batch, :max_cached_len, :]

            # reshape martix to enable efficient einsum
            w_uk = self.k_up_proj.weight.view(self.num_heads, self.head_dim, self.kv_lora_rank)
            w_uv = self.v_up_proj.weight.view(self.num_heads, self.head_dim, self.kv_lora_rank)

            # absorb w_uk into q and compute attention score
            q_absorbed = torch.einsum('bqhd, hdk -> bqhk', q_nope, w_uk)
            attn_score = torch.einsum('bqhk, btk -> bhqt', q_absorbed, cached_kv)
            rope_score = torch.einsum('bqhd, btd -> bhqt', q_rope, cached_pe)
            score = (attn_score + rope_score) / (self.head_dim + self.rope_dim) ** 0.5

            # create per-sequence attention mask for batched inference 
            # for each sequence, mask out positions beyond its valid length
            attn_mask = torch.arange(max_cached_len, device=x.device).unsqueeze(0)  # (1, max_cached_len)
            valid_mask = attn_mask < self.cache_seqlens[:batch].unsqueeze(1)        # (bsz, max_cached_len)

            if mask is not None:
                # expand mask shape to (seq_len, max_cached_len)
                expanded_mask = torch.ones((seq_len, max_cached_len), device=x.device, dtype=torch.bool)
                expanded_mask[:, :seq_len] = mask  # apply causal mask to current tokens
                # combine with per-sequence valid mask by broadcasting
                valid_mask = valid_mask.unsqueeze(1) & expanded_mask.unsqueeze(0)
            else:
                # at decode Stage, just use per-sequence valid mask
                valid_mask = valid_mask.unsqueeze(1).expand(batch, seq_len, max_cached_len)

            # apply mask directly using masked_fill (more memory efficient than float mask)
            score = score.masked_fill(~valid_mask.unsqueeze(1), float('-inf'))

            # absorb w_uv into o and compute attention output
            attn_latent = torch.einsum('bhqt, btk -> bhqk', F.softmax(score, dim=-1), cached_kv)
            attn_output = torch.einsum('bhqk, hdk -> bqhd', attn_latent, w_uv)
        
        # ==============================
        # Training: separate computation
        # ==============================
        else:
            k_nope = self.k_up_proj(kv_compressed).view(batch, seq_len, self.num_heads, self.head_dim)
            v = self.v_up_proj(kv_compressed).view(batch, seq_len, self.num_heads, self.head_dim)

            # replicate k_rope to match each head and concatenate with k_nope
            k_rope = k_rope.unsqueeze(2).expand(batch, seq_len, self.num_heads, self.rope_dim)
            k = torch.cat([k_nope, k_rope], dim=-1)  # (bsz, seq_len, num_heads, head_dim+rope_dim)
            q = torch.cat([q_nope, q_rope], dim=-1)  # (bsz, seq_len, num_heads, head_dim+rope_dim)
            
            q = q.transpose(1, 2)  # (bsz, num_heads, seq_len, head_dim+rope_dim)
            k = k.transpose(1, 2)  # (bsz, num_heads, seq_len, head_dim+rope_dim)
            v = v.transpose(1, 2)  # (bsz, num_heads, seq_len, head_dim)

            attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
            attn_output = attn_output.transpose(1, 2)  # (bsz, seq_len, num_heads, head_dim)

        if self.use_gate:
            head_gates = torch.sigmoid(self.head_gates_logits)    # (num_heads,)
            gate_view = head_gates.view(1, 1, self.num_heads, 1)  # (1, 1, num_heads, 1)
            attn_output = attn_output * gate_view  # element-wise multiplication

        attn_output = attn_output.contiguous().view(batch, seq_len, -1)
        return self.output_proj(attn_output)  # (bsz, seq_len, d_model)

    def reset_cache(self):
        """Reset KV cache and sequence length tracking. Call this before each new generation session."""
        if self.cache_enabled:
            self.kv_cache.zero_()
            self.pe_cache.zero_()
            self.cache_seqlens.zero_()
