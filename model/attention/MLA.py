import torch
import torch.nn as nn
from typing import Optional

from .utils import RotaryPositionalEmbedding, store_mla_cache

from torch.nn import functional as F


class MultiHeadLatentAttention(nn.Module):
    """
    Multi-Head Latent Attention (MLA) as used in DeepSeek-v3.

    Key innovations:
    1. Low-rank compression of KV cache to reduce memory usage
    2. Decoupled RoPE: Split into RoPE part and non-RoPE part, concatenate after RoPE

    Supports two modes:
    1. Training mode: Use forward() for standard attention with optional causal mask
    2. Inference mode: Use inference() for paged attention with vLLM-style engine
       - Set paged_attention=True and assign k_cache/v_cache from ModelRunner
       - k_cache stores kv_compressed (kv_lora_rank)
       - v_cache stores k_rope (rope_dim)
    """

    def __init__(
        self,
        d_model: int,
        head_num: int,
        rope: RotaryPositionalEmbedding = None,
        rope_dim: int = None,
        q_lora_rank: int = None,
        kv_lora_rank: int = None,
        device=None,
        dtype=None
    ):
        super().__init__()
        assert d_model % head_num == 0, "d_model must be divisible by head_num"

        self.d_model = d_model
        self.num_heads = head_num
        self.head_dim = d_model // head_num

        # parameter for low-rank compression
        self.rope_dim = rope_dim if rope_dim is not None else 8
        self.kv_lora_rank = kv_lora_rank if kv_lora_rank is not None else (
            d_model // 4)
        self.q_lora_rank = q_lora_rank if q_lora_rank is not None else self.kv_lora_rank

        # Paged attention mode (for inference with engine)
        self.paged_attention = False
        # will store kv_compressed (num_blocks, block_size, kv_lora_rank)
        self.k_cache = None
        # will store k_rope (num_blocks, block_size, rope_dim)
        self.v_cache = None
        self.block_size = 256

        # Pre-compute attention scale factor
        self.scale = 1.0 / ((self.head_dim + self.rope_dim) ** 0.5)

        # q projection path with explicit dtype:
        self.q_down_proj = nn.Linear(
            d_model, self.q_lora_rank, device=device, dtype=dtype)
        self.q_up_proj_fused = nn.Linear(
            self.q_lora_rank, d_model + head_num * self.rope_dim,
            device=device, dtype=dtype
        )

        # kv projection path with explicit dtype:
        self.kv_down_proj = nn.Linear(
            d_model, self.kv_lora_rank, device=device, dtype=dtype)
        self.kv_up_proj_fused = nn.Linear(
            self.kv_lora_rank, 2 * d_model,
            device=device, dtype=dtype
        )
        self.k_rope_proj = nn.Linear(
            d_model, self.rope_dim, device=device, dtype=dtype)

        # initalize normalization and output projection (RMSNorm uses FP32 internally)
        self.q_norm = nn.RMSNorm(self.q_lora_rank,  device=device)
        self.kv_norm = nn.RMSNorm(self.kv_lora_rank, device=device)
        self.rope = rope

        self.output_proj = nn.Linear(
            head_num*self.head_dim, d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for training mode.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional attention mask

        Returns:
            output: Output tensor with same shape as input
        """
        batch, seq_len, _ = x.shape
        # Generate token positions (always starting from 0 for training)
        token_positions = torch.arange(seq_len, device=x.device)

        # =========
        # Process Q
        # =========
        q_compressed = self.q_norm(self.q_down_proj(x))
        q_fused = self.q_up_proj_fused(q_compressed)
        q_nope = q_fused[..., :self.d_model].view(
            batch, seq_len, self.num_heads, self.head_dim)
        q_rope = q_fused[..., self.d_model:].view(
            batch, seq_len, self.num_heads, self.rope_dim)

        q_rope = q_rope.transpose(1, 2)
        q_rope = self.rope(q_rope, token_positions)
        q_rope = q_rope.transpose(1, 2)

        # ===============
        # Process K and V
        # ===============
        kv_compressed = self.kv_norm(self.kv_down_proj(x))
        k_rope = self.k_rope_proj(x)

        k_rope = k_rope.unsqueeze(1)
        k_rope = self.rope(k_rope, token_positions)
        k_rope = k_rope.squeeze(1)

        # Training: fused computation
        kv_fused = self.kv_up_proj_fused(kv_compressed)
        k_nope = kv_fused[..., :self.d_model].view(
            batch, seq_len, self.num_heads, self.head_dim)
        v = kv_fused[..., self.d_model:].view(
            batch, seq_len, self.num_heads, self.head_dim)

        k_rope = k_rope.unsqueeze(2).expand(
            batch, seq_len, self.num_heads, self.rope_dim)
        k = torch.cat([k_nope, k_rope], dim=-1)
        q = torch.cat([q_nope, q_rope], dim=-1)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        attn_output = attn_output.transpose(
            1, 2).contiguous().view(batch, seq_len, -1)
        return self.output_proj(attn_output)

    def inference(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inference pass using paged attention for efficient inference.

        MLA stores compressed representations:
        - k_cache: kv_compressed (kv_lora_rank)
        - v_cache: k_rope (rope_dim)

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            output: Output tensor with same shape as input
        """
        from utils.context import get_context
        context = get_context()

        bsz, seq_len, _ = x.shape
        total_tokens = bsz * seq_len
        x_flat = x.view(total_tokens, self.d_model)

        # Compute token positions based on context
        if context.is_prefill and context.cu_seqlens_q is not None:
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
            token_positions = context.context_lens - 1

        # =========
        # Process Q
        # =========
        q_compressed = self.q_norm(self.q_down_proj(
            x_flat))  # (total_tokens, q_lora_rank)
        # (total_tokens, d_model + num_heads * rope_dim)
        q_fused = self.q_up_proj_fused(q_compressed)
        q_nope = q_fused[..., :self.d_model].view(
            total_tokens, self.num_heads, self.head_dim)
        q_rope = q_fused[..., self.d_model:].view(
            total_tokens, self.num_heads, self.rope_dim)

        # Apply RoPE to q_rope: (tokens, heads, rope_dim) -> (heads, tokens, rope_dim) for RoPE
        q_rope = self.rope(q_rope.transpose(
            0, 1), token_positions).transpose(0, 1)

        # ===============
        # Process K and V
        # ===============
        kv_compressed = self.kv_norm(self.kv_down_proj(
            x_flat))  # (total_tokens, kv_lora_rank)
        # (total_tokens, rope_dim)
        k_rope = self.k_rope_proj(x_flat)

        # Apply RoPE to k_rope: (tokens, rope_dim) -> (1, tokens, rope_dim) for RoPE
        k_rope = self.rope(k_rope.unsqueeze(0), token_positions).squeeze(0)

        # Store to paged cache
        if context.slot_mapping is not None:
            store_mla_cache(
                kv_compressed, k_rope,
                self.k_cache, self.v_cache,
                context.slot_mapping, self.block_size
            )

        if context.is_prefill:
            # Prefill: use standard attention
            # Expand kv_compressed to full K and V
            kv_fused = self.kv_up_proj_fused(
                kv_compressed)  # (total_tokens, 2 * d_model)
            k_nope = kv_fused[..., :self.d_model].view(
                total_tokens, self.num_heads, self.head_dim)
            v = kv_fused[..., self.d_model:].view(
                total_tokens, self.num_heads, self.head_dim)

            # Replicate k_rope to match each head and concatenate
            k_rope_expanded = k_rope.unsqueeze(1).expand(
                total_tokens, self.num_heads, self.rope_dim)
            # (tokens, heads, head_dim+rope_dim)
            k = torch.cat([k_nope, k_rope_expanded], dim=-1)
            # (tokens, heads, head_dim+rope_dim)
            q = torch.cat([q_nope, q_rope], dim=-1)

            # For prefill with varlen, use per-sequence attention
            cu_seqlens = context.cu_seqlens_q
            if cu_seqlens is None:
                cu_seqlens = torch.tensor(
                    [0, total_tokens], dtype=torch.int32, device=x.device)

            # Use batched attention for each sequence
            attn_output = self._varlen_attention(q, k, v, cu_seqlens)
        else:
            # Decode: matrix absorption
            attn_output = self._paged_decode_attention(
                q_nope, q_rope, context.block_tables, context.context_lens, bsz
            )

        output = self.output_proj(attn_output.view(total_tokens, self.d_model))
        return output.view(bsz, seq_len, self.d_model)

    def _varlen_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, cu_seqlens: torch.Tensor
    ) -> torch.Tensor:
        """
        Variable-length attention for prefill phase.

        Args:
            q: (total_tokens, num_heads, head_dim + rope_dim)
            k: (total_tokens, num_heads, head_dim + rope_dim)
            v: (total_tokens, num_heads, head_dim)
            cu_seqlens: cumulative sequence lengths

        Returns:
            output: (total_tokens, num_heads, head_dim)
        """
        num_seqs = cu_seqlens.shape[0] - 1
        outputs = []

        for i in range(num_seqs):
            start = cu_seqlens[i].item()
            end = cu_seqlens[i + 1].item()
            seq_len = end - start

            q_seq = q[start:end].transpose(
                0, 1).unsqueeze(0)  # (1, heads, seq, dim)
            k_seq = k[start:end].transpose(
                0, 1).unsqueeze(0)  # (1, heads, seq, dim)
            v_seq = v[start:end].transpose(0, 1).unsqueeze(
                0)  # (1, heads, seq, head_dim)

            # Causal mask
            mask = torch.tril(torch.ones(seq_len, seq_len,
                              device=q.device, dtype=torch.bool))

            out = F.scaled_dot_product_attention(
                q_seq, k_seq, v_seq, attn_mask=mask)
            out = out.squeeze(0).transpose(0, 1)  # (seq, heads, head_dim)
            outputs.append(out)

        return torch.cat(outputs, dim=0)  # (total_tokens, heads, head_dim)

    def _paged_decode_attention(
        self,
        q_nope: torch.Tensor,
        q_rope: torch.Tensor,
        block_tables: torch.Tensor,
        context_lens: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """
        Paged attention decode using matrix absorption for MLA.

        Args:
            q_nope: (batch_size, num_heads, head_dim)
            q_rope: (batch_size, num_heads, rope_dim)
            block_tables: (batch_size, max_num_blocks)
            context_lens: (batch_size,)

        Returns:
            output: (batch_size, num_heads, head_dim)
        """
        q_nope = q_nope.view(batch_size, self.num_heads, self.head_dim)
        q_rope = q_rope.view(batch_size, self.num_heads, self.rope_dim)

        max_context_len = context_lens.max().item()
        max_num_blocks = block_tables.shape[1]

        # Gather cached KV and PE from paged cache
        # k_cache: (num_blocks, block_size, kv_lora_rank)
        # v_cache: (num_blocks, block_size, rope_dim)
        cached_kv_list = []
        cached_pe_list = []

        for b in range(batch_size):
            ctx_len = context_lens[b].item()
            num_blocks_needed = (
                ctx_len + self.block_size - 1) // self.block_size

            kv_tokens = []
            pe_tokens = []
            for block_idx in range(num_blocks_needed):
                physical_block = block_tables[b, block_idx].item()
                if physical_block == -1:
                    continue

                if block_idx == num_blocks_needed - 1:
                    # Last block may be partial
                    tokens_in_block = ctx_len - block_idx * self.block_size
                else:
                    tokens_in_block = self.block_size

                kv_tokens.append(
                    self.k_cache[physical_block, :tokens_in_block])
                pe_tokens.append(
                    self.v_cache[physical_block, :tokens_in_block])

            if kv_tokens:
                # (ctx_len, kv_lora_rank)
                cached_kv_list.append(torch.cat(kv_tokens, dim=0))
                # (ctx_len, rope_dim)
                cached_pe_list.append(torch.cat(pe_tokens, dim=0))
            else:
                cached_kv_list.append(torch.zeros(
                    0, self.kv_lora_rank, device=q_nope.device))
                cached_pe_list.append(torch.zeros(
                    0, self.rope_dim, device=q_nope.device))

        # Pad to max_context_len
        cached_kv = torch.zeros(batch_size, max_context_len,
                                self.kv_lora_rank, device=q_nope.device, dtype=q_nope.dtype)
        cached_pe = torch.zeros(batch_size, max_context_len,
                                self.rope_dim, device=q_nope.device, dtype=q_nope.dtype)

        for b in range(batch_size):
            ctx_len = cached_kv_list[b].shape[0]
            if ctx_len > 0:
                cached_kv[b, :ctx_len] = cached_kv_list[b]
                cached_pe[b, :ctx_len] = cached_pe_list[b]

        # Matrix absorption attention
        w_uk = self.kv_up_proj_fused.weight[:self.d_model, :].view(
            self.num_heads, self.head_dim, self.kv_lora_rank)
        w_uv = self.kv_up_proj_fused.weight[self.d_model:, :].view(
            self.num_heads, self.head_dim, self.kv_lora_rank)

        # q_nope: (bsz, heads, head_dim), w_uk: (heads, head_dim, kv_lora_rank)
        # (bsz, heads, kv_lora_rank)
        q_absorbed = torch.einsum('bhd, hdk -> bhk', q_nope, w_uk)

        # Attention scores
        # (bsz, heads, ctx_len)
        attn_score = torch.einsum('bhk, btk -> bht', q_absorbed, cached_kv)
        # (bsz, heads, ctx_len)
        rope_score = torch.einsum('bhd, btd -> bht', q_rope, cached_pe)
        score = (attn_score + rope_score) * self.scale

        # Apply mask for valid positions
        valid_mask = torch.arange(max_context_len, device=q_nope.device).unsqueeze(
            0) < context_lens.unsqueeze(1)
        score = score.masked_fill(~valid_mask.unsqueeze(1), float('-inf'))

        # Softmax and weighted sum
        attn_weights = F.softmax(score, dim=-1)  # (bsz, heads, ctx_len)
        # (bsz, heads, kv_lora_rank)
        attn_latent = torch.einsum('bht, btk -> bhk', attn_weights, cached_kv)

        # Output projection via matrix absorption
        # (bsz, heads, head_dim)
        attn_output = torch.einsum('bhk, hdk -> bhd', attn_latent, w_uv)

        return attn_output
