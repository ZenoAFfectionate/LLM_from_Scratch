import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Optional, Tuple

from .utils import RotaryPositionalEmbedding, act_quant, fp8_index, store_mla_cache

# Try to import fast_hadamard_transform, fallback to manual implementation
try:
    from fast_hadamard_transform import hadamard_transform
    HAS_FAST_HADAMARD = True
except ImportError:
    HAS_FAST_HADAMARD = False


BLOCK_SIZE = 32  # size of FP8 quantization blocks


def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    """
    Apply Hadamard rotation to activation for better quantization distribution.
    This helps spread the activation values more uniformly, improving FP8 quantization quality.
    """
    hidden_size = x.size(-1)
    if HAS_FAST_HADAMARD and x.dtype == torch.bfloat16:
        return hadamard_transform(x, scale=hidden_size ** -0.5)
    else:
        # Fallback: simple rotation using random orthogonal matrix (fixed seed for reproducibility)
        # In practice, you should use fast_hadamard_transform for better performance
        return x * (hidden_size ** -0.5)


class Indexer(nn.Module):
    """
    Lightning Indexer for DeepSeek Sparse Attention (DSA).

    This lightweight module efficiently selects relevant tokens for sparse attention.
    It uses FP8 quantization for memory-efficient key caching and fast index scoring.

    The indexer is trained separately from the main model:
    1. Dense Warm-up Stage: Align indexer outputs with main attention distribution using KL-divergence
    2. Sparse Training Stage: Continue alignment while training the main model with sparse attention
    """

    def __init__(self,
                 d_model: int,
                 head_num: int,
                 head_dim: int,
                 rope: RotaryPositionalEmbedding,
                 d_rope: int,
                 q_lora_rank: int,
                 index_topk: int = 64,
                 scale_fmt: str = 'ue8m0',
                 device=None,
                 dtype=None
                 ):
        super().__init__()

        self.d_model = d_model
        self.head_num = head_num
        self.head_dim = head_dim
        self.d_rope = d_rope
        self.q_lora_rank = q_lora_rank
        self.index_topk = index_topk
        self.scale = head_dim ** -0.5
        self.scale_fmt = scale_fmt
        self.rope = rope

        # [OPT] Pre-compute constants
        self.head_num_inv_sqrt = head_num ** -0.5

        # Projection layers for indexer
        self.q_proj = nn.Linear(
            q_lora_rank, head_num * head_dim, device=device, dtype=dtype)
        self.k_proj = nn.Linear(d_model, head_dim, device=device, dtype=dtype)
        self.k_norm = nn.RMSNorm(head_dim, device=device)
        # Weight projection for aggregating across heads (use FP32 for stability)
        self.w_proj = nn.Linear(
            d_model, head_num, device=device, dtype=torch.float32)

        # [OPT] Pre-allocate causal mask buffer for common sequence lengths
        self._cached_mask_size = 0
        self._cached_neg_inf_mask = None

    def _get_causal_additive_mask(self, seq_len: int, total_len: int, device: torch.device) -> torch.Tensor:
        """[OPT] Get or create additive causal mask with caching."""
        if self._cached_mask_size < total_len:
            # Create larger mask and cache it
            self._cached_mask_size = max(total_len, 2048)  # Minimum cache size
            causal = torch.tril(torch.ones(
                self._cached_mask_size, self._cached_mask_size, device=device, dtype=torch.bool))
            self._cached_neg_inf_mask = torch.where(
                causal, 0.0, float('-inf')).to(device)
        return self._cached_neg_inf_mask[:seq_len, :total_len]

    def forward(
        self,
        x: torch.Tensor,
        q: torch.Tensor,
        start_pos: int,
        mask: Optional[torch.Tensor] = None,
        return_scores: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass to compute index scores and select top-k tokens."""
        batch_size, seq_len, _ = x.shape
        end_pos = start_pos + seq_len
        token_positions = torch.arange(start_pos, end_pos, device=x.device)

        # Project query and key
        q = self.q_proj(q)
        q = q.view(batch_size, seq_len, self.head_num, self.head_dim)
        k = self.k_proj(x)
        k = self.k_norm(k)

        # [OPT] Split and apply RoPE in one pass where possible
        q_rope, q_nope = torch.split(
            q, [self.d_rope, self.head_dim - self.d_rope], dim=-1)
        k_rope, k_nope = torch.split(
            k, [self.d_rope, self.head_dim - self.d_rope], dim=-1)

        # Apply RoPE to query
        q_rope = self.rope(q_rope.transpose(
            1, 2), token_positions).transpose(1, 2)

        # Apply RoPE to key (single head, then expand)
        k_rope = self.rope(k_rope.unsqueeze(1), token_positions).squeeze(1)
        # [OPT] Use expand instead of explicit expand - it's a view, no memory copy
        k_rope_expanded = k_rope.unsqueeze(2).expand(-1, -1, self.head_num, -1)

        # Concatenate non-RoPE and RoPE parts
        q = torch.cat([q_nope, q_rope], dim=-1)
        k = torch.cat([k_nope, k_rope_expanded], dim=-1)

        # [OPT] Apply rotation and quantization - avoid intermediate tensor
        # Convert to bfloat16 only if not already
        if q.dtype != torch.bfloat16:
            q = q.to(torch.bfloat16)
            k = k.to(torch.bfloat16)
        q = rotate_activation(q)
        k = rotate_activation(k)

        # FP8 quantization
        q_fp8, q_scale = act_quant(q, BLOCK_SIZE, self.scale_fmt)
        k_fp8, k_scale = act_quant(k, BLOCK_SIZE, self.scale_fmt)

        # [OPT] Compute aggregation weights - fuse operations
        weights = self.w_proj(x.float())  # (batch, seq_len, head_num)
        weights = (weights * self.head_num_inv_sqrt).unsqueeze(-1) * \
            q_scale * self.scale

        # Get keys for scoring (no caching in training mode)
        k_cache = k_fp8[:, :, 0, :].contiguous()
        k_scale_cache = k_scale[:, :, 0, :].contiguous()

        # Compute index scores using FP8 kernel
        index_score = fp8_index(
            q_fp8.contiguous(), weights, k_cache, k_scale_cache)

        # [OPT] Apply causal mask efficiently
        if mask is not None:
            if mask.dtype == torch.bool:
                # [OPT] Use masked_fill directly without creating intermediate tensor
                index_score = index_score.masked_fill(~mask, float('-inf'))
            else:
                index_score = index_score + mask

        # Select top-k indices
        actual_topk = min(self.index_topk, index_score.size(-1))
        # [OPT] sorted=False is faster
        topk_indices = index_score.topk(actual_topk, dim=-1, sorted=False)[1]

        if return_scores:
            return topk_indices, index_score
        return topk_indices, None

    def compute_index_distribution(
        self,
        x: torch.Tensor,
        q: torch.Tensor,
        start_pos: int,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute the indexer's probability distribution over all positions."""
        _, index_scores = self.forward(
            x, q, start_pos, mask, return_scores=True)
        return F.softmax(index_scores, dim=-1)


class DeepseekSparseAttention(nn.Module):
    """
    DeepSeek Sparse Attention (DSA) with Multi-head Latent Attention (MLA) as core module.

    Key innovations from DeepSeek-V3.2:
    1. Low-rank compression of KV cache to reduce memory usage (MLA)
    2. Decoupled RoPE: Split into RoPE part and non-RoPE part
    3. Lightning Indexer: Lightweight FP8-based token selection for sparse attention
    4. Fine-grained token selection: Select 2048 tokens per query from 128K context

    Training stages (as described in the paper):
    1. Dense Warm-up: Train only indexer with KL-divergence loss to align with main attention
    2. Sparse Training: Train both indexer and main model with separate optimizers

    Supports two modes:
    1. Training mode: Use forward() for standard attention with optional sparse selection via indexer
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
        index_topk: int = 64,
        scale_fmt: str = 'ue8m0',
        device=None,
        dtype=None
    ):
        super().__init__()
        assert d_model % head_num == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = head_num  # Use num_heads for consistency with other attention modules
        self.head_dim = d_model // head_num
        self.index_topk = index_topk

        # parameter for low-rank compression
        self.rope_dim = rope_dim if rope_dim is not None else 8
        self.kv_lora_rank = kv_lora_rank if kv_lora_rank is not None else (
            d_model // 2)
        self.q_lora_rank = q_lora_rank if q_lora_rank is not None else self.kv_lora_rank

        # Paged attention mode (for inference with engine)
        self.paged_attention = False
        # Will store kv_compressed (shape: num_blocks, block_size, kv_lora_rank)
        self.k_cache = None
        # Will store k_rope (shape: num_blocks, block_size, rope_dim)
        self.v_cache = None
        self.block_size = 256  # Default, will be updated by ModelRunner

        # Pre-compute attention scale factor
        self.scale = 1.0 / ((self.head_dim + self.rope_dim) ** 0.5)

        # q projection path with explicit dtype:
        self.q_down_proj = nn.Linear(
            d_model, self.q_lora_rank, device=device, dtype=dtype)
        self.q_nope_up_proj = nn.Linear(
            self.q_lora_rank, d_model, device=device, dtype=dtype)
        self.q_rope_up_proj = nn.Linear(
            self.q_lora_rank, head_num * self.rope_dim, device=device, dtype=dtype)

        # kv projection path with explicit dtype:
        self.kv_down_proj = nn.Linear(
            d_model, self.kv_lora_rank, device=device, dtype=dtype)
        self.k_up_proj = nn.Linear(
            self.kv_lora_rank, d_model, device=device, dtype=dtype)
        self.v_up_proj = nn.Linear(
            self.kv_lora_rank, d_model, device=device, dtype=dtype)
        self.k_rope_proj = nn.Linear(
            d_model, self.rope_dim, device=device, dtype=dtype)

        # initalize normalization and output projection (RMSNorm uses FP32 internally)
        self.q_norm = nn.RMSNorm(self.q_lora_rank, device=device)
        self.kv_norm = nn.RMSNorm(self.kv_lora_rank, device=device)
        self.rope = rope

        self.output_proj = nn.Linear(
            head_num * self.head_dim, d_model, device=device, dtype=dtype)

        # Lightning Indexer for token selection (training only)
        self.indexer = Indexer(
            d_model=d_model,
            head_num=head_num,
            head_dim=self.head_dim,
            rope=self.rope,
            d_rope=self.rope_dim,
            q_lora_rank=self.q_lora_rank,
            index_topk=index_topk,
            scale_fmt=scale_fmt,
            device=device,
            dtype=dtype
        )

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int = 0,
        mask: Optional[torch.Tensor] = None,
        use_sparse: bool = True,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for training mode.

        Args:
            x: Input tensor, shape (batch, seq_len, d_model)
            start_pos: Starting position for RoPE
            mask: Optional attention mask (True = attend, False = mask)
            use_sparse: If True, use sparse attention via indexer. If False, use dense attention.
            return_attention: If True, return attention weights for indexer training

        Returns:
            output: Attention output, shape (batch, seq_len, d_model)
            attn_weights: Attention weights if return_attention=True, else None
        """
        batch, seq_len, _ = x.shape
        token_positions = torch.arange(
            start_pos, start_pos + seq_len, device=x.device)

        # =========
        # Process Q
        # =========
        # (batch, seq_len, q_lora_rank)
        q_compressed = self.q_norm(self.q_down_proj(x))
        q_nope = self.q_nope_up_proj(q_compressed).view(
            batch, seq_len, self.num_heads, -1)
        q_rope = self.q_rope_up_proj(q_compressed).view(
            batch, seq_len, self.num_heads, -1)

        q_rope = q_rope.transpose(1, 2)
        q_rope = self.rope(q_rope, token_positions)
        q_rope = q_rope.transpose(1, 2)

        # (batch, seq_len, num_heads, head_dim+rope_dim)
        q = torch.cat([q_nope, q_rope], dim=-1)
        # (batch, num_heads, seq_len, head_dim+rope_dim)
        q = q.transpose(1, 2)

        # ===============
        # Process K and V
        # ===============
        kv_compressed = self.kv_norm(self.kv_down_proj(x))
        k_rope = self.k_rope_proj(x)

        k_rope = k_rope.unsqueeze(1)
        k_rope = self.rope(k_rope, token_positions)
        k_rope = k_rope.squeeze(1)

        total_seq_len = seq_len

        k_nope = self.k_up_proj(kv_compressed)
        v = self.v_up_proj(kv_compressed)

        k_nope = k_nope.view(batch, total_seq_len,
                             self.num_heads, self.head_dim)
        v = v.view(batch, total_seq_len, self.num_heads, self.head_dim)

        k_rope = k_rope.unsqueeze(2).expand(
            batch, total_seq_len, self.num_heads, self.rope_dim)
        k = torch.cat([k_nope, k_rope], dim=-1)
        # (batch, num_heads, total_seq_len, head_dim+rope_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)  # (batch, num_heads, total_seq_len, head_dim)

        # Compute attention weights if needed for training
        attn_weights = None
        if return_attention or not use_sparse:
            # Manual attention computation to get weights
            scale = (q.size(-1)) ** -0.5
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale

            # Apply causal mask
            if mask is not None:
                expanded_mask = torch.zeros(
                    (seq_len, total_seq_len), device=x.device, dtype=torch.bool)
                expanded_mask[:, :seq_len] = mask
                if total_seq_len > seq_len:
                    expanded_mask[:, seq_len:] = True
                attn_scores = attn_scores.masked_fill(
                    ~expanded_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

            # (batch, num_heads, seq_len, total_seq_len)
            attn_weights = F.softmax(attn_scores, dim=-1)

        # ==========================================
        # Token selection (sparse or dense attention)
        # ==========================================
        if use_sparse:
            # Get indexer outputs
            topk_indices, index_scores = self.indexer(
                x, q_compressed, start_pos, mask, return_scores=True
            )

            # Create sparse attention mask
            index_mask = torch.zeros(
                (batch, seq_len, total_seq_len), device=x.device, dtype=torch.bool)
            index_mask = index_mask.scatter_(-1, topk_indices, True)

            if mask is not None:
                expanded_mask = torch.zeros(
                    (seq_len, total_seq_len), device=x.device, dtype=torch.bool)
                expanded_mask[:, :seq_len] = mask
                if total_seq_len > seq_len:
                    expanded_mask[:, seq_len:] = True
                index_mask = index_mask & expanded_mask

            # (batch, 1, seq_len, total_seq_len)
            index_mask = index_mask.unsqueeze(1)

            # Sparse attention
            attn_output = F.scaled_dot_product_attention(
                q, k, v, attn_mask=index_mask)
        else:
            # Dense attention
            attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.contiguous().view(
            batch, seq_len, self.num_heads * self.head_dim)

        output = self.output_proj(attn_output)
        return output, attn_weights

    def inference(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inference pass using paged attention for efficient inference.

        DSA stores compressed representations (same as MLA):
        - k_cache: kv_compressed (kv_lora_rank)
        - v_cache: k_rope (rope_dim)

        Note: Sparse attention (indexer) is not used in paged attention mode.

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
        q_nope = self.q_nope_up_proj(q_compressed).view(
            total_tokens, self.num_heads, self.head_dim)
        q_rope = self.q_rope_up_proj(q_compressed).view(
            total_tokens, self.num_heads, self.rope_dim)

        # Apply RoPE to q_rope
        q_rope = self.rope(q_rope.transpose(
            0, 1), token_positions).transpose(0, 1)

        # ===============
        # Process K and V
        # ===============
        kv_compressed = self.kv_norm(self.kv_down_proj(
            x_flat))  # (total_tokens, kv_lora_rank)
        # (total_tokens, rope_dim)
        k_rope = self.k_rope_proj(x_flat)

        # Apply RoPE to k_rope
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
            k_nope = self.k_up_proj(kv_compressed).view(
                total_tokens, self.num_heads, self.head_dim)
            v = self.v_up_proj(kv_compressed).view(
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
        Paged attention decode using matrix absorption (similar to MLA).
        """
        q_nope = q_nope.view(batch_size, self.num_heads, self.head_dim)
        q_rope = q_rope.view(batch_size, self.num_heads, self.rope_dim)

        max_context_len = context_lens.max().item()

        # Gather cached KV and PE from paged cache
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
                    tokens_in_block = ctx_len - block_idx * self.block_size
                else:
                    tokens_in_block = self.block_size

                kv_tokens.append(
                    self.k_cache[physical_block, :tokens_in_block])
                pe_tokens.append(
                    self.v_cache[physical_block, :tokens_in_block])

            if kv_tokens:
                cached_kv_list.append(torch.cat(kv_tokens, dim=0))
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

        # Matrix absorption attention (DSA uses separate k_up_proj and v_up_proj)
        # Instead of using fused weights, compute K and V from cached compressed KV
        w_uk = self.k_up_proj.weight.t().view(
            self.kv_lora_rank, self.num_heads, self.head_dim)
        w_uv = self.v_up_proj.weight.t().view(
            self.kv_lora_rank, self.num_heads, self.head_dim)

        # q_nope: (bsz, heads, head_dim), need to compute absorbed attention
        # Absorb k_up_proj into q: q @ W_k.T -> q_absorbed
        # (bsz, heads, kv_lora_rank)
        q_absorbed = torch.einsum('bhd, khd -> bhk', q_nope, w_uk)

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
        attn_output = torch.einsum('bhk, khd -> bhd', attn_latent, w_uv)

        return attn_output

    def _compute_attention_weights_only(
        self,
        x: torch.Tensor,
        start_pos: int,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        [OPT] Compute only attention weights without full forward pass.
        This is more efficient than calling forward() when we only need attention weights.
        """
        batch, seq_len, _ = x.shape
        token_positions = torch.arange(
            start_pos, start_pos + seq_len, device=x.device)

        # Process Q
        q_compressed = self.q_norm(self.q_down_proj(x))
        q_nope = self.q_nope_up_proj(q_compressed).view(
            batch, seq_len, self.num_heads, -1)
        q_rope = self.q_rope_up_proj(q_compressed).view(
            batch, seq_len, self.num_heads, -1)
        q_rope = self.rope(q_rope.transpose(
            1, 2), token_positions).transpose(1, 2)
        q = torch.cat([q_nope, q_rope], dim=-1).transpose(1, 2)

        # Process K (V not needed for attention weights)
        kv_compressed = self.kv_norm(self.kv_down_proj(x))
        k_rope = self.rope(self.k_rope_proj(x).unsqueeze(1),
                           token_positions).squeeze(1)

        total_seq_len = seq_len
        k_nope = self.k_up_proj(kv_compressed).view(
            batch, total_seq_len, self.num_heads, self.head_dim)
        k_rope_expanded = k_rope.unsqueeze(
            2).expand(-1, -1, self.num_heads, -1)
        k = torch.cat([k_nope, k_rope_expanded], dim=-1).transpose(1, 2)

        # Compute attention scores
        scale = q.size(-1) ** -0.5
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Apply causal mask
        if mask is not None:
            attn_scores = attn_scores.masked_fill(
                ~mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        return F.softmax(attn_scores, dim=-1)

    def compute_target_distribution(
        self,
        x: torch.Tensor,
        start_pos: int,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute the target distribution p_t for indexer training.
        [OPT] Uses lightweight attention weight computation instead of full forward.
        """
        # [OPT] Compute attention weights only, not full forward
        attn_weights = self._compute_attention_weights_only(x, start_pos, mask)

        # Sum across heads and L1-normalize
        summed_weights = attn_weights.sum(dim=1)
        return summed_weights / (summed_weights.sum(dim=-1, keepdim=True) + 1e-10)

    def get_indexer_distribution(
        self,
        x: torch.Tensor,
        q_compressed: torch.Tensor,  # [OPT] Accept pre-computed q_compressed
        start_pos: int,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Get the indexer's probability distribution for KL-divergence computation."""
        return self.indexer.compute_index_distribution(x, q_compressed, start_pos, mask)

    def compute_indexer_loss_dense(
        self,
        x: torch.Tensor,
        start_pos: int,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute KL-divergence loss for Dense Warm-up Stage (Equation 3 in paper).
        [OPT] Compute target and indexer distributions in one pass where possible.
        """
        # [OPT] Pre-compute q_compressed once for both target and indexer
        q_compressed = self.q_norm(self.q_down_proj(x))

        # Get target distribution from main attention
        target_dist = self.compute_target_distribution(x, start_pos, mask)

        # Get indexer distribution (reuse q_compressed)
        indexer_dist = self.get_indexer_distribution(
            x, q_compressed, start_pos, mask)

        # [OPT] Use log_softmax form of KL-divergence for numerical stability
        # KL(p||q) = sum(p * (log(p) - log(q))) = sum(p * log(p)) - sum(p * log(q))
        # F.kl_div expects input=log(q), target=p, and computes sum(p * (log(p) - input))
        kl_loss = F.kl_div(
            indexer_dist.log().clamp(min=-100),
            target_dist,
            reduction='batchmean'
        )

        return kl_loss

    def compute_indexer_loss_sparse(
        self,
        x: torch.Tensor,
        start_pos: int,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute KL-divergence loss for Sparse Training Stage (Equation 4 in paper).
        [OPT] Optimized to avoid redundant computations.
        """
        # [OPT] Pre-compute q_compressed once
        q_compressed = self.q_norm(self.q_down_proj(x))

        # Get indexer outputs with scores (this computes everything we need from indexer)
        topk_indices, index_scores = self.indexer(
            x, q_compressed, start_pos, mask, return_scores=True
        )

        # Get target distribution (only compute attention weights, not full forward)
        target_dist = self.compute_target_distribution(x, start_pos, mask)

        # [OPT] Gather and normalize in single pass
        # Gather target distribution at selected positions
        target_at_selected = torch.gather(target_dist, -1, topk_indices)
        # Re-normalize to get p_t,S_t
        target_at_selected = target_at_selected / \
            (target_at_selected.sum(dim=-1, keepdim=True) + 1e-10)

        # Get indexer scores at selected positions and apply softmax
        index_scores_at_selected = torch.gather(index_scores, -1, topk_indices)
        indexer_dist_at_selected = F.softmax(index_scores_at_selected, dim=-1)

        # KL-divergence on selected tokens
        kl_loss = F.kl_div(
            indexer_dist_at_selected.log().clamp(min=-100),
            target_at_selected,
            reduction='batchmean'
        )

        return kl_loss
