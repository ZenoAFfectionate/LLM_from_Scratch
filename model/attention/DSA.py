import torch
import torch.nn as nn
from torch.nn import functional as F

from .utils import *
from fast_hadamard_transform import hadamard_transform 


block_size = 32  # size of FP8 quantization blocks


def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    '''  '''
    assert x.dtype == torch.bfloat16
    hidden_size = x.size(-1)
    return hadamard_transform(x, scale=hidden_size ** -0.5)


class Indexer(nn.Module):
    """ An light-weight Indexer for efficient relevant token selection """

    def __init__(self, 
                 d_model,
                 head_num,
                 head_dim,
                 rope,
                 d_rope,
                 q_lora_rank,
                 index_topk=64,
                 scale_fmt='ue8m0',
                 max_batch_size=8,
                 max_seq_len=2048,
                 cache_enable=False
                ):
        self.d_model = d_model    # model dimension
        self.head_num = head_num  # attention head of the indexer
        self.head_dim = head_dim  # head dimension of the indexer
        self.d_rope = d_rope      # dimension for RoPE part
        self.q_lora_rank = q_lora_rank  # 
        self.index_topk = index_topk    # 
        self.scale = head_dim ** -0.5   #
        self.scale_fmt = scale_fmt      #
        self.cache_enable = cache_enable
        self.rope = rope

        self.q_proj = nn.Linear(q_lora_rank, head_num*head_dim)  # 
        self.k_proj = nn.Linear(d_model, head_dim)               # 
        self.k_norm = nn.RMSNorm(head_dim)                       # 
        self.w_proj = nn.Linear(d_model, self.head_num, dtype=torch.float32)  # 

        self.register_buffer(
            'k_cache', 
            torch.zeros(max_batch_size, max_seq_len, head_dim, dtype=torch.float8_e4m3fn), 
            persistent=False
        )
        self.register_buffer(
            'k_scale', 
            torch.zeros(max_batch_size, max_seq_len, head_dim // block_size, dtype=torch.float32), 
            persistent=False
        )

    def forward(self, x: torch.Tensor, q: torch.Tensor, start_pos: int, mask=None):
        batch_size, seq_len, _ = x.shape
        end_pos = start_pos + seq_len
        # generate token positions based on start_pos
        token_positions = torch.arange(start_pos, end_pos, device=x.device)

        q = self.q_proj(q)
        q = q.view(batch_size, seq_len, self.head_num, self.head_dim)
        k = self.k_proj(x)
        k = self.k_norm(k)
        
        # apply RoPE for the rope part of both q and k
        q_rope, q_nope = torch.split(q, [self.d_rope, self.head_dim - self.d_rope], dim=-1)
        k_rope, k_nope = torch.split(k, [self.d_rope, self.head_dim - self.d_rope], dim=-1)

        q_rope = q_rope.transpose(1, 2)  # (batch, head_num, seq_len, d_rope)
        q_rope = self.rope(q_rope, token_positions)  # apply RoPE
        q_rope = q_rope.transpose(1, 2)  # (batch, seq_len, head_num, d_rope)
        k_rope = k_rope.unsqueeze(1)     # ()
        k_rope = self.rope(k_rope, token_positions)  # apply RoPE
        k_rope = k_rope.squeeze(1)       # ()

        # [WARNING] expand k_rope to match head_num dimension
        k_rope = k_rope.unsqueeze(2).expand(batch_size, seq_len, self.head_num, self.d_rope)

        q = torch.cat([q_nope, q_rope], dim=-1)
        k = torch.cat([k_nope, k_rope], dim=-1)
        q = rotate_activation(q)  # apply rotate activation
        k = rotate_activation(k)  # apply rotate activation

        q_fp8, q_scale = act_quant(q, block_size, self.scale_fmt)  # FP8 quant
        k_fp8, k_scale = act_quant(k, block_size, self.scale_fmt)  # FP8 quant

        if self.cache_enable:  # update kv cache if enable
            self.k_cache[:batch_size, start_pos:end_pos] = k_fp8
            self.k_scale[:batch_size, start_pos:end_pos] = k_scale
        
        # compute aggregation weights
        weights = self.w_proj(x.float()) * (self.head_num ** -0.5) # (batch, seq_len, head_num)
        weights = weights.unsqueeze(-1) * q_scale * self.scale     # (batch, seq_len, head_num, 1)
        # compute index score for each token
        k_cache = self.k_cache[:batch_size, :end_pos].contiguous()
        k_scale = self.k_scale[:batch_size, :end_pos].contiguous()
        index_score = fp8_index(q_fp8.contiguous(), weights, k_cache, k_scale)
        if mask is not None: index_score += mask  # apply mask
        # select top-k indices
        return index_score.topk(min(self.index_topk, end_pos), dim=-1)[1]


class DeepseekSparseAttention(nn.Module):
    """
    DeepSeek Sparse Attention (DSA) with Multi-Query Attention (MQA) as core module.

    Key innovations:
    1. Low-rank compression of KV cache to reduce memory usage
    2. Decoupled RoPE: Split into RoPE part and non-RoPE part, concatenate after RoPE
    3. Gating mechanism for multi-head attention output
    4. Grouped Query Attention: Multiple query heads share fewer KV heads

    Args:
        d_model: Model dimension
        num_query_heads: Number of query attention heads
        num_kv_heads: Number of key/value heads (must divide num_query_heads)
        rope: Rotary Position Embedding module
        d_rope: Dimension for RoPE (default: 4)
        kv_lora_rank: Rank for KV compression (default: d_model // 2)
        q_lora_rank:  Rank for non-RoPE query compression (default: kv_lora_rank)
        use_gate: Whether to use head-wise gating mechanism (default: True)
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
        index_topk: int = 64,
        scale_fmt: str = 'ue8m0',
        max_batch_size: int = 8,
        max_seq_len: int = 2048,
        cache_enabled: bool = False,
        device=None,
        dtype=None
    ):
        super().__init__()
        assert d_model % head_num == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.head_num = head_num
        self.head_dim = d_model // head_num
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

        self.output_proj = nn.Linear(head_num*self.head_dim, d_model, device=device, dtype=dtype)

        self.indexer = Indexer(
            d_model=d_model,
            rope=self.rope,
            rope_dim=self.rope_dim,
            head_num=self.head_num,
            head_dim=self.head_dim,
            q_lora_rank=self.q_lora_rank,
            index_topk=index_topk,
            scale_fmt=scale_fmt,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            cache_enable=cache_enabled
        )

        # KV cache buffers (only used during inference when cache_enabled=True)
        # In DSA, we cache the COMPRESSED representations, not the full K and V
        cache_dtype = dtype if dtype is not None else torch.float32
        if self.cache_enabled:
            self.register_buffer(
                "kv_cache",
                torch.zeros(max_batch_size, max_seq_len, self.kv_lora_rank, device=device, dtype=cache_dtype),
                persistent=False
            )
            self.register_buffer(
                "pe_cache",
                torch.zeros(max_batch_size, max_seq_len, self.d_rope, device=device, dtype=cache_dtype),
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
        q_compressed = self.q_norm(self.q_down_proj(x))  # (batch, seq_len, q_lora_rank)
        q_nope = self.q_nope_up_proj(q_compressed).view(batch, seq_len, self.head_num, -1)  # head_dim
        q_rope = self.q_rope_up_proj(q_compressed).view(batch, seq_len, self.head_num, -1)  # rope_dim

        q_rope = q_rope.transpose(1, 2)  # (batch, head_num, seq_len, rope_dim)
        q_rope = self.rope(q_rope, token_positions)  # apply RoPE
        q_rope = q_rope.transpose(1, 2)  # (batch, seq_len, head_num, rope_dim)

        # concatenate non-RoPE and RoPE parts
        q = torch.cat([q_nope, q_rope], dim=-1)  # (batch, seq_len, head_num, head_dim+rope_dim)
        q = q.transpose(1, 2)                    # (batch, head_num, seq_len, head_dim+rope_dim)

        # ===============
        # Process K and V
        # ===============
        kv_compressed = self.kv_norm(self.kv_down_proj(x))  # (batch, seq_len, kv_lora_rank)
        k_rope = self.k_rope_proj(x)                        # (batch, seq_len, rope_dim)

        k_rope = k_rope.unsqueeze(1)  # (batch, 1, seq_len, rope_dim)
        k_rope = self.rope(k_rope, token_positions)  # apply RoPE
        k_rope = k_rope.squeeze(1)    # (batch, seq_len, rope_dim)

        # handle KV cache during inference - cache the single k_rope (not replicated)
        if self.cache_enabled:
            self.kv_cache[:batch, start_pos:start_pos+seq_len, :] = kv_compressed
            self.pe_cache[:batch, start_pos:start_pos+seq_len, :] = k_rope
            total_seq_len = start_pos + seq_len  # total length including cache
            kv_compressed = self.kv_cache[:batch, :total_seq_len, :]
            k_rope = self.pe_cache[:batch, :total_seq_len, :]
        else:
            total_seq_len = seq_len

        k_nope = self.k_up_proj(kv_compressed)  # (batch, total_seq_len, d_model)
        v = self.v_up_proj(kv_compressed)       # (batch, total_seq_len, d_model)

        k_nope = k_nope.view(batch, total_seq_len, self.head_num, self.head_dim)
        v = v.view(batch, total_seq_len, self.head_num, self.head_dim)

        # replicate k_rope to match each head and concatenate with k_nope
        k_rope = k_rope.unsqueeze(2).expand(batch, total_seq_len, self.head_num, self.d_rope)
        k = torch.cat([k_nope, k_rope], dim=-1)  # (batch, total_seq_len, head_num, head_dim+d_rope)
        k = k.transpose(1, 2)                    # (batch, head_num, total_seq_len, head_dim+d_rope)
        v = v.transpose(1, 2)                    # (batch, head_num, total_seq_len, head_dim)

        # ==========================================
        # Perform Indexing to select relevant tokens
        # ==========================================
        topk_indices = self.indexer(x, q_compressed, start_pos, mask)  # (batch, seq_len, index_topk)
        # create boolean index_mask and scatter True to top-k positions
        index_mask = torch.zeros((batch, seq_len, total_seq_len), device=x.device, dtype=torch.bool)
        index_mask = index_mask.scatter_(-1, topk_indices, True)  # (batch, seq_len, total_seq_len)

        if mask is not None:
            # epxand causal mask to match total_seq_len and copy original causal mask to it
            expanded_mask = torch.zeros((seq_len, total_seq_len), device=x.device, dtype=torch.bool)
            expanded_mask[:, :seq_len] = mask
            # allow full attention for cached positions beyond current sequence
            if total_seq_len > seq_len: expanded_mask[:, seq_len:] = True
            index_mask = index_mask & expanded_mask  # combine via logical AND

        index_mask = index_mask.unsqueeze(1)  # add head dimension for broadcasting

        # ============================
        # Scaled dot-product attention
        # ============================
        attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=index_mask)
        attn_output = attn_output.transpose(1, 2)  # (batch, seq_len, head_num, head_dim)
        attn_output = attn_output.contiguous().view(batch, seq_len, self.head_num*self.head_dim)

        return self.output_proj(attn_output)  # (batch, seq_len, d_model)
