import torch
import torch.nn as nn
from .utils import Linear, softmax


# -------------------------------------------------------
#  Problem 6: Implement Rotary Position Embedding Module
# -------------------------------------------------------
class RotaryPositionalEmbedding(nn.Module):
    """ PyTorch implementation of Rotary Position Embedding module """
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        # generate inverse frequency to handle the rotation
        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device).float() / d_k))
        position = torch.arange(max_seq_len, device=device).float().unsqueeze(1)
        angles = position @ inv_freq.unsqueeze(0)     # (max_seq_len, d_k / 2)
        angles = angles.repeat_interleave(2, dim=-1)  # (max_seq_len, d_k)
        # register buffers for cached cosine and sine values
        self.register_buffer('cos_cached', torch.cos(angles), persistent=False)  # (max_seq_len, d_k)
        self.register_buffer('sin_cached', torch.sin(angles), persistent=False)  # (max_seq_len, d_k)
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        ''' use token_positions to slice your cos and sin tensors along seq_len dimension '''
        # get cos and sin value from cache and broadcast to match x shape:
        cos = self.cos_cached[token_positions]  # (seq_len, d_k)
        sin = self.sin_cached[token_positions]  # (seq_len, d_k)
        cos = cos.view(*([1] * (x.ndim - 2)), *cos.shape)  # (batch_size, ..., seq_len, d_k)
        sin = sin.view(*([1] * (x.ndim - 2)), *sin.shape)  # (batch_size, ..., seq_len, d_k)
        # group the last dimension of x into pairs of two
        x_reshaped = x.float().reshape(*x.shape[:-1], -1, 2)
        # split into two tensors abd perform the rotation
        x1, x2 = x_reshaped[..., 0], x_reshaped[..., 1]
        x_rotated = torch.stack((-x2, x1), dim=-1).reshape(*x.shape)
        return x * cos + x_rotated * sin  # perform the rotation


# ---------------------------------------------------
#  Problem 7: Implement Scaled Dot-Product Attention
# ---------------------------------------------------
def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask=None):
    """ implement scaled dot-product attention mechanism """
    d_k, d_v = q.size(-1), v.size(-1)
    assert k.size(-2) == v.size(-2), "key and value must have the same sequence length"
    # compute attention weights (consider mask if given)
    attn_scores = (q @ k.transpose(-2, -1)) / (d_k ** 0.5)  # (batch_size, ..., d_k, d_k)
    if mask is not None: attn_scores = attn_scores.masked_fill(mask==0, float('-inf'))
    attn_weights = softmax(attn_scores, dim=-1)             # (batch_size, ..., d_k, d_k)
    return attn_weights @ v  # (batch_size, ..., seq_len, d_v)


# -------------------------------------------------------
#  Problem 8: Implement Multi-Head Self-Attention Module
# -------------------------------------------------------
class MultiHeadSelfAttention(nn.Module):
    """ Causal Multi-Head Self-Attention module with RoPE """

    def __init__(self, d_model: int, num_heads: int, rope: RotaryPositionalEmbedding, device=None): 
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        # initialize the projection layers
        self.q_proj = Linear(d_model, d_model, device=device)
        self.k_proj = Linear(d_model, d_model, device=device) 
        self.v_proj = Linear(d_model, d_model, device=device)
        self.output_proj = Linear(d_model, d_model, device=device)
        # initialize Rotary Positional Embedding module
        self.rope = rope

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor = None) -> torch.Tensor:
        batch_size, *prefix, seq_len, d_model = x.shape

        q = self.q_proj(x)  # (optimization) add RMSNorm for q
        k = self.k_proj(x)  # (optimization) add RMSNorm for k
        v = self.v_proj(x)

        # reshape q, k, v for multi-head attention
        q = q.view(batch_size, *prefix, seq_len, self.num_heads, self.head_dim).transpose(-2, -3)
        k = k.view(batch_size, *prefix, seq_len, self.num_heads, self.head_dim).transpose(-2, -3)
        v = v.view(batch_size, *prefix, seq_len, self.num_heads, self.head_dim).transpose(-2, -3)

        # apply Rotary Position Embedding on q and k
        if self.rope is not None:
            if token_positions is None: token_positions = torch.arange(seq_len, device=x.device)
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        # apply scaled dot-product attention with causal mask
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).bool()
        attn_output = scaled_dot_product_attention(q, k, v, causal_mask)

        # reshape attn_output and apply final output projection
        attn_output = attn_output.transpose(-2, -3).contiguous().view(batch_size, *prefix, seq_len, d_model)
        return self.output_proj(attn_output)
