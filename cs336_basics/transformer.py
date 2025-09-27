import math
import torch
import torch.nn as nn

from .utils import *
from .attention import MultiHeadSelfAttention, RotaryPositionalEmbedding


# ----------------------------------------
#  Problem 9: Implement Transformer Block
# ----------------------------------------
class TransformerBlock(nn.Module):
    """ Transformer Block with Multi-Head Self-Attention and Feed-Forward Network """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, rope: RotaryPositionalEmbedding, drop_p: float | None = None, device=None):
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model, num_heads, rope, device=device)
        self.ffn = SwiGLU_FFN(d_model, d_ff, device=device)
        self.ln1 = RMSNorm(d_model, device=device)
        self.ln2 = RMSNorm(d_model, device=device)
        # Add dropout layers if needed...
        self.dropout1 = nn.Dropout(drop_p) if drop_p else nn.Identity()
        self.dropout2 = nn.Dropout(drop_p) if drop_p else nn.Identity()

    def forward(self, x: torch.Tensor):
        # Apply RMSNorm and Multi-Head Attention
        residual = x
        x = self.ln1(x)
        x = self.attn(x)
        x = self.dropout1(x)
        x = x + residual
        # Apply RMSNorm and Feed-Forward Network
        residual = x
        x = self.ln2(x)
        x = self.ffn(x)
        x = self.dropout2(x)
        x = x + residual
        return x


# --------------------------------------------------
#  Problem 10: Implement Transformer Language Model
# --------------------------------------------------
class TransformerLM(nn.Module):
    """Language Model based on stacked Transformer Decoder Blocks"""

    def __init__(self, vocab_size: int, context_length: int, d_model: int, num_layers: int, num_heads: int, d_ff: int, rope_theta: float, drop_p: float | None = None, device=None):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model, device=device)
        self.rope = RotaryPositionalEmbedding(rope_theta, d_model // num_heads, context_length, device=device)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, self.rope, drop_p, device=device)
            for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model, device=device)
        self.lm_head = Linear(d_model, vocab_size, device=device)

    def forward(self, x: torch.Tensor):
        # apply token embedding
        x = self.token_embeddings(x)
        # pass through the transformer blocks
        for block in self.layers:
            x = block(x)
        # modeling token probability distribution
        logits = self.lm_head(self.ln_final(x))
        return logits
