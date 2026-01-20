import torch
import torch.nn as nn

from model.utils import *
from model.architecture.mlp import MLP
from model.attention.MHA import MultiHeadSelfAttention
from model.attention.GQA import GroupedQueryAttention
from model.attention.MLA import MultiHeadLatentAttention
# from model.attention.DSA import DeepseekSparseAttention
from model.attention.utils import RotaryPositionalEmbedding
from model.architecture.moe import MOE


# ----------------------------------------
#  Problem 9: Implement Transformer Block
# ----------------------------------------
class Block(nn.Module):
    """ Transformer Block with Grouped Query Attention and Feed-Forward Network """

    def __init__(
        self,
        d_model: int,
        head_num: int,
        d_ff: int,
        rope: RotaryPositionalEmbedding,
        drop_p: float,
        use_moe: bool,
        n_routed_experts: int,
        num_experts_per_tok: int,
        n_shared_experts: int,
        aux_seq_loss_alpha: float,
        num_kv_heads: int,
        attention_type: str = "GQA",
        rope_dim: int = None,
        kv_lora_rank: int = None,
        q_lora_rank:  int = None,
        index_topk: int = 128,
        bias_update_speed: float = 0.01,
        device=None,
        dtype=None
    ):
        super().__init__()
        self.use_moe = use_moe
        self.attention_type = attention_type

        # Select attention mechanism based on attention_type
        if attention_type == "MHA":
            self.att = MultiHeadSelfAttention(d_model, head_num, rope, use_gate=False, device=device, dtype=dtype)
        elif attention_type == "GQA":
            self.att = GroupedQueryAttention(d_model, head_num, num_kv_heads, rope, use_gate=False, device=device, dtype=dtype)
        elif attention_type == "MLA":
            self.att = MultiHeadLatentAttention(
                d_model=d_model,
                head_num=head_num,
                rope=rope,
                rope_dim=rope_dim,
                q_lora_rank=q_lora_rank,
                kv_lora_rank=kv_lora_rank,
                use_gate=False,
                device=device,
                dtype=dtype
            )
        # elif attention_type == "DSA":
        #     self.att = DeepseekSparseAttention(
        #         d_model=d_model,
        #         head_num=head_num,
        #         rope=rope,
        #         rope_dim=rope_dim,
        #         q_lora_rank=q_lora_rank,
        #         kv_lora_rank=kv_lora_rank,
        #         index_topk=index_topk,
        #         device=device,
        #         dtype=dtype
        #     )

        # Choose between standard FFN and MoE FFN
        if use_moe:
            self.ffn = MOE(
                d_model=d_model,
                d_ff=d_ff,
                n_routed_experts=n_routed_experts,
                num_experts_per_tok=num_experts_per_tok,
                n_shared_experts=n_shared_experts,
                aux_seq_loss_alpha=aux_seq_loss_alpha,
                bias_update_speed=bias_update_speed,
                device=device,
                dtype=dtype  # Pass dtype to MoE
            )
        else:
            self.ffn = MLP(d_model, d_ff, device=device, dtype=dtype)

        # RMSNorm always uses FP32 weights (handled internally)
        self.att_norm = RMSNorm(d_model, device=device)  #
        self.ffn_norm = RMSNorm(d_model, device=device)  #
        self.dropout = nn.Dropout(drop_p) if drop_p else nn.Identity()

    def forward(self, x: torch.Tensor, residual: torch.Tensor, start_pos: int = 0,
                mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass with Fused Add & Norm optimization.

        Args:
            x: input tensor
            residual: residual tensor from previous layer (or None for first layer)
            start_pos: starting position for RoPE and KV cache
            mask: optional causal attention mask (shared across all layers)

        Returns:
            - output tensor
            - updated residual tensor
        """
        # Fused Add & Norm for ATTA
        if residual is None:
            x, residual = self.att_norm(x), x
        else:
            x, residual = self.att_norm(x, residual)
        x = self.att(x, start_pos, mask)
        x = self.dropout(x)

        # Fused Add & Norm for FFN
        x, residual = self.ffn_norm(x, residual)
        x = self.ffn(x)
        x = self.dropout(x)

        return x, residual


# --------------------------------------------------
#  Problem 10: Implement Transformer Language Model
# --------------------------------------------------
class TransformerLM(nn.Module):
    """Language Model based on stacked Transformer Decoder Blocks"""

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        drop_p: float,
        use_moe: bool,
        moe_layers: list,
        n_routed_experts: int,
        num_experts_per_tok: int,
        n_shared_experts: int,
        aux_seq_loss_alpha: float,
        num_kv_heads: int,
        attention_type: str = "GQA",
        rope_dim: int = None,
        kv_lora_rank: int = None,
        q_lora_rank: int = None,
        bias_update_speed: float = 0.01,
        device=None,
        dtype=None
    ):
        super().__init__()
        self.use_moe = use_moe
        self.attention_type = attention_type
        # Model weights are always FP32 for stability. Mixed precision only affects forward pass via autocast.
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.rope = RotaryPositionalEmbedding(rope_theta, d_model // num_heads, context_length, device=device)

        # Build Transformer layers with optional MoE and configurable attention
        self.layers = nn.ModuleList([
            Block(
                d_model,
                num_heads,
                d_ff,
                self.rope,
                drop_p,
                use_moe=(use_moe and (moe_layers is None or i in moe_layers)),
                n_routed_experts=n_routed_experts,
                num_experts_per_tok=num_experts_per_tok,
                n_shared_experts=n_shared_experts,
                aux_seq_loss_alpha=aux_seq_loss_alpha,
                num_kv_heads=num_kv_heads,
                attention_type=attention_type,
                rope_dim=rope_dim,
                kv_lora_rank=kv_lora_rank,
                q_lora_rank=q_lora_rank,
                bias_update_speed=bias_update_speed,
                device=device,
                dtype=dtype
            )
            for i in range(num_layers)
        ])
        self.final_norm = RMSNorm(d_model, device=device)
        self.lm_head  = nn.Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, start_pos: int = 0):
        """
        Forward pass through the transformer language model with Fused Add & Norm.

        The causal attention mask is constructed once here and shared across all layers
        for efficiency. This avoids redundant mask construction in each attention layer.
        The residual connection is fused with normalization for memory efficiency.
        """
        seq_len = x.size(1)

        x = self.token_embeddings(x)  # apply token embedding

        mask = None
        if seq_len > 1:  # construct boolean causal mask once for training (True means allowed)
            mask = torch.tril(torch.ones((seq_len, seq_len), device=x.device, dtype=torch.bool))

        residual = None
        for block in self.layers:
            x, residual = block(x, residual, start_pos, mask)

        x, _ = self.final_norm(x, residual)
        return self.lm_head(x)  # model token probability distribution

    def update_moe_biases(self):
        """Update expert biases for all MoE layers (auxiliary-loss-free load balancing)"""
        if not self.use_moe:
            return
        for layer in self.layers:
            if hasattr(layer, 'use_moe') and layer.use_moe:
                if hasattr(layer.ffn, 'update_expert_bias'):
                    layer.ffn.update_expert_bias()
