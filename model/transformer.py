import torch
import torch.nn as nn

from model.utils import *
from model.config import Config
from model.attention.utils import RotaryPositionalEmbedding
from model.architecture.mlp import MLP
from model.architecture.moe import MOE
from model.attention.MHA import MultiHeadSelfAttention
from model.attention.GQA import GroupedQueryAttention
from model.attention.MLA import MultiHeadLatentAttention
from model.attention.DSA import DeepseekSparseAttention


# ----------------------------------------
#  Problem 9: Implement Transformer Block
# ----------------------------------------
class Block(nn.Module):
    """ Transformer Block with Grouped Query Attention and Feed-Forward Network """

    def __init__(
        self,
        config: Config,
        rope: RotaryPositionalEmbedding,
        use_moe: bool,
        device=None,
        dtype=None
    ):
        super().__init__()
        self.use_moe = use_moe
        self.attention_type = config.attention_type

        # Select attention mechanism based on attention_type
        if config.attention_type == "MHA":
            self.att = MultiHeadSelfAttention(
                config.d_model, config.num_heads, rope,
                device=device, dtype=dtype
            )
        elif config.attention_type == "GQA":
            self.att = GroupedQueryAttention(
                config.d_model, config.num_heads, config.num_kv_heads, rope,
                device=device, dtype=dtype
            )
        elif config.attention_type == "MLA":
            self.att = MultiHeadLatentAttention(
                d_model=config.d_model,
                head_num=config.num_heads,
                rope=rope,
                rope_dim=config.rope_dim,
                q_lora_rank=config.q_lora_rank,
                kv_lora_rank=config.kv_lora_rank,
                device=device,
                dtype=dtype
            )
        elif config.attention_type == "DSA":
            # DeepSeek Sparse Attention - extends MLA with lightning indexer
            index_topk = getattr(config, 'index_topk', 256)
            scale_fmt = getattr(config, 'scale_fmt', 'ue8m0')
            self.att = DeepseekSparseAttention(
                d_model=config.d_model,
                head_num=config.num_heads,
                rope=rope,
                rope_dim=config.rope_dim,
                q_lora_rank=config.q_lora_rank,
                kv_lora_rank=config.kv_lora_rank,
                index_topk=index_topk,
                scale_fmt=scale_fmt,
                device=device,
                dtype=dtype
            )

        # Choose between standard FFN and MoE FFN
        if use_moe:
            self.ffn = MOE(
                d_model=config.d_model,
                d_ff=config.d_ff,
                n_routed_experts=config.n_routed_experts,
                num_experts_per_tok=config.num_experts_per_tok,
                n_shared_experts=config.n_shared_experts,
                aux_seq_loss_alpha=config.aux_seq_loss_alpha,
                bias_update_speed=config.bias_update_speed,
                device=device,
                dtype=dtype
            )
        else:
            self.ffn = MLP(config.d_model, config.d_ff, device=device, dtype=dtype)

        # RMSNorm always uses FP32 weights (handled internally)
        self.att_norm = RMSNorm(config.d_model, device=device)
        self.ffn_norm = RMSNorm(config.d_model, device=device)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, residual: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass with Fused Add & Norm optimization.

        Args:
            x: input tensor
            residual: residual tensor from previous layer (or None for first layer)
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
        x = self.att(x, mask)
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

    def __init__(self, config: Config, device=None, dtype=None):
        super().__init__()
        self.config = config
        self.use_moe = config.use_moe
        self.attention_type = config.attention_type
        self.context_length = config.context_length

        # Model weights are always FP32 for stability
        self.token_embeddings = Embedding(config.vocab_size, config.d_model, device=device, dtype=dtype)

        rope_dim = config.d_model // config.num_heads
        if config.attention_type in ("MLA", "DSA"):
            rope_dim = config.rope_dim
        self.rope = RotaryPositionalEmbedding(
            config.rope_theta,
            rope_dim,
            config.context_length,
            device=device
        )

        # Build Transformer layers with optional MoE and configurable attention
        self.layers = nn.ModuleList([
            Block(
                config=config,
                rope=self.rope,
                use_moe=(config.use_moe and (config.moe_layers is None or i in config.moe_layers)),
                device=device,
                dtype=dtype
            )
            for i in range(config.num_layers)
        ])
        self.final_norm = RMSNorm(config.d_model, device=device)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor):
        """
        Forward pass through the transformer language model with Fused Add & Norm.

        MHA/GQA/MLA use is_causal=True so no explicit mask tensor is needed.
        DSA requires an explicit mask for its sparse indexer, so we only create
        the mask when the attention type is DSA.
        """
        seq_len = x.size(1)

        x = self.token_embeddings(x)  # apply token embedding

        mask = None  # Only DSA needs an explicit causal mask (for its sparse indexer).
        if self.attention_type == "DSA" and seq_len > 1:
            mask = torch.tril(torch.ones((seq_len, seq_len), device=x.device, dtype=torch.bool))

        residual = None
        for block in self.layers:
            x, residual = block(x, residual, mask)

        x, _ = self.final_norm(x, residual)
        return self.lm_head(x)  # model token probability distribution


    def update_moe_biases(self):
        """Update expert biases for aux-loss-free load balance"""
        if not self.use_moe: return

        for layer in self.layers:
            if hasattr(layer, 'use_moe') and layer.use_moe:
                if hasattr(layer.ffn, 'update_expert_bias'):
                    layer.ffn.update_expert_bias()
