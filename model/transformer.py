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
from model.attention.CCA import CompressedConvAttention
from model.attention.GDA import GatedDeltaAttention


# ----------------------------------------
#  Problem 9: Implement Transformer Block
# ----------------------------------------
class Block(nn.Module):
    """
    Transformer Block with learned residual scaling (ZAYA1-8B Eq. 6):

        xl+1 = Res-scale_res(xl) + Res-scale_out(Layer(RMSnorm(xl)))

    where Res-scale(x) = α·x + β (α init=1, β init=0 → identity).
    """

    def __init__(
        self,
        config: Config,
        rope: RotaryPositionalEmbedding,
        use_moe: bool,
        use_gda: bool = False,
        device=None,
        dtype=None
    ):
        super().__init__()
        self.use_moe = use_moe
        # When use_gda=True, the attention sub-block is GatedDeltaAttention
        # regardless of `config.attention_type`. This is used by TransformerLM
        # to build a 1:1 hybrid (primary attention ↔ GDA) without touching the
        # surrounding RMSNorm / ResScale / dropout structure.
        self.attention_type = "GDA" if use_gda else config.attention_type

        # ── Attention ──
        if use_gda:
            # GDA is intrinsically causal & has its own positional handling
            # (short conv + delta-rule recurrence) — no RoPE is needed.
            self.att = GatedDeltaAttention(
                d_model=config.d_model,
                num_v_heads=config.gda_num_v_heads,
                num_k_heads=config.gda_num_k_heads,
                head_k_dim=config.gda_head_k_dim,
                head_v_dim=config.gda_head_v_dim,
                conv_kernel_size=config.gda_conv_kernel_size,
                device=device, dtype=dtype,
            )
        elif config.attention_type == "MHA":
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
        elif config.attention_type == "CCA":
            c_dim = (config.cca_compressed_dim
                     if config.cca_compressed_dim is not None
                     else config.d_model // config.num_heads // 2)
            kv_heads = (config.cca_num_kv_heads
                        if config.cca_num_kv_heads is not None
                        else config.num_heads // 4)
            self.att = CompressedConvAttention(
                d_model=config.d_model,
                num_query_heads=config.num_heads,
                num_kv_heads=kv_heads,
                c_dim=c_dim,
                rope=rope,
                conv_kernel_size=config.cca_conv_kernel_size,
                device=device,
                dtype=dtype
            )

        # ── FFN ──
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

        # ── Normalization (without residual add — residual scaling handles that) ──
        self.att_norm = RMSNorm(config.d_model, device=device)
        self.ffn_norm = RMSNorm(config.d_model, device=device)

        # ── Learned residual scaling (paper Eq. 6) ──
        # res-scale for the residual stream (applied to xl before each sub-block)
        self.res_scale_att = ResScale(config.d_model, device=device)
        self.res_scale_ffn = ResScale(config.d_model, device=device)
        # out-scale for each sub-block's output
        self.out_scale_att = ResScale(config.d_model, device=device)
        self.out_scale_ffn = ResScale(config.d_model, device=device)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, xl: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass with learned residual scaling.

            xl_mid = Res-scale_res_att(xl) + Res-scale_out_att(Attn(RMSnorm(xl)))
            xl_out  = Res-scale_res_ffn(xl_mid) + Res-scale_out_ffn(FFN(RMSnorm(xl_mid)))

        Args:
            xl: residual stream input  (batch, seq, d_model)
            mask: optional attention mask (unused; is_causal=True handles it)
        Returns:
            xl: updated residual stream
        """
        # ── Attention sub-block ──
        normed = self.att_norm(xl)                        # RMSNorm(xl)
        att_out = self.att(normed, mask)                  # Attn(RMSnorm(xl))
        att_out = self.dropout(att_out)
        xl = self.res_scale_att(xl) + self.out_scale_att(att_out)
        # xl_mid = Res-scale_res(xl) + Res-scale_out(Attn(RMSnorm(xl)))

        # ── FFN sub-block ──
        normed = self.ffn_norm(xl)                        # RMSNorm(xl_mid)
        ffn_out = self.ffn(normed)                         # FFN(RMSnorm(xl_mid))
        ffn_out = self.dropout(ffn_out)
        xl = self.res_scale_ffn(xl) + self.out_scale_ffn(ffn_out)
        # xl_out = Res-scale_res(xl_mid) + Res-scale_out(FFN(RMSnorm(xl_mid)))

        return xl


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

        self.token_embeddings = Embedding(
            config.vocab_size, config.d_model, device=device, dtype=dtype)

        # RoPE dimension varies by attention type
        rope_dim = config.d_model // config.num_heads
        if config.attention_type == "MLA":
            rope_dim = config.rope_dim
        elif config.attention_type == "CCA":
            c_dim = (config.cca_compressed_dim
                     if config.cca_compressed_dim is not None
                     else config.d_model // config.num_heads // 2)
            rope_dim = c_dim // 2  # 50% RoPE
        self.rope = RotaryPositionalEmbedding(
            config.rope_theta, rope_dim,
            config.context_length, device=device
        )

        # Build Transformer layers.
        # Hybrid GDA: when enabled and the primary attention is one of
        # {MHA, GQA, MLA}, alternate 1:1 — even layers use the primary
        # attention, odd layers use GDA. CCA already has local sequence-mixing
        # convolutions, so it is *never* paired with GDA: when attention_type
        # is "CCA", the hybrid flag is silently ignored and every layer stays
        # CCA. (`gda_layer_indices` is also stored for downstream inspection.)
        is_hybrid = config.use_gda_hybrid and config.attention_type != "CCA"
        gda_layer_indices = (
            [i for i in range(config.num_layers) if i % 2 == 1]
            if is_hybrid else []
        )
        self.gda_layer_indices = gda_layer_indices

        self.layers = nn.ModuleList([
            Block(
                config=config,
                rope=self.rope,
                use_moe=(config.use_moe and (
                    config.moe_layers is None or i in config.moe_layers)),
                use_gda=(i in gda_layer_indices),
                device=device,
                dtype=dtype
            )
            for i in range(config.num_layers)
        ])
        self.final_norm = RMSNorm(config.d_model, device=device)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor):
        """
        Forward pass with residual stream xl propagating through blocks.
        All attention types use is_causal=True — no explicit mask needed.
        """
        xl = self.token_embeddings(x)
        mask = None
        for block in self.layers:
            xl = block(xl, mask)
        xl = self.final_norm(xl)
        return self.lm_head(xl)

    def update_moe_biases(self):
        """Update expert biases for aux-loss-free load balance"""
        if not self.use_moe:
            return
        for layer in self.layers:
            if hasattr(layer, 'use_moe') and layer.use_moe:
                if hasattr(layer.ffn, 'update_expert_bias'):
                    layer.ffn.update_expert_bias()


# Legacy alias so external code / adapters can `from model.transformer import TransformerBlock`.
TransformerBlock = Block
