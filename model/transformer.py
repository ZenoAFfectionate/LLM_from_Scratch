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
#  GDA layer scheduling (Qwen3-Next style)
# ----------------------------------------
_GDA_BLOCK_PATTERNS = {
    "none": [False, False, False, False],
    "3:1":  [True,  True,  True,  False],
    "1:1":  [True,  True,  False, False],
    "1:3":  [True,  False, False, False],
}


def build_gda_layer_mask(num_layers: int, gda_ratio: str) -> list:
    """Return a list of booleans indicating which layers use GDA.

    The pattern repeats every 4 layers. For `num_layers` not divisible by 4,
    the tail is truncated from the same repeating pattern. Within each 4-layer
    block, GDA layers come first, primary attention at the end.

    Args:
        num_layers: total number of transformer layers.
        gda_ratio:  one of "none", "3:1", "1:1", "1:3".

    Returns:
        list[bool] of length `num_layers` — True = GDA, False = primary.
    """
    if gda_ratio not in _GDA_BLOCK_PATTERNS:
        raise ValueError(
            f"gda_ratio must be one of {list(_GDA_BLOCK_PATTERNS)}, got {gda_ratio!r}"
        )
    block = _GDA_BLOCK_PATTERNS[gda_ratio]
    return [block[i % 4] for i in range(num_layers)]


# ----------------------------------------
#  Problem 9: Implement Transformer Block
# ----------------------------------------
class Block(nn.Module):
    """
    Transformer Block — supports two residual schemes:

    - `residual_type="vanilla"` (standard pre-norm):
        x_mid = x + Attn(RMSNorm(x))
        x_out = x_mid + FFN(RMSNorm(x_mid))

    - `residual_type="resscale"` (ZAYA1-8B Eq. 6, learned residual scaling):
        x_mid = ResScale_res_att(x) + ResScale_out_att(Attn(RMSNorm(x)))
        x_out = ResScale_res_ffn(x_mid) + ResScale_out_ffn(FFN(RMSNorm(x_mid)))
    """

    def __init__(
        self,
        config: Config,
        rope: RotaryPositionalEmbedding,
        use_moe: bool,
        use_gda: bool = False,
        residual_type: str = "resscale",
        device=None,
        dtype=None
    ):
        super().__init__()
        self.use_moe = use_moe
        if residual_type not in ("vanilla", "resscale"):
            raise ValueError(
                f"Block residual_type must be 'vanilla' or 'resscale'; got {residual_type!r}"
            )
        self.residual_type = residual_type
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
                        else max(1, config.num_heads // 4))
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
                swiglu_limit=config.swiglu_limit,
                device=device,
                dtype=dtype
            )
        else:
            self.ffn = MLP(
                config.d_model, config.d_ff,
                swiglu_limit=config.swiglu_limit,
                device=device, dtype=dtype,
            )

        # ── Normalization ──
        self.att_norm = RMSNorm(config.d_model, device=device)
        self.ffn_norm = RMSNorm(config.d_model, device=device)

        # ── Residual scaling (only for "resscale") ──
        if residual_type == "resscale":
            self.res_scale_att = ResScale(config.d_model, device=device)
            self.res_scale_ffn = ResScale(config.d_model, device=device)
            self.out_scale_att = ResScale(config.d_model, device=device)
            self.out_scale_ffn = ResScale(config.d_model, device=device)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, xl: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """Forward pass dispatched by `residual_type`."""
        if self.residual_type == "vanilla":
            # Standard pre-norm residual.
            xl = xl + self.dropout(self.att(self.att_norm(xl), mask))
            xl = xl + self.dropout(self.ffn(self.ffn_norm(xl)))
            return xl

        # ── ZAYA1 ResScale residual ──
        normed = self.att_norm(xl)
        att_out = self.dropout(self.att(normed, mask))
        xl = self.res_scale_att(xl) + self.out_scale_att(att_out)

        normed = self.ffn_norm(xl)
        ffn_out = self.dropout(self.ffn(normed))
        xl = self.res_scale_ffn(xl) + self.out_scale_ffn(ffn_out)
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
        # Hybrid GDA: when `config.gda_ratio != "none"` and the primary attention
        # is one of {MHA, GQA, MLA}, layers are scheduled per the Qwen3-Next style
        # 4-layer pattern (see `build_gda_layer_mask`). CCA already has local
        # sequence-mixing convolutions, so it is *never* paired with GDA: when
        # `attention_type == "CCA"`, GDA is silently disabled and every layer
        # stays CCA. (`gda_layer_indices` is also stored for downstream inspection.)
        gda_ratio = (
            config.gda_ratio if config.attention_type != "CCA" else "none"
        )
        gda_mask = build_gda_layer_mask(config.num_layers, gda_ratio)
        gda_layer_indices = [i for i, use in enumerate(gda_mask) if use]
        self.gda_layer_indices = gda_layer_indices

        residual_type = (
            config.residual_type if config.residual_type != "mhc" else "resscale"
        )

        self.layers = nn.ModuleList([
            Block(
                config=config,
                rope=self.rope,
                use_moe=(config.use_moe and (
                    config.moe_layers is None or i in config.moe_layers)),
                use_gda=(i in gda_layer_indices),
                residual_type=residual_type,
                device=device,
                dtype=dtype
            )
            for i in range(config.num_layers)
        ])
        self.final_norm = RMSNorm(config.d_model, device=device)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, device=device, dtype=dtype)

        # Weight tying: share the (vocab, d_model) embedding matrix with the LM head.
        # Both `Embedding.weight` and `nn.Linear.weight` are stored as (vocab, d_model),
        # so the assignment is a direct alias. Acts as ~5M-param regularizer on this
        # 35M model and is standard for small-data pretraining (GPT-2, Llama).
        self.lm_head.weight = self.token_embeddings.weight

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
