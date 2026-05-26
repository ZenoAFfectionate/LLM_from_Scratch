"""
mHC TransformerLM — same architecture as ``model/transformer.py`` but the
ZAYA1-style ``ResScale`` residual is replaced by Manifold-Constrained
Hyper-Connections (Xie et al., 2026; see ``model/architecture/mHC.py``):
``HCBlock`` carries H parallel residual streams ``(B, S, H, D)`` and uses one
``HyperConnection`` per sub-block (attention, FFN); ``mHCTransformerLM`` adds a
``StreamEmbed`` after the token embedding to expand the single stream into H
and a ``HyperHead`` before ``final_norm`` to collapse them back. mHC hyperparams
(``hc_mult``, ``hc_sinkhorn_iters``, ``hc_eps``) come from ``model.config.Config``.
"""

import torch
import torch.nn as nn

from model.utils import Embedding, RMSNorm
from model.config import Config
from model.transformer import build_gda_layer_mask
from model.attention.utils import RotaryPositionalEmbedding
from model.architecture.mlp import MLP
from model.architecture.moe import MOE
from model.architecture.mHC import HyperConnection, HyperHead, StreamEmbed
from model.attention.MHA import MultiHeadSelfAttention
from model.attention.GQA import GroupedQueryAttention
from model.attention.MLA import MultiHeadLatentAttention
from model.attention.CCA import CompressedConvAttention
from model.attention.GDA import GatedDeltaAttention


# =========================================================================== Block

class HCBlock(nn.Module):
    """Transformer Block whose residual is a Manifold-Constrained Hyper-Connection.

    Replaces ``transformer.Block``'s ZAYA1-style learned residual scaling::

        xl_mid = ResScale_res_att(xl)     + ResScale_out_att(Attn(RMSNorm(xl)))
        xl_out  = ResScale_res_ffn(xl_mid) + ResScale_out_ffn(FFN (RMSNorm(xl_mid)))

    with the mHC update (paper §2.2 eq. 8) on H parallel streams::

        post, comb, collapsed = HC_att(streams)
        streams = HC.merge(streams, Attn(RMSNorm(collapsed)), post, comb)

        post, comb, collapsed = HC_ffn(streams)
        streams = HC.merge(streams, FFN(RMSNorm(collapsed)), post, comb)

    Equivalence to a vanilla residual: when ``comb = I`` and ``post = 1`` (which
    is approximately what mHC produces with `hc_mult = 1` and zero-mean init),
    each sub-block's contribution becomes ``stream + sublayer_out``, recovering
    the standard pre-norm transformer up to the learned ``pre`` gate.
    """

    def __init__(
        self,
        config: Config,
        rope: RotaryPositionalEmbedding,
        use_moe: bool,
        use_gda: bool = False,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.use_moe = use_moe
        # Same GDA-override semantics as transformer.Block: GDA replaces the
        # primary attention regardless of config.attention_type.
        self.attention_type = "GDA" if use_gda else config.attention_type

        # ── Attention ──
        if use_gda:
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
                device=device, dtype=dtype,
            )
        elif config.attention_type == "GQA":
            self.att = GroupedQueryAttention(
                config.d_model, config.num_heads, config.num_kv_heads, rope,
                device=device, dtype=dtype,
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
                dtype=dtype,
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
                dtype=dtype,
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
                dtype=dtype,
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

        # ── mHC residual (one HyperConnection per sub-block) ──
        self.hc_att = HyperConnection.from_config(config, device=device, dtype=dtype)
        self.hc_ffn = HyperConnection.from_config(config, device=device, dtype=dtype)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden_streams: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args
        ----
        hidden_streams : ``(B, S, H, D)`` — parallel residual streams.
        mask : optional attention mask (unused, like in ``transformer.Block``).

        Returns
        -------
        ``(B, S, H, D)`` — updated streams after one attention + FFN sub-block.
        """
        # ── Attention sub-block ──
        post, comb, collapsed = self.hc_att(hidden_streams)
        att_out = self.att(self.att_norm(collapsed), mask)
        att_out = self.dropout(att_out)
        hidden_streams = HyperConnection.merge(
            hidden_streams, att_out, post, comb,
        )

        # ── FFN sub-block ──
        post, comb, collapsed = self.hc_ffn(hidden_streams)
        ffn_out = self.ffn(self.ffn_norm(collapsed))
        ffn_out = self.dropout(ffn_out)
        hidden_streams = HyperConnection.merge(
            hidden_streams, ffn_out, post, comb,
        )

        return hidden_streams


# =================================================================== TransformerLM

class mHCTransformerLM(nn.Module):
    """Decoder-only LM with mHC residual streams.

    Differs from ``TransformerLM`` only in three places:
      1. A ``StreamEmbed`` after the token embedding expands the single hidden
         state into ``H = config.hc_mult`` parallel streams.
      2. Each layer is an ``HCBlock`` (mHC residual) instead of a ``Block``
         (ResScale residual).
      3. A ``HyperHead`` before ``final_norm`` collapses the ``H`` streams back
         to a single sequence for the LM head.
    """

    def __init__(self, config: Config, device=None, dtype=None):
        super().__init__()
        self.config = config
        self.use_moe = config.use_moe
        self.attention_type = config.attention_type
        self.context_length = config.context_length
        self.hc_mult = getattr(config, "hc_mult", 1)

        self.token_embeddings = Embedding(
            config.vocab_size, config.d_model, device=device, dtype=dtype,
        )

        # Expand the token embedding into H residual streams. ``channel_first=True``
        # produces an explicit H axis just before D (so streams are
        # ``(B, S, H, D)``), and ``expand_to_streams=True`` duplicates the input
        # across H with a per-stream learned offset.
        self.stream_embed = StreamEmbed(
            num_streams=self.hc_mult,
            dim=config.d_model,
            channel_first=True,
            expand_to_streams=True,
            device=device,
            dtype=dtype,
        )

        # RoPE dimension varies by attention type — same logic as TransformerLM.
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
            config.context_length, device=device,
        )

        # Hybrid GDA — identical wiring to TransformerLM, using the shared
        # `build_gda_layer_mask` helper (Qwen3-Next style, primary at block end).
        gda_ratio = (
            config.gda_ratio if config.attention_type != "CCA" else "none"
        )
        gda_mask = build_gda_layer_mask(config.num_layers, gda_ratio)
        gda_layer_indices = [i for i, use in enumerate(gda_mask) if use]
        self.gda_layer_indices = gda_layer_indices

        self.layers = nn.ModuleList([
            HCBlock(
                config=config,
                rope=self.rope,
                use_moe=(config.use_moe and (
                    config.moe_layers is None or i in config.moe_layers)),
                use_gda=(i in gda_layer_indices),
                device=device,
                dtype=dtype,
            )
            for i in range(config.num_layers)
        ])

        # Collapse H streams → single sequence before the final RMSNorm.
        self.hyper_head = HyperHead.from_config(config, device=device, dtype=dtype)

        self.final_norm = RMSNorm(config.d_model, device=device)
        self.lm_head = nn.Linear(
            config.d_model, config.vocab_size, device=device, dtype=dtype,
        )

        # Weight tying: share the embedding matrix with the LM head (see TransformerLM).
        self.lm_head.weight = self.token_embeddings.weight

    def forward(self, x: torch.Tensor):
        """Forward pass through token → streams → blocks → collapse → lm_head."""
        xl = self.token_embeddings(x)              # (B, S, D)
        streams = self.stream_embed(xl)            # (B, S, H, D)
        mask = None
        for block in self.layers:
            streams = block(streams, mask)         # (B, S, H, D)
        xl = self.hyper_head(streams)              # (B, S, D)
        xl = self.final_norm(xl)
        return self.lm_head(xl)

    def update_moe_biases(self):
        """Update expert biases for aux-loss-free load balance — same as TransformerLM."""
        if not self.use_moe:
            return
        for layer in self.layers:
            if hasattr(layer, "use_moe") and layer.use_moe:
                if hasattr(layer.ffn, "update_expert_bias"):
                    layer.ffn.update_expert_bias()


# Convenience alias so external code matches the `transformer` module's style.
mHCBlock = HCBlock
