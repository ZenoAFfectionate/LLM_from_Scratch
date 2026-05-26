"""
Multi-Token Prediction (MTP) — DeepSeek-V3 style.

Carries a chain of additional prediction depths after the main transformer:
each depth fuses the previous depth's hidden state with the *next* token's
embedding, runs one transformer block on the fused state, and emits a new
hidden state that the trainer maps back to logits via the shared LM head.

This module was originally written against an older Transformer Block
signature (``Block(d_model, num_heads, d_ff, rope, drop_p, ...)``) and was
broken by the refactor that switched ``model/transformer.py`` to a Config-
driven ``Block(config, rope, use_moe, use_gda)``. This file now matches the
current Block API:
    * ``__init__`` takes a project ``Config`` (plus the shared rope) and
      forwards into ``Block(config, rope, use_moe)``.
    * ``forward`` uses the Block's single-tensor signature (no kv_cache,
      no use_cache; the Block forward returns a tensor, not a tuple).
"""

import torch
import torch.nn as nn

from model.config import Config
from model.utils import RMSNorm


class MTPModule(nn.Module):
    """One MTP depth (DeepSeek-V3 §2.3).

    Projects ``[RMSNorm(h_{k-1}) ; RMSNorm(Emb(t_{i+k}))]`` from ``2 * d_model``
    down to ``d_model``, then runs a transformer block. Uses the project's
    ``Config``-driven ``TransformerBlock``.

    Args:
        config: Project Config; provides d_model, attention type, FFN choice,
                and all the (de)tuning knobs the inner Block needs.
        rope:   Shared RotaryPositionalEmbedding instance from the main model.
        use_moe: Whether THIS depth should use MoE for its FFN.
        device, dtype: passed through to all sub-modules.
    """

    def __init__(
        self,
        config: Config,
        rope,
        use_moe: bool = False,
        device=None,
        dtype=None,
    ):
        super().__init__()

        # Import here to avoid the model.transformer → model.mtp circular import
        # that would otherwise happen at module load time.
        from model.transformer import TransformerBlock

        d_model = config.d_model
        # M_k: projects [RMSNorm(h_{k-1}) ; RMSNorm(Emb(t_{i+k}))] (2d) → d.
        self.projection = nn.Linear(2 * d_model, d_model, device=device, dtype=dtype)

        # Transformer block at this depth — single tensor in, single tensor out.
        # GDA is intentionally NOT used in MTP depths: the recurrent state
        # would need its own per-depth cache and the hybrid pattern is a
        # whole-model knob, not a per-depth one.
        self.transformer_block = TransformerBlock(
            config=config,
            rope=rope,
            use_moe=use_moe,
            use_gda=False,
            device=device,
            dtype=dtype,
        )

        # RMSNorm layers for the two inputs (DeepSeek-V3 §2.3 Eq. 19).
        self.norm_h = RMSNorm(d_model, device=device)
        self.norm_emb = RMSNorm(d_model, device=device)

    def forward(
        self,
        h_prev: torch.Tensor,          # (batch_size, seq_len, d_model)
        token_embeddings: torch.Tensor  # (batch_size, seq_len, d_model)
    ) -> torch.Tensor:
        """
        Forward pass of one MTP depth.

        Args:
            h_prev: Representations from the previous depth, (B, S, d_model).
            token_embeddings: Embeddings of the *future* tokens at this depth,
                              (B, S, d_model).

        Returns:
            h_k: Representations at the current depth, (B, S, d_model).
        """
        h_prev_norm = self.norm_h(h_prev)
        token_emb_norm = self.norm_emb(token_embeddings)

        # Concatenate along feature dim, project back to d_model.
        combined = torch.cat([h_prev_norm, token_emb_norm], dim=-1)  # (B, S, 2d)
        h_prime = self.projection(combined)                          # (B, S, d)

        # Current Block.forward signature: (xl, mask=None) -> tensor
        return self.transformer_block(h_prime)


class MultiTokenPredictor(nn.Module):
    """Stack of ``num_depths`` MTPModules — one per MTP prediction depth.

    Used by ``train_mtp.py`` after the main backbone. Each depth consumes the
    previous depth's representation and predicts one token further in the
    future, sharing the embedding (and LM head) with the main model.

    Args:
        num_depths:  D, number of additional prediction depths (paper §2.3).
        config:      Project Config; same instance used by the main model.
        rope:        Shared rope instance from the main TransformerLM.
        moe_layers:  optional list of depth indices that should use MoE FFN;
                     None ⇒ every depth uses MoE iff ``config.use_moe`` is True.
        device, dtype: forwarded to every MTPModule.
    """

    def __init__(
        self,
        num_depths: int,
        config: Config,
        rope,
        moe_layers=None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.num_depths = num_depths

        self.mtp_modules = nn.ModuleList([
            MTPModule(
                config=config,
                rope=rope,
                use_moe=(config.use_moe and (
                    moe_layers is None or i in moe_layers)),
                device=device,
                dtype=dtype,
            )
            for i in range(num_depths)
        ])

    def forward(
        self,
        h_main: torch.Tensor,           # Main model representations
        token_ids: torch.Tensor,        # Input token IDs
        embedding_layer: nn.Module,     # Shared embedding layer
    ):
        """
        Forward pass through all MTP depths (DeepSeek-V3 §2.3 Eq. 18-22).

        Args:
            h_main: Main model output representations, (B, S, d_model).
            token_ids: Input token IDs, (B, S).
            embedding_layer: Shared embedding layer from the main model.

        Returns:
            List of representations at each depth: [h_1, h_2, ..., h_D].
        """
        _, seq_len, _ = h_main.shape

        depth_representations = []
        h_prev = h_main  # h_0 is the main model output

        for k, mtp_module in enumerate(self.mtp_modules):
            # At code-depth k (paper depth k+1), position i fuses h_prev[i] with
            # emb(t_{i+k+1}) and predicts t_{i+k+2}. We therefore need BOTH
            # the fusion token (i+k+1 ≤ S-1) AND the target token (i+k+2 ≤ S-1).
            # The stricter constraint i ≤ S-k-3 gives valid_len = S - k - 2
            # positions [0, S-k-3]. Producing only loss-valid positions keeps the
            # downstream cross-entropy shape-aligned (see train_mtp.compute_mtp_loss).
            valid_len = seq_len - k - 2
            if valid_len <= 0:
                break

            future_token_ids = token_ids[:, k + 1 : k + 1 + valid_len]    # (B, S-k-2)
            future_token_embeddings = embedding_layer(future_token_ids)   # (B, S-k-2, d)
            h_prev_sliced = h_prev[:, :valid_len, :]                      # (B, S-k-2, d)

            h_k = mtp_module(h_prev_sliced, future_token_embeddings)

            depth_representations.append(h_k)
            h_prev = h_k

        return depth_representations
