"""
Manifold-Constrained Hyper-Connections (mHC) — Xie et al., 2026.

Replaces the standard residual `x_{l+1} = x_l + sublayer(norm(x_l))` with `H`
parallel hidden streams that are mixed via learned, doubly-stochastic weights
projected onto the manifold via Sinkhorn-Knopp.

This module is self-contained and intentionally does NOT modify
`model/transformer.py` — integration is the caller's responsibility. See the
"Integration sketch" docstring on `HyperConnection` for the wiring.

Components
----------
UnweightedRMSNorm
    Helper RMSNorm without learnable scale, used inside HyperConnection /
    HyperHead so the mix-logits computation does not introduce a redundant
    `gamma` next to the per-output `scale` parameter.

StreamEmbed
    Per-stream learnable embedding. Used at the input of the transformer to
    expand the token embedding from a single stream to `H` distinct streams
    (so the streams don't all start identical and immediately degenerate).
    Supports two input layouts (4D `(B, S, H, D)` for HyperConnection
    consumers, 3D `(B*H, S, D)` for batch-flattened pipelines).

HyperConnection
    Per-sub-block module. Takes the `H` parallel streams, produces the
    sublayer input (`collapsed`), and the placement/combine weights
    (`post`, `comb`) to merge the sublayer output back into the streams.
    `comb` is projected onto the doubly-stochastic manifold via Sinkhorn-Knopp.

HyperHead
    Final HC-stream collapse. Mirrors the `pre` path of HyperConnection
    (sigmoid-gated weighted sum across the stream axis) since there's no
    further sublayer to feed.

Adapted from the transformers library `DeepseekV4HyperConnection` reference
implementation with this project's API conventions:
    - explicit per-arg constructor (no required `Config` object)
    - `from_config(config)` classmethod for ergonomic integration
    - explicit shape annotations
    - public `merge(streams, sublayer_out, post, comb)` staticmethod so the
      caller doesn't have to know the einsum mechanics

Shape conventions
-----------------
    B = batch
    S = sequence length
    H = hc_mult (number of parallel streams)
    D = d_model (hidden size per stream)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# =================================================================== UnweightedRMSNorm

class UnweightedRMSNorm(nn.Module):
    """RMSNorm without a learnable `weight`/`gamma` — pure root-mean-square scaling.

    Used inside HyperConnection / HyperHead so the mix-logit computation
    does NOT introduce a redundant per-feature gain next to the per-output
    `scale` parameter (which is already learned).
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(-1, keepdim=True)
        return (x * torch.rsqrt(var + self.eps)).to(dtype)


# =================================================================== StreamEmbed

class StreamEmbed(nn.Module):
    """Per-stream learnable embedding.

    Adds a small learned offset to each of the `H` residual streams so they
    don't start out identical. Critical for mHC training stability — without
    it, the `H` streams collapse to a single one early in training.

    Two layout families (selected by `channel_first`) × two expansion modes
    (`expand_to_streams`):

    +---------------------+-----------------------+----------------------------+
    |                     | channel_first=False   | channel_first=True         |
    |                     | (batch-flattened)     | (H-as-axis)                |
    +---------------------+-----------------------+----------------------------+
    | expand_to_streams=  | input  (B*H, ..., D)  | input  (..., H, D)         |
    |   False             | output (B*H, ..., D)  | output (..., H, D)         |
    +---------------------+-----------------------+----------------------------+
    | expand_to_streams=  | input  (B,   ..., D)  | input  (..., D)            |
    |   True              | output (B*H, ..., D)  | output (..., H, D)         |
    +---------------------+-----------------------+----------------------------+

    The `channel_first=True, expand_to_streams=True` mode is the one used by
    HyperConnection / HyperHead consumers in this file: it takes a normal
    `(B, S, D)` embedding and produces `(B, S, H, D)` streams.

    Parameters
    ----------
    num_streams : H — number of parallel residual streams.
    dim         : D — hidden size per stream.
    channel_first :
        False → H is tiled into the leading batch dim;
        True  → H is an explicit inner axis just before D.
    expand_to_streams :
        False → input already has H stream slots; only add the embedding.
        True  → duplicate input across H and add per-stream embedding.
    init_std : truncated-normal std for the embedding init (default 0.02).
    """

    def __init__(
        self,
        num_streams: int,
        dim: int,
        channel_first: bool = False,
        expand_to_streams: bool = False,
        init_std: float = 0.02,
        device=None,
        dtype=None,
    ):
        super().__init__()
        assert num_streams >= 1, "num_streams (H) must be >= 1"
        self.num_streams = num_streams
        self.dim = dim
        self.channel_first = channel_first
        self.expand_to_streams = expand_to_streams
        self.stream_embed = nn.Parameter(
            torch.empty(num_streams, dim, device=device, dtype=dtype)
        )
        nn.init.trunc_normal_(self.stream_embed, mean=0.0, std=init_std,
                              a=-3 * init_std, b=3 * init_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, D = self.num_streams, self.dim

        if self.expand_to_streams:
            if self.channel_first:
                # (..., D) → (..., H, D): unsqueeze and broadcast-add.
                x_exp = x.unsqueeze(-2).expand(*x.shape[:-1], H, D)
                emb_shape = (1,) * (x_exp.ndim - 2) + (H, D)
                return x_exp + self.stream_embed.view(emb_shape)
            else:
                # (B, ..., D) → (B*H, ..., D): tile along batch, add per-stream.
                B = x.shape[0]
                tail = x.shape[1:]
                x_exp = x.unsqueeze(1).expand(B, H, *tail).reshape(B * H, *tail)
                emb_shape = (B * H,) + (1,) * (len(tail) - 1) + (D,)
                emb = self.stream_embed.unsqueeze(0).expand(B, H, D).reshape(emb_shape)
                return x_exp + emb
        else:
            if self.channel_first:
                # (..., H, D) → same. Broadcast (H, D) across leading dims.
                emb_shape = (1,) * (x.ndim - 2) + (H, D)
                return x + self.stream_embed.view(emb_shape)
            else:
                # (B*H, ..., D) → same. View → broadcast-add → view back.
                BH = x.shape[0]
                assert BH % H == 0, (
                    f"batch dim {BH} not divisible by num_streams={H}"
                )
                B = BH // H
                view = x.view(B, H, *x.shape[1:])  # (B, H, ..., D)
                emb_shape = (1, H) + (1,) * (view.ndim - 3) + (D,)
                return (view + self.stream_embed.view(emb_shape)).view_as(x)


# =================================================================== HyperConnection

class HyperConnection(nn.Module):
    r"""Manifold-Constrained Hyper-Connection — one site (attention or FFN) per layer.

    Given `H` parallel residual streams ``hidden_streams: (B, S, H, D)``, this
    module computes three quantities used by the surrounding decoder layer:

      * **collapsed** ``(B, S, D)`` — the single sequence fed into the
        sublayer (attention or MLP). Built as ``Σ_i pre_i * stream_i``.
      * **post** ``(B, S, H)`` — per-stream placement weight (in ``[0, 2]``)
        applied to the sublayer's output before it is added back to each
        stream.
      * **comb** ``(B, S, H, H)`` — doubly-stochastic matrix mixing streams
        among themselves. After Sinkhorn-Knopp projection both row and column
        sums approach 1.

    The next streams are then::

        new_streams[..., i, :] = Σ_j comb[..., i, j] * hidden_streams[..., j, :]
                               + post[..., i] * sublayer_out[..., :]

    The convenience staticmethod ``HyperConnection.merge`` performs that
    update so the caller never has to know the einsum.

    Math (mirrors Xie et al., 2026, §2.2 eq. 8)
    -------------------------------------------
    Let ``flat = RMSNorm(stream.flatten(start_dim=2))`` of shape ``(B, S, H*D)``.
    A single Linear ``fn`` produces ``(2+H)*H`` mix logits, split as
    ``[pre_w (H), post_w (H), comb_w (H*H)]``. With learned ``base`` (same
    split) and three learned scalar scales::

        pre  = sigmoid(pre_w  * pre_scale  + pre_b)  + eps        ∈ (eps, 1+eps)
        post = 2 * sigmoid(post_w * post_scale + post_b)          ∈ (0, 2)
        comb = sinkhorn( softmax(comb_w * comb_scale + comb_b, dim=-1) + eps )
                                                                  ≈ doubly stochastic

    Sinkhorn-Knopp: start from the softmax (rows already ≈ 1), column-normalise
    once, then alternate (row, col) normalisation for ``hc_sinkhorn_iters - 1``
    further rounds. With even 1 iteration the matrix is reasonably balanced;
    2 iterations match the DeepseekV4 reference; 5+ converge tightly.

    Integration sketch
    ------------------
    ::

        class Block(nn.Module):
            def __init__(self, config):
                ...
                self.hc_att = HyperConnection.from_config(config)
                self.hc_ffn = HyperConnection.from_config(config)
                ...

            def forward(self, hidden_streams, mask=None):
                # attention sub-block
                post, comb, collapsed = self.hc_att(hidden_streams)
                sublayer_out = self.att(self.att_norm(collapsed), mask)
                hidden_streams = HyperConnection.merge(
                    hidden_streams, sublayer_out, post, comb,
                )
                # FFN sub-block
                post, comb, collapsed = self.hc_ffn(hidden_streams)
                sublayer_out = self.ffn(self.ffn_norm(collapsed))
                hidden_streams = HyperConnection.merge(
                    hidden_streams, sublayer_out, post, comb,
                )
                return hidden_streams

    Parameters
    ----------
    d_model            : D — hidden size per stream.
    hc_mult            : H — number of parallel residual streams (>= 1).
    hc_sinkhorn_iters  : Sinkhorn-Knopp iterations (>= 1).
    hc_eps             : numerical-stability epsilon added to sigmoid /
                         softmax / normalisation denominators.
    rms_norm_eps       : eps for the internal UnweightedRMSNorm.
    """

    def __init__(
        self,
        d_model: int,
        hc_mult: int = 4,
        hc_sinkhorn_iters: int = 2,
        hc_eps: float = 1e-4,
        rms_norm_eps: float = 1e-6,
        device=None,
        dtype=None,
    ):
        super().__init__()
        assert hc_mult >= 1, f"hc_mult (H) must be >= 1, got {hc_mult}"
        assert hc_sinkhorn_iters >= 1, (
            f"hc_sinkhorn_iters must be >= 1, got {hc_sinkhorn_iters}"
        )
        self.d_model = d_model
        self.hc_mult = hc_mult
        self.hc_sinkhorn_iters = hc_sinkhorn_iters
        self.hc_eps = hc_eps

        self.input_norm = UnweightedRMSNorm(eps=rms_norm_eps)

        mix = (2 + hc_mult) * hc_mult  # = 2H + H²: pre_w (H) + post_w (H) + comb_w (H²)
        self.fn = nn.Parameter(torch.empty(
            mix, hc_mult * d_model, device=device, dtype=dtype))
        self.base = nn.Parameter(torch.empty(mix, device=device, dtype=dtype))
        # 3 outputs of the mHC mapping: pre, post, comb — each gets its own scale.
        self.scale = nn.Parameter(torch.empty(3, device=device, dtype=dtype))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Small truncated-gaussian for ``fn``, zero for ``base``, one for ``scale``.

        With this init the network starts near identity behaviour:
        ``pre  ≈ sigmoid(0) + eps  = 0.5 + eps`` (uniform stream collapse),
        ``post ≈ 2 * sigmoid(0)    = 1.0`` (full sublayer placement on every stream),
        ``comb ≈ softmax(0) + eps  = 1/H + eps`` after Sinkhorn (uniform stream mix).
        """
        nn.init.trunc_normal_(self.fn, mean=0.0, std=0.02, a=-0.06, b=0.06)
        nn.init.zeros_(self.base)
        nn.init.ones_(self.scale)

    @classmethod
    def from_config(cls, config, device=None, dtype=None) -> "HyperConnection":
        """Build from a project ``Config`` instance. Honors optional fields
        ``hc_sinkhorn_iters``, ``hc_eps``, ``rms_norm_eps`` if present.
        """
        return cls(
            d_model=config.d_model,
            hc_mult=getattr(config, "hc_mult", 4),
            hc_sinkhorn_iters=getattr(config, "hc_sinkhorn_iters", 2),
            hc_eps=getattr(config, "hc_eps", 1e-4),
            rms_norm_eps=getattr(config, "rms_norm_eps", 1e-6),
            device=device,
            dtype=dtype,
        )

    def forward(self, hidden_streams: torch.Tensor):
        """
        Args
        ----
        hidden_streams : ``(B, S, H, D)`` parallel residual streams.

        Returns
        -------
        post      : ``(B, S, H)``      — placement weights ∈ ``(0, 2)``.
        comb      : ``(B, S, H, H)``   — doubly-stochastic stream-mix matrix.
        collapsed : ``(B, S, D)``      — single sequence for the sublayer.

        All internal computation is in fp32 for stability; ``post``, ``comb``
        and ``collapsed`` are cast back to ``hidden_streams.dtype`` on return.
        """
        H = self.hc_mult
        assert hidden_streams.ndim == 4, (
            f"expected (B, S, H, D), got shape {tuple(hidden_streams.shape)}"
        )
        assert hidden_streams.shape[-2] == H, (
            f"expected H={H} streams, got {hidden_streams.shape[-2]}"
        )
        assert hidden_streams.shape[-1] == self.d_model, (
            f"expected D={self.d_model}, got {hidden_streams.shape[-1]}"
        )

        out_dtype = hidden_streams.dtype
        streams_fp32 = hidden_streams.float()

        # (B, S, H*D) — flatten the (H, D) tail so the linear can read all streams.
        flat = self.input_norm(streams_fp32.flatten(start_dim=2))
        mix_logits = F.linear(flat, self.fn.float())  # (B, S, (2+H)*H)

        # Split into the three logit groups.
        pre_w, post_w, comb_w = mix_logits.split([H, H, H * H], dim=-1)
        pre_b, post_b, comb_b = self.base.float().split([H, H, H * H])
        pre_scale, post_scale, comb_scale = self.scale.float().unbind(0)

        # pre  ∈ (eps, 1+eps): stream-collapse weights.
        pre = torch.sigmoid(pre_w * pre_scale + pre_b) + self.hc_eps

        # post ∈ (0, 2): placement weight (×2 lets each stream optionally
        # receive twice the sublayer signal).
        post = 2.0 * torch.sigmoid(post_w * post_scale + post_b)

        # comb: H×H doubly-stochastic matrix via Sinkhorn-Knopp.
        comb_logits = (
            comb_w.view(*comb_w.shape[:-1], H, H) * comb_scale
            + comb_b.view(H, H)
        )
        # softmax along columns (last dim) gives rows summing ≈ 1.
        comb = torch.softmax(comb_logits, dim=-1) + self.hc_eps
        # one col-normalisation kicks off Sinkhorn (matches DeepseekV4 reference).
        comb = comb / (comb.sum(dim=-2, keepdim=True) + self.hc_eps)
        for _ in range(self.hc_sinkhorn_iters - 1):
            comb = comb / (comb.sum(dim=-1, keepdim=True) + self.hc_eps)  # rows
            comb = comb / (comb.sum(dim=-2, keepdim=True) + self.hc_eps)  # cols

        # Collapse: weighted sum across H, drops the stream axis.
        # (B, S, H, 1) * (B, S, H, D) → sum over H → (B, S, D)
        collapsed = (pre.unsqueeze(-1) * streams_fp32).sum(dim=2)

        return post.to(out_dtype), comb.to(out_dtype), collapsed.to(out_dtype)

    @staticmethod
    def merge(
        hidden_streams: torch.Tensor,
        sublayer_out: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ) -> torch.Tensor:
        """Update the residual streams given the sublayer output.

        Implements::

            new_streams[..., i, :] = Σ_j comb[..., i, j] * hidden_streams[..., j, :]
                                   + post[..., i] * sublayer_out[..., :]

        Args
        ----
        hidden_streams : ``(B, S, H, D)``
        sublayer_out   : ``(B, S, D)``
        post           : ``(B, S, H)``
        comb           : ``(B, S, H, H)``

        Returns
        -------
        ``(B, S, H, D)`` — the new streams.
        """
        # ``...ij,...jd → ...id``: combine input streams (j) into output streams (i).
        mixed = torch.einsum("...ij,...jd->...id", comb, hidden_streams)
        # Broadcast sublayer_out across the H axis, weighted per-stream.
        added = post.unsqueeze(-1) * sublayer_out.unsqueeze(-2)
        return mixed + added


# =================================================================== HyperHead

class HyperHead(nn.Module):
    r"""Final HC-stream collapse: ``(B, S, H, D) → (B, S, D)``.

    Mirrors the ``pre`` path of HyperConnection — sigmoid-gated weighted sum
    across the stream axis — but without ``post`` / ``comb`` since there is no
    further sublayer to feed.

    Used just before the LM head's final RMSNorm / output projection.
    """

    def __init__(
        self,
        d_model: int,
        hc_mult: int = 4,
        hc_eps: float = 1e-4,
        rms_norm_eps: float = 1e-6,
        device=None,
        dtype=None,
    ):
        super().__init__()
        assert hc_mult >= 1, f"hc_mult (H) must be >= 1, got {hc_mult}"
        self.d_model = d_model
        self.hc_mult = hc_mult
        self.eps = hc_eps

        self.input_norm = UnweightedRMSNorm(eps=rms_norm_eps)
        self.hc_fn = nn.Parameter(torch.empty(
            hc_mult, hc_mult * d_model, device=device, dtype=dtype))
        self.hc_base = nn.Parameter(torch.empty(hc_mult, device=device, dtype=dtype))
        self.hc_scale = nn.Parameter(torch.empty(1, device=device, dtype=dtype))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.hc_fn, mean=0.0, std=0.02, a=-0.06, b=0.06)
        nn.init.zeros_(self.hc_base)
        nn.init.ones_(self.hc_scale)

    @classmethod
    def from_config(cls, config, device=None, dtype=None) -> "HyperHead":
        return cls(
            d_model=config.d_model,
            hc_mult=getattr(config, "hc_mult", 4),
            hc_eps=getattr(config, "hc_eps", 1e-4),
            rms_norm_eps=getattr(config, "rms_norm_eps", 1e-6),
            device=device,
            dtype=dtype,
        )

    def forward(self, hidden_streams: torch.Tensor) -> torch.Tensor:
        """
        Args
        ----
        hidden_streams : ``(B, S, H, D)``

        Returns
        -------
        ``(B, S, D)`` — collapsed single sequence.
        """
        H = self.hc_mult
        assert hidden_streams.ndim == 4
        assert hidden_streams.shape[-2] == H
        assert hidden_streams.shape[-1] == self.d_model

        out_dtype = hidden_streams.dtype
        streams_fp32 = hidden_streams.float()

        flat = self.input_norm(streams_fp32.flatten(start_dim=2))
        mixes = F.linear(flat, self.hc_fn.float())  # (B, S, H)
        pre = torch.sigmoid(
            mixes * self.hc_scale.float() + self.hc_base.float()
        ) + self.eps  # (B, S, H)

        collapsed = (pre.unsqueeze(-1) * streams_fp32).sum(dim=2)
        return collapsed.to(out_dtype)
