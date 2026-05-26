"""
Tests for model/architecture/mHC.py — Manifold-Constrained Hyper-Connections
(Xie et al., 2026; mirrors transformers' DeepseekV4HyperConnection).

Coverage:
    UnweightedRMSNorm
        - no learnable parameters
        - shape & dtype preservation (fp32 / fp16 / bf16)
        - matches a manual RMSNorm reference
        - eps prevents division-by-zero on all-zero input

    StreamEmbed
        - all 4 mode combinations (channel_first × expand_to_streams)
        - learnable parameter with finite gradient
        - per-stream embeddings are distinct (broadcast happens along H, not D)
        - trunc-normal init range honoured
        - assertion fires when B*H is not divisible by H

    HyperConnection (core mHC module)
        - shapes: post (B,S,H), comb (B,S,H,H), collapsed (B,S,D)
        - post values lie in (0, 2)            (paper §2.2 eq. 8)
        - comb is row- & column-stochastic after Sinkhorn-Knopp
        - identity-like behaviour at init (pre≈0.5, post≈1.0, comb≈1/H)
        - matches a step-by-step manual reproduction of the math
        - gradients flow to fn / base / scale
        - dtype: fp16/bf16 inputs return fp16/bf16, internals in fp32
        - shape validation rejects wrong rank / wrong H / wrong D
        - from_config respects the project's Config class (defaults & overrides)
        - varying H (1, 2, 4, 8) — including the H=1 degenerate-residual case
        - sinkhorn_iters convergence: more iters → tighter row/col sums
        - sinkhorn_iters=1 still produces a finite balanced matrix
        - determinism in eval()
        - reset_parameters() restores init state

    HyperConnection.merge (staticmethod)
        - matches the einsum + outer-product reference
        - identity-comb + post=0 returns hidden_streams unchanged
        - post=0 + arbitrary comb is equivalent to comb @ hidden_streams
        - identity-comb passes through; only sublayer_out contributes
        - does not modify inputs in-place
        - works under torch.no_grad

    HyperHead
        - output shape (B, S, D)
        - at init collapsed ≈ 0.5 * Σ_h streams[..., h, :]
        - gradient flow to hc_fn / hc_base / hc_scale
        - dtype preservation
        - from_config integrates with Config

    Integration sketch
        - composing N HyperConnections + sublayers stays finite and produces
          gradients on every HyperConnection's parameters
"""
from __future__ import annotations
import math
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.architecture.mHC import (
    UnweightedRMSNorm,
    StreamEmbed,
    HyperConnection,
    HyperHead,
)


# Shared shape constants (deliberately small so tests stay fast).
B, S, H, D = 2, 5, 4, 16


# ============================================================== helpers / fixtures

class _ConfigLike:
    """Minimal stand-in for the project's ``Config`` for ``from_config`` tests.

    We don't import ``model.config.Config`` because its default ``hc_mult`` is 1
    and constructing it pulls in dozens of unrelated fields. ``from_config``
    only uses ``getattr`` on a few names, so any namespace works.
    """

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


@pytest.fixture
def streams() -> torch.Tensor:
    """(B, S, H, D) hidden streams with a deterministic seed."""
    torch.manual_seed(0)
    return torch.randn(B, S, H, D)


@pytest.fixture
def hc() -> HyperConnection:
    """Default-config HyperConnection, freshly initialised."""
    torch.manual_seed(0)
    return HyperConnection(d_model=D, hc_mult=H, hc_sinkhorn_iters=2)


@pytest.fixture
def head() -> HyperHead:
    torch.manual_seed(0)
    return HyperHead(d_model=D, hc_mult=H)


def _manual_forward(hc: HyperConnection, streams: torch.Tensor):
    """Reproduce HyperConnection.forward step by step. Used to cross-check the
    module against the paper math (a small typo in the implementation would
    show up here)."""
    Hc = hc.hc_mult
    flat = hc.input_norm(streams.float().flatten(start_dim=2))
    mix_logits = F.linear(flat, hc.fn.float())
    pre_w, post_w, comb_w = mix_logits.split([Hc, Hc, Hc * Hc], dim=-1)
    pre_b, post_b, comb_b = hc.base.float().split([Hc, Hc, Hc * Hc])
    pre_s, post_s, comb_s = hc.scale.float().unbind(0)

    pre = torch.sigmoid(pre_w * pre_s + pre_b) + hc.hc_eps
    post = 2.0 * torch.sigmoid(post_w * post_s + post_b)
    cl = comb_w.view(*comb_w.shape[:-1], Hc, Hc) * comb_s + comb_b.view(Hc, Hc)
    comb = torch.softmax(cl, dim=-1) + hc.hc_eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + hc.hc_eps)
    for _ in range(hc.hc_sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + hc.hc_eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + hc.hc_eps)
    collapsed = (pre.unsqueeze(-1) * streams.float()).sum(dim=2)
    return post, comb, collapsed, pre


# ============================================================== UnweightedRMSNorm

def test_uwrms_has_no_learnable_parameters():
    """Whole point of `Unweighted` — gamma is absorbed into mHC's `scale`."""
    layer = UnweightedRMSNorm()
    assert list(layer.parameters()) == []


def test_uwrms_preserves_shape_and_dtype():
    layer = UnweightedRMSNorm()
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        x = torch.randn(2, 3, 5, dtype=dtype)
        y = layer(x)
        assert y.shape == x.shape and y.dtype == dtype


def test_uwrms_matches_manual_rms_reference():
    layer = UnweightedRMSNorm(eps=1e-6)
    x = torch.randn(2, 3, 8).double()  # fp64 reference to make tolerance tight
    y = layer(x.float()).double()
    ref = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
    torch.testing.assert_close(y, ref, atol=1e-5, rtol=1e-5)


def test_uwrms_handles_zero_input_without_nan():
    """eps must keep rsqrt finite when the variance is zero."""
    layer = UnweightedRMSNorm(eps=1e-6)
    y = layer(torch.zeros(4, 8))
    assert torch.isfinite(y).all()
    assert (y == 0).all()


# ===================================================================== StreamEmbed

def test_streamembed_channel_first_expand_shape():
    """The mode used by HyperConnection consumers: (B, S, D) → (B, S, H, D)."""
    layer = StreamEmbed(num_streams=H, dim=D,
                        channel_first=True, expand_to_streams=True)
    x = torch.randn(B, S, D)
    y = layer(x)
    assert y.shape == (B, S, H, D)


def test_streamembed_channel_first_passthrough_shape():
    layer = StreamEmbed(num_streams=H, dim=D,
                        channel_first=True, expand_to_streams=False)
    x = torch.randn(B, S, H, D)
    assert layer(x).shape == x.shape


def test_streamembed_batch_flattened_expand_shape():
    layer = StreamEmbed(num_streams=H, dim=D,
                        channel_first=False, expand_to_streams=True)
    x = torch.randn(B, S, D)
    y = layer(x)
    assert y.shape == (B * H, S, D)


def test_streamembed_batch_flattened_passthrough_shape():
    layer = StreamEmbed(num_streams=H, dim=D,
                        channel_first=False, expand_to_streams=False)
    x = torch.randn(B * H, S, D)
    assert layer(x).shape == x.shape


def test_streamembed_per_stream_offsets_differ():
    """The embedding must be applied per-stream — not the same constant on every
    stream — otherwise the H copies would stay identical and mHC degenerates."""
    layer = StreamEmbed(num_streams=H, dim=D,
                        channel_first=True, expand_to_streams=True)
    with torch.no_grad():
        # Forcing the embedding to be a clear pattern makes the check unambiguous.
        layer.stream_embed.zero_()
        layer.stream_embed.add_(torch.arange(H, dtype=torch.float32)
                                .unsqueeze(-1).expand(H, D))
    x = torch.zeros(1, 1, D)
    y = layer(x)  # (1, 1, H, D)
    per_stream_means = y[0, 0].mean(dim=-1)
    torch.testing.assert_close(per_stream_means, torch.arange(H, dtype=torch.float32))


def test_streamembed_is_learnable():
    layer = StreamEmbed(num_streams=3, dim=4)
    assert isinstance(layer.stream_embed, nn.Parameter)
    x = torch.randn(6, 4)  # B*H = 6, H = 3 → B = 2
    layer(x).pow(2).sum().backward()
    assert layer.stream_embed.grad is not None
    assert layer.stream_embed.grad.abs().sum() > 0


def test_streamembed_trunc_normal_init_bounds():
    """Default std=0.02, a/b at ±3σ — values must lie strictly inside [-0.06, 0.06]."""
    layer = StreamEmbed(num_streams=64, dim=32, init_std=0.02)
    e = layer.stream_embed
    assert e.abs().max().item() <= 3 * 0.02 + 1e-6
    assert e.std().item() < 0.05  # very loose upper bound; std should be ~0.02


def test_streamembed_bad_batch_raises():
    """B*H must divide cleanly into H groups in batch-flattened pass-through mode."""
    layer = StreamEmbed(num_streams=3, dim=4,
                        channel_first=False, expand_to_streams=False)
    with pytest.raises(AssertionError):
        layer(torch.randn(7, 4))  # 7 % 3 != 0


# ================================================================ HyperConnection
# ---- output shapes ------------------------------------------------------------

def test_hc_output_shapes(hc, streams):
    post, comb, collapsed = hc(streams)
    assert post.shape == (B, S, H)
    assert comb.shape == (B, S, H, H)
    assert collapsed.shape == (B, S, D)


# ---- value ranges -------------------------------------------------------------

def test_hc_post_in_zero_two_range(hc, streams):
    """post = 2·σ(·) lives strictly in (0, 2)."""
    post, _, _ = hc(streams)
    assert post.min() > 0
    assert post.max() < 2.0


def test_hc_comb_row_stochastic(hc, streams):
    """After Sinkhorn-Knopp, rows must sum to ~1 (slightly under, due to eps)."""
    _, comb, _ = hc(streams)
    row_sums = comb.sum(dim=-1)
    # eps=1e-4 → row sums sit ~1-eps; allow generous slack but reject any drift
    # outside [0.99, 1.01].
    assert (row_sums >= 0.99).all() and (row_sums <= 1.01).all()


def test_hc_comb_col_stochastic(hc, streams):
    _, comb, _ = hc(streams)
    col_sums = comb.sum(dim=-2)
    assert (col_sums >= 0.99).all() and (col_sums <= 1.01).all()


def test_hc_comb_doubly_stochastic_after_more_iters(streams):
    """More Sinkhorn iterations → tighter convergence on both axes."""
    torch.manual_seed(0)
    hc_low = HyperConnection(d_model=D, hc_mult=H, hc_sinkhorn_iters=1)
    torch.manual_seed(0)
    hc_high = HyperConnection(d_model=D, hc_mult=H, hc_sinkhorn_iters=10)
    # share the same `fn`/`base`/`scale` so only the iters differ
    hc_high.load_state_dict(hc_low.state_dict())

    _, c1, _ = hc_low(streams)
    _, c10, _ = hc_high(streams)
    dev_1 = max(
        (c1.sum(-1) - 1).abs().max().item(),
        (c1.sum(-2) - 1).abs().max().item(),
    )
    dev_10 = max(
        (c10.sum(-1) - 1).abs().max().item(),
        (c10.sum(-2) - 1).abs().max().item(),
    )
    # convergence is monotonic in expectation
    assert dev_10 <= dev_1 + 1e-6


# ---- numerical match against manual recomputation ----------------------------

def test_hc_forward_matches_manual_math(hc, streams):
    post, comb, coll = hc(streams)
    m_post, m_comb, m_coll, _ = _manual_forward(hc, streams)
    torch.testing.assert_close(post, m_post, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(comb, m_comb, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(coll, m_coll, atol=1e-6, rtol=1e-6)


# ---- init behaviour ----------------------------------------------------------

def test_hc_init_pre_near_half(hc, streams):
    """At init: fn ~ small noise, base = 0, scale = 1 → sigmoid input ~ 0
    → pre ≈ 0.5 + eps. The tolerance accounts for trunc-normal `fn` noise
    accumulating over H*D=64 input dims (so the pre-sigmoid logits drift to
    roughly ±0.5, giving sigmoid deviations of ~0.12 from 0.5)."""
    _, _, _, pre = _manual_forward(hc, streams)
    diff = (pre - 0.5).abs()
    # also confirm activations are far from saturation (the real risk at init)
    assert diff.max().item() < 0.2
    assert pre.min().item() > 0.2 and pre.max().item() < 0.8


def test_hc_init_post_near_one(hc, streams):
    """post = 2·σ(·) is centered at 1.0 with double the sigmoid spread, so its
    natural range at init is ~[0.6, 1.4]."""
    post, _, _ = hc(streams)
    assert (post - 1.0).abs().max().item() < 0.4
    # not pinned to either extreme of the (0, 2) bound
    assert post.min().item() > 0.4 and post.max().item() < 1.6


def test_hc_init_comb_near_uniform(hc, streams):
    _, comb, _ = hc(streams)
    target = 1.0 / H
    # post-Sinkhorn the matrix sits within ~10% of 1/H at random init
    assert (comb - target).abs().max().item() < 0.1


# ---- gradient flow -----------------------------------------------------------

def test_hc_gradients_flow_to_all_params(hc, streams):
    streams = streams.detach().requires_grad_(True)
    post, comb, collapsed = hc(streams)
    loss = collapsed.pow(2).sum() + post.sum() + comb.sum()
    loss.backward()
    for name, p in hc.named_parameters():
        assert p.grad is not None, f"no grad on {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad on {name}"
        assert p.grad.abs().sum() > 0, f"zero grad on {name}"
    assert streams.grad is not None and torch.isfinite(streams.grad).all()


# ---- dtype handling ----------------------------------------------------------

@pytest.mark.parametrize(
    "dtype",
    [torch.float32, torch.float16, torch.bfloat16],
)
def test_hc_dtype_preserved(dtype, hc):
    """Internals compute in fp32, outputs cast back to the input dtype."""
    x = torch.randn(B, S, H, D, dtype=dtype)
    post, comb, collapsed = hc(x)
    assert post.dtype == dtype
    assert comb.dtype == dtype
    assert collapsed.dtype == dtype


# ---- shape validation --------------------------------------------------------

def test_hc_rejects_wrong_rank(hc):
    with pytest.raises(AssertionError, match="expected"):
        hc(torch.randn(B, S, D))  # missing H axis


def test_hc_rejects_wrong_h(hc):
    with pytest.raises(AssertionError, match="streams"):
        hc(torch.randn(B, S, H + 1, D))


def test_hc_rejects_wrong_d(hc):
    with pytest.raises(AssertionError, match="D="):
        hc(torch.randn(B, S, H, D + 1))


# ---- from_config -------------------------------------------------------------

def test_hc_from_config_defaults():
    cfg = _ConfigLike(d_model=D)  # everything else hits getattr defaults
    layer = HyperConnection.from_config(cfg)
    assert layer.d_model == D
    assert layer.hc_mult == 4  # default
    assert layer.hc_sinkhorn_iters == 2  # default
    assert layer.hc_eps == 1e-4  # default


def test_hc_from_config_overrides():
    cfg = _ConfigLike(d_model=D, hc_mult=3, hc_sinkhorn_iters=5,
                      hc_eps=1e-3, rms_norm_eps=1e-5)
    layer = HyperConnection.from_config(cfg)
    assert layer.hc_mult == 3
    assert layer.hc_sinkhorn_iters == 5
    assert layer.hc_eps == 1e-3
    assert layer.input_norm.eps == 1e-5


def test_hc_from_config_with_project_config():
    """Real model.config.Config integrates cleanly (uses its hc_mult field)."""
    from model.config import Config
    cfg = Config(d_model=D, hc_mult=3, num_layers=2, vocab_size=100)
    layer = HyperConnection.from_config(cfg)
    assert layer.d_model == D and layer.hc_mult == 3
    out = layer(torch.randn(B, S, 3, D))
    assert all(t.shape[:2] == (B, S) for t in out)


# ---- varying H ---------------------------------------------------------------

@pytest.mark.parametrize("h_val", [1, 2, 4, 8])
def test_hc_various_h(h_val):
    """The mapping width scales with H — make sure all the index arithmetic works."""
    torch.manual_seed(0)
    layer = HyperConnection(d_model=D, hc_mult=h_val, hc_sinkhorn_iters=2)
    x = torch.randn(B, S, h_val, D)
    post, comb, coll = layer(x)
    assert post.shape == (B, S, h_val)
    assert comb.shape == (B, S, h_val, h_val)
    assert coll.shape == (B, S, D)
    # For H=1 the doubly-stochastic constraint forces comb≡[[1]] — verify.
    if h_val == 1:
        torch.testing.assert_close(
            comb, torch.ones_like(comb), atol=2e-4, rtol=0
        )


def test_hc_sinkhorn_iters_one_still_finite(streams):
    """Even with the minimum 1 iteration, comb must be finite. With iters=1 the
    rows are NOT exactly 1 (only the col-normalisation runs), so check finiteness
    and a loose row/col upper bound."""
    layer = HyperConnection(d_model=D, hc_mult=H, hc_sinkhorn_iters=1)
    _, comb, _ = layer(streams)
    assert torch.isfinite(comb).all()
    # column sums are exactly normalised; rows are just bounded
    col_sums = comb.sum(-2)
    assert (col_sums >= 0.99).all() and (col_sums <= 1.01).all()


# ---- determinism / no-grad ---------------------------------------------------

def test_hc_eval_determinism(hc, streams):
    hc.eval()
    with torch.no_grad():
        a = hc(streams)
        b = hc(streams)
    for x, y in zip(a, b):
        torch.testing.assert_close(x, y, atol=0.0, rtol=0.0)


def test_hc_reset_parameters_is_repeatable():
    layer = HyperConnection(d_model=D, hc_mult=H)
    # The init is stochastic, but base/scale are deterministic — check them.
    layer.fn.data.fill_(1.0)
    layer.base.data.fill_(7.0)
    layer.scale.data.fill_(13.0)
    layer.reset_parameters()
    assert (layer.base == 0).all()
    assert (layer.scale == 1).all()
    # fn is trunc-normal noise — should be small.
    assert layer.fn.abs().max().item() <= 0.06 + 1e-6


# ============================================================ HyperConnection.merge

def test_merge_matches_einsum_reference(hc, streams):
    post, comb, _ = hc(streams)
    sub = torch.randn(B, S, D)
    out = HyperConnection.merge(streams, sub, post, comb)
    ref = (torch.einsum("...ij,...jd->...id", comb, streams)
           + post.unsqueeze(-1) * sub.unsqueeze(-2))
    torch.testing.assert_close(out, ref, atol=1e-6, rtol=1e-6)


def test_merge_identity_comb_zero_post_returns_streams(streams):
    """If comb is the identity and post is zero, the residual stream is
    untouched — the standard 'skip' fallback."""
    identity_comb = torch.eye(H).expand(B, S, H, H).contiguous()
    zero_post = torch.zeros(B, S, H)
    sub = torch.randn(B, S, D)
    out = HyperConnection.merge(streams, sub, zero_post, identity_comb)
    torch.testing.assert_close(out, streams, atol=0.0, rtol=0.0)


def test_merge_post_zero_drops_sublayer(streams):
    """post=0 must completely discard the sublayer output — only stream mixing
    matters (this is what a 'gate-closed' state looks like)."""
    _, comb, _ = HyperConnection(d_model=D, hc_mult=H)(streams)
    sub = torch.randn(B, S, D)
    out = HyperConnection.merge(streams, sub, torch.zeros(B, S, H), comb)
    mixed_only = torch.einsum("...ij,...jd->...id", comb, streams)
    torch.testing.assert_close(out, mixed_only, atol=1e-6, rtol=1e-6)


def test_merge_identity_comb_passes_streams_and_adds_sublayer(streams):
    """When comb = identity, each stream receives only itself + post*sublayer.
    This is the regime where mHC reduces to a standard residual."""
    identity_comb = torch.eye(H).expand(B, S, H, H).contiguous()
    post = torch.ones(B, S, H) * 0.7
    sub = torch.randn(B, S, D)
    out = HyperConnection.merge(streams, sub, post, identity_comb)
    expected = streams + 0.7 * sub.unsqueeze(-2)
    torch.testing.assert_close(out, expected, atol=1e-6, rtol=1e-6)


def test_merge_does_not_modify_inputs_in_place(streams):
    streams_clone = streams.clone()
    sub = torch.randn(B, S, D)
    sub_clone = sub.clone()
    post = torch.ones(B, S, H)
    comb = torch.eye(H).expand(B, S, H, H).contiguous()
    HyperConnection.merge(streams, sub, post, comb)
    torch.testing.assert_close(streams, streams_clone, atol=0.0, rtol=0.0)
    torch.testing.assert_close(sub, sub_clone, atol=0.0, rtol=0.0)


def test_merge_under_no_grad(streams):
    post = torch.ones(B, S, H, requires_grad=False)
    comb = torch.eye(H).expand(B, S, H, H).contiguous()
    sub = torch.randn(B, S, D)
    with torch.no_grad():
        out = HyperConnection.merge(streams, sub, post, comb)
    assert not out.requires_grad


# ===================================================================== HyperHead

def test_head_output_shape(head, streams):
    out = head(streams)
    assert out.shape == (B, S, D)


def test_head_output_near_half_sum_at_init(head, streams):
    """At init, pre ≈ 0.5+eps → collapsed ≈ 0.5 * Σ_h streams[..., h, :]."""
    out = head(streams)
    expected = 0.5 * streams.sum(dim=2)
    # absolute slack: pre noise (~0.02 sigmoid-input variance) ≈ small fraction
    # of streams' magnitude.
    diff = (out - expected).abs()
    # streams std ≈ 1 → 0.05 absolute is well within the noise floor of the init
    assert diff.mean().item() < 0.2


def test_head_gradients_flow(head, streams):
    streams = streams.detach().requires_grad_(True)
    head(streams).pow(2).sum().backward()
    for name, p in head.named_parameters():
        assert p.grad is not None, f"no grad on {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad on {name}"
    assert streams.grad is not None and torch.isfinite(streams.grad).all()


@pytest.mark.parametrize(
    "dtype",
    [torch.float32, torch.float16, torch.bfloat16],
)
def test_head_dtype_preserved(dtype, head):
    x = torch.randn(B, S, H, D, dtype=dtype)
    out = head(x)
    assert out.dtype == dtype


def test_head_rejects_wrong_h(head):
    with pytest.raises(AssertionError):
        head(torch.randn(B, S, H + 1, D))


def test_head_from_config_defaults():
    cfg = _ConfigLike(d_model=D)
    layer = HyperHead.from_config(cfg)
    assert layer.d_model == D
    assert layer.hc_mult == 4
    assert layer.eps == 1e-4


def test_head_reset_parameters():
    layer = HyperHead(d_model=D, hc_mult=H)
    layer.hc_base.data.fill_(5.0)
    layer.hc_scale.data.fill_(-3.0)
    layer.reset_parameters()
    assert (layer.hc_base == 0).all()
    assert (layer.hc_scale == 1).all()
    assert layer.hc_fn.abs().max().item() <= 0.06 + 1e-6


# ================================================================ Integration

def test_multi_layer_composition_finite_and_grads():
    """Mini-block: two HyperConnection sites + dummy sublayers + a HyperHead.

    Mirrors the integration sketch in HyperConnection's docstring (attention
    then FFN, both sites going through their own HC). Verifies the streams flow
    through unchanged in rank, every HC's parameters receive a gradient, and
    no NaN/Inf escapes the pipeline.
    """
    torch.manual_seed(0)
    hc_att = HyperConnection(d_model=D, hc_mult=H)
    hc_ffn = HyperConnection(d_model=D, hc_mult=H)
    head = HyperHead(d_model=D, hc_mult=H)
    sub_att = nn.Linear(D, D, bias=False)
    sub_ffn = nn.Linear(D, D, bias=False)

    streams = torch.randn(B, S, H, D, requires_grad=True)
    post, comb, collapsed = hc_att(streams)
    sub_out = sub_att(collapsed)
    streams2 = HyperConnection.merge(streams, sub_out, post, comb)

    post, comb, collapsed = hc_ffn(streams2)
    sub_out = sub_ffn(collapsed)
    streams3 = HyperConnection.merge(streams2, sub_out, post, comb)

    final = head(streams3)
    assert final.shape == (B, S, D)
    assert torch.isfinite(final).all()

    final.pow(2).sum().backward()
    for mod_name, mod in (("hc_att", hc_att), ("hc_ffn", hc_ffn), ("head", head)):
        for p_name, p in mod.named_parameters():
            assert p.grad is not None, f"no grad on {mod_name}.{p_name}"
            assert torch.isfinite(p.grad).all(), f"non-finite grad on {mod_name}.{p_name}"


def test_streams_through_stream_embed_then_hc():
    """StreamEmbed → HyperConnection is the typical 'inject H streams at the
    start of the transformer' wiring. Verify the shapes line up."""
    emb = StreamEmbed(num_streams=H, dim=D, channel_first=True,
                      expand_to_streams=True)
    hc = HyperConnection(d_model=D, hc_mult=H)
    token_emb = torch.randn(B, S, D)
    streams = emb(token_emb)
    assert streams.shape == (B, S, H, D)
    post, comb, collapsed = hc(streams)
    assert post.shape == (B, S, H)
    assert comb.shape == (B, S, H, H)
    assert collapsed.shape == (B, S, D)
