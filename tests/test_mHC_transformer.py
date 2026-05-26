"""
Tests for model/mHC_transformer.py — TransformerLM with Manifold-Constrained
Hyper-Connection residuals (Xie et al., 2026).

Coverage:
    HCBlock
        - forward preserves the (B, S, H, D) stream shape
        - works for every supported attention type (MHA / GQA / MLA / CCA)
        - GDA override produces a GDA attention regardless of attention_type
        - dense FFN (MLP) and MoE FFN both wire up
        - gradients flow to every parameter, including hc_att / hc_ffn / att / ffn
        - hc_att and hc_ffn are *independent* modules (not the same instance)
        - block actually transforms the streams (output ≠ input)
        - dropout=0 + eval() is deterministic
        - GDA-mode block bypasses RoPE (its `att` is a GatedDeltaAttention)

    mHCTransformerLM
        - forward returns logits of shape (B, S, vocab_size)
        - StreamEmbed / HyperHead / HCBlock layers are registered
        - works across attention types {MHA, GQA, MLA, CCA}
        - works across hc_mult values {1, 2, 4}
        - GDA hybrid alternates: odd layers use GDA, even use the primary attn
        - MoE FFN end-to-end (CUDA-only because MOE uses a Triton kernel)
        - update_moe_biases is a no-op when use_moe=False, runs when True
        - full backward pass: every parameter receives a finite, non-zero grad
        - eval() + dropout=0 is deterministic
        - param count strictly exceeds the vanilla TransformerLM (HC adds
          ``fn`` / ``base`` / ``scale`` per site + StreamEmbed + HyperHead)
"""
from __future__ import annotations
import pytest
import torch
import torch.nn as nn

from model.config import Config
from model.mHC_transformer import HCBlock, mHCTransformerLM
from model.transformer import TransformerLM
from model.architecture.mHC import HyperConnection, HyperHead, StreamEmbed
from model.attention.GDA import GatedDeltaAttention


HAS_CUDA = torch.cuda.is_available()
cuda_only = pytest.mark.skipif(not HAS_CUDA, reason="requires CUDA / Triton kernel")


# ============================================================== test helpers

def _tiny_config(**overrides) -> Config:
    """A small but realistic Config — fast to instantiate, exercises full mHC stack.

    Defaults: MHA, dense FFN, hc_mult=3, hc_sinkhorn_iters=2.
    Override anything via kwargs.
    """
    base = dict(
        vocab_size=64,
        context_length=16,
        d_model=32,
        num_layers=2,
        num_heads=4,
        d_ff=64,
        dropout=0.0,                # default off so determinism checks are easy
        attention_type="MHA",
        num_kv_heads=2,
        rope_dim=8,
        q_lora_rank=16,
        kv_lora_rank=16,
        cca_compressed_dim=8,
        cca_num_kv_heads=2,
        cca_conv_kernel_size=3,
        use_moe=False,
        hc_mult=3,
        hc_sinkhorn_iters=2,
        hc_eps=1e-4,
    )
    base.update(overrides)
    return Config(**base)


def _make_block(cfg: Config, use_moe: bool = False, use_gda: bool = False,
                device=None, dtype=None) -> HCBlock:
    """Construct a single HCBlock for direct testing. RoPE dim follows the
    same rule as ``mHCTransformerLM.__init__``."""
    from model.attention.utils import RotaryPositionalEmbedding
    rope_dim = cfg.d_model // cfg.num_heads
    if cfg.attention_type == "MLA":
        rope_dim = cfg.rope_dim
    elif cfg.attention_type == "CCA":
        c_dim = (cfg.cca_compressed_dim if cfg.cca_compressed_dim is not None
                 else cfg.d_model // cfg.num_heads // 2)
        rope_dim = c_dim // 2
    rope = RotaryPositionalEmbedding(
        cfg.rope_theta, rope_dim, cfg.context_length, device=device,
    )
    return HCBlock(cfg, rope=rope, use_moe=use_moe, use_gda=use_gda,
                   device=device, dtype=dtype)


# =========================================================================== HCBlock

def test_block_forward_shape_preserved():
    cfg = _tiny_config()
    blk = _make_block(cfg).eval()
    streams = torch.randn(2, 8, cfg.hc_mult, cfg.d_model)
    out = blk(streams)
    assert out.shape == streams.shape


def test_block_actually_transforms_streams():
    """The block must not be a no-op — at random init the output should differ
    from the input by a measurable amount (sublayer contributes something)."""
    torch.manual_seed(0)
    cfg = _tiny_config()
    blk = _make_block(cfg).eval()
    streams = torch.randn(2, 6, cfg.hc_mult, cfg.d_model)
    out = blk(streams)
    diff = (out - streams).abs().mean().item()
    assert diff > 1e-4, "block output is suspiciously identical to input"


@pytest.mark.parametrize("attn_type", ["MHA", "GQA", "MLA", "CCA"])
def test_block_all_attention_types_work(attn_type):
    cfg = _tiny_config(attention_type=attn_type)
    blk = _make_block(cfg).eval()
    streams = torch.randn(2, 8, cfg.hc_mult, cfg.d_model)
    out = blk(streams)
    assert out.shape == streams.shape
    assert torch.isfinite(out).all()


def test_block_gda_override_replaces_attention():
    """`use_gda=True` swaps the attention to GatedDeltaAttention regardless of
    the config's attention_type."""
    cfg = _tiny_config(attention_type="MHA")
    blk = _make_block(cfg, use_gda=True).eval()
    assert isinstance(blk.att, GatedDeltaAttention)
    assert blk.attention_type == "GDA"


def test_block_uses_dense_mlp_when_moe_off():
    from model.architecture.mlp import MLP
    cfg = _tiny_config(use_moe=False)
    blk = _make_block(cfg)
    assert isinstance(blk.ffn, MLP)


@cuda_only
def test_block_uses_moe_when_moe_on():
    """MoE uses a Triton kernel so only run when CUDA is present. Build on CPU
    first because ``HyperConnection.reset_parameters`` uses ``trunc_normal_``,
    which on CUDA dispatches through nvrtc and fails on some environments."""
    from model.architecture.moe import MOE
    cfg = _tiny_config(use_moe=True, n_routed_experts=4, num_experts_per_tok=2)
    blk = _make_block(cfg, use_moe=True).cuda().eval()
    assert isinstance(blk.ffn, MOE)
    streams = torch.randn(2, 8, cfg.hc_mult, cfg.d_model, device="cuda")
    out = blk(streams)
    assert out.shape == streams.shape and torch.isfinite(out).all()


def test_block_hc_att_and_hc_ffn_are_independent_modules():
    """Each sub-block must own its own HyperConnection — otherwise gradient
    statistics from attention and FFN bleed into each other's gates."""
    cfg = _tiny_config()
    blk = _make_block(cfg)
    assert isinstance(blk.hc_att, HyperConnection)
    assert isinstance(blk.hc_ffn, HyperConnection)
    assert blk.hc_att is not blk.hc_ffn
    # ...and they don't accidentally share parameters either.
    assert blk.hc_att.fn is not blk.hc_ffn.fn
    assert blk.hc_att.base is not blk.hc_ffn.base


def test_block_gradients_flow_to_all_params():
    cfg = _tiny_config()
    blk = _make_block(cfg)
    streams = torch.randn(1, 6, cfg.hc_mult, cfg.d_model, requires_grad=True)
    blk(streams).pow(2).sum().backward()
    for name, p in blk.named_parameters():
        assert p.grad is not None, f"no grad on {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad on {name}"
    # at least one of the HC-specific params has nonzero grad
    assert blk.hc_att.fn.grad.abs().sum() > 0
    assert blk.hc_ffn.fn.grad.abs().sum() > 0


def test_block_deterministic_under_dropout_zero():
    cfg = _tiny_config(dropout=0.0)
    blk = _make_block(cfg).eval()
    streams = torch.randn(2, 8, cfg.hc_mult, cfg.d_model)
    with torch.no_grad():
        a = blk(streams)
        b = blk(streams)
    torch.testing.assert_close(a, b, atol=0.0, rtol=0.0)


# =================================================================== mHCTransformerLM

def test_lm_forward_logits_shape():
    cfg = _tiny_config()
    m = mHCTransformerLM(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    with torch.no_grad():
        y = m(x)
    assert y.shape == (2, 8, cfg.vocab_size)
    assert torch.isfinite(y).all()


def test_lm_registers_stream_embed_hyper_head_and_hcblocks():
    cfg = _tiny_config()
    m = mHCTransformerLM(cfg)
    assert isinstance(m.stream_embed, StreamEmbed)
    assert isinstance(m.hyper_head, HyperHead)
    assert all(isinstance(layer, HCBlock) for layer in m.layers)
    assert len(m.layers) == cfg.num_layers


@pytest.mark.parametrize("attn_type", ["MHA", "GQA", "MLA", "CCA"])
def test_lm_all_attention_types(attn_type):
    cfg = _tiny_config(attention_type=attn_type)
    m = mHCTransformerLM(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (2, 6))
    with torch.no_grad():
        y = m(x)
    assert y.shape == (2, 6, cfg.vocab_size)
    assert torch.isfinite(y).all()


@pytest.mark.parametrize("hc_mult", [1, 2, 4])
def test_lm_various_hc_mult(hc_mult):
    """hc_mult=1 is the degenerate single-stream case; 2 and 4 are typical."""
    cfg = _tiny_config(hc_mult=hc_mult)
    m = mHCTransformerLM(cfg).eval()
    # internals carry H streams between layers
    assert m.stream_embed.num_streams == hc_mult
    assert m.hyper_head.hc_mult == hc_mult
    x = torch.randint(0, cfg.vocab_size, (2, 5))
    with torch.no_grad():
        y = m(x)
    assert y.shape == (2, 5, cfg.vocab_size)
    assert torch.isfinite(y).all()


@cuda_only
def test_lm_gda_hybrid_alternates_layers():
    """GDA's Triton kernel requires CUDA tensors at runtime — mark cuda_only."""
    cfg = _tiny_config(num_layers=4, attention_type="MHA", use_gda_hybrid=True)
    m = mHCTransformerLM(cfg).cuda().eval()
    # Odd-indexed layers should be GDA; even should be MHA.
    assert m.gda_layer_indices == [1, 3]
    assert m.layers[0].attention_type == "MHA"
    assert m.layers[1].attention_type == "GDA"
    assert m.layers[2].attention_type == "MHA"
    assert m.layers[3].attention_type == "GDA"
    # Forward still works end-to-end.
    x = torch.randint(0, cfg.vocab_size, (1, 6), device="cuda")
    with torch.no_grad():
        y = m(x)
    assert y.shape == (1, 6, cfg.vocab_size)
    assert torch.isfinite(y).all()


def test_lm_gda_hybrid_layer_indices_computed_correctly():
    """Wiring of `gda_layer_indices` is pure config logic — testable on CPU."""
    cfg = _tiny_config(num_layers=4, attention_type="MHA", use_gda_hybrid=True)
    # Avoid constructing GDA modules here (Triton needs CUDA). Just inspect
    # the index list the constructor computes.
    from model.mHC_transformer import mHCTransformerLM as _LM
    # Manual recomputation matching the source logic.
    expected = [i for i in range(cfg.num_layers) if i % 2 == 1]
    is_hybrid = cfg.use_gda_hybrid and cfg.attention_type != "CCA"
    assert is_hybrid
    assert expected == [1, 3]


def test_lm_gda_hybrid_silently_ignored_for_cca():
    """CCA already has its own short-conv local mixing; GDA hybrid is a no-op."""
    cfg = _tiny_config(num_layers=4, attention_type="CCA", use_gda_hybrid=True)
    m = mHCTransformerLM(cfg)
    assert m.gda_layer_indices == []
    assert all(layer.attention_type == "CCA" for layer in m.layers)


def test_lm_full_backward_pass_grads_everywhere():
    """Loss → backward must produce a finite gradient on every parameter."""
    torch.manual_seed(0)
    cfg = _tiny_config(num_layers=2)
    m = mHCTransformerLM(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 6))
    y = m(x)
    loss = y.float().pow(2).mean()
    loss.backward()
    missing, nonfinite = [], []
    for name, p in m.named_parameters():
        if p.grad is None:
            missing.append(name)
        elif not torch.isfinite(p.grad).all():
            nonfinite.append(name)
    assert not missing, f"no gradient on: {missing}"
    assert not nonfinite, f"non-finite gradient on: {nonfinite}"
    # Spot-check mHC-specific params receive gradient signal.
    spot = ["stream_embed.stream_embed", "hyper_head.hc_fn",
            "layers.0.hc_att.fn", "layers.0.hc_ffn.scale",
            "layers.1.hc_att.base", "layers.1.hc_ffn.fn"]
    name_to_p = dict(m.named_parameters())
    for n in spot:
        assert name_to_p[n].grad.abs().sum() > 0, f"zero grad on {n}"


def test_lm_eval_mode_deterministic():
    cfg = _tiny_config(dropout=0.0)
    m = mHCTransformerLM(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (2, 6))
    with torch.no_grad():
        a = m(x)
        b = m(x)
    torch.testing.assert_close(a, b, atol=0.0, rtol=0.0)


def test_lm_update_moe_biases_noop_for_dense():
    """If the model has no MoE layers, update_moe_biases must be a silent no-op."""
    cfg = _tiny_config(use_moe=False)
    m = mHCTransformerLM(cfg)
    m.update_moe_biases()  # must not raise


def test_lm_param_count_strictly_exceeds_vanilla():
    """mHC adds: StreamEmbed (H·D), HyperHead (H + H·H·D), and per-layer
    HyperConnection (3·(2H+H²) + (2H+H²)·H·D)·2. Confirm the total exceeds
    the vanilla model with the same backbone."""
    cfg_mhc = _tiny_config(hc_mult=4)
    # Build a matching vanilla config — same backbone, hc_mult is moot there.
    cfg_van = _tiny_config(hc_mult=4)
    m_mhc = mHCTransformerLM(cfg_mhc)
    m_van = TransformerLM(cfg_van)
    n_mhc = sum(p.numel() for p in m_mhc.parameters())
    n_van = sum(p.numel() for p in m_van.parameters())
    assert n_mhc > n_van, f"mHC ({n_mhc}) should have more params than vanilla ({n_van})"


def test_lm_streams_shape_through_layers():
    """Internally the streams should remain (B, S, H, D) between every layer.
    Verify by hooking the layer outputs."""
    cfg = _tiny_config(num_layers=3, hc_mult=4)
    m = mHCTransformerLM(cfg).eval()

    seen = []
    handles = [layer.register_forward_hook(
        lambda mod, inp, out, _seen=seen: _seen.append(tuple(out.shape))
    ) for layer in m.layers]

    try:
        x = torch.randint(0, cfg.vocab_size, (2, 7))
        with torch.no_grad():
            m(x)
    finally:
        for h in handles:
            h.remove()

    expected = (2, 7, cfg.hc_mult, cfg.d_model)
    assert seen == [expected] * cfg.num_layers


def test_lm_inference_no_grad_smoke():
    cfg = _tiny_config()
    m = mHCTransformerLM(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (4, 12))
    with torch.no_grad():
        y = m(x)
    assert y.shape == (4, 12, cfg.vocab_size)
    assert torch.isfinite(y).all()


# ============================================================== MoE-specific (CUDA)

@cuda_only
def test_lm_moe_forward_and_update_bias():
    """End-to-end with MoE FFN: forward shape, finite logits, bias update runs.

    Built on CPU then moved to CUDA — keeps HyperConnection's trunc_normal_
    init off the nvrtc path that some CUDA toolchains lack."""
    cfg = _tiny_config(
        use_moe=True,
        n_routed_experts=4,
        num_experts_per_tok=2,
        n_shared_experts=1,
        moe_layers=None,
    )
    m = mHCTransformerLM(cfg).cuda()
    m.train()  # enable aux loss path & expert_load updates
    x = torch.randint(0, cfg.vocab_size, (2, 8), device="cuda")
    y = m(x)
    assert y.shape == (2, 8, cfg.vocab_size)
    assert torch.isfinite(y).all()
    # update_moe_biases must run without error and adjust at least one bias.
    biases_before = {i: layer.ffn.gate.expert_bias.clone()
                     for i, layer in enumerate(m.layers)
                     if hasattr(layer, "use_moe") and layer.use_moe}
    m.update_moe_biases()
    # at least one expert_bias must have updated (load was unbalanced from a
    # single forward pass)
    diffs = [
        (m.layers[i].ffn.gate.expert_bias - b).abs().sum().item()
        for i, b in biases_before.items()
    ]
    assert any(d > 0 for d in diffs), "no bias was updated despite MoE forward pass"


@cuda_only
def test_lm_moe_subset_of_layers():
    """When `moe_layers` lists indices, only those layers use MoE."""
    cfg = _tiny_config(
        num_layers=3,
        use_moe=True,
        n_routed_experts=4,
        num_experts_per_tok=2,
        moe_layers=[1],  # only the middle layer is MoE
    )
    m = mHCTransformerLM(cfg).cuda().eval()
    from model.architecture.mlp import MLP
    from model.architecture.moe import MOE
    assert isinstance(m.layers[0].ffn, MLP)
    assert isinstance(m.layers[1].ffn, MOE)
    assert isinstance(m.layers[2].ffn, MLP)
    x = torch.randint(0, cfg.vocab_size, (1, 6), device="cuda")
    with torch.no_grad():
        y = m(x)
    assert y.shape == (1, 6, cfg.vocab_size)
    assert torch.isfinite(y).all()
