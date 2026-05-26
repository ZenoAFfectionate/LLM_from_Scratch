"""
Tests for model/architecture/* — feed-forward, MoE, short conv, kernels.

Coverage:
    mlp.py:
        - Linear        (output shape, weight loading, dtype check, error on bad shape)
        - silu          (matches torch.nn.functional.silu)
        - MLP (SwiGLU)  (forward shape, gradients, zero-weight degeneracy)

    moe.py:
        - Gate          (output shapes, topk distribution, aux loss training-only)
        - MOE           (forward shape, output identical to dense MLP under deterministic routing)

    mHC.py:
        - StreamEmbed   (output shape changes batch by num_streams; learnable embedding)

    kernels.py:
        - segment_reduce_weighted / fused_scatter_add_weighted
                        (matches a pure-PyTorch reference up to fp16 tolerance)

    utils.ShortConv:
        - forward shape preserved, causal mask removes future leakage.
"""
import math
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.architecture.mlp import Linear, MLP, silu
from model.utils import ShortConv

HAS_CUDA = torch.cuda.is_available()
cuda_only = pytest.mark.skipif(not HAS_CUDA, reason="requires CUDA / Triton kernel")


# ===================================================================== Linear

def test_linear_forward_shape():
    layer = Linear(8, 5)
    x = torch.randn(2, 3, 8)
    y = layer(x)
    assert y.shape == (2, 3, 5)


def test_linear_matches_matmul():
    layer = Linear(8, 5)
    x = torch.randn(4, 8)
    y = layer(x)
    expected = x @ layer.weight.t()
    torch.testing.assert_close(y, expected, atol=1e-6, rtol=1e-5)


def test_linear_rejects_mismatched_last_dim():
    layer = Linear(8, 5)
    with pytest.raises(RuntimeError):
        layer(torch.randn(2, 7))


def test_linear_weight_dtype_is_float32_by_default():
    layer = Linear(4, 4)
    assert layer.weight.dtype == torch.float32


# ====================================================================== silu

def test_silu_matches_torch():
    x = torch.randn(5, 7) * 3.0
    torch.testing.assert_close(silu(x), F.silu(x), atol=1e-6, rtol=1e-5)


def test_silu_zero_zero():
    """silu(0) = 0 * sigmoid(0) = 0."""
    x = torch.zeros(3, 4)
    assert (silu(x) == 0).all()


# ====================================================================== MLP

def test_mlp_forward_shape():
    layer = MLP(d_model=16, d_ff=32)
    x = torch.randn(2, 5, 16)
    y = layer(x)
    assert y.shape == x.shape


def test_mlp_gradients_flow_to_all_weights():
    layer = MLP(d_model=8, d_ff=16)
    x = torch.randn(1, 3, 8)
    layer(x).pow(2).sum().backward()
    for n, p in layer.named_parameters():
        assert p.grad is not None, f"no grad for {n}"
        assert torch.isfinite(p.grad).all()


def test_mlp_zero_w2_makes_output_zero():
    """If the down-projection w2 is zero, output is identically zero."""
    layer = MLP(d_model=8, d_ff=16)
    with torch.no_grad():
        layer.w2.weight.zero_()
        if layer.w2.bias is not None:
            layer.w2.bias.zero_()
    out = layer(torch.randn(1, 4, 8))
    assert torch.equal(out, torch.zeros_like(out))


def test_mlp_uses_fused_swiglu_projection():
    """MLP in this codebase fuses w1 and w3 into a single Linear of out=2*d_ff."""
    d_model, d_ff = 8, 16
    layer = MLP(d_model, d_ff)
    # w1: (2*d_ff, d_model), w2: (d_model, d_ff)
    assert layer.w1.weight.shape == (2 * d_ff, d_model)
    assert layer.w2.weight.shape == (d_model, d_ff)


def test_mlp_default_matches_unclamped_path():
    """swiglu_limit=0 (default) must reproduce the previous unclamped behavior bit-exactly."""
    torch.manual_seed(0)
    layer = MLP(d_model=8, d_ff=16)
    x = torch.randn(2, 3, 8)
    # Manual SwiGLU reference using the same fused weights.
    gate, up = layer.w1(x).chunk(2, dim=-1)
    expected = layer.w2(F.silu(gate) * up)
    torch.testing.assert_close(layer(x), expected, atol=0.0, rtol=0.0)


def test_mlp_swiglu_clamp_bounds_intermediate_activations():
    """With swiglu_limit > 0, the clamped pre-activations must respect the bounds."""
    limit = 0.5
    torch.manual_seed(0)
    layer = MLP(d_model=8, d_ff=16, swiglu_limit=limit)
    # Blow up the gate/up projection so unclamped activations clearly exceed `limit`.
    with torch.no_grad():
        layer.w1.weight.mul_(50.0)
    x = torch.randn(4, 8)
    gate_raw, up_raw = layer.w1(x).chunk(2, dim=-1)
    # Sanity: at least some raw activations exceed the limit, so the clamp is exercised.
    assert (up_raw.abs() > limit).any() and (gate_raw > limit).any()
    # Reproduce the layer's internal clamped product and confirm bounds.
    up_clamped = torch.clamp(up_raw, min=-limit, max=limit)
    gate_clamped = torch.clamp(gate_raw, max=limit)
    assert up_clamped.abs().max().item() <= limit + 1e-6
    assert gate_clamped.max().item() <= limit + 1e-6
    # End-to-end output must equal the manually clamped reference.
    expected = layer.w2(F.silu(gate_clamped) * up_clamped)
    torch.testing.assert_close(layer(x), expected, atol=0.0, rtol=0.0)


def test_mlp_routing_weights_scale_output_linearly():
    """Passing routing weights in forward must scale the pre-down activations."""
    torch.manual_seed(0)
    layer = MLP(d_model=8, d_ff=16)
    x = torch.randn(2, 4, 8)
    # Scalar weight broadcast — compare against manual reference (w2 has bias,
    # so the layer is affine, not linear; verify against the exact computation).
    s = 0.25
    gate, up = layer.w1(x).chunk(2, dim=-1)
    expected_scalar = layer.w2(F.silu(gate) * up * s)
    torch.testing.assert_close(
        layer(x, weights=torch.tensor(s)), expected_scalar, atol=1e-6, rtol=1e-5
    )
    # Per-token weight broadcast over the hidden dim.
    w = torch.rand(2, 4, 1)
    expected = layer.w2(F.silu(gate) * up * w)
    torch.testing.assert_close(layer(x, weights=w), expected, atol=1e-6, rtol=1e-5)


def test_mlp_weights_none_is_identity():
    """weights=None must be bit-identical to omitting the argument."""
    torch.manual_seed(0)
    layer = MLP(d_model=8, d_ff=16)
    x = torch.randn(2, 4, 8)
    torch.testing.assert_close(layer(x), layer(x, weights=None), atol=0.0, rtol=0.0)


# ====================================================================== MoE

def test_gate_output_shapes():
    from model.architecture.moe import Gate
    n_exp, top_k = 4, 2
    gate = Gate(hidden_size=8, n_routed_experts=n_exp, num_experts_per_tok=top_k)
    x = torch.randn(2, 5, 8)
    gate.train()
    topk_idx, topk_w, aux_loss, counts = gate(x)
    # 2 * 5 = 10 tokens, each picks top_k experts
    assert topk_idx.shape == (10, top_k)
    assert topk_w.shape == (10, top_k)
    assert counts.shape == (n_exp,)
    assert counts.sum().item() == 10 * top_k


def test_gate_indices_in_range_and_unique_per_token():
    from model.architecture.moe import Gate
    n_exp, top_k = 6, 3
    gate = Gate(hidden_size=8, n_routed_experts=n_exp, num_experts_per_tok=top_k)
    topk_idx, _, _, _ = gate(torch.randn(1, 7, 8))
    assert int(topk_idx.min()) >= 0 and int(topk_idx.max()) < n_exp
    # Each token's top-k indices must be unique (no expert picked twice)
    for row in topk_idx:
        assert row.unique().numel() == top_k


def test_gate_aux_loss_only_in_training():
    from model.architecture.moe import Gate
    gate = Gate(hidden_size=8, n_routed_experts=3, num_experts_per_tok=1,
                aux_seq_loss_alpha=0.01)
    gate.eval()
    _, _, aux, counts = gate(torch.randn(1, 5, 8))
    assert aux is None and counts is None
    gate.train()
    _, _, aux, counts = gate(torch.randn(1, 5, 8))
    assert aux is not None and counts is not None


def test_gate_weights_renormalised_for_topk_gt_1():
    """When top_k > 1 the chosen weights must sum to ~1 along the last dim."""
    from model.architecture.moe import Gate
    gate = Gate(hidden_size=8, n_routed_experts=4, num_experts_per_tok=2)
    _, topk_w, _, _ = gate(torch.randn(1, 6, 8))
    sums = topk_w.sum(dim=-1)
    torch.testing.assert_close(sums, torch.ones_like(sums), atol=1e-5, rtol=1e-5)


def test_gate_update_bias_balances_load():
    """update_bias should pull bias *down* for over-utilised experts."""
    from model.architecture.moe import Gate
    gate = Gate(hidden_size=8, n_routed_experts=4, num_experts_per_tok=1,
                bias_update_speed=0.1)
    gate.train()
    # Mock an imbalanced load: expert 0 got way too many tokens, expert 3 got zero
    gate.expert_load = torch.tensor([20, 4, 4, 0])
    initial_bias = gate.expert_bias.clone()
    gate.update_bias(total_tokens=28)
    # Expert 0 was overloaded → its bias should decrease.
    # Expert 3 was underloaded → its bias should increase.
    assert gate.expert_bias[0] < initial_bias[0]
    assert gate.expert_bias[3] > initial_bias[3]


@cuda_only
def test_moe_forward_shape():
    """MOE uses a Triton segment-reduce kernel under the hood — requires CUDA."""
    from model.architecture.moe import MOE
    layer = MOE(
        d_model=16, d_ff=32,
        n_routed_experts=4, num_experts_per_tok=2,
        n_shared_experts=0,
        device="cuda", dtype=torch.float32,
    ).cuda()
    x = torch.randn(2, 8, 16, device="cuda")
    y = layer(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


@cuda_only
def test_moe_shared_expert_contributes():
    """With n_shared_experts > 0, the output must change vs the no-shared case
    on the same router decisions."""
    from model.architecture.moe import MOE
    torch.manual_seed(0)
    layer_no_shared = MOE(
        d_model=16, d_ff=32,
        n_routed_experts=4, num_experts_per_tok=2, n_shared_experts=0,
        device="cuda", dtype=torch.float32,
    ).cuda()
    torch.manual_seed(0)
    layer_with_shared = MOE(
        d_model=16, d_ff=32,
        n_routed_experts=4, num_experts_per_tok=2, n_shared_experts=1,
        device="cuda", dtype=torch.float32,
    ).cuda()
    x = torch.randn(1, 4, 16, device="cuda")
    y_no = layer_no_shared(x)
    y_yes = layer_with_shared(x)
    # outputs must differ
    assert not torch.allclose(y_no, y_yes, atol=1e-5)


# ============================================== DeepSeek-V4 MoE gate features

def test_gate_default_score_func_is_v4_sqrtsoftplus():
    """Default gating should be DeepSeek-V4 (sqrt(softplus))."""
    from model.architecture.moe import Gate
    gate = Gate(hidden_size=8, n_routed_experts=4, num_experts_per_tok=2)
    assert gate.score_func == "sqrtsoftplus"
    assert gate.route_scale == 1.0
    # SUPPORTED_SCORE_FUNCS contract preserved
    assert "softmax" in gate.SUPPORTED_SCORE_FUNCS
    assert "sigmoid" in gate.SUPPORTED_SCORE_FUNCS
    assert "sqrtsoftplus" in gate.SUPPORTED_SCORE_FUNCS


def test_gate_score_function_matches_formula():
    """_apply_score_function must implement the three DeepSeek formulas exactly."""
    from model.architecture.moe import Gate
    logits = torch.randn(7, 5) * 2.0
    gate_sm = Gate(hidden_size=8, n_routed_experts=5, num_experts_per_tok=1, score_func="softmax")
    gate_sg = Gate(hidden_size=8, n_routed_experts=5, num_experts_per_tok=1, score_func="sigmoid")
    gate_sp = Gate(hidden_size=8, n_routed_experts=5, num_experts_per_tok=1, score_func="sqrtsoftplus")
    torch.testing.assert_close(gate_sm._apply_score_function(logits), F.softmax(logits, dim=-1))
    torch.testing.assert_close(gate_sg._apply_score_function(logits), torch.sigmoid(logits))
    torch.testing.assert_close(gate_sp._apply_score_function(logits), F.softplus(logits).sqrt())


def test_gate_rejects_unknown_score_func():
    """Unknown score_func must raise ValueError listing supported options."""
    from model.architecture.moe import Gate
    with pytest.raises(ValueError, match="Unsupported score_func"):
        Gate(hidden_size=8, n_routed_experts=4, num_experts_per_tok=1,
             score_func="gelu")


@pytest.mark.parametrize("score_func", ["softmax", "sigmoid", "sqrtsoftplus"])
def test_gate_forward_works_for_all_score_funcs(score_func):
    """All three score functions must produce finite, shape-correct outputs."""
    from model.architecture.moe import Gate
    n_exp, top_k = 4, 2
    gate = Gate(hidden_size=8, n_routed_experts=n_exp, num_experts_per_tok=top_k,
                score_func=score_func)
    gate.train()
    topk_idx, topk_w, _, counts = gate(torch.randn(2, 5, 8))
    assert topk_idx.shape == (10, top_k) and topk_w.shape == (10, top_k)
    assert torch.isfinite(topk_w).all()
    assert int(topk_idx.min()) >= 0 and int(topk_idx.max()) < n_exp
    # Indices are valid expert IDs; counts cover the right total.
    assert counts.sum().item() == 10 * top_k


def test_gate_non_softmax_weights_normalized_in_topk():
    """sigmoid / sqrtsoftplus paths must yield per-token weight sum = 1 (route_scale=1)."""
    from model.architecture.moe import Gate
    for sf in ("sigmoid", "sqrtsoftplus"):
        gate = Gate(hidden_size=8, n_routed_experts=4, num_experts_per_tok=2, score_func=sf)
        _, topk_w, _, _ = gate(torch.randn(3, 6, 8))
        sums = topk_w.sum(dim=-1)
        torch.testing.assert_close(sums, torch.ones_like(sums), atol=1e-5, rtol=1e-5)


def test_gate_softmax_path_skips_renormalization():
    """For softmax, gathered top-k weights are raw softmax probabilities (sum < 1 in general)."""
    from model.architecture.moe import Gate
    torch.manual_seed(123)
    gate = Gate(hidden_size=8, n_routed_experts=6, num_experts_per_tok=2, score_func="softmax")
    _, topk_w, _, _ = gate(torch.randn(4, 4, 8))
    # top-2 of a 6-way softmax: sum should be in (0, 1] but virtually never 1.
    sums = topk_w.sum(dim=-1)
    assert (sums <= 1.0 + 1e-5).all()
    assert (sums < 1.0).any()


def test_gate_route_scale_multiplies_weights():
    """route_scale must scale the (normalized) top-k weights linearly."""
    from model.architecture.moe import Gate
    torch.manual_seed(7)
    x = torch.randn(1, 5, 8)
    gate_one = Gate(hidden_size=8, n_routed_experts=4, num_experts_per_tok=2,
                    score_func="sqrtsoftplus", route_scale=1.0)
    gate_two = Gate(hidden_size=8, n_routed_experts=4, num_experts_per_tok=2,
                    score_func="sqrtsoftplus", route_scale=2.5)
    # Use the same weights so routing decisions match exactly.
    with torch.no_grad():
        gate_two.weight.copy_(gate_one.weight)
    _, w1, _, _ = gate_one(x)
    _, w2, _, _ = gate_two(x)
    torch.testing.assert_close(w2, w1 * 2.5, atol=1e-6, rtol=1e-5)


def test_gate_bias_added_to_activated_scores_not_logits():
    """V4 semantics: large negative bias on expert k must push k out of top-k
    even when its raw logits are largest, because bias acts in activated-score range."""
    from model.architecture.moe import Gate
    gate = Gate(hidden_size=4, n_routed_experts=3, num_experts_per_tok=1, score_func="sqrtsoftplus")
    # Construct a hidden so that logits clearly prefer expert 0.
    with torch.no_grad():
        gate.weight.zero_()
        gate.weight[0] = torch.tensor([1.0, 0.0, 0.0, 0.0])
        gate.weight[1] = torch.tensor([0.5, 0.0, 0.0, 0.0])
        gate.weight[2] = torch.tensor([0.1, 0.0, 0.0, 0.0])
    x = torch.tensor([[[10.0, 0.0, 0.0, 0.0]]])  # logits ≈ [10, 5, 1]
    # No bias: expert 0 wins.
    topk_idx, _, _, _ = gate(x)
    assert topk_idx.flatten().tolist() == [0]
    # Heavily penalize expert 0 in the *activated* score range to flip the choice.
    with torch.no_grad():
        gate.expert_bias[0] = -10.0
    topk_idx, _, _, _ = gate(x)
    assert topk_idx.flatten().tolist() != [0], "Negative bias on activated scores must demote expert 0"


def test_gate_weights_use_unbiased_scores():
    """The returned routing weights must be derived from the *un-biased* activated scores,
    so that adding a uniform positive bias to all experts does NOT change the weights
    (it can only change which experts get selected, not their weight magnitudes)."""
    from model.architecture.moe import Gate
    torch.manual_seed(0)
    gate = Gate(hidden_size=8, n_routed_experts=4, num_experts_per_tok=2, score_func="sqrtsoftplus")
    x = torch.randn(1, 5, 8)
    _, w_a, _, _ = gate(x)
    with torch.no_grad():
        gate.expert_bias += 5.0  # uniform shift → preserves ordering → same indices
    _, w_b, _, _ = gate(x)
    torch.testing.assert_close(w_a, w_b, atol=1e-6, rtol=1e-5)


def test_gate_sqrtsoftplus_outputs_nonnegative_and_finite():
    """sqrt(softplus(x)) is well-defined (>0) for any finite logit — sanity for very negative inputs."""
    from model.architecture.moe import Gate
    gate = Gate(hidden_size=4, n_routed_experts=4, num_experts_per_tok=1, score_func="sqrtsoftplus")
    out = gate._apply_score_function(torch.tensor([[-1e3, -1.0, 0.0, 1e3]]))
    assert torch.isfinite(out).all()
    assert (out >= 0).all()


@cuda_only
@pytest.mark.parametrize("score_func", ["softmax", "sigmoid", "sqrtsoftplus"])
def test_moe_forward_works_for_all_score_funcs(score_func):
    """End-to-end MOE forward must succeed for each V2/V3/V4 score function."""
    from model.architecture.moe import MOE
    layer = MOE(
        d_model=16, d_ff=32,
        n_routed_experts=4, num_experts_per_tok=2, n_shared_experts=1,
        score_func=score_func, route_scale=1.0,
        device="cuda", dtype=torch.float32,
    ).cuda()
    x = torch.randn(2, 8, 16, device="cuda")
    y = layer(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


@cuda_only
def test_moe_forward_route_scale_changes_output():
    """route_scale != 1.0 must scale the routed contribution proportionally."""
    from model.architecture.moe import MOE
    torch.manual_seed(0)
    layer_a = MOE(
        d_model=16, d_ff=32,
        n_routed_experts=4, num_experts_per_tok=2, n_shared_experts=0,
        score_func="sqrtsoftplus", route_scale=1.0,
        device="cuda", dtype=torch.float32,
    ).cuda()
    torch.manual_seed(0)
    layer_b = MOE(
        d_model=16, d_ff=32,
        n_routed_experts=4, num_experts_per_tok=2, n_shared_experts=0,
        score_func="sqrtsoftplus", route_scale=2.0,
        device="cuda", dtype=torch.float32,
    ).cuda()
    x = torch.randn(1, 4, 16, device="cuda")
    y_a = layer_a(x)
    y_b = layer_b(x)
    # Same routing decisions but doubled weight on routed experts → expect ~2x.
    torch.testing.assert_close(y_b, 2.0 * y_a, atol=1e-4, rtol=1e-4)


def test_moe_propagates_v4_kwargs_to_gate():
    """MOE.__init__ must thread score_func / route_scale through to the Gate instance."""
    from model.architecture.moe import MOE
    layer = MOE(
        d_model=16, d_ff=32,
        n_routed_experts=4, num_experts_per_tok=2,
        n_shared_experts=0,
        score_func="sigmoid", route_scale=2.5,
    )
    assert layer.score_func == "sigmoid"
    assert layer.route_scale == 2.5
    assert layer.gate.score_func == "sigmoid"
    assert layer.gate.route_scale == 2.5


# =============================================================== StreamEmbed

def test_stream_embed_shape_round_trip():
    """Without expand_to_streams: input (b*s, ..., d) -> (b*s, ..., d)."""
    from model.architecture.mHC import StreamEmbed
    layer = StreamEmbed(num_streams=4, dim=8, channel_first=False,
                        expand_to_streams=False)
    x = torch.randn(8, 5, 8)  # 8 = 2 batches × 4 streams
    y = layer(x)
    assert y.shape == x.shape


def test_stream_embed_expansion():
    """expand_to_streams duplicates batch dim by num_streams."""
    from model.architecture.mHC import StreamEmbed
    layer = StreamEmbed(num_streams=4, dim=6, channel_first=False,
                        expand_to_streams=True)
    x = torch.randn(2, 3, 6)
    y = layer(x)
    assert y.shape == (8, 3, 6)


def test_stream_embed_is_learnable():
    from model.architecture.mHC import StreamEmbed
    layer = StreamEmbed(num_streams=3, dim=4)
    assert isinstance(layer.stream_embed, nn.Parameter)
    x = torch.randn(6, 4)
    y = layer(x)
    y.sum().backward()
    assert layer.stream_embed.grad is not None
    assert layer.stream_embed.grad.abs().sum() > 0


# =================================================== utils.ShortConv

def test_shortconv_forward_shape_preserved():
    layer = ShortConv(hidden_size=8, kernel_size=4, hc_mult=2)
    x = torch.randn(1, 6, 2, 8)              # (B, T, G=hc_mult, C)
    y = layer(x)
    assert y.shape == x.shape


def test_shortconv_is_causal():
    """Perturbing only future tokens (>= t+1) must not change output at position t."""
    layer = ShortConv(hidden_size=4, kernel_size=4, hc_mult=2,
                      activation=False).eval()
    T = 10
    x = torch.randn(1, T, 2, 4)
    with torch.no_grad():
        y_a = layer(x)
        x2 = x.clone()
        x2[:, T // 2 + 1:] += torch.randn_like(x2[:, T // 2 + 1:])  # future perturb
        y_b = layer(x2)
    # past (up to T//2) must be identical
    torch.testing.assert_close(y_a[:, : T // 2 + 1], y_b[:, : T // 2 + 1],
                               atol=1e-5, rtol=1e-5)


def test_shortconv_groups_validated():
    layer = ShortConv(hidden_size=4, kernel_size=3, hc_mult=2)
    # input has 3 groups but layer expects 2 → assertion error
    with pytest.raises(AssertionError):
        layer(torch.randn(1, 5, 3, 4))


# ============================================== Triton kernel: segment_reduce_weighted

@cuda_only
def test_segment_reduce_weighted_matches_pytorch_ref():
    """segment_reduce_weighted must match a brute-force PyTorch implementation."""
    from model.architecture.kernels import segment_reduce_weighted
    n_tokens, top_k, d_model = 6, 3, 32
    expert_out = torch.randn(n_tokens * top_k, d_model, device="cuda")
    weights = torch.rand(n_tokens * top_k, device="cuda")
    y = segment_reduce_weighted(expert_out, weights, n_tokens, top_k)

    # Reference: weight each row, then sum every top_k rows.
    ref = (expert_out * weights[:, None]).view(n_tokens, top_k, d_model).sum(dim=1)
    torch.testing.assert_close(y, ref, atol=1e-3, rtol=1e-3)


@cuda_only
def test_fused_scatter_add_weighted_resorts_correctly():
    """fused_scatter_add_weighted handles unsorted target indices via argsort."""
    from model.architecture.kernels import fused_scatter_add_weighted
    n_tokens, top_k, d_model = 4, 2, 16
    n_sorted = n_tokens * top_k
    expert_out = torch.randn(n_sorted, d_model, device="cuda")
    weights = torch.rand(n_sorted, device="cuda")
    # token indices in random order: each token appears exactly top_k times
    sorted_token_idx = torch.tensor([2, 0, 1, 2, 3, 1, 3, 0],
                                    dtype=torch.int64, device="cuda")
    y = fused_scatter_add_weighted(
        expert_out, sorted_token_idx, weights, n_tokens, top_k,
    )
    # Reference: accumulate manually
    ref = torch.zeros(n_tokens, d_model, device="cuda")
    for i in range(n_sorted):
        ref[sorted_token_idx[i]] += expert_out[i] * weights[i]
    torch.testing.assert_close(y, ref, atol=1e-3, rtol=1e-3)
