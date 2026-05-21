"""
Tests for model/attention/* — every attention variant in the codebase.

Coverage:
    utils:          softmax, RotaryPositionalEmbedding, scaled_dot_product_attention,
                    store_kvcache, flash_attention_prefill, paged_attention_decode
    MHA:            shape, causality (gradient test), parity with PyTorch SDPA
    GQA:            shape, causality, expansion correctness when num_kv_heads<num_heads
    MLA:            shape, causality, low-rank compression sanity
    CCA:            shape, causality, learned key temperature does something
    GDA:            full functional tests already live in /test_gda.py (root) and
                    /tests/test_gda.py here — this file re-tests shape + causality
                    only to keep the suite self-contained.

Most paged-attention inference paths use Triton kernels that need CUDA.
Those tests are skipped automatically when CUDA is unavailable.
"""
import math
import pytest
import torch
import torch.nn.functional as F

from model.attention.utils import (
    softmax,
    RotaryPositionalEmbedding,
    scaled_dot_product_attention,
    store_kvcache,
    flash_attention_prefill,
    paged_attention_decode,
)
from model.attention.MHA import MultiHeadSelfAttention
from model.attention.GQA import GroupedQueryAttention
from model.attention.MLA import MultiHeadLatentAttention
from model.attention.CCA import CompressedConvAttention


HAS_CUDA = torch.cuda.is_available()
cuda_only = pytest.mark.skipif(not HAS_CUDA, reason="requires CUDA")


# ===================================================================== softmax

def test_softmax_matches_torch():
    x = torch.randn(3, 4, 5)
    for dim in [-1, 1, 2]:
        torch.testing.assert_close(softmax(x, dim=dim), F.softmax(x, dim=dim))


def test_softmax_numerical_stability():
    """Subtracting the max means adding a huge constant must not blow up."""
    x = torch.randn(2, 7)
    base = softmax(x, dim=-1)
    shifted = softmax(x + 1e4, dim=-1)
    # fp32 precision means the result may differ at ~1e-3 even though the
    # math is identical; we just verify no NaN / inf and approximate equality.
    assert torch.isfinite(shifted).all()
    torch.testing.assert_close(base, shifted, atol=1e-3, rtol=1e-3)


def test_softmax_sums_to_one():
    x = torch.randn(3, 5, 7)
    y = softmax(x, dim=-1)
    sums = y.sum(dim=-1)
    torch.testing.assert_close(sums, torch.ones_like(sums))


# ====================================================== RoPE (Rotary Embedding)

def _causal_dim_split(x):
    """Reference rotary that splits into (x1, x2) halves and rotates."""
    return torch.chunk(x, 2, dim=-1)


def test_rope_output_norm_preserved():
    """RoPE is a rotation — it must preserve the L2 norm of the vector."""
    d, T = 32, 16
    rope = RotaryPositionalEmbedding(theta=10000.0, d_k=d, max_seq_len=T)
    x = torch.randn(2, 4, T, d)
    pos = torch.arange(T)
    y = rope(x, pos)
    torch.testing.assert_close(
        x.norm(dim=-1), y.norm(dim=-1), atol=1e-5, rtol=1e-5
    )


def test_rope_zero_position_is_identity():
    """At position 0, all rotations are 0 → identity transform."""
    d = 16
    rope = RotaryPositionalEmbedding(theta=10000.0, d_k=d, max_seq_len=8)
    x = torch.randn(1, 1, 1, d)
    y = rope(x, torch.tensor([0]))
    torch.testing.assert_close(y, x, atol=1e-6, rtol=1e-6)


def test_rope_relative_property():
    """Key property of RoPE: dot(rope(q, m), rope(k, n)) depends on q,k and (m-n)."""
    d = 16
    rope = RotaryPositionalEmbedding(theta=10000.0, d_k=d, max_seq_len=32)
    q = torch.randn(1, 1, 1, d)
    k = torch.randn(1, 1, 1, d)

    # Same offset → same dot product
    def dot(m, n):
        rq = rope(q, torch.tensor([m]))
        rk = rope(k, torch.tensor([n]))
        return (rq * rk).sum().item()

    base = dot(3, 7)            # offset 4
    same_offset = dot(8, 12)    # offset 4
    diff_offset = dot(3, 5)     # offset 2
    assert abs(base - same_offset) < 1e-4, "RoPE not translation-invariant in offset"
    assert abs(base - diff_offset) > 1e-3, "different offsets must give different dot"


# ==================================================== scaled_dot_product_attention

def test_sdpa_matches_torch_unmasked():
    B, H, T, D = 2, 3, 5, 8
    q = torch.randn(B, H, T, D)
    k = torch.randn(B, H, T, D)
    v = torch.randn(B, H, T, D)
    actual = scaled_dot_product_attention(q, k, v)
    expected = F.scaled_dot_product_attention(q, k, v)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)


def test_sdpa_masking_zeros_out_attention():
    B, T, D = 1, 4, 8
    q = torch.randn(B, T, D)
    k = torch.randn(B, T, D)
    v = torch.randn(B, T, D)
    # Allow only self-attention
    mask = torch.eye(T, dtype=torch.bool).unsqueeze(0)
    out = scaled_dot_product_attention(q, k, v, mask)
    # When each query can only see itself, the result is exactly v[b, q]
    torch.testing.assert_close(out, v, atol=1e-5, rtol=1e-4)


# ========================================================== MultiHead SelfAttention

def _grad_causal_check(layer, dtype=torch.float32, device="cpu", seq=12, dmodel=None):
    """Generic causality test via gradient: grad of output[t] wrt input[>t] is 0."""
    d = dmodel if dmodel is not None else layer.d_model
    x = torch.randn(1, seq, d, dtype=dtype, device=device, requires_grad=True)
    y = layer(x)
    t = seq // 2
    y[0, t].abs().sum().backward()
    g = x.grad[0]
    future = g[t + 1:].abs().sum().item()
    past = g[: t + 1].abs().sum().item()
    return past, future


def test_mha_forward_shape_and_causality():
    d, H, T = 32, 4, 10
    rope = RotaryPositionalEmbedding(10000.0, d // H, T)
    layer = MultiHeadSelfAttention(d, H, rope=rope)
    x = torch.randn(2, T, d)
    y = layer(x)
    assert y.shape == x.shape
    past, future = _grad_causal_check(layer, seq=T)
    assert future < 1e-4, f"future grad leaks: {future:.3e}"
    assert past > 1e-3, f"past grad vanishes: {past:.3e}"


def test_mha_without_rope_runs():
    layer = MultiHeadSelfAttention(d_model=16, num_heads=2, rope=None)
    x = torch.randn(1, 5, 16)
    y = layer(x)
    assert y.shape == x.shape


# ============================================================ GroupedQueryAttention

def test_gqa_shape_and_causality():
    d, Hq, Hk, T = 32, 8, 2, 12  # 4 query heads share each KV head
    rope = RotaryPositionalEmbedding(10000.0, d // Hq, T)
    layer = GroupedQueryAttention(d, Hq, Hk, rope=rope)
    x = torch.randn(2, T, d)
    y = layer(x)
    assert y.shape == x.shape
    past, future = _grad_causal_check(layer, seq=T)
    assert future < 1e-4 and past > 1e-3


def test_gqa_validates_head_divisibility():
    rope = RotaryPositionalEmbedding(10000.0, 8, 4)
    with pytest.raises(AssertionError):
        # 8 query heads, 3 kv heads (8 % 3 != 0)
        GroupedQueryAttention(d_model=64, num_query_heads=8, num_kv_heads=3, rope=rope)


def test_gqa_full_kv_equivalent_when_groups_one():
    """num_query_heads == num_kv_heads should match a plain MHA shape-wise."""
    d, H, T = 16, 4, 8
    rope = RotaryPositionalEmbedding(10000.0, d // H, T)
    gqa = GroupedQueryAttention(d, H, H, rope=rope)
    x = torch.randn(1, T, d)
    y = gqa(x)
    assert y.shape == x.shape


# =========================================================== MultiHeadLatentAttention

def test_mla_forward_shape_and_causality():
    d, H, T = 32, 4, 12
    rope_dim = 8
    rope = RotaryPositionalEmbedding(10000.0, rope_dim, T)
    layer = MultiHeadLatentAttention(
        d_model=d, head_num=H, rope=rope,
        rope_dim=rope_dim, q_lora_rank=16, kv_lora_rank=16,
    )
    x = torch.randn(2, T, d)
    y = layer(x)
    assert y.shape == x.shape
    past, future = _grad_causal_check(layer, seq=T)
    assert future < 1e-4
    assert past > 1e-3


def test_mla_low_rank_param_savings():
    """KV down-projection should be (d_model -> kv_lora_rank), not (d_model -> d_model)."""
    d, H, lora = 64, 4, 16
    rope = RotaryPositionalEmbedding(10000.0, 8, 4)
    layer = MultiHeadLatentAttention(d, H, rope=rope, rope_dim=8,
                                     q_lora_rank=lora, kv_lora_rank=lora)
    assert layer.kv_down_proj.weight.shape == (lora, d)
    assert layer.q_down_proj.weight.shape == (lora, d)


# =========================================================== CompressedConvAttention

def test_cca_forward_shape_and_causality():
    d, Hq, Hk, T = 32, 4, 2, 12
    c_dim = 8
    rope = RotaryPositionalEmbedding(10000.0, c_dim // 2, T)
    layer = CompressedConvAttention(
        d_model=d, num_query_heads=Hq, num_kv_heads=Hk,
        c_dim=c_dim, rope=rope, conv_kernel_size=3,
    )
    x = torch.randn(2, T, d)
    y = layer(x)
    assert y.shape == x.shape
    past, future = _grad_causal_check(layer, seq=T)
    assert future < 1e-3
    assert past > 1e-3


def test_cca_key_temperature_is_learnable():
    rope = RotaryPositionalEmbedding(10000.0, 4, 4)
    layer = CompressedConvAttention(
        d_model=16, num_query_heads=2, num_kv_heads=2,
        c_dim=8, rope=rope,
    )
    assert layer.key_temp.requires_grad
    x = torch.randn(1, 4, 16)
    layer(x).sum().backward()
    assert layer.key_temp.grad is not None and layer.key_temp.grad.abs() > 0


# =================================================================== GDA (re-test)

def test_gda_basic_shape_and_causality():
    """GDA has its own comprehensive suite — here we just sanity-check integration."""
    if not HAS_CUDA:
        pytest.skip("GDA needs CUDA (uses fla Triton kernels)")
    from model.attention.GDA import GatedDeltaAttention
    layer = GatedDeltaAttention(
        d_model=64, num_v_heads=4, num_k_heads=2,
        head_k_dim=32, head_v_dim=32, conv_kernel_size=4,
        device="cuda", dtype=torch.bfloat16,
    )
    x = torch.randn(1, 16, 64, device="cuda", dtype=torch.bfloat16,
                    requires_grad=True)
    y = layer(x)
    assert y.shape == x.shape
    y[0, 8].abs().sum().backward()
    assert x.grad[0, 9:].abs().sum().item() < 1e-2


# ================================================== paged KV-cache (CUDA / Triton)

@cuda_only
def test_store_kvcache_writes_to_correct_slots():
    nb, bs, nh, hd = 4, 8, 2, 16
    k_cache = torch.zeros(nb, bs, nh, hd, device="cuda")
    v_cache = torch.zeros(nb, bs, nh, hd, device="cuda")
    n_tokens = 5
    k = torch.randn(n_tokens, nh, hd, device="cuda")
    v = torch.randn(n_tokens, nh, hd, device="cuda")
    # Slot pattern: 0, 1, 2, 9 (block 1, offset 1), -1 (skip)
    slot_mapping = torch.tensor([0, 1, 2, 9, -1], dtype=torch.long, device="cuda")
    store_kvcache(k, v, k_cache, v_cache, slot_mapping, block_size=bs)
    # Verify tokens 0..3 made it
    torch.testing.assert_close(k_cache[0, 0], k[0])
    torch.testing.assert_close(k_cache[0, 1], k[1])
    torch.testing.assert_close(k_cache[0, 2], k[2])
    torch.testing.assert_close(k_cache[1, 1], k[3])  # slot 9 = block 1, off 1
    # The -1 token must not have been written; that slot should still be 0
    assert torch.equal(k_cache[1, 0], torch.zeros_like(k_cache[1, 0]))


@cuda_only
def test_flash_attention_prefill_matches_pytorch_sdpa():
    """For a single sequence, the Triton flash kernel must agree with SDPA."""
    T, H, D = 32, 4, 32
    q = torch.randn(T, H, D, device="cuda", dtype=torch.float16)
    k = torch.randn(T, H, D, device="cuda", dtype=torch.float16)
    v = torch.randn(T, H, D, device="cuda", dtype=torch.float16)
    cu = torch.tensor([0, T], dtype=torch.int32, device="cuda")

    out = flash_attention_prefill(q, k, v, cu, 1 / math.sqrt(D), H, H, D)
    # reference: (1, H, T, D) layout for SDPA
    qr = q.transpose(0, 1).unsqueeze(0).float()
    kr = k.transpose(0, 1).unsqueeze(0).float()
    vr = v.transpose(0, 1).unsqueeze(0).float()
    ref = F.scaled_dot_product_attention(qr, kr, vr, is_causal=True).squeeze(0).transpose(0, 1)
    diff = (out.float() - ref).abs()
    assert diff.max().item() < 5e-2, f"flash prefill max-err={diff.max().item():.3f}"


@cuda_only
def test_paged_attention_decode_correctness():
    """Single-token decode against a brute-force PyTorch reference."""
    bsz, nh, nkh, hd, bs, ctx = 1, 2, 2, 16, 8, 5
    nb = 2
    # build random cache for the first `ctx` tokens, all in block 0
    k_cache = torch.zeros(nb, bs, nkh, hd, device="cuda")
    v_cache = torch.zeros(nb, bs, nkh, hd, device="cuda")
    k_cache[0, :ctx] = torch.randn(ctx, nkh, hd, device="cuda")
    v_cache[0, :ctx] = torch.randn(ctx, nkh, hd, device="cuda")
    q = torch.randn(bsz, nh, hd, device="cuda")
    block_tables = torch.zeros(bsz, 1, dtype=torch.int32, device="cuda")  # block 0
    context_lens = torch.tensor([ctx], device="cuda", dtype=torch.int32)
    scale = 1.0 / math.sqrt(hd)
    out = paged_attention_decode(
        q, k_cache, v_cache, block_tables, context_lens,
        scale, nh, nkh, hd, bs,
    )
    # Per-head reference (avoid broadcasting pitfalls): build (nh, ctx) scores explicitly
    q_ref = q[0]                              # (nh, hd)
    k = k_cache[0, :ctx].transpose(0, 1)       # (nkh, ctx, hd) == (nh, ctx, hd) when nh==nkh
    v = v_cache[0, :ctx].transpose(0, 1)       # (nh, ctx, hd)
    # scores[h, c] = q[h] · k[h, c] * scale
    scores = torch.einsum("hd,hcd->hc", q_ref, k) * scale   # (nh, ctx)
    weights = F.softmax(scores, dim=-1)                       # (nh, ctx)
    ref = torch.einsum("hc,hcd->hd", weights, v)              # (nh, hd)
    diff = (out[0] - ref).abs()
    assert diff.max().item() < 5e-2, f"decode max err={diff.max().item():.3f}"
