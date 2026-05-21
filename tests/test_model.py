from einops import rearrange
import numpy
import torch
import torch.nn.functional as F

from .adapters import (
    run_multihead_self_attention_with_rope,
    run_rope,
    run_silu,
    run_multihead_self_attention,
    run_swiglu,
    run_rmsnorm,
    run_scaled_dot_product_attention,
    run_transformer_block,
    run_transformer_lm,
    run_linear,
    run_embedding,
)


def test_linear(numpy_snapshot, ts_state_dict, in_embeddings, d_model, d_ff):
    w1_weight = ts_state_dict[0]["layers.0.ffn.w1.weight"]
    output = run_linear(
        d_in=d_model,
        d_out=d_ff,
        weights=w1_weight,
        in_features=in_embeddings,
    )
    numpy_snapshot.assert_match(output)


def test_embedding(numpy_snapshot, ts_state_dict, in_indices, vocab_size, d_model):
    embedding_weight = ts_state_dict[0]["token_embeddings.weight"]
    output = run_embedding(
        vocab_size=vocab_size,
        d_model=d_model,
        weights=embedding_weight,
        token_ids=in_indices,
    )
    numpy_snapshot.assert_match(output)


def test_swiglu(numpy_snapshot, ts_state_dict, in_embeddings, d_model, d_ff):
    w1_weight, w2_weight, w3_weight = [ts_state_dict[0][f"layers.0.ffn.{k}.weight"] for k in ["w1", "w2", "w3"]]

    actual_output = run_swiglu(
        d_model=d_model,
        d_ff=d_ff,
        w1_weight=w1_weight,
        w2_weight=w2_weight,
        w3_weight=w3_weight,
        in_features=in_embeddings,
    )
    numpy_snapshot.assert_match(actual_output, atol=1e-5)


def test_scaled_dot_product_attention(numpy_snapshot, q, k, v, mask):
    actual_output = run_scaled_dot_product_attention(Q=q, K=k, V=v, mask=mask)
    numpy_snapshot.assert_match(
        actual_output,
        atol=1e-6,
    )


def test_4d_scaled_dot_product_attention(numpy_snapshot, q, k, v, mask):
    # Shape: (batch_size, num_heads, seq_len, d_k)
    q, k, v = (rearrange(x, "(batch head) seq d -> batch head seq d", head=2) for x in (q, k, v))
    mask = rearrange(mask, "(batch head) query key -> batch head query key", head=2)

    actual_output = run_scaled_dot_product_attention(Q=q, K=k, V=v, mask=mask)
    numpy_snapshot.assert_match(
        actual_output,
        atol=1e-6,
    )


def test_multihead_self_attention(numpy_snapshot, in_embeddings, d_model, n_heads, ts_state_dict):
    d, _ = ts_state_dict
    q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight = [
        d[f"layers.0.attn.{k}_proj.weight"] for k in ["q", "k", "v", "output"]
    ]
    actual_output = run_multihead_self_attention(
        d_model=d_model,
        num_heads=n_heads,
        q_proj_weight=q_proj_weight,
        k_proj_weight=k_proj_weight,
        v_proj_weight=v_proj_weight,
        o_proj_weight=o_proj_weight,
        in_features=in_embeddings,
    )
    numpy_snapshot.assert_match(actual_output, atol=1e-6)


def test_multihead_self_attention_with_rope(
    numpy_snapshot, in_embeddings, d_model, n_heads, ts_state_dict, n_keys, theta, pos_ids
):
    d, _ = ts_state_dict
    q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight = [
        d[f"layers.0.attn.{k}_proj.weight"] for k in ["q", "k", "v", "output"]
    ]
    pos_ids = rearrange(pos_ids, "seq -> 1 seq")
    actual_output = run_multihead_self_attention_with_rope(
        d_model=d_model,
        num_heads=n_heads,
        max_seq_len=n_keys,
        theta=theta,
        q_proj_weight=q_proj_weight,
        k_proj_weight=k_proj_weight,
        v_proj_weight=v_proj_weight,
        o_proj_weight=o_proj_weight,
        in_features=in_embeddings,
        token_positions=pos_ids,
    )
    numpy_snapshot.assert_match(actual_output, atol=1e-6)


def test_transformer_lm(
    numpy_snapshot, vocab_size, n_keys, d_model, n_layers, n_heads, d_ff, theta, ts_state_dict, in_indices
):
    state_dict, _ = ts_state_dict

    actual_output = run_transformer_lm(
        vocab_size=vocab_size,
        context_length=n_keys,
        d_model=d_model,
        num_layers=n_layers,
        num_heads=n_heads,
        d_ff=d_ff,
        rope_theta=theta,
        weights=state_dict,
        in_indices=in_indices,
    )
    numpy_snapshot.assert_match(actual_output, atol=1e-4, rtol=1e-2)


def test_transformer_lm_truncated_input(
    numpy_snapshot, vocab_size, n_keys, d_model, n_layers, n_heads, d_ff, theta, ts_state_dict, in_indices
):
    in_indices_truncated = in_indices[..., : in_indices.shape[-1] // 2]
    truncated_actual_output = run_transformer_lm(
        vocab_size=vocab_size,
        context_length=n_keys,
        d_model=d_model,
        num_layers=n_layers,
        num_heads=n_heads,
        d_ff=d_ff,
        rope_theta=theta,
        weights=ts_state_dict[0],
        in_indices=in_indices_truncated,
    )

    numpy_snapshot.assert_match(
        truncated_actual_output,
        atol=1e-4,
    )


def test_transformer_block(numpy_snapshot, ts_state_dict, in_embeddings, d_model, n_heads, d_ff, n_keys, theta):
    block_weights = {k.replace("layers.0.", ""): v for k, v in ts_state_dict[0].items() if "layers.0." in k}

    actual_output = run_transformer_block(
        d_model=d_model,
        num_heads=n_heads,
        d_ff=d_ff,
        max_seq_len=n_keys,
        theta=theta,
        weights=block_weights,
        in_features=in_embeddings,
    )
    numpy_snapshot.assert_match(
        actual_output,
        atol=1e-6,
    )


def test_rmsnorm(numpy_snapshot, ts_state_dict, in_embeddings):
    state_dict, _ = ts_state_dict
    reference_weights = state_dict["layers.1.ln1.weight"]
    d_model = reference_weights.shape[0]

    actual_output = run_rmsnorm(d_model=d_model, eps=1e-5, weights=reference_weights, in_features=in_embeddings)

    numpy_snapshot.assert_match(actual_output, atol=1e-6)


def test_rope(numpy_snapshot, in_embeddings, d_model, theta, n_queries, pos_ids):
    output = run_rope(
        d_model, theta=theta, max_seq_len=n_queries, in_query_or_key=in_embeddings, token_positions=pos_ids
    )
    numpy_snapshot.assert_match(output, atol=1e-6)


def test_silu_matches_pytorch():
    x = torch.tensor(
        [
            [0.2352, 0.9259, 0.5189, 0.4725, 0.9730],
            [0.7581, 0.9692, 0.2129, 0.9345, 0.0149],
        ]
    )
    expected_output = F.silu(x)
    actual_output = run_silu(x)
    numpy.testing.assert_allclose(actual_output.detach().numpy(), expected_output.detach().numpy(), atol=1e-6)


# =============================================================================
# Additional integration tests for the higher-level model objects:
#   - Config: defaults / serialization / update
#   - Block + TransformerLM: forward shape, causal property, parameter sharing
#
# These intentionally bypass `tests/adapters.py` and instantiate the modules
# directly so they remain valid as the adapter layer drifts.
# =============================================================================

import json as _json
import os as _os
import pytest as _pytest


def _build_config(**over):
    """Build a tiny Config that's cheap to instantiate and run."""
    from model.config import Config
    base = dict(
        vocab_size=64, context_length=32, d_model=32, num_layers=2,
        num_heads=4, d_ff=64, dropout=0.0,
        attention_type="GQA", num_kv_heads=2, rope_theta=10000.0,
        use_moe=False, use_engram=False,
    )
    base.update(over)
    return Config(**base)


def test_config_defaults_and_normalisation():
    from model.config import Config
    cfg = Config()
    # Some baseline sanity invariants
    assert cfg.d_model > 0 and cfg.num_heads > 0
    assert cfg.d_model % cfg.num_heads == 0
    # `num_kv_heads` defaults to num_heads when None is passed
    assert cfg.num_kv_heads == cfg.num_heads
    # optimizer name is lowercased
    cfg2 = Config(optimizer="ADAMW")
    assert cfg2.optimizer == "adamw"


def test_config_unknown_kwargs_are_stored():
    """Config stashes unknown kwargs as attributes (forward-compat for new fields)."""
    from model.config import Config
    cfg = Config(custom_field=123, vocab_size=10)
    assert cfg.custom_field == 123
    assert cfg.vocab_size == 10


def test_config_json_roundtrip(tmp_path):
    from model.config import Config
    cfg = _build_config(vocab_size=128, d_model=64, attention_type="MHA")
    out = tmp_path / "cfg.json"
    cfg.save(out)
    loaded = Config.from_json(out)
    # All values from the saved dict are restored
    for k, v in cfg.to_dict().items():
        assert getattr(loaded, k) == v


def test_config_update_validates_keys():
    cfg = _build_config()
    cfg.update(d_model=256)
    assert cfg.d_model == 256
    with _pytest.raises(AttributeError):
        cfg.update(this_field_does_not_exist=1)


@_pytest.mark.parametrize("attention_type,extra", [
    ("MHA", {}),
    ("GQA", {"num_kv_heads": 2}),
    ("MLA", {"q_lora_rank": 16, "kv_lora_rank": 16, "rope_dim": 8}),
    ("CCA", {"cca_compressed_dim": 8, "cca_num_kv_heads": 2}),
])
def test_transformer_lm_forward_shape_all_attention_types(attention_type, extra):
    """End-to-end forward through TransformerLM for every attention variant."""
    from model.transformer import TransformerLM
    cfg = _build_config(attention_type=attention_type, **extra)
    model = TransformerLM(cfg)
    B, T = 2, 16
    ids = torch.randint(0, cfg.vocab_size, (B, T))
    logits = model(ids)
    assert logits.shape == (B, T, cfg.vocab_size)
    assert torch.isfinite(logits).all()


def test_transformer_lm_is_causal():
    """Output at position t should not depend on tokens at positions > t."""
    from model.transformer import TransformerLM
    cfg = _build_config(attention_type="MHA", dropout=0.0)
    model = TransformerLM(cfg).eval()
    T = 12
    ids_a = torch.randint(0, cfg.vocab_size, (1, T))
    ids_b = ids_a.clone()
    # Perturb the second half of the sequence; outputs in the first half
    # must remain bit-identical.
    ids_b[0, T // 2:] = (ids_b[0, T // 2:] + 7) % cfg.vocab_size
    with torch.no_grad():
        out_a = model(ids_a)
        out_b = model(ids_b)
    torch.testing.assert_close(out_a[0, : T // 2], out_b[0, : T // 2], atol=1e-5, rtol=1e-5)


def test_transformer_lm_truncated_input_smoke():
    """A short input (well below context_length) must still go through cleanly."""
    from model.transformer import TransformerLM
    cfg = _build_config(attention_type="GQA", num_kv_heads=2)
    model = TransformerLM(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 4))
    with torch.no_grad():
        out = model(ids)
    assert out.shape == (1, 4, cfg.vocab_size)


def test_transformer_lm_backward_param_grads_finite():
    from model.transformer import TransformerLM
    cfg = _build_config(attention_type="GQA", num_kv_heads=2)
    model = TransformerLM(cfg)
    ids = torch.randint(0, cfg.vocab_size, (2, 8))
    logits = model(ids)
    loss = logits.float().pow(2).mean()
    loss.backward()
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"no grad: {name}"
            assert torch.isfinite(p.grad).all(), f"non-finite grad: {name}"


def test_transformer_block_residual_path_is_identity_at_zero_branch():
    """If the attention and FFN sub-blocks are forced to output 0 at init,
    the Block degenerates to the identity (Res-scale is identity at init)."""
    from model.transformer import Block
    cfg = _build_config(attention_type="MHA")
    from model.attention.utils import RotaryPositionalEmbedding
    rope = RotaryPositionalEmbedding(
        theta=cfg.rope_theta, d_k=cfg.d_model // cfg.num_heads,
        max_seq_len=cfg.context_length,
    )
    block = Block(cfg, rope=rope, use_moe=False).eval()

    # Zero both the weight AND the bias of the last linear in each sub-block.
    # (nn.Linear defaults to bias=True; ignoring bias would leave a constant offset.)
    with torch.no_grad():
        block.att.output_proj.weight.zero_()
        if block.att.output_proj.bias is not None:
            block.att.output_proj.bias.zero_()
        block.ffn.w2.weight.zero_()
        if block.ffn.w2.bias is not None:
            block.ffn.w2.bias.zero_()
    x = torch.randn(1, 8, cfg.d_model)
    with torch.no_grad():
        y = block(x)
    # With α=1, β=0 in ResScale, and sub-block outputs zero, y == x
    torch.testing.assert_close(y, x, atol=1e-5, rtol=1e-5)


# ----------------------------------------------------------------------------
# Hybrid GDA tests: TransformerLM with 1:1 alternation between primary
# attention and GatedDeltaAttention.  GDA uses fla Triton kernels at forward
# time, so the forward-pass tests are CUDA-only; construction tests work on
# CPU.
# ----------------------------------------------------------------------------

@_pytest.mark.parametrize("primary", ["MHA", "GQA", "MLA"])
def test_hybrid_layer_pattern_is_alternating(primary):
    """Construction-only: every odd layer becomes GDA, even layers stay
    primary. CCA opt-out is verified separately."""
    from model.transformer import TransformerLM
    extra = {
        "MHA": {},
        "GQA": {"num_kv_heads": 2},
        "MLA": {"q_lora_rank": 16, "kv_lora_rank": 16, "rope_dim": 8},
    }[primary]
    cfg = _build_config(
        attention_type=primary, num_layers=4, use_gda_hybrid=True, **extra,
    )
    model = TransformerLM(cfg)
    # gda_layer_indices == [1, 3]
    assert model.gda_layer_indices == [1, 3]
    assert model.layers[0].attention_type == primary
    assert model.layers[1].attention_type == "GDA"
    assert model.layers[2].attention_type == primary
    assert model.layers[3].attention_type == "GDA"


def test_hybrid_disabled_for_cca():
    """Setting use_gda_hybrid alongside attention_type='CCA' must be a no-op:
    every layer stays CCA, no GDA layers are inserted."""
    from model.transformer import TransformerLM
    cfg = _build_config(
        attention_type="CCA", num_layers=4, use_gda_hybrid=True,
        cca_compressed_dim=8, cca_num_kv_heads=2,
    )
    model = TransformerLM(cfg)
    assert model.gda_layer_indices == []
    for layer in model.layers:
        assert layer.attention_type == "CCA"


def test_gda_defaults_are_moderate():
    """The default GDA dims should be ~standard-attention size:
    value_dim == d_model, key_dim ≈ d_model / 2 (GVA 2:1)."""
    from model.config import Config
    cfg = Config(d_model=128, num_heads=8, use_gda_hybrid=True,
                 attention_type="MHA")
    assert cfg.gda_num_v_heads * cfg.gda_head_v_dim == cfg.d_model  # value_dim
    # key_dim = num_k * head_k = 4 * 16 = 64 = d_model / 2
    assert cfg.gda_num_k_heads * cfg.gda_head_k_dim == cfg.d_model // 2


def test_hybrid_forward_shape_cuda():
    """Full end-to-end forward through a hybrid LM (CUDA-only because GDA's
    chunk-gated-delta-rule kernel needs Triton)."""
    if not torch.cuda.is_available():
        _pytest.skip("GDA hybrid forward requires CUDA")
    from model.transformer import TransformerLM
    cfg = _build_config(
        attention_type="GQA", num_kv_heads=2, num_layers=4, use_gda_hybrid=True,
    )
    model = TransformerLM(cfg, device="cuda", dtype=torch.bfloat16).cuda()
    B, T = 1, 16
    ids = torch.randint(0, cfg.vocab_size, (B, T), device="cuda")
    with torch.no_grad():
        logits = model(ids)
    assert logits.shape == (B, T, cfg.vocab_size)
    assert torch.isfinite(logits).all()


def test_hybrid_param_count_within_reasonable_bound_of_pure_primary():
    """Sanity check: hybrid (half GDA half MHA) shouldn't bloat the model
    versus pure MHA. Allow up to 1.5x parameter count growth."""
    from model.transformer import TransformerLM
    cfg_pure = _build_config(attention_type="MHA", num_layers=4)
    cfg_hyb = _build_config(attention_type="MHA", num_layers=4, use_gda_hybrid=True)
    pure = sum(p.numel() for p in TransformerLM(cfg_pure).parameters())
    hyb = sum(p.numel() for p in TransformerLM(cfg_hyb).parameters())
    # GDA layer is moderately larger than MHA (conv + extra gate projections),
    # but should be within ~1.5x per replaced layer → overall <1.5x.
    ratio = hyb / pure
    assert 0.8 < ratio < 1.5, f"hybrid/pure param ratio = {ratio:.2f}"
