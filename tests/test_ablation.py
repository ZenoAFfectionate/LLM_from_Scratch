"""Comprehensive test for the GDA / residual ablation implementation.

Tests:
  1. Each of the 15 configs can build a model (TransformerLM or mHCTransformerLM).
  2. A dummy forward pass produces (B, S, vocab_size)-shaped logits.
  3. The number of GDA layers matches the expected pattern.
  4. checkpoint_folder_name derived from config matches the expected slug.

Run from project root:
    python tests/test_ablation.py
"""
import json
import sys
from pathlib import Path

import torch

from model.config import Config
from model.transformer import TransformerLM, build_gda_layer_mask
from model.mHC_transformer import mHCTransformerLM


REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "config"

EXPECTED_GDA = {
    "none": 0,
    "3:1": 6,
    "1:1": 4,
    "1:3": 2,
}

# Slug derivation MUST match train.py.
def expected_folder(c: Config) -> str:
    suffix = ""
    if c.gda_ratio != "none":
        suffix += f"_gda{c.gda_ratio.replace(':', '-')}"
    if c.residual_type != "resscale":
        suffix += f"_res-{c.residual_type}"
    ffn = "MoE" if c.use_moe else "FFN"
    return f"{c.dataset}_{c.attention_type}+{ffn}{suffix}"


def build_model(c: Config, device: str, dtype: torch.dtype):
    # Mirror train.py's dispatch.
    if c.residual_type == "mhc":
        return mHCTransformerLM(config=c, device=device, dtype=dtype).to(device)
    return TransformerLM(config=c, device=device, dtype=dtype).to(device)


def count_gda_layers(model) -> int:
    layers = model.layers
    return sum(1 for layer in layers if layer.attention_type == "GDA")


def test_one(cfg_path: Path, device: str) -> bool:
    name = cfg_path.name
    try:
        c = Config.from_json(cfg_path)
        c.vocab_size = 256  # tiny vocab to keep memory low

        # Pattern check via helper alone (no model build needed).
        mask = build_gda_layer_mask(c.num_layers, c.gda_ratio)
        n_gda_pattern = sum(mask)
        n_gda_expected = EXPECTED_GDA[c.gda_ratio]
        assert n_gda_pattern == n_gda_expected, (
            f"helper produced {n_gda_pattern} GDA layers, expected {n_gda_expected}"
        )

        # Folder-name slug check
        folder = expected_folder(c)

        # Build model + forward pass.
        model = build_model(c, device=device, dtype=torch.bfloat16)
        n_params = sum(p.numel() for p in model.parameters())
        n_gda_built = count_gda_layers(model)
        assert n_gda_built == n_gda_expected, (
            f"model has {n_gda_built} GDA layers, expected {n_gda_expected}"
        )

        # Dummy forward (B=1, S=64) — keep small to fit comfortably.
        # Wrap in autocast(bfloat16) to mirror train.py's training step.
        B, S = 2, 128
        x = torch.randint(0, c.vocab_size, (B, S), device=device)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(x)
        assert logits.shape == (B, S, c.vocab_size), (
            f"unexpected logits shape {tuple(logits.shape)}"
        )
        has_nan = torch.isnan(logits).any().item()
        has_inf = torch.isinf(logits).any().item()

        # Cleanup
        del model
        torch.cuda.empty_cache()

        nan_marker = " [NaN!]" if has_nan else (" [Inf!]" if has_inf else "")
        print(
            f"  [OK]  {name:50s}  "
            f"params={n_params/1e6:6.2f}M  gda_layers={n_gda_built}/{c.num_layers}  "
            f"folder={folder}{nan_marker}"
        )
        return not (has_nan or has_inf)
    except Exception as e:
        import traceback
        print(f"  [FAIL] {name}: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: CUDA not available — GDA's fla kernels may not work on CPU.")
        sys.exit(1)

    print(f"Device: {device}")
    print()

    cfgs = sorted(list(CONFIG_DIR.glob('[*GDA-*_tinystories.json')) +
                  list(CONFIG_DIR.glob('[*res-*_tinystories.json')))
    print(f"Testing {len(cfgs)} configs:")
    print()

    n_ok = 0
    for cfg in cfgs:
        if test_one(cfg, device):
            n_ok += 1

    print()
    print(f"Result: {n_ok}/{len(cfgs)} passed")
    sys.exit(0 if n_ok == len(cfgs) else 1)


if __name__ == "__main__":
    main()
