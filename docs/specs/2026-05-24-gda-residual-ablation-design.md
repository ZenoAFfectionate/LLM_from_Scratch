# GDA / Residual Connection Ablation — Design Spec

_Date: 2026-05-24_

## Goal

Diagnose root cause of model performance drop by running two ablation studies on TinyStories:

1. **GDA ablation** — for each primary attention (MHA / GQA / MLA), compare 4 GDA-hybrid ratios:
   `none`, `GDA:Primary = 3:1`, `1:1`, `1:3` (per 4 layers). 12 experiments total.
2. **Residual ablation** — on GQA+MoE fixed config, compare 3 residual schemes:
   `vanilla` pre-norm, `resscale` (ZAYA1), `mhc` (Manifold-Constrained Hyper-Connection, H=2). 3 experiments.

## GDA Layer Pattern (Qwen3-Next style, per-4-layer block, num_layers=8)

| `gda_ratio` | 8-layer pattern (G=GDA, A=Primary) | #GDA | #Primary |
|-------------|-----------------------------------|------|----------|
| `none`      | A,A,A,A,A,A,A,A                   | 0    | 8        |
| `3:1`       | G,G,G,A,G,G,G,A                   | 6    | 2        |
| `1:1`       | G,G,A,A,G,G,A,A                   | 4    | 4        |
| `1:3`       | G,A,A,A,G,A,A,A                   | 2    | 6        |

Rule: within each 4-layer block, all GDA layers come first, primary at the end.

## Code Changes

### `model/config.py`
- Add `gda_ratio: str = "none"` — replaces hard-coded `use_gda_hybrid` logic. Accepts `"none" | "3:1" | "1:1" | "1:3"`.
- Add `residual_type: str = "resscale"` — accepts `"vanilla" | "resscale" | "mhc"`.
- Keep `use_gda_hybrid` for backward compatibility (deprecated alias: maps to `gda_ratio="1:1"` if True).

### `model/transformer.py`
- New helper `build_gda_layer_mask(num_layers: int, ratio: str) -> List[bool]` — returns the Qwen3-Next pattern.
- `Block.__init__` accepts `residual_type`. When `"vanilla"`, skip ResScale construction and use `x + sublayer(norm(x))` in `forward`. When `"resscale"`, keep current behavior.
- `TransformerLM` uses the helper to compute `gda_layer_indices` from `config.gda_ratio`.

### `model/mHC_transformer.py`
- Use the same `build_gda_layer_mask` helper for `gda_layer_indices`. (No residual change needed — mHC is the residual type.)

### `train.py`
- After loading config, dispatch model construction:
  ```python
  if config.residual_type == "mhc":
      from model.mHC_transformer import mHCTransformerLM
      model = mHCTransformerLM(config=config, ...)
  else:
      model = TransformerLM(config=config, ...)
  ```
- Modify `checkpoint_folder_name` to add suffixes for GDA / residual variants (so different ablations do not collide):
  - Default (`gda_ratio="none"`, `residual_type="resscale"`): no suffix (backward compatible).
  - Otherwise append `_gda{ratio}` and/or `_res-{type}` (sanitize `:` → `-`).
  - Example: `TinyStories_GQA+MoE_gda3-1`, `TinyStories_GQA+MoE_res-mhc`.

## Config Files (15 total)

All inherit hyperparameters from `[GQA+MoE]train_tinystories.json` (batch_size=128, max_iterations=20000, etc.) except attention-specific fields.

**GDA ablation (12):**
- `[{MHA,GQA,MLA}+MoE]GDA-{none,3-1,1-1,1-3}_tinystories.json`

**Residual ablation (3):**
- `[GQA+MoE]res-{vanilla,resscale,mhc}_tinystories.json`

Notes:
- For MHA: `num_kv_heads = num_heads = 16`.
- For GQA: `num_kv_heads = 4` (matches existing config).
- For MLA: `q_lora_rank = 128`, `kv_lora_rank = 256`, `rope_dim = 16` (matches existing).
- For mHC: `hc_mult = 2`.

## Scripts (2 master scripts)

Both follow the structure of `scripts/run_tinystories_experiments.sh`:
- Iterate over a list of configs, run `train.py --config <cfg>` sequentially on GPU 0.
- After all runs, parse `record.txt` of each checkpoint folder for the final validation loss/PPL.
- Emit a markdown summary table.

- `scripts/run_gda_ablation.sh` — 12 configs, summary → `results_summary_gda.md`.
- `scripts/run_residual_ablation.sh` — 3 configs, summary → `results_summary_residual.md`.

## Out of Scope

- No changes to optimizer, dataset, MoE config, or other hyperparameters.
- No CCA in this ablation (CCA never pairs with GDA, and is already covered by `[CCA+MoE]`).
- No MTP changes (use_mtp / mtp_lambda inherited from base config).
