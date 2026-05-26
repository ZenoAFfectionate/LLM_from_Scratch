#!/usr/bin/env python3
"""Generate ablation config files for GDA and residual experiments.

Run from project root:
    python scripts/_gen_ablation_configs.py
"""
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "config"

# Hyperparameters shared by all 15 ablation experiments
# (inherited from [GQA+MoE]train_tinystories.json).
BASE = {
    "seed": 42,
    "dataset": "TinyStories",
    "data_dir": str(REPO_ROOT / "data" / "TinyStories"),
    "train_file": "ts2_train.txt",
    "valid_file": "ts2_valid.txt",
    "vocab_file": str(REPO_ROOT / "data" / "TinyStories" / "vocab.json"),
    "merges_file": str(REPO_ROOT / "data" / "TinyStories" / "merges.txt"),
    "special_tokens": ["<|endoftext|>"],
    "context_length": 512,
    "d_model": 512,
    "num_layers": 8,
    "num_heads": 16,
    "d_ff": 1344,
    "dropout": 0.1,
    "rope_theta": 10000.0,
    "use_moe": True,
    "moe_layers": [1, 2, 3, 4, 5, 6, 7],
    "n_routed_experts": 4,
    "num_experts_per_tok": 2,
    "n_shared_experts": 1,
    "aux_seq_loss_alpha": 0.01,
    "bias_update_speed": 0.01,
    "swiglu_limit": 7.0,
    "use_mtp": True,
    "mtp_num_depths": 2,
    "mtp_lambda": 0.1,
    "z_loss_alpha": 1e-4,
    "optimizer": "muon",
    "muon_lr": 0.02,
    "muon_min_lr": 0.002,
    "muon_momentum": 0.95,
    "muon_nesterov": True,
    "muon_ns_steps": 5,
    "muon_weight_decay": 0.0,
    "batch_size": 128,
    "max_iterations": 20000,
    "max_lr": 5e-4,
    "min_lr": 5e-5,
    "warmup_iterations": 500,
    "beta1": 0.9,
    "beta2": 0.999,
    "eps": 1e-8,
    "weight_decay": 0.1,
    "max_grad_norm": 1.0,
    "use_amp": True,
    "log_interval": 100,
    "eval_interval": 1000,
    "eval_batches": 100,
    "checkpoint_dir": str(REPO_ROOT / "checkpoints"),
}


def attention_fields(attn: str) -> dict:
    """Return attention-specific config fields."""
    if attn == "MHA":
        return {"attention_type": "MHA", "num_kv_heads": 16}
    if attn == "GQA":
        return {"attention_type": "GQA", "num_kv_heads": 4}
    if attn == "MLA":
        return {
            "attention_type": "MLA",
            "num_kv_heads": 16,  # ignored by MLA but kept for completeness
            "rope_dim": 16,
            "q_lora_rank": 128,
            "kv_lora_rank": 256,
        }
    raise ValueError(attn)


def gda_ablation_configs() -> list:
    """12 configs: 3 attention types × 4 GDA ratios."""
    out = []
    ratios = [("none", "none"), ("3:1", "3-1"), ("1:1", "1-1"), ("1:3", "1-3")]
    for attn in ("MHA", "GQA", "MLA"):
        for ratio_value, ratio_slug in ratios:
            cfg = dict(BASE)
            cfg.update(attention_fields(attn))
            cfg["gda_ratio"] = ratio_value
            cfg["residual_type"] = "resscale"
            cfg["run_name"] = f"[{attn}+MoE] GDA={ratio_value} on TinyStories"
            filename = f"[{attn}+MoE]GDA-{ratio_slug}_tinystories.json"
            out.append((filename, cfg))
    return out


def residual_ablation_configs() -> list:
    """3 configs: GQA+MoE × {vanilla, resscale, mhc}."""
    out = []
    for res_type in ("vanilla", "resscale", "mhc"):
        cfg = dict(BASE)
        cfg.update(attention_fields("GQA"))
        cfg["gda_ratio"] = "none"
        cfg["residual_type"] = res_type
        if res_type == "mhc":
            cfg["hc_mult"] = 2  # H=2 parallel streams
        cfg["run_name"] = f"[GQA+MoE] residual={res_type} on TinyStories"
        filename = f"[GQA+MoE]res-{res_type}_tinystories.json"
        out.append((filename, cfg))
    return out


def main():
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    for filename, cfg in gda_ablation_configs() + residual_ablation_configs():
        path = CONFIG_DIR / filename
        with open(path, "w") as f:
            json.dump(cfg, f, indent=4)
        written.append(path.name)
    print(f"Wrote {len(written)} configs to {CONFIG_DIR}:")
    for n in written:
        print(f"  {n}")


if __name__ == "__main__":
    main()
