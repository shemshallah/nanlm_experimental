#!/usr/bin/env python3
"""Reproducible z3 training presets.

The preset values are CLI destinations, not dataclass field names.  A preset
only fills arguments the caller did not supply, so every value can still be
overridden explicitly.

``115m`` is the Blackwell-class model: 10 layers, d_model 320, three
12-expert MoE layers, 256-vertex concept objects, per-lag MTP branches, and a
64k factorized embedding.  It is intentionally the same model on every GPU;
only batch size / activation checkpointing should vary with hardware.
"""

from __future__ import annotations

from typing import Any

# Expected model parameter count for the preset below.  train_v3.py prints a
# warning if the constructed model drifts from this number.
PRESET_PARAM_COUNTS: dict[str, int] = {
    "115m": 115_238_237,
}

PRESETS: dict[str, dict[str, Any]] = {
    "115m": {
        # Model geometry.
        "vocab_size": 64060,
        "ctx": 512,
        "d_model": 320,
        "n_layers": 10,
        "n_heads": 10,
        "n_kv_heads": 2,
        "swiglu": 1280,
        "window": 512,
        "global_every": 4,
        "moe_layers": 3,
        "n_experts": 12,
        "topk": 2,
        "shared_experts": 2,
        "expert_hidden": 1280,
        "moe_group_size": 4,
        "moe_max_rows": 512,
        "moe_hierarchical": True,
        # 12 experts / 4 branches = 3 experts per branch, so top-2 is a real
        # within-branch choice (not the degenerate 6 x 2 layout).
        "moe_branches": 4,
        "moe_context_router": True,
        "moe_route_feedback": True,
        "moe_route_to_attention": True,
        "moe_hyperdimensional": True,
        "moe_hd_dim": 256,
        "moe_hd_scale": 0.5,
        "moe_hd_diversity_coef": 1e-3,
        "moe_associative_memory": True,
        "moe_memory_keys": 16,
        "moe_memory_dim": 0,
        # Embedding / attention.
        "factorized_embedding": True,
        "embedding_rank": 128,
        "mtp_tie_heads": True,
        "attn_gate": True,
        "max_ctx": 8192,
        "rope_scaling": "auto",
        "rope_factor": 8.0,
        "rope_original_ctx": 1024,
        # Concept loop.
        "concept_loop": True,
        "concept_loop_scale": 1.0,
        "concept_loop_attention": False,
        "concept_router_tokens": True,
        "concept_router_global": True,
        "feedback_token_residual": True,
        "feedback_route_concept": True,
        "feedback_object_carry": True,
        "feedback_memory_mtp": True,
        "feedback_mtp_carry": True,
        "concept_vertices": 256,
        "concept_dim": 128,
        "concept_top_vertices": 8,
        "concept_top_pairs": 4,
        "concept_top_triples": 2,
        "object_id_embedding": True,
        "object_id_dim": 64,
        "composition_head": True,
        "object_balance_coef": 1e-3,
        # Diagnostics (cheap, and useful in every run).
        "byte_probe": True,
        "byte_probe_full": True,
        "word_probe_size": 0,
        "probe_detach": True,
        # MTP: mtp=3 means t+2 and t+3 (two heads).
        "mtp": 3,
        "mtp_adapter_rank": 32,
        "mtp_block": True,
        "mtp_block_experts": 8,
        "mtp_block_hidden": 1024,
        "mtp_attention_global": True,
        "mtp_attention_gate": False,
        "mtp_teacher_prob": 0.75,
        "mtp_gt_mix_start": 0.85,
        "mtp_gt_mix_end": 0.25,
        "mtp_gt_mix_steps": 0,
        "feedback_gate_warmup_steps": 500,
        "mtp_aux_layers": "3,7",
        "mtp_aux_weight": 0.2,
        "mtp_gate_coef": 0.1,
        "mtp_gate_prior": 0.7,
        "mtp_gate": True,
        "mtp_gate_hidden": 256,
        "mtp_gate_use_hidden": True,
        "mtp_hd_time": True,
        "mtp_branch_norm": True,
        "mtp_weight": 0.3,
        "mtp_lr_scale": 0.5,
        # Memory.  The learned 256-slot path is on; the hashed n-gram shortcut
        # is off by default because it is a training-corpus lookup.
        "use_memory": True,
        "ngram_memory": False,
        "memory_slots": 256,
        "memory_heads": 4,
        "memory_size": 1 << 20,
        "memory_order": 4,
        "memory_min_count": 5,
        "memory_alpha_init": 0.1,
        # Data / loss balance.
        "sidecar": True,
        "document_framing": True,
        "task_vocab_size": 16,
        "dropout": 0.1,
        "composition_weight": 0.05,
        "byte_probe_weight": 0.02,
        "byte_sequence_weight": 0.02,
        "word_probe_weight": 0.02,
        "object_pointer_weight": 0.02,
        "object_triple_pointer_weight": 0.01,
        "mtp_object_pointer_weight": 0.01,
        "mtp_object_triple_pointer_weight": 0.005,
        "mtp_composition_weight": 0.02,
        "load_balance_coef": 0.01,
        "router_z_coef": 0.001,
        # Optimization.  10k steps x batch 8 x ctx 512 = 41.9M tokens.
        "batch_size": 8,
        "steps": 10_000,
        "lr": 2e-4,
        "lr_horizon": "wsd",
        "lr_total_steps": 10_000,
        "warmup_steps": 300,
        "decay_frac": 0.1,
        "min_lr_frac": 0.05,
        "weight_decay": 0.05,
        "grad_clip": 1.0,
        "ema_decay": 0.999,
        "ema_start_frac": 0.3,
        "grad_checkpoint": True,
        "amp": True,
        # Evaluation / artifacts.
        "eval_every": 500,
        "eval_batches": 8,
        "log_every": 50,
        "save_every": 500,
        "best_every": 500,
        "seed": 3,
        "threads": 4,
        "device": "auto",
        "out": "runs/z3_115m",
    },
}


# Kaggle 10-minute profile.  This keeps the exact 115M model geometry and
# parameter count; it changes only the short-run schedule, loss weights, and
# the concept top-k.  Diagnostic heads stay in the parameter count but their
# forward compute is skipped when their loss weight is zero (see V3LM.forward).
PRESETS["kaggle10m"] = {
    **PRESETS["115m"],
    "tokenizer": "bin",
    "layered_model": "z3/tokenizers/layered_v1/layered_tokenizer.json",
    "ctx": 512,
    "batch_size": 4,
    "steps": 1000,
    "max_seconds": 600.0,
    "warmup_steps": 30,
    "lr_total_steps": 1000,
    "eval_every": 100,
    "eval_batches": 2,
    "log_every": 10,
    "save_every": 100,
    "best_every": 100,
    "no_ema": True,
    "dropout": 0.05,
    "feedback_gate_warmup_steps": 30,
    "mtp_weight": 0.1,
    "mtp_aux_weight": 0.0,
    "byte_sequence_weight": 0.0,
    "byte_probe_weight": 0.01,
    "cross_resolution_weight": 0.0,
    "object_pointer_weight": 0.005,
    "object_triple_pointer_weight": 0.0,
    "mtp_object_pointer_weight": 0.005,
    "mtp_object_triple_pointer_weight": 0.0,
    "composition_weight": 0.02,
    "mtp_composition_weight": 0.01,
    "concept_top_vertices": 4,
    "concept_top_pairs": 2,
    "concept_top_triples": 0,
    "device": "cuda",
    "amp": True,
    "grad_checkpoint": True,
    "threads": 2,
    "out": "runs/z3_115m_kaggle10m",
}
PRESET_PARAM_COUNTS["kaggle10m"] = PRESET_PARAM_COUNTS["115m"]

# Controlled next-move experiments.  Each inherits the kaggle10m schedule and
# changes exactly one architecture/initialization knob.
PRESETS["kaggle10m_obj"] = {
    **PRESETS["kaggle10m"],
    "object_gate_init": 0.05,
    "out": "runs/z3_115m_kaggle10m_obj",
}
PRESETS["kaggle10m_route"] = {
    **PRESETS["kaggle10m"],
    "gate_route_attention": True,
    "out": "runs/z3_115m_kaggle10m_route",
}
PRESETS["kaggle10m_mtp64"] = {
    **PRESETS["kaggle10m"],
    "mtp_output_rank": 64,
    "out": "runs/z3_115m_kaggle10m_mtp64",
}
for _name in ("kaggle10m_obj", "kaggle10m_route"):
    PRESET_PARAM_COUNTS[_name] = PRESET_PARAM_COUNTS["115m"]
PRESET_PARAM_COUNTS["kaggle10m_mtp64"] = 111_097_437

# LR sweep winners from the 600-second Kaggle round.  lr3e4 is the new
# recommended short-run profile; it keeps the same 115M geometry.
PRESETS["kaggle10m_lr3e4"] = {
    **PRESETS["kaggle10m"],
    "lr": 3e-4,
    "out": "runs/z3_115m_kaggle10m_lr3e4",
}
PRESETS["kaggle10m_lr3e4_decay"] = {
    **PRESETS["kaggle10m_lr3e4"],
    "steps": 600,
    "lr_total_steps": 600,
    "decay_frac": 0.2,
    "out": "runs/z3_115m_kaggle10m_lr3e4_decay",
}
PRESETS["kaggle10m_lr4e4"] = {
    **PRESETS["kaggle10m"],
    "lr": 4e-4,
    "out": "runs/z3_115m_kaggle10m_lr4e4",
}
for _name in (
    "kaggle10m_lr3e4", "kaggle10m_lr3e4_decay", "kaggle10m_lr4e4",
):
    PRESET_PARAM_COUNTS[_name] = PRESET_PARAM_COUNTS["115m"]


def get_preset(name: str) -> dict[str, Any]:
    key = str(name).strip().lower()
    if key not in PRESETS:
        raise KeyError(f"unknown preset {name!r}; choose {sorted(PRESETS)}")
    return dict(PRESETS[key])


def expected_parameter_count(name: str) -> int | None:
    return PRESET_PARAM_COUNTS.get(str(name).strip().lower())
