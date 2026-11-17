#!/usr/bin/env python3
"""Static 115M profile: parameter counts and per-token MAC estimates.

This does not train.  It builds the fixed ``115m`` model, groups parameters by
top-level module, and estimates multiply-accumulate work per token from the
resolved config.  The MAC estimate is intentionally mechanical: it is useful
for ranking bottlenecks, not for claiming hardware throughput.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train_v3 import apply_preset, build_parser, make_config  # noqa: E402
from v3_model import V3LM  # noqa: E402


def _count(module: torch.nn.Module | None) -> int:
    if module is None:
        return 0
    return sum(parameter.numel() for parameter in module.parameters())


def parameter_breakdown(model: V3LM) -> dict[str, int]:
    groups: dict[str, int] = {}
    for name, module in model.named_children():
        groups[name] = _count(module)
    blocks = []
    for index, block in enumerate(model.blocks):
        blocks.append(
            {
                "layer": index,
                "total": _count(block),
                "attention": _count(block.attn),
                "mlp": _count(block.mlp),
                "feedback": sum(
                    _count(getattr(block, name, None))
                    for name in (
                        "route_attn_proj",
                        "route_moe_proj",
                        "route_to_state",
                        "loop_to_state",
                    )
                ),
            }
        )
    mtp_blocks = [
        {
            "lag": index + 2,
            "total": _count(block),
            "attention": _count(block.attn),
            "moe": _count(block.moe),
        }
        for index, block in enumerate(model.mtp_blocks)
    ]
    concept = {}
    if model.concept_loop is not None:
        for name, module in model.concept_loop.named_children():
            concept[name] = _count(module)
    return {
        "total": model.parameter_count(),
        "groups": groups,
        "blocks": blocks,
        "mtp_blocks": mtp_blocks,
        "concept_loop": concept,
    }


def mac_estimate(model: V3LM) -> dict[str, float]:
    cfg = model.cfg
    t = int(cfg.ctx)
    d = int(cfg.d_model)
    rank = int(model.embedding_rank)
    vocab = int(cfg.vocab_size)
    swiglu = int(cfg.swiglu)
    expert_hidden = int(cfg.expert_hidden)
    moe_layers = int(cfg.moe_layers)
    dense_layers = max(0, int(cfg.n_layers) - moe_layers)
    topk = int(cfg.topk)
    shared = int(cfg.shared_experts)
    kv_dim = int(cfg.n_kv_heads) * (d // int(cfg.n_heads))
    n_mtp = max(0, int(cfg.mtp) - 1)
    aux_layers = len(
        [layer for layer in cfg.mtp_aux_layers if 0 <= int(layer) < int(cfg.n_layers)]
    )
    attention = 2 * t * d * (d + 2 * kv_dim) + 2 * t * d * d
    attention += 2 * t * d * d if cfg.attn_gate else 0
    if cfg.window >= t or cfg.global_every <= 1:
        attention += 4 * t * t * d
    else:
        attention += 4 * t * min(t, int(cfg.window)) * d
    dense_ffn = 3 * t * d * swiglu
    moe_ffn = (topk + shared) * 3 * t * d * expert_hidden
    moe_ffn += t * d * int(cfg.n_experts)
    concept_encoder = 2 * t * d * d * 2 + 2 * t * (4 * d) * d
    if cfg.subword_embedding:
        concept_encoder += 2 * t * (5 * d) * d - 2 * t * (4 * d) * d
    concept_object = t * int(cfg.concept_loop_vertices) * int(cfg.concept_loop_dim)
    feedback = 4 * 2 * t * d * d
    main_head = 2 * t * d * rank + 2 * t * rank * vocab
    mtp_head = main_head
    aux_head = main_head
    byte_probe = 2 * t * d * 256
    byte_sequence = 2 * t * d * 48 * 256 if cfg.byte_probe_full else 0
    mtp_block = 0
    if cfg.mtp_block:
        mtp_block = 2 * t * d * (d + 2 * kv_dim) + 2 * t * d * d
        mtp_block += 4 * t * t * d
        mtp_block += 4 * 3 * t * d * int(cfg.mtp_block_hidden)
        mtp_block += 2 * t * d * int(cfg.mtp_block_experts)
        mtp_block *= n_mtp
    mtp_context = 2 * 2 * t * d * d * n_mtp
    return {
        "tokens": t,
        "embedding_up": 2 * t * rank * d,
        "attention_per_layer": attention,
        "attention_total": attention * int(cfg.n_layers),
        "dense_ffn_total": dense_ffn * dense_layers,
        "moe_ffn_total": moe_ffn * moe_layers,
        "concept_encoder_total": concept_encoder * int(cfg.n_layers),
        "concept_object_total": concept_object * int(cfg.n_layers),
        "feedback_total": feedback * int(cfg.n_layers),
        "main_head": main_head,
        "mtp_heads": mtp_head * n_mtp,
        "mtp_aux_heads": aux_head * aux_layers,
        "mtp_blocks": mtp_block,
        "mtp_context": mtp_context,
        "byte_probe": byte_probe,
        "byte_sequence": byte_sequence,
        "composition_head": 2 * t * d * 20,
        "memory_cross": 2 * t * d * d * 2,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", default="115m")
    parser.add_argument("--ctx", type=int, default=512)
    parser.add_argument("--json", default="")
    args = parser.parse_args()
    argv = ["--preset", args.preset, "--corpus", "unused", "--ctx", str(args.ctx)]
    cli = build_parser()
    parsed = cli.parse_args(argv)
    apply_preset(parsed, argv)
    cfg = make_config(parsed, 64060)
    model = V3LM(cfg)
    params = parameter_breakdown(model)
    macs = mac_estimate(model)
    total = sum(
        value for key, value in macs.items()
        if isinstance(value, (int, float)) and key not in ("tokens",)
    )
    # The per-layer terms are totals; the remaining keys are one head each.
    macs["total_estimate"] = total
    macs["main_head_share"] = macs["main_head"] / max(1, total)
    macs["mtp_head_share"] = macs["mtp_heads"] / max(1, total)
    macs["aux_head_share"] = macs["mtp_aux_heads"] / max(1, total)
    result = {"params": params, "macs": macs}
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    if args.json:
        Path(args.json).write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
