"""Audit and reopen Terminal-RSI feedback gates.

The model has many learned scalar gates.  A gate at zero is not merely small:
the corresponding hierarchy/MoE/feedback path is disconnected.  This module
centralizes the names so audits and fresh runs can deliberately open paths
instead of silently training a dead graph.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

EXACT_GATES = (
    "upgrades.hierarchy.bottom_up_gates",
    "upgrades.hierarchy.top_down_gates",
    "upgrades.hierarchy.lateral_gates",
    "upgrades.hierarchy.boundary_gate",
    "upgrades.hierarchy.pair_memory.gate",
    "upgrades.hierarchy.top_moe.gate",
    "upgrades.hierarchy.bottom_extensions.gate",
    "upgrades.symbiosis_gate",
    "upgrades.symbiosis_output_gate",
    "upgrades.fast_weight.scale",
)


def is_feedback_gate(name: str) -> bool:
    return name in EXACT_GATES or name.endswith(".moe.gate")


def reopen_gates(model: torch.nn.Module, value: float) -> dict[str, float]:
    """Set every hierarchy/MoE/feedback scalar gate to ``value``."""
    if not (-10.0 <= float(value) <= 10.0):
        raise ValueError("gate value outside tanh-safe range")
    changed = {}
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if is_feedback_gate(name) and parameter.numel() >= 1:
                parameter.fill_(float(value))
                changed[name] = float(value)
    return changed


def audit_gates(model: torch.nn.Module, threshold: float = 0.01) -> dict:
    """Return gate values and mark effectively disconnected paths."""
    rows = {}
    dead = []
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if not is_feedback_gate(name):
                continue
            values = parameter.detach().float().flatten()
            mean = float(values.mean())
            rows[name] = {
                "mean": round(mean, 6),
                "min": round(float(values.min()), 6),
                "max": round(float(values.max()), 6),
                "tanh_mean": round(float(torch.tanh(values).mean()), 6),
                "dead": bool(values.abs().max() < threshold),
            }
            if rows[name]["dead"]:
                dead.append(name)
    return {"gates": rows, "dead": dead,
            "verdict": "disconnected paths present" if dead else "all gates open"}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.01)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    checkpoint = torch.load(args.checkpoint, map_location="cpu",
                            weights_only=True)
    model_state = checkpoint.get("model", {})
    # Audit the serialized tensors directly; no full model allocation needed.
    gates = {}
    dead = []
    for name, value in model_state.items():
        if not is_feedback_gate(name) or not torch.is_tensor(value):
            continue
        values = value.detach().float().flatten()
        gates[name] = {
            "mean": round(float(values.mean()), 6),
            "min": round(float(values.min()), 6),
            "max": round(float(values.max()), 6),
            "tanh_mean": round(float(torch.tanh(values).mean()), 6),
            "dead": bool(values.abs().max() < args.threshold),
        }
        if gates[name]["dead"]:
            dead.append(name)
    report = {"checkpoint": str(args.checkpoint), "gates": gates,
              "dead": dead,
              "verdict": "disconnected paths present" if dead else "all gates open"}
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
