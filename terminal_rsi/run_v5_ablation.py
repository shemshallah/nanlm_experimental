"""Short hierarchy ablation runner for the v5 unified graph.

Runs the same parent-warm-start smoke under several hierarchy variants and
writes a comparable JSON/markdown table.  This is deliberately short (default
45 s per variant), not a long training run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .run_blackwell_smoke import main as smoke_main

VARIANTS = (
    "full", "no_loops", "no_bottom_up", "no_top_down", "no_lateral",
    "no_pair", "no_units", "no_boundary_pooling",
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variants", default="full,no_loops,no_pair,no_units")
    parser.add_argument("--seconds", type=float, default=45.0)
    parser.add_argument("--corpus-dir", type=Path)
    parser.add_argument("--legacy-checkpoint", type=Path)
    parser.add_argument("--allow-parent-mismatch", action="store_true")
    parser.add_argument("--train-bytes", type=int, default=2_000_000)
    parser.add_argument("--batch", type=int, default=2048)
    parser.add_argument("--microbatch", type=int, default=1024)
    parser.add_argument("--hierarchical-eval-books", type=int, default=1)
    parser.add_argument("--hierarchical-eval-max-positions", type=int, default=128)
    args = parser.parse_args(argv)
    variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    unknown = [item for item in variants if item not in VARIANTS]
    if unknown:
        parser.error(f"Unknown ablation variants: {unknown}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for variant in variants:
        out = args.output_dir / variant
        out.mkdir(parents=True, exist_ok=True)
        command = [
            "--output-dir", str(out),
            "--seconds", str(args.seconds),
            "--batch", str(args.batch),
            "--microbatch", str(args.microbatch),
            "--upgrade-loss-weight", "1.0",
            "--boundary-loss-weight", "0.02",
            "--unit-loss-weight", "0.1",
            "--distill-weight", "0.5",
            "--moe-loss-weight", "0.01",
            "--ablation", variant,
            "--rsi-version", "off",
            "--max-dev-positions", "32",
            "--topk-probes", "1",
            "--hierarchical-eval-books", str(args.hierarchical_eval_books),
            "--hierarchical-eval-max-positions",
            str(args.hierarchical_eval_max_positions),
            "--generation-steps", "0",
        ]
        if args.corpus_dir:
            command += ["--corpus-dir", str(args.corpus_dir)]
        if args.legacy_checkpoint:
            command += [
                "--legacy-checkpoint", str(args.legacy_checkpoint),
                "--legacy-core-lr-scale", "0.1",
            ]
            if args.allow_parent_mismatch:
                command.append("--allow-parent-mismatch")
        print(f"=== ablation {variant} ===", flush=True)
        try:
            status = smoke_main(command)
        except SystemExit as exc:
            status = int(exc.code) if exc.code is not None else 1
        receipt_path = out / "blackwell_smoke_metrics.json"
        receipt = (json.loads(receipt_path.read_text())
                   if receipt_path.exists() else None)
        byte = ((receipt or {}).get("hierarchical") or {}).get("byte") or {}
        units = receipt.get("unit_heads_dev") if receipt else None
        rows.append({
            "variant": variant,
            "status": status,
            "before_bpb": (receipt or {}).get("before", {}).get("bpb"),
            "after_bpb": (receipt or {}).get("after", {}).get("bpb"),
            "byte_bits": byte.get("bits_per_unit"),
            "byte_top5": byte.get("top5"),
            "unit_heads": units,
            "updates": (receipt or {}).get("updates_this_round"),
            "gpu": (receipt or {}).get("gpu"),
        })
    summary = {"schema": "terminal-rsi-v5-ablation-v1", "rows": rows}
    (args.output_dir / "ablation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    lines = [
        "# Terminal-RSI v5 hierarchy ablation",
        "",
        "| variant | status | before BPB | after BPB | byte bits | byte top-5 | updates |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['variant']} | {row['status']} | "
            f"{row['before_bpb']} | {row['after_bpb']} | "
            f"{row['byte_bits']} | {row['byte_top5']} | {row['updates']} |")
    (args.output_dir / "ABLATION.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0 if all(row["status"] == 0 for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
