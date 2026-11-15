#!/usr/bin/env python3
"""Run one WikiText-2 CPU architecture ablation row.

This is the per-matrix worker used by the GitHubActions workflow.  It keeps the
base model/config fixed and changes only the feature under test.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "z3" / "train_v3.py"

ROWS = {
    "concept_full": [],
    "baseline_no_concept": [
        "--no-concept-loop",
        "--no-composition-head",
        "--task-vocab-size", "0",
    ],
    "concept_no_mtp_structure": [
        "--mtp-composition-weight", "0",
        "--mtp-object-pointer-weight", "0",
        "--mtp-object-triple-pointer-weight", "0",
    ],
    "concept_pair_only": ["--concept-top-triples", "0"],
    "concept_no_composition": ["--no-composition-head"],
    "concept_no_sidecar": ["--no-sidecar"],
    "concept_loopscale10": ["--concept-loop-scale", "0.1"],
    "concept_router_only": ["--no-concept-loop-attention"],
}


def base_args(
    train_bin: Path,
    valid_bin: Path,
    out: Path,
    steps: int,
    ctx: int,
) -> list[str]:
    return [
        "--ctx", str(ctx),
        "--d-model", "192",
        "--n-layers", "6",
        "--n-heads", "6",
        "--n-kv-heads", "2",
        "--swiglu", "512",
        "--window", str(ctx),
        "--global-every", "2",
        "--moe-layers", "3",
        "--n-experts", "8",
        "--topk", "2",
        "--shared-experts", "1",
        "--expert-hidden", "512",
        "--moe-group-size", "4",
        "--no-use-memory",
        "--mtp", "2",
        "--mtp-block",
        "--mtp-block-experts", "4",
        "--mtp-block-hidden", "512",
        "--mtp-aux-layers", "2,4",
        "--mtp-weight", "0.3",
        "--concept-top-vertices", "8",
        "--concept-top-pairs", "4",
        "--concept-top-triples", "2",
        "--batch-size", "1",
        "--steps", str(steps),
        "--eval-every", str(max(1, steps)),
        "--eval-batches", "2",
        "--log-every", "25",
        "--lr", "2e-4",
        "--warmup-steps", "20",
        "--no-ema",
        "--device", "cpu",
        "--threads", "2",
        "--tokenizer", "bin",
        "--vocab-size", "64060",
        "--token-dtype", "uint16",
        "--corpus", str(train_bin),
        "--val-corpus", str(valid_bin),
        "--out", str(out),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--row", choices=tuple(ROWS), required=True)
    parser.add_argument("--train-bin", required=True)
    parser.add_argument("--valid-bin", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--ctx", type=int, default=256)
    parser.add_argument("--timeout", type=int, default=3000)
    args = parser.parse_args()
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(TRAIN),
        *base_args(
            Path(args.train_bin).expanduser(),
            Path(args.valid_bin).expanduser(),
            out,
            args.steps,
            args.ctx,
        ),
        *ROWS[args.row],
    ]
    log = out / "run.log"
    started = time.time()
    with log.open("w", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(command) + "\n")
        handle.flush()
        try:
            proc = subprocess.run(
                command,
                cwd=str(ROOT),
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=args.timeout,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
            code = proc.returncode
        except subprocess.TimeoutExpired:
            handle.write(f"\n[timeout] after {args.timeout}s\n")
            code = 124
    result_path = out / "result.json"
    row = {
        "row": args.row,
        "returncode": code,
        "elapsed_seconds": round(time.time() - started, 2),
        "log": str(log),
    }
    if result_path.is_file():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        final = result.get("final") or {}
        train = result.get("last_train") or {}
        row.update(
            {
                "ok": bool(
                    code == 0
                    and result.get("status") == "completed"
                    and int(result.get("nonfinite_skips") or 0) == 0
                ),
                "status": result.get("status"),
                "steps_completed": result.get("steps_completed"),
                "nonfinite_skips": result.get("nonfinite_skips"),
                "params": result.get("params"),
                "elapsed_train": result.get("elapsed_seconds"),
                "final_loss": final.get("loss"),
                "final_bits_per_token": final.get("bits_per_token"),
                "final_bits_per_byte": final.get("bits_per_byte"),
                "top1": final.get("accuracy"),
                "top5": final.get("top5"),
                "train_loss": train.get("loss"),
                "composition": train.get("composition"),
                "object_pointer": train.get("object_pointer"),
                "object_triple_pointer": train.get("object_triple_pointer"),
                "mtp_composition": train.get("mtp_composition"),
                "mtp_object_pointer": train.get("mtp_object_pointer"),
                "mtp_object_triple_pointer": train.get("mtp_object_triple_pointer"),
                "data": result.get("data"),
            }
        )
    else:
        row.update({"ok": False, "error": "missing result.json"})
    (out / "row_result.json").write_text(
        json.dumps(row, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(row, indent=2, sort_keys=True), flush=True)
    return 0 if row.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
