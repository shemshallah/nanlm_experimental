#!/usr/bin/env python3
"""Run the small CPU Z4 ablation matrix used by GitHub Actions.

The matrix is deliberately tiny: it is a wiring and stability gate, not a
quality benchmark.  Every row uses the same generated math corpus and differs
only in the feature under test.

Example::

    python z3/run_ablation_matrix.py --out /tmp/z3_ablation --steps 8

Outputs ``summary.json`` and per-run logs under the output directory.  The
command exits non-zero if any row fails, reports non-finite steps, or does not
complete the requested number of steps.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "z3" / "train_v3.py"
BUILD_CORPUS = ROOT / "z3" / "build_math_corpus.py"


def _run(command: list[str], log_path: Path, timeout: int) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(command) + "\n")
        handle.flush()
        try:
            process = subprocess.run(
                command,
                cwd=str(ROOT),
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
            return process.returncode
        except subprocess.TimeoutExpired:
            handle.write(f"\n[timeout] {timeout}s after {time.time() - started:.1f}s\n")
            return 124


def _build_corpus(out: Path, examples: int, seed: int) -> Path:
    target = out / f"corpus_{seed}"
    command = [
        sys.executable,
        str(BUILD_CORPUS),
        "--examples", str(examples),
        "--seed", str(seed),
        "--out", str(target),
    ]
    log = out / f"corpus_{seed}.log"
    code = _run(command, log, timeout=120)
    if code != 0:
        raise SystemExit(f"corpus build failed ({code}); see {log}")
    return target


def _base_args(
    out: Path,
    corpus: list[Path],
    steps: int,
    ctx: int,
) -> list[str]:
    args = [
        "--ctx", str(ctx),
        "--d-model", "128",
        "--n-layers", "4",
        "--n-heads", "4",
        "--n-kv-heads", "2",
        "--swiglu", "256",
        "--window", str(ctx),
        "--global-every", "2",
        "--moe-layers", "2",
        "--n-experts", "4",
        "--topk", "2",
        "--shared-experts", "1",
        "--expert-hidden", "256",
        "--moe-group-size", "2",
        "--no-use-memory",
        "--mtp", "2",
        "--mtp-block",
        "--mtp-block-experts", "2",
        "--mtp-block-hidden", "256",
        "--mtp-aux-layers", "1,3",
        "--mtp-weight", "0.2",
        "--concept-top-vertices", "4",
        "--concept-top-pairs", "2",
        "--concept-top-triples", "1",
        "--batch-size", "1",
        "--steps", str(steps),
        "--eval-every", str(max(1, steps)),
        "--eval-batches", "1",
        "--log-every", str(max(1, steps)),
        "--lr", "1e-4",
        "--warmup-steps", "1",
        "--no-ema",
        "--device", "cpu",
        "--threads", "2",
        "--tokenizer", "layered",
        "--corpus", *[str(path / "train.txt") for path in corpus],
        "--out", str(out),
    ]
    return args


def _rows(corpus_a: Path, corpus_b: Path) -> list[tuple[str, dict[str, Any]]]:
    return [
        (
            "baseline_no_concept",
            {
                "corpus": [corpus_a],
                "extra": [
                    "--no-concept-loop",
                    "--no-composition-head",
                    "--task-vocab-size", "0",
                ],
            },
        ),
        (
            "concept_full",
            {"corpus": [corpus_a], "extra": []},
        ),
        (
            "concept_no_mtp_structure",
            {
                "corpus": [corpus_a],
                "extra": [
                    "--mtp-composition-weight", "0",
                    "--mtp-object-pointer-weight", "0",
                    "--mtp-object-triple-pointer-weight", "0",
                ],
            },
        ),
        (
            "concept_pair_only",
            {
                "corpus": [corpus_a],
                "extra": ["--concept-top-triples", "0"],
            },
        ),
        (
            "domain_mixture",
            {
                "corpus": [corpus_a, corpus_b],
                "extra": [
                    "--domain-mixture",
                    "--corpus-task-ids", "0,1",
                    "--task-vocab-size", "4",
                ],
            },
        ),
    ]


def _extract_result(name: str, out: Path, code: int, elapsed: float) -> dict[str, Any]:
    result_path = out / "result.json"
    row: dict[str, Any] = {
        "name": name,
        "returncode": int(code),
        "elapsed_seconds": round(elapsed, 2),
        "log": str(out / "run.log"),
    }
    if not result_path.is_file():
        row["ok"] = False
        row["error"] = "missing result.json"
        return row
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
            "per_task": final.get("per_task"),
            "data": result.get("data"),
        }
    )
    return row


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="runs/z3_ablation")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--ctx", type=int, default=128)
    parser.add_argument("--examples", type=int, default=300)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument(
        "--only", default="",
        help="comma-separated row names; empty runs all rows",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.steps < 1 or args.ctx < 8:
        raise SystemExit("--steps must be >=1 and --ctx >=8")
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    corpus_a = _build_corpus(out, args.examples, 20261115)
    corpus_b = _build_corpus(out, max(50, args.examples // 2), 20261116)
    wanted = {
        item.strip() for item in args.only.split(",") if item.strip()
    }
    rows = [
        row for row in _rows(corpus_a, corpus_b)
        if not wanted or row[0] in wanted
    ]
    if not rows:
        raise SystemExit("no ablation rows selected")
    summary = []
    failed = False
    for name, spec in rows:
        row_out = out / name
        row_out.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            str(TRAIN),
            *_base_args(
                row_out, list(spec["corpus"]), args.steps, args.ctx
            ),
            *list(spec["extra"]),
        ]
        started = time.time()
        code = _run(command, row_out / "run.log", args.timeout)
        row = _extract_result(
            name, row_out, code, time.time() - started
        )
        summary.append(row)
        failed = failed or not row["ok"]
        print(
            f"[ablate] {name:26s} ok={row['ok']} "
            f"loss={row.get('final_loss')} "
            f"bpb={row.get('final_bits_per_byte')}",
            flush=True,
        )
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
