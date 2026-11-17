#!/usr/bin/env python3
"""Run the z3 115M build -> train -> test -> validate pipeline.

The stages are deliberately separate so each can be resumed or inspected:

    build     build_domain_bins.py
    train     train_v3.py --preset 115m
    eval      eval_domains.py
    wt103     eval_v3.py on the external WikiText-103 test
    validate  validate_run.py promotion gates

Examples::

    # print the commands without running anything
    python z3/run_115m_pipeline.py --dry-run \
      --corpus arxiv=corpora/domains/arxiv.txt \
      --corpus code=corpora/domains/code.txt

    # train/eval/validate an already-built corpus
    python z3/run_115m_pipeline.py --stages train,eval,validate \
      --data-dir data/z3_domains_115m --out runs/z3_115m
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent


def _stage_list(raw: str) -> list[str]:
    stages = [item.strip() for item in raw.split(",") if item.strip()]
    allowed = {"build", "train", "eval", "wt103", "validate"}
    unknown = sorted(set(stages) - allowed)
    if unknown:
        raise SystemExit(f"unknown stages: {unknown}; allowed={sorted(allowed)}")
    return stages


def _run(command: list[str], *, dry_run: bool) -> None:
    print("$ " + " ".join(shlex.quote(item) for item in command), flush=True)
    if dry_run:
        return
    subprocess.run(command, cwd=str(REPO), check=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stages",
        default="train,eval,validate",
        help="comma-separated build,train,eval,wt103,validate",
    )
    parser.add_argument("--data-dir", default="data/z3_domains_115m")
    parser.add_argument("--out", default="runs/z3_115m")
    parser.add_argument(
        "--corpus",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="domain corpus for the build stage",
    )
    parser.add_argument("--test-dir", default="")
    parser.add_argument("--manifest", default="")
    parser.add_argument("--wt103", default="", help="external wiki.test.raw")
    parser.add_argument("--baseline-test", default="")
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--ctx", type=int, default=512)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--grad-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--max-bytes-per-domain", type=int, default=50_000_000)
    parser.add_argument(
        "--train-arg",
        action="append",
        default=[],
        help="extra raw argument appended to train_v3.py",
    )
    parser.add_argument(
        "--eval-arg",
        action="append",
        default=[],
        help="extra raw argument appended to eval_domains.py",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    stages = _stage_list(args.stages)
    data_dir = Path(args.data_dir).expanduser()
    out_dir = Path(args.out).expanduser()
    test_dir = Path(args.test_dir or (data_dir / "test")).expanduser()
    manifest = Path(
        args.manifest or (data_dir / "manifest.json")
    ).expanduser()
    python = sys.executable

    if "build" in stages:
        if not args.corpus:
            raise SystemExit("--stage build requires --corpus NAME=PATH")
        command = [
            python, str(ROOT / "build_domain_bins.py"),
            *sum((["--corpus", spec] for spec in args.corpus), []),
            "--out", str(data_dir),
            "--max-bytes-per-domain", str(args.max_bytes_per_domain),
            "--train-fraction", "0.98",
            "--valid-fraction", "0.01",
        ]
        _run(command, dry_run=args.dry_run)

    if "train" in stages:
        command = [
            python, str(ROOT / "train_v3.py"),
            "--preset", "115m",
            "--corpus", str(data_dir / "train.bin"),
            "--val-corpus", str(data_dir / "valid.bin"),
            "--data-manifest", str(manifest),
            "--tokenizer", "bin",
            "--ctx", str(args.ctx),
            "--batch-size", str(args.batch_size),
            "--steps", str(args.steps),
            "--device", args.device,
            "--out", str(out_dir),
        ]
        if args.grad_checkpoint:
            command.append("--grad-checkpoint")
        command.extend(args.train_arg)
        _run(command, dry_run=args.dry_run)

    if "eval" in stages:
        command = [
            python, str(ROOT / "eval_domains.py"),
            "--checkpoint", str(out_dir / "best.pt"),
            "--test-dir", str(test_dir),
            "--data-manifest", str(manifest),
            "--tokenizer", "layered",
            "--document-framing",
            "--sidecar",
            "--require-sidecar",
            "--ctx", str(args.ctx),
            "--batch-size", str(args.batch_size),
            "--device", args.device,
            "--out", str(out_dir / "test_domains.json"),
        ]
        command.extend(args.eval_arg)
        _run(command, dry_run=args.dry_run)

    if "wt103" in stages:
        if not args.wt103:
            raise SystemExit("--stage wt103 requires --wt103 PATH")
        command = [
            python, str(ROOT / "eval_v3.py"),
            "--checkpoint", str(out_dir / "best.pt"),
            "--corpus", f"wt103={args.wt103}",
            "--tokenizer", "layered",
            "--document-framing",
            "--sidecar",
            "--require-sidecar",
            "--full",
            "--ctx", str(args.ctx),
            "--batch-size", str(args.batch_size),
            "--device", args.device,
            "--out", str(out_dir / "eval_wt103.json"),
        ]
        _run(command, dry_run=args.dry_run)

    if "validate" in stages:
        command = [
            python, str(ROOT / "validate_run.py"),
            "--run", str(out_dir),
            "--test", str(out_dir / "test_domains.json"),
            "--out", str(out_dir / "validation.json"),
        ]
        if args.baseline_test:
            command.extend(["--baseline", args.baseline_test])
        _run(command, dry_run=args.dry_run)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
