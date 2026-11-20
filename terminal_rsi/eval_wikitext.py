#!/usr/bin/env python3
"""External WikiText byte/token/word held-out evaluation for Terminal-RSI.

The 6-hour corpus deliberately excludes WikiText, so these raw test files are
never seen during training.  This evaluator reuses the training-side BPE,
causal feature builder and hierarchical metric code, and scores fixed,
evenly-spaced excerpts (no cherry-picking, no target leakage).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

from .hierarchical_metrics import evaluate_dev_excerpts
from .legacy_v103_reference import read_bpe_ranks
from .unified import device_for
from .verbatim_v103 import VerbatimV103RSI


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _clean_wikitext(raw: bytes) -> bytes:
    """Drop section-heading lines and collapse blank lines (GPT-2 style)."""
    text = raw.decode("utf-8", errors="ignore")
    lines = [line for line in text.splitlines() if not line.startswith("=")]
    cleaned = "\n".join(line for line in lines if line.strip())
    return (cleaned.strip() + "\n").encode("utf-8")


def _prepare(name: str, raw_path: Path, work: Path, books: int,
             length: int) -> dict:
    cleaned = _clean_wikitext(raw_path.read_bytes())
    if len(cleaned) < length + 4096:
        raise ValueError(f"{name}: cleaned text too short ({len(cleaned)} bytes)")
    out = work / f"{name}.clean.txt"
    out.write_bytes(cleaned)
    span = max(1, len(cleaned) - length)
    starts = [int(index * span / max(1, books - 1)) for index in range(books)]
    selected = []
    for index, start in enumerate(starts):
        selected.append({
            "gutenberg_id": f"{name}-{index:02d}",
            "relative_path": out.name,
            "excerpt_start": int(start),
            "length": int(length),
        })
    manifest = work / f"{name}.manifest.json"
    manifest.write_text(json.dumps({
        "schema": "terminal-rsi-external-wikitext-v1",
        "name": name,
        "raw_path": str(raw_path),
        "raw_sha256": _sha256(raw_path),
        "clean_path": str(out),
        "clean_sha256": _sha256(out),
        "clean_bytes": len(cleaned),
        "books": books,
        "excerpt_length": length,
        "selected_books": selected,
    }, indent=2) + "\n")
    return {
        "name": name,
        "manifest": str(manifest),
        "raw_sha256": _sha256(raw_path),
        "clean_sha256": _sha256(out),
        "clean_bytes": len(cleaned),
        "excerpts": books,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--bpe-merges", type=Path, required=True)
    parser.add_argument("--raw", action="append", required=True,
                        metavar="NAME=PATH",
                        help="External raw test file, e.g. wt2=/data/wiki.test.raw")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--books", type=int, default=12)
    parser.add_argument("--excerpt-length", type=int, default=4096)
    parser.add_argument("--max-positions", type=int, default=256)
    parser.add_argument("--context", type=int, default=256)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--mode", default=None,
                        help="Override checkpoint mode (cpu/t4/blackwell)")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    ranks = read_bpe_ranks(args.bpe_merges)
    if not ranks:
        raise SystemExit("BPE merge ranks are empty")
    checkpoint = torch.load(args.checkpoint, map_location="cpu",
                            weights_only=True)
    if checkpoint.get("schema") != (
            "terminal-rsi-verbatim-v103-plus-upgrades-v1"):
        raise SystemExit("Not a unified Terminal-RSI checkpoint")
    mode = args.mode or checkpoint.get("mode") or checkpoint["config"]["name"]
    device = device_for(mode)
    model = VerbatimV103RSI(mode).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    print(f"loaded {args.checkpoint} mode={mode} step={checkpoint.get('steps')} "
          f"device={device}", flush=True)

    args.work_dir.mkdir(parents=True, exist_ok=True)
    prepared = []
    pending = []
    for item in args.raw:
        if "=" not in item:
            raise SystemExit(f"--raw needs NAME=PATH, got {item!r}")
        name, path = item.split("=", 1)
        info = _prepare(name, Path(path), args.work_dir,
                        args.books, args.excerpt_length)
        prepared.append(info)
        pending.append((name, info["manifest"]))

    report = {
        "schema": "terminal-rsi-external-wikitext-eval-v1",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "checkpoint_step": int(checkpoint.get("steps", 0)),
        "mode": mode,
        "device": str(device),
        "bpe_merges": str(args.bpe_merges),
        "bpe_sha256": _sha256(args.bpe_merges),
        "context": args.context,
        "max_positions": args.max_positions,
        "batch": args.batch,
        "prepared": prepared,
        "tests": {},
    }
    for name, manifest_path in pending:
        started = time.perf_counter()
        result = evaluate_dev_excerpts(
            model,
            corpus_dir=args.work_dir,
            dev_manifest=manifest_path,
            bpe_merges=args.bpe_merges,
            context=args.context,
            books=args.books,
            max_positions=args.max_positions,
            eval_batch=args.batch,
            device=device,
        )
        result["eval_seconds"] = round(time.perf_counter() - started, 2)
        report["tests"][name] = result
        byte = result["byte"]
        token = result.get("token") or {}
        word = result.get("word") or {}
        print(f"{name}: bpb={byte['bpb']:.5f} byte_top5={byte['top5']:.4f} "
              f"token_top5={token.get('top5', float('nan')):.4f} "
              f"word_top5={word.get('top5', float('nan')):.4f} "
              f"positions={result['positions']}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print("wrote", args.output, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
