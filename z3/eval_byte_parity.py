#!/usr/bin/env python3
"""Byte-marginal parity evaluation for a 64k-subword checkpoint.

Token top-1/top-5 over a 64,060-way vocabulary are NOT comparable to the old
256-way byte model's top-1/top-5.  This script maps the token distribution to
the next target token's first-byte marginal and reports 256-way byte top-k,
which is the comparable metric.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

try:
    from .eval_v3 import encode_file, load_checkpoint, load_tokenizer
    from .layered_tokenizer import load_artifact
except ImportError:  # script execution from z3/
    from eval_v3 import encode_file, load_checkpoint, load_tokenizer
    from layered_tokenizer import load_artifact


def first_byte_table(vocab_size: int, layered_model: Path, device: torch.device):
    layered = load_artifact(layered_model)
    raw = [int(item[0]) if item else 0 for item in layered.subword.token_bytes]
    if len(raw) < vocab_size:
        raw = raw + [0] * (vocab_size - len(raw))
    return torch.tensor(raw[:vocab_size], dtype=torch.long, device=device)


@torch.inference_mode()
def evaluate_corpus(
    model,
    ids: list[int],
    table: torch.Tensor,
    *,
    ctx: int,
    batch_size: int,
    device: torch.device,
) -> dict:
    starts = list(range(0, len(ids) - 1, ctx))
    full = [start for start in starts if start + ctx + 1 <= len(ids)]
    tail = [start for start in starts if start + ctx + 1 > len(ids)]
    correct1 = correct5 = total = 0

    def consume(windows: list[list[int]]) -> None:
        nonlocal correct1, correct5, total
        xb = torch.tensor(windows, dtype=torch.long, device=device)
        logits, _mtp, _aux = model(xb)
        probs = torch.softmax(logits[:, :-1].float(), dim=-1)
        byte_probs = torch.zeros(
            *probs.shape[:2], 256, device=device, dtype=probs.dtype
        ).scatter_add_(
            2, table.view(1, 1, -1).expand_as(probs), probs
        )
        targets = table[xb[:, 1:]]
        total += int(targets.numel())
        correct1 += int((byte_probs.argmax(-1) == targets).sum())
        correct5 += int(
            (byte_probs.topk(5, dim=-1).indices
             == targets.unsqueeze(-1)).any(-1).sum()
        )

    for index in range(0, len(full), batch_size):
        consume([ids[start:start + ctx + 1] for start in full[index:index + batch_size]])
    for start in tail:
        consume([ids[start:]])
    if total < 1:
        raise RuntimeError("corpus produced no target bytes")
    return {
        "targets": total,
        "byte_top1": correct1 / total,
        "byte_top5": correct5 / total,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--corpus", action="append", nargs="+", required=True,
        metavar="[NAME=]PATH",
    )
    parser.add_argument("--tokenizer", choices=("v4", "layered"), default="v4")
    parser.add_argument("--ctx", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default="byte_parity.json")
    parser.add_argument(
        "--layered-model",
        default=str(Path(__file__).resolve().with_name("tokenizers")
                    / "layered_v1" / "layered_tokenizer.json"),
    )
    args = parser.parse_args()
    device = torch.device(args.device)
    model, cfg, meta = load_checkpoint(args.checkpoint)
    model = model.to(device).eval()
    tokenizer = load_tokenizer(args.tokenizer)
    table = first_byte_table(
        cfg.vocab_size, Path(args.layered_model), device
    )
    specs = []
    for group in args.corpus:
        for item in group:
            if "=" in item:
                name, path = item.split("=", 1)
            else:
                path = item
                name = Path(item).stem
            specs.append((name, Path(path).expanduser()))
    results = {}
    for name, path in specs:
        ids = encode_file(path, tokenizer)
        results[name] = evaluate_corpus(
            model, ids, table, ctx=args.ctx,
            batch_size=args.batch_size, device=device,
        )
        print(name, results[name], flush=True)
    payload = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": meta.get("step"),
        "ctx": args.ctx,
        "tokenizer": tokenizer.kind,
        "corpora": results,
    }
    Path(args.out).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
