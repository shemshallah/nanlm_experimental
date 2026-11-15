#!/usr/bin/env python3
"""Compute exact byte/character denominators for the fixed eval windows."""

from __future__ import annotations

import argparse
import bisect
import math
from pathlib import Path

import numpy as np

try:
    from .layered_tokenizer import _char_byte_offsets, load_artifact
except ImportError:  # script execution from z3/
    from layered_tokenizer import _char_byte_offsets, load_artifact  # type: ignore


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--valid-text", required=True)
    parser.add_argument("--ctx", type=int, default=256)
    parser.add_argument("--eval-batches", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--token-loss", type=float, required=True)
    parser.add_argument(
        "--layered-model",
        default=str(
            Path(__file__).resolve().with_name("tokenizers")
            / "layered_v1" / "layered_tokenizer.json"
        ),
    )
    args = parser.parse_args()
    text = Path(args.valid_text).expanduser().read_text(
        encoding="utf-8", errors="ignore"
    )
    tokenizer = load_artifact(Path(args.layered_model).expanduser())
    native = getattr(tokenizer.subword, "native", None)
    if native is not None:
        encoded = native.encode(text)
        ids = np.asarray(encoded.ids, dtype=np.int64)
        token_bytes = [
            tokenizer.subword.token_bytes[int(token_id)]
            for token_id in ids.tolist()
        ]
        char_lengths = [
            int(end) - int(start) for start, end in encoded.offsets
        ]
    else:
        encoded = tokenizer.subword.encode(text)
        ids = np.asarray(encoded.ids, dtype=np.int64)
        token_bytes = encoded.token_bytes
        spans = encoded.spans
        offsets = _char_byte_offsets(text)
        char_lengths = [
            bisect.bisect_left(offsets, end)
            - bisect.bisect_left(offsets, start)
            for start, end in spans
        ]
    positions = np.random.default_rng(args.seed + 100_003).integers(
        args.ctx, len(ids), size=args.eval_batches * args.batch_size
    )
    target_tokens = 0
    target_bytes = 0
    target_chars = 0
    for position in positions:
        start = int(position) - args.ctx
        target_tokens += args.ctx - 1
        for index in range(start + 1, int(position)):
            target_bytes += len(token_bytes[index])
            target_chars += char_lengths[index]
    bits = args.token_loss / math.log(2)
    print(
        {
            "tokens": target_tokens,
            "bytes": target_bytes,
            "chars": target_chars,
            "bytes_per_token": target_bytes / max(1, target_tokens),
            "chars_per_token": target_chars / max(1, target_tokens),
            "bits_per_byte": bits * target_tokens / max(1, target_bytes),
            "bits_per_char": bits * target_tokens / max(1, target_chars),
            "eval_positions": positions.tolist(),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
