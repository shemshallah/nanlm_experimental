#!/usr/bin/env python3
"""Retune the layered word vocabulary from v4c token bins.

The original layered artifact was fitted on a 5MB FineWeb sample.  This tool
streams the uint16 token bins used for training, reconstructs the exact text
through the byte-complete subword layer, counts word spans, and writes a new
layered artifact with the selected word vocabulary.

It deliberately does not modify the token IDs or the subword vocabulary.  The
output is alignment metadata for the concept loop, not a new text tokenizer.

Example::

    python z3/tune_word_vocab_bins.py \
        --artifact z3/tokenizers/layered_v1/layered_tokenizer.json \
        --token-bin data/fineweb_v4c_train.bin \
        --vocab-size 64000 \
        --out runs/layered_v4c/layered_tokenizer.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    from .layered_tokenizer import (
        WordLayer,
        load_artifact,
        write_artifact,
    )
except ImportError:  # script execution from z3/
    from layered_tokenizer import (  # type: ignore
        WordLayer,
        load_artifact,
        write_artifact,
    )


_DTYPES = {
    "uint16": np.uint16,
    "uint32": np.uint32,
    "int32": np.int32,
}


def _iter_chunks(path: Path, dtype: np.dtype, chunk_tokens: int,
                 max_tokens: int) -> Iterable[np.ndarray]:
    remaining = max_tokens if max_tokens > 0 else None
    with path.open("rb") as handle:
        while remaining is None or remaining > 0:
            count = chunk_tokens if remaining is None else min(
                chunk_tokens, remaining
            )
            ids = np.fromfile(handle, dtype=dtype, count=count)
            if ids.size == 0:
                break
            if remaining is not None:
                remaining -= int(ids.size)
            yield ids


def tune(
    token_bins: list[Path],
    artifact: Path,
    out: Path,
    *,
    vocab_size: int = 64_000,
    dtype: str = "uint16",
    chunk_tokens: int = 2_000_000,
    max_tokens: int = 0,
    report_path: Path | None = None,
) -> dict:
    tokenizer = load_artifact(artifact)
    subword = tokenizer.subword
    token_dtype = np.dtype(_DTYPES[dtype])
    counts: Counter[str] = Counter()
    n_tokens = 0
    n_chunks = 0
    n_occurrences = 0
    remaining = max_tokens if max_tokens > 0 else None

    for path in token_bins:
        path = Path(path).expanduser()
        if not path.is_file():
            raise SystemExit(f"token bin not found: {path}")
        if remaining == 0:
            break
        for ids in _iter_chunks(
            path, token_dtype, chunk_tokens,
            remaining if remaining is not None else 0,
        ):
            n_tokens += int(ids.size)
            n_chunks += 1
            if remaining is not None:
                remaining -= int(ids.size)
            # Decoding a token chunk is exact at the token-byte level.  A
            # UTF-8 code point or word crossing a chunk boundary can be split;
            # that affects at most one word per chunk and is negligible for a
            # corpus-scale vocabulary count.
            text = subword.decode(ids.tolist())
            spans = WordLayer().spans(text)
            chunk_counts = Counter(
                piece for piece, _, _, is_word in spans if is_word
            )
            counts.update(chunk_counts)
            n_occurrences += sum(chunk_counts.values())
            print(
                f"[words] chunks={n_chunks} tokens={n_tokens:,} "
                f"types={len(counts):,} occurrences={n_occurrences:,}",
                flush=True,
            )
            if remaining == 0:
                break

    if not counts:
        raise SystemExit("no word tokens found in the supplied bins")
    selected = [
        word for word, _ in counts.most_common(max(1, int(vocab_size)))
    ]
    fitted = WordLayer(
        vocab_size=len(selected),
        vocab={word: index for index, word in enumerate(selected)},
    )
    total = sum(counts.values())
    covered = sum(counts[word] for word in selected)
    tokenizer.word = fitted
    destination = write_artifact(tokenizer, out, include_word_vocab=True)
    report = {
        "artifact": str(destination),
        "source_artifact": str(Path(artifact).expanduser()),
        "token_bins": [str(Path(path).expanduser()) for path in token_bins],
        "dtype": dtype,
        "tokens_seen": n_tokens,
        "chunks": n_chunks,
        "word_types": len(counts),
        "word_occurrences": total,
        "vocab_size": fitted.vocab_size,
        "coverage": covered / max(1, total),
        "min_count": min(
            (counts[word] for word in selected), default=0
        ),
        "max_count": max(
            (counts[word] for word in selected), default=0
        ),
    }
    if report_path is not None:
        target = Path(report_path).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact", required=True,
        help="existing layered tokenizer artifact to update",
    )
    parser.add_argument(
        "--token-bin", nargs="+", required=True,
        help="one or more v4c uint16/uint32 token bins",
    )
    parser.add_argument("--out", required=True, help="output artifact path")
    parser.add_argument("--vocab-size", type=int, default=64_000)
    parser.add_argument(
        "--dtype", choices=tuple(_DTYPES), default="uint16"
    )
    parser.add_argument("--chunk-tokens", type=int, default=2_000_000)
    parser.add_argument("--max-tokens", type=int, default=0)
    parser.add_argument("--report", default="")
    args = parser.parse_args()
    report = tune(
        [Path(path) for path in args.token_bin],
        Path(args.artifact),
        Path(args.out),
        vocab_size=args.vocab_size,
        dtype=args.dtype,
        chunk_tokens=max(1, args.chunk_tokens),
        max_tokens=max(0, args.max_tokens),
        report_path=Path(args.report) if args.report else None,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
