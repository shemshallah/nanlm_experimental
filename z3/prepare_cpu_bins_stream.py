#!/usr/bin/env python3
"""Memory-bounded parallel tokenizer for large text bins.

The original preparation script encodes the whole corpus in one process.  On a
32 GB container a 500 MB text slice can transiently use 30 GB.  This version
tokenizes newline-aligned chunks in a small process pool and writes uint16 bins
incrementally.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

try:
    from .layered_tokenizer import load_artifact
except ImportError:  # script execution from z3/
    from layered_tokenizer import load_artifact  # type: ignore


_WORKER_TOKENIZER = None
_WORKER_DTYPE = "uint16"


def _init_worker(model_path: str, dtype: str) -> None:
    global _WORKER_TOKENIZER, _WORKER_DTYPE
    _WORKER_TOKENIZER = load_artifact(Path(model_path))
    _WORKER_DTYPE = dtype


def _encode_chunk(text: str) -> np.ndarray:
    assert _WORKER_TOKENIZER is not None
    native = getattr(_WORKER_TOKENIZER.subword, "native", None)
    if native is not None:
        ids = np.asarray(native.encode(text).ids, dtype=np.int64)
    else:
        ids = np.asarray(
            _WORKER_TOKENIZER.subword.encode(text).ids, dtype=np.int64
        )
    dtype = np.dtype(_WORKER_DTYPE)
    if ids.size and int(ids.max()) >= np.iinfo(dtype).max:
        raise ValueError(
            f"token id {int(ids.max())} does not fit in {_WORKER_DTYPE}"
        )
    return ids.astype(dtype, copy=False)


def _read_chunks(path: Path, chunk_bytes: int):
    """Yield decoded text chunks whose boundaries fall on newlines."""
    with path.open("rb") as handle:
        buffer = bytearray()
        for line in handle:
            buffer.extend(line)
            if len(buffer) >= chunk_bytes:
                yield buffer.decode("utf-8", errors="ignore")
                buffer.clear()
        if buffer:
            yield buffer.decode("utf-8", errors="ignore")


def encode_file(
    path: Path,
    out_path: Path,
    *,
    model_path: Path,
    dtype: str,
    workers: int,
    chunk_bytes: int,
) -> dict:
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    total = 0
    max_id = 0
    with ProcessPoolExecutor(
        max_workers=max(1, workers),
        initializer=_init_worker,
        initargs=(str(model_path), dtype),
    ) as pool:
        with tmp.open("wb") as out:
            for ids in pool.map(
                _encode_chunk,
                _read_chunks(path, chunk_bytes),
                chunksize=1,
            ):
                if ids.size:
                    max_id = max(max_id, int(ids.max()))
                    ids.tofile(out)
                    total += int(ids.size)
    tmp.replace(out_path)
    return {"tokens": total, "max_token_id": max_id}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-text", required=True)
    parser.add_argument("--valid-text", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--layered-model",
        default=str(
            Path(__file__).resolve().with_name("tokenizers")
            / "layered_v1" / "layered_tokenizer.json"
        ),
    )
    parser.add_argument("--dtype", choices=("uint16", "uint32"), default="uint16")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--chunk-mb", type=int, default=8)
    args = parser.parse_args()
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    model_path = Path(args.layered_model).expanduser().resolve()
    chunk_bytes = max(1, args.chunk_mb) * 1024 * 1024
    train = encode_file(
        Path(args.train_text).expanduser(), out / "train.bin",
        model_path=model_path, dtype=args.dtype,
        workers=args.workers, chunk_bytes=chunk_bytes,
    )
    valid = encode_file(
        Path(args.valid_text).expanduser(), out / "valid.bin",
        model_path=model_path, dtype=args.dtype,
        workers=args.workers, chunk_bytes=chunk_bytes,
    )
    tokenizer = load_artifact(model_path)
    meta = {
        "dtype": args.dtype,
        "vocab_size": tokenizer.vocab_size,
        "train_tokens": train["tokens"],
        "valid_tokens": valid["tokens"],
        "train_max_token_id": train["max_token_id"],
        "valid_max_token_id": valid["max_token_id"],
        "subword_backend": tokenizer.subword.manifest().get("backend"),
        "preparation": "parallel_newline_chunks",
        "workers": args.workers,
        "chunk_mb": args.chunk_mb,
        "train_bin": str(out / "train.bin"),
        "valid_bin": str(out / "valid.bin"),
    }
    (out / "meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(meta, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
