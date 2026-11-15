#!/usr/bin/env python3
"""Prepare token bins for the GitHub-hosted CPU architecture run.

The workflow downloads the small public WikiText-2 raw files on the runner and
uses this script to encode them with the bundled layered tokenizer.  No corpus
or checkpoint is downloaded to the phone.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from .layered_tokenizer import load_artifact
except ImportError:  # script execution from z3/
    from layered_tokenizer import load_artifact  # type: ignore


def encode_text(path: Path, tokenizer, dtype: str) -> np.ndarray:
    text = path.read_text(encoding="utf-8", errors="ignore")
    # Only subword IDs are needed for the flat training bin.  Do not build
    # word/sentence/math objects over the entire corpus during preparation.
    encoded = tokenizer.subword.encode(text)
    ids = np.asarray(encoded.ids, dtype=np.int64)
    dtype_value = np.dtype(dtype)
    if ids.size and int(ids.max()) >= np.iinfo(dtype_value).max:
        raise SystemExit(
            f"token id {int(ids.max())} does not fit in {dtype}"
        )
    return ids.astype(dtype_value, copy=False)


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
    args = parser.parse_args()
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = load_artifact(Path(args.layered_model).expanduser())
    train = encode_text(Path(args.train_text).expanduser(), tokenizer, args.dtype)
    valid = encode_text(Path(args.valid_text).expanduser(), tokenizer, args.dtype)
    train_path = out / "train.bin"
    valid_path = out / "valid.bin"
    train.tofile(train_path)
    valid.tofile(valid_path)
    meta = {
        "dtype": args.dtype,
        "vocab_size": tokenizer.vocab_size,
        "train_tokens": int(train.size),
        "valid_tokens": int(valid.size),
        "subword_backend": tokenizer.subword.manifest().get("backend"),
        "train_bin": str(train_path),
        "valid_bin": str(valid_path),
    }
    (out / "meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(meta, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
