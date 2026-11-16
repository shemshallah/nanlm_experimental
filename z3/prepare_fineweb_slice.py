#!/usr/bin/env python3
"""Split a range-limited FineWeb-Edu shard sample into train/valid text.

The workflow asks the public Kaggle API for only the first N bytes of a shard,
so no full corpus is downloaded.  The final partial line is discarded and the
train/valid cut is placed on a newline so neither split starts mid-document.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def split_text(raw: bytes, valid_fraction: float) -> tuple[bytes, bytes]:
    if not raw:
        raise ValueError("input is empty")
    # Drop the final partial line created by the byte-range request.
    last_newline = raw.rfind(b"\n")
    if last_newline > 0:
        raw = raw[: last_newline + 1]
    if len(raw) < 2:
        raise ValueError("input is too small after trimming to a newline")
    target = int(len(raw) * (1.0 - valid_fraction))
    target = min(max(target, 1), len(raw) - 1)
    split = raw.rfind(b"\n", 0, target)
    if split <= 0:
        split = target
    else:
        split += 1
    train, valid = raw[:split], raw[split:]
    if not train or not valid:
        raise ValueError("train or validation split is empty")
    return train, valid


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="raw text sample")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--valid-fraction", type=float, default=0.04)
    parser.add_argument("--slice-mb", type=int, default=0,
                        help="optional second cap in MiB (0 disables)")
    args = parser.parse_args()
    raw = Path(args.input).expanduser().read_bytes()
    if args.slice_mb > 0:
        raw = raw[: args.slice_mb * 1024 * 1024]
    train, valid = split_text(raw, args.valid_fraction)
    out = Path(args.out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "train.txt").write_bytes(train)
    (out / "valid.txt").write_bytes(valid)
    stats = {
        "input_bytes": len(raw),
        "train_bytes": len(train),
        "valid_bytes": len(valid),
        "valid_fraction": args.valid_fraction,
        "train_path": str(out / "train.txt"),
        "valid_path": str(out / "valid.txt"),
    }
    print(json.dumps(stats, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
