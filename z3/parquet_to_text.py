#!/usr/bin/env python3
"""Flatten a HuggingFace wikitext parquet split to plain text.

Used only on the GitHub runner for the public WikiText-103 test split.  The
parquet is read in batches so the conversion does not duplicate the whole table
in memory.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--column", default="text")
    parser.add_argument("--batch-size", type=int, default=4096)
    args = parser.parse_args()
    import pyarrow.parquet as pq

    source = Path(args.input).expanduser()
    target = Path(args.output).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    written = 0
    parquet = pq.ParquetFile(source)
    with target.open("w", encoding="utf-8") as handle:
        for batch in parquet.iter_batches(
            batch_size=args.batch_size, columns=[args.column]
        ):
            for value in batch.column(0).to_pylist():
                rows += 1
                if value is None:
                    continue
                text = value if isinstance(value, str) else str(value)
                if not text:
                    continue
                handle.write(text)
                if not text.endswith("\n"):
                    handle.write("\n")
                written += len(text)
    print({"rows": rows, "characters": written, "output": str(target)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
