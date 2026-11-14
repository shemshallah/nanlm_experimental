#!/usr/bin/env python3
"""Stream a FineWeb-Edu subset into document files for the diamond engine."""
import argparse
import os
import sys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="corpus root; writes fineweb/*.txt")
    ap.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    ap.add_argument("--config", default="sample-10BT")
    ap.add_argument("--split", default="train")
    ap.add_argument("--max-docs", type=int, default=4000)
    ap.add_argument("--max-chars", type=int, default=80_000_000)
    ap.add_argument("--chars-per-doc", type=int, default=20000)
    args = ap.parse_args()

    try:
        from datasets import load_dataset
    except Exception as e:
        raise SystemExit(
            "datasets is not installed. Run: pip install -q datasets\n"
            f"import error: {e}"
        )

    doc_dir = os.path.join(args.out, "fineweb")
    os.makedirs(doc_dir, exist_ok=True)
    for name in os.listdir(doc_dir):
        if name.endswith(".txt"):
            os.remove(os.path.join(doc_dir, name))

    print(
        f"streaming {args.dataset} config={args.config} split={args.split}",
        flush=True,
    )
    ds = load_dataset(
        args.dataset, args.config, split=args.split, streaming=True
    )

    written = 0
    chars = 0
    for row in ds:
        text = row.get("text") or row.get("content") or ""
        text = text.strip()
        if len(text) < 64:
            continue
        text = text[: args.chars_per_doc]
        path = os.path.join(doc_dir, f"{written:05d}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        written += 1
        chars += len(text)
        if written % 250 == 0:
            print(f"  {written} docs, {chars:,} chars", flush=True)
        if written >= args.max_docs or chars >= args.max_chars:
            break

    print(
        f"wrote {written} FineWeb-Edu documents, {chars:,} chars to {doc_dir}",
        flush=True,
    )
    if written < 10:
        raise SystemExit("too few FineWeb-Edu documents; check dataset access/internet")


if __name__ == "__main__":
    main()
