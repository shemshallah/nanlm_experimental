#!/usr/bin/env python3
"""Split WikiText-103 raw text into article-level documents for the diamond engine.

The engine treats each file as one document, so writing one file per WikiText
article gives a real whole-document held-out split. Long articles are left
intact; the engine reads the first --file-read-chars characters of each file.
"""
import argparse
import os
import re
import sys
import urllib.request


def is_article_title(line: str) -> bool:
    s = line.rstrip("\n")
    # WikiText article titles are " = Title = "; section headings are
    # " = = Section = = " and must not start a new document.
    return (
        s.startswith(" = ")
        and s.endswith(" = ")
        and not s.startswith(" = = ")
    )


def download(url: str, dest: str) -> None:
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"using existing {dest} ({os.path.getsize(dest):,} bytes)")
        return
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    print(f"downloading {url} -> {dest}", flush=True)
    urllib.request.urlretrieve(url, dest)
    print(f"downloaded {os.path.getsize(dest):,} bytes", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="wiki.train.raw path")
    ap.add_argument("--out", required=True, help="output corpus root")
    ap.add_argument("--download-url", default="")
    ap.add_argument("--max-articles", type=int, default=4000)
    ap.add_argument("--max-chars", type=int, default=80_000_000)
    args = ap.parse_args()

    if args.download_url:
        download(args.download_url, args.raw)
    if not os.path.exists(args.raw):
        raise SystemExit(f"missing raw file: {args.raw}")

    doc_dir = os.path.join(args.out, "wikitext103")
    os.makedirs(doc_dir, exist_ok=True)
    # Remove stale documents from a previous split so document IDs are stable.
    for name in os.listdir(doc_dir):
        if name.endswith(".txt"):
            os.remove(os.path.join(doc_dir, name))

    current = []
    current_chars = 0
    written = 0
    written_chars = 0
    skipped_chars = 0

    def flush() -> bool:
        nonlocal current, current_chars, written, written_chars
        if not current:
            return True
        text = "".join(current).strip()
        if len(text) >= 32:
            if written >= args.max_articles or written_chars >= args.max_chars:
                return False
            path = os.path.join(doc_dir, f"{written:05d}.txt")
            with open(path, "w", encoding="utf-8") as f:
                f.write(text + "\n")
            written += 1
            written_chars += len(text)
        current = []
        current_chars = 0
        return True

    with open(args.raw, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if is_article_title(line):
                if current and not flush():
                    break
                current = [line]
                current_chars = len(line)
            else:
                if current:
                    current.append(line)
                    current_chars += len(line)
                else:
                    skipped_chars += len(line)
            if written >= args.max_articles or written_chars >= args.max_chars:
                break
    flush()

    print(
        f"wrote {written} documents, {written_chars:,} chars "
        f"to {doc_dir} (skipped prefix {skipped_chars:,} chars)",
        flush=True,
    )
    if written < 10:
        raise SystemExit("too few documents produced; check the raw WikiText format")


if __name__ == "__main__":
    main()
