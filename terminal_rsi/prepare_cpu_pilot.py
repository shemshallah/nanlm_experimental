"""Prepare a reproducible 1,000-update Terminal-RSI CPU pilot from NaNROUTER.

Creates NEW English-only train bytes and a disjoint 64-book DEV. No original
checkpoint, historic DEV, sealed TEST, or C oracle is accessed. Intended for
a disposable fresh-start benchmark; continue protected v103 separately.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

from terminal_rsi.build_fresh_dev import (
    _write_if_unchanged_or_new, build, language_rejection,
    metadata, validate_fresh_dev,
)

ID = re.compile(r"^(\d+)--")


def build_cpu_pilot(*, corpus_dir: Path, output_dir: Path,
                    train_bytes: int = 2_000_000,
                    train_chunk_bytes: int = 65_536,
                    dev_books: int = 64, dev_positions: int = 1024,
                    seed: int = 103, min_remaining_books: int = 64) -> dict:
    if train_bytes < 4096 or train_chunk_bytes < 4096:
        raise ValueError("Need enough independent training bytes for v103 history")
    if dev_books < 1:
        raise ValueError("DEV must contain at least one distinct book")
    books = {}
    for path in sorted(corpus_dir.rglob("*.txt")):
        match = ID.match(path.name)
        if not match:
            continue
        book_id = str(int(match.group(1)))
        if book_id in books:
            raise ValueError("Duplicate Gutenberg book ID: " + book_id)
        books[book_id] = path
    sources = metadata(corpus_dir)
    candidates = []
    rejected = {}
    for bid, path in books.items():
        content = path.read_bytes()
        reason = language_rejection(content, path.name, sources.get(bid, {}))
        if reason is not None:
            rejected[reason] = rejected.get(reason, 0) + 1
            continue
        if len(content) < 4096:
            rejected["too_short"] = rejected.get("too_short", 0) + 1
            continue
        candidates.append((bid, path, content))
    # Reserve at least DEV book_count eligible identities before TRAIN selection.
    if len(candidates) < dev_books + 1:
        raise ValueError("Insufficient English non-Russian books for disjoint TRAIN and DEV")
    order = sorted(candidates, key=lambda x: (
        hashlib.sha256(f"cpu-train:{seed}:{x[0]}".encode()).hexdigest(), int(x[0])))
    train = bytearray()
    training_ids = []
    chunks = []
    for bid, path, content in order[:max(0, len(order) - max(dev_books, min_remaining_books))]:
        if len(train) >= train_bytes:
            break
        count = min(train_bytes - len(train), train_chunk_bytes, len(content))
        if count < 4096:
            break
        begin = len(train)
        train.extend(content[:count])
        training_ids.append(bid)
        chunks.append({"gutenberg_id": bid, "bytes": count, "start": begin,
                       "end": len(train), "book_sha256": hashlib.sha256(content).hexdigest()})
    if len(train) < train_bytes:
        raise ValueError(f"Only collected {len(train)} of {train_bytes} requested TRAIN bytes")
    output_dir.mkdir(parents=True, exist_ok=True)
    train_file = output_dir / "train_cpu_pilot.bin"
    train_id_file = output_dir / "train_book_ids.json"
    _write_if_unchanged_or_new(train_file, bytes(train))
    _write_if_unchanged_or_new(
        train_id_file,
        (json.dumps({"book_ids": training_ids}, sort_keys=True, indent=2) + "\n").encode())
    dev = build(
        corpus_dir=corpus_dir, train_stream=train_file,
        output_dir=output_dir, book_count=dev_books,
        positions_per_book=dev_positions, skip=512, seed=seed,
        train_book_ids=train_id_file,
        source_release="corpora-short-stories-0004",
    )
    validate_fresh_dev(
        output_dir / "win_dev.bin", output_dir / "win_dev.manifest.json",
        train_sha256=hashlib.sha256(train).hexdigest())
    chosen_ids = {v["gutenberg_id"] for v in dev["selected_books"]}
    if chosen_ids.intersection(training_ids):
        raise AssertionError("TRAIN and DEV Gutenberg book identities overlap")
    report = {
        "schema": "terminal-rsi-nanrouter-cpu-pilot-v1",
        "source": "shemshallah/NaNROUTER/corpora/short_stories",
        "seed": seed,
        "train_bytes": len(train),
        "train_sha256": hashlib.sha256(train).hexdigest(),
        "train_books": len(training_ids),
        "train_segments": chunks,
        "dev_books": dev_books,
        "dev_scored_bytes": dev["scored_positions"],
        "dev_sha256": dev["dev_sha256"],
        "book_disjoint_train_dev_verified_by_gutenberg_ids": True,
        "russian_exclusion": dev["russian_exclusion"],
        "source_rejection_counts": rejected,
        "sealed_test_never_opened": True,
        "historic_dev_never_restored": True,
    }
    _write_if_unchanged_or_new(
        output_dir / "cpu_pilot.manifest.json",
        (json.dumps(report, sort_keys=True, indent=2) + "\n").encode())
    return {k: v for k, v in report.items() if k != "train_segments"}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus-dir", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--train-bytes", type=int, default=2_000_000)
    p.add_argument("--train-chunk-bytes", type=int, default=65_536)
    p.add_argument("--dev-books", type=int, default=64)
    p.add_argument("--dev-positions", type=int, default=1024)
    p.add_argument("--seed", type=int, default=103)
    args = p.parse_args(argv)
    print(json.dumps(build_cpu_pilot(
        corpus_dir=args.corpus_dir, output_dir=args.output_dir,
        train_bytes=args.train_bytes,
        train_chunk_bytes=args.train_chunk_bytes,
        dev_books=args.dev_books, dev_positions=args.dev_positions,
        seed=args.seed), sort_keys=True, indent=2), flush=True)


if __name__ == "__main__":
    main()
