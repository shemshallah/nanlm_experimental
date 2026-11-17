"""Build a NEW, larger Terminal-RSI DEV from NaNROUTER English short stories.

No archived DEV recovery, TEST access, model imports, or training updates.
Input is the extracted NaNROUTER short_stories release (recommended 0004).
Output uses the original v103 <II count,skip> + <I length> + raw bytes format.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import tempfile

SCHEMA = "terminal-rsi-fresh-nanrouter-dev-v1"
ID_PATTERN = re.compile(r"^(\d+)--")
CYRILLIC = re.compile(r"[\u0400-\u052f\u1c80-\u1c8f\u2de0-\u2dff\ua640-\ua69f]")
RUSSIAN_TAG = re.compile(
    r"(?:^|[^a-z])(?:ru|rus|russian)(?:$|[^a-z])", re.I)
RUSSIAN_AUTHOR = re.compile(
    r"dostoe?vsk|dostoievsk|tolsto[yi]|chekhov|tchekhov|"
    r"turgenev|gogol|pushkin|gork[yi]|lermontov|"
    r"garshin|kuprin|korolenko|sholokhov|bun[in]|"
    r"andreye?v|leskov", re.I)
ENGLISH_WORDS = frozenset(
    ("the", "and", "of", "to", "in", "that", "it", "was", "he", "she",
     "with", "his", "her", "had", "they", "as", "is", "for", "on"))
WORDS = re.compile(r"[A-Za-z']+")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_ids(path: Path | None) -> set[str]:
    """Accept numeric Gutenberg IDs, plain lines, or an explicit JSON ID list."""
    if path is None:
        return set()
    raw = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        value = json.loads(raw)
        if isinstance(value, dict):
            value = value.get("book_ids", value.get("gutenberg_ids"))
        if not isinstance(value, list):
            raise ValueError("Exclusion JSON requires book_ids/gutenberg_ids array")
        ids = [str(v) for v in value]
    else:
        ids = [s.strip() for s in raw.splitlines() if s.strip() and not s.startswith("#")]
    if any(not s.isdigit() for s in ids):
        raise ValueError("Exclusion lists must contain Gutenberg numeric IDs only")
    return {str(int(v)) for v in ids}


def metadata(root: Path) -> dict[str, dict]:
    out = {}
    for path in sorted(root.rglob("SOURCES.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            bid = rec.get("gutenberg_id", rec.get("book_id", rec.get("id")))
            if bid is not None and str(bid).isdigit():
                out[str(int(bid))] = rec
    return out


def language_rejection(data: bytes, name: str, meta: dict) -> str | None:
    """Conservative Russian exclusion, plus positive English-text evidence."""
    langs = meta.get("languages", meta.get("language", []))
    if isinstance(langs, str):
        langs = [langs]
    if not isinstance(langs, list):
        langs = [str(langs)]
    if any(RUSSIAN_TAG.search(str(lang)) for lang in langs):
        return "russian_language_metadata"
    title_author = " ".join(
        (name, str(meta.get("title", "")), str(meta.get("author", ""))))
    if CYRILLIC.search(title_author) or RUSSIAN_TAG.search(title_author):
        return "russian_title_or_filename"
    if RUSSIAN_AUTHOR.search(title_author):
        return "russian_author_conservative_exclusion"
    try:
        text = data.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        return "invalid_utf8"
    if CYRILLIC.search(text):
        return "cyrillic_text"
    # Do not confuse a Latin-transliterated foreign text with English.
    tokens = [m.group(0).lower() for m in WORDS.finditer(text[:12000])]
    if len(tokens) < 120 or sum(t in ENGLISH_WORDS for t in tokens) < max(12, len(tokens) // 14):
        return "insufficient_english_evidence"
    return None


def overlap_with_train(data: bytes, train: bytes, stride: int = 256) -> bool:
    """Conservative exclusion whenever long candidate content occurs in TRAIN.

    A raw-stream substring check cannot *prove* ID-level book separation if
    TRAIN lacked a book-ID manifest. That limitation is recorded in metadata.
    """
    width = 64
    if len(data) < width:
        return False
    offsets = set(range(0, len(data) - width + 1, stride))
    offsets.add(len(data) - width)
    for offset in offsets:
        if data[offset:offset + width] in train:
            return True
    return False


def choose_excerpt(raw: bytes, book_id: str, seed: int, total: int):
    if len(raw) < total:
        raise ValueError("Book too short")
    token = f"{seed}:{book_id}:{sha256(raw)}".encode()
    offset = int.from_bytes(hashlib.sha256(token).digest()[:8], "little") % (len(raw) - total + 1)
    return offset, raw[offset:offset + total]


def _write_if_unchanged_or_new(path: Path, data: bytes):
    if path.exists():
        if path.read_bytes() == data:
            return
        raise FileExistsError(f"Refusing to overwrite different DEV artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".fresh_dev_", delete=False) as temp:
        temp.write(data)
        name = temp.name
    try:
        # Exclusive create ensures concurrent builders cannot replace a file.
        with open(path, "xb") as output, open(name, "rb") as src:
            while True:
                chunk = src.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
    finally:
        os.unlink(name)


def build(*, corpus_dir: Path, train_stream: Path, output_dir: Path,
          book_count: int = 64, positions_per_book: int = 1024,
          skip: int = 512, seed: int = 103,
          train_book_ids: Path | None = None,
          sealed_test_book_ids: Path | None = None,
          extra_excluded_book_ids: set[str] | None = None,
          source_release: str = "corpora-short-stories-0004") -> dict:
    if book_count < 1 or positions_per_book < 1 or skip < 32:
        raise ValueError("Positive book/position counts and >=32 warm-up required")
    if skip + positions_per_book > 1_048_576:
        raise ValueError("Original v103 reader limits each stored book to 1 MiB")
    if not corpus_dir.is_dir() or not train_stream.is_file():
        raise FileNotFoundError("Need extracted NaNROUTER short stories and original TRAIN stream")
    train = train_stream.read_bytes()
    if not train:
        raise ValueError("Empty TRAIN stream")
    train_digest = sha256(train)
    old_train_ids = read_ids(train_book_ids)
    old_test_ids = read_ids(sealed_test_book_ids)
    excluded = old_train_ids | old_test_ids | set(extra_excluded_book_ids or ())
    sources = metadata(corpus_dir)
    by_id: dict[str, Path] = {}
    for path in sorted(corpus_dir.rglob("*.txt")):
        m = ID_PATTERN.match(path.name)
        if m is None:
            continue
        bid = str(int(m.group(1)))
        if bid in by_id:
            raise ValueError(f"Duplicate Gutenberg ID {bid} in corpus input")
        by_id[bid] = path
    # Order independent of archive extraction or filesystem enumeration.
    order = sorted(by_id, key=lambda x: (sha256(f"{seed}:{x}".encode()), int(x)))
    selection = []
    payload = []
    rejected = Counter()
    fingerprints: set[bytes] = set()
    for bid in order:
        if len(selection) == book_count:
            break
        if bid in excluded:
            rejected["excluded_known_train_or_test_id"] += 1
            continue
        path = by_id[bid]
        raw = path.read_bytes()
        reason = language_rejection(raw, path.name, sources.get(bid, {}))
        if reason:
            rejected[reason] += 1
            continue
        if len(raw) < skip + positions_per_book:
            rejected["book_too_short"] += 1
            continue
        # Scan the FULL book against TRAIN; do not allow overlapping passages.
        if overlap_with_train(raw, train):
            rejected["candidate_training_substring_overlap"] += 1
            continue
        offset, excerpt = choose_excerpt(raw, bid, seed, skip + positions_per_book)
        # Distinct book IDs alone are insufficient against duplicate editions.
        marks = {excerpt[i:i + 64] for i in (0, skip, skip + positions_per_book - 64)}
        if fingerprints.intersection(marks):
            rejected["duplicate_book_passage"] += 1
            continue
        fingerprints.update(marks)
        selection.append({
            "gutenberg_id": bid,
            "relative_path": path.relative_to(corpus_dir).as_posix(),
            "book_sha256": sha256(raw),
            "excerpt_sha256": sha256(excerpt),
            "excerpt_start": offset,
            "length": len(excerpt),
        })
        payload.append(excerpt)
    if len(selection) != book_count:
        raise ValueError(
            f"Only {len(selection)}/{book_count} eligible English non-Russian books; "
            f"rejections={dict(rejected)}")
    win = bytearray(struct.pack("<II", book_count, skip))
    for chunk in payload:
        win += struct.pack("<I", len(chunk))
        win += chunk
    dev_bytes = bytes(win)
    manifest = {
        "schema": SCHEMA,
        "dataset": "new-english-only-dev-not-historic-v103",
        "source": "shemshallah/NaNROUTER/corpora/short_stories",
        "release_tag": source_release,
        "seed": seed,
        "books": book_count,
        "skip": skip,
        "positions_per_book": positions_per_book,
        "scored_positions": book_count * positions_per_book,
        "dev_sha256": sha256(dev_bytes),
        "source_train_sha256": train_digest,
        "train_id_exclusion_supplied": train_book_ids is not None,
        "sealed_test_id_exclusion_supplied": sealed_test_book_ids is not None,
        "id_level_train_disjoint_proven": train_book_ids is not None,
        "id_level_test_disjoint_proven": sealed_test_book_ids is not None,
        "train_passage_overlap_check": "full_book_64_byte_shingles_stride_256",
        "russian_exclusion": "all_cyrillic_text; Russian_language_metadata; Russian_title; conservative_Russian_author",
        "excluded_reason_counts": dict(sorted(rejected.items())),
        "selected_books": selection,
        "sealed_test_opened": False,
    }
    output = output_dir / "win_dev.bin"
    receipt = output_dir / "win_dev.manifest.json"
    _write_if_unchanged_or_new(output, dev_bytes)
    _write_if_unchanged_or_new(receipt, (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode())
    return manifest


def validate_fresh_dev(dev_path: Path, manifest_path: Path,
                       *, train_sha256: str | None = None) -> dict:
    """Reader-side gate against swapping DEV data or reusing historic C oracles."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA:
        raise ValueError("Not a new NaNROUTER DEV manifest")
    payload = dev_path.read_bytes()
    if sha256(payload) != manifest.get("dev_sha256"):
        raise ValueError("New DEV SHA-256 differs from its manifest")
    if train_sha256 is not None and train_sha256 != manifest.get("source_train_sha256"):
        raise ValueError("New DEV was built against a DIFFERENT TRAIN stream")
    if len(payload) < 8:
        raise ValueError("Truncated new DEV")
    books, skip = struct.unpack_from("<II", payload)
    per = manifest.get("positions_per_book")
    if books != manifest.get("books") or not isinstance(per, int) or per < 1 or skip != manifest.get("skip") or skip < 32:
        raise ValueError("New DEV shape/skip does not match manifest")
    if manifest.get("scored_positions") != books * per:
        raise ValueError("Incorrect new DEV scored-position count")
    selected = manifest.get("selected_books", [])
    if len(selected) != books or len({b["gutenberg_id"] for b in selected}) != books:
        raise ValueError("Missing/duplicate DEV Gutenberg book IDs")
    at = 8
    for meta in selected:
        if at + 4 > len(payload):
            raise ValueError("Truncated DEV book header")
        length = struct.unpack_from("<I", payload, at)[0]
        at += 4
        if length != skip + per or at + length > len(payload):
            raise ValueError("DEV book length does not match fixed evaluation window")
        if sha256(payload[at:at + length]) != meta["excerpt_sha256"]:
            raise ValueError("DEV book content differs from manifest")
        if CYRILLIC.search(payload[at:at + length].decode("utf-8", "ignore")):
            raise ValueError("DEV contains Cyrillic text")
        at += length
    if at != len(payload):
        raise ValueError("DEV contains trailing unscored bytes")
    if manifest.get("sealed_test_opened") is not False:
        raise ValueError("Unexpected TEST access in new DEV manifest")
    return manifest


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus-dir", type=Path, required=True,
                   help="Extracted NaNROUTER short_stories batch (NOT arbitrary texts)")
    p.add_argument("--train-stream", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--books", type=int, default=64)
    p.add_argument("--positions-per-book", type=int, default=1024)
    p.add_argument("--skip", type=int, default=512)
    p.add_argument("--seed", type=int, default=103)
    p.add_argument("--train-book-ids", type=Path,
                   help="Optional original TRAIN Gutenberg ID manifest for provable ID exclusion")
    p.add_argument("--sealed-test-book-ids", type=Path,
                   help="Optional sealed TEST *IDs only*, never reads TEST samples")
    p.add_argument("--source-release", default="corpora-short-stories-0004")
    args = p.parse_args(argv)
    manifest = build(corpus_dir=args.corpus_dir, train_stream=args.train_stream,
                     output_dir=args.output_dir, book_count=args.books,
                     positions_per_book=args.positions_per_book, skip=args.skip,
                     seed=args.seed, train_book_ids=args.train_book_ids,
                     sealed_test_book_ids=args.sealed_test_book_ids,
                     source_release=args.source_release)
    validate_fresh_dev(args.output_dir / "win_dev.bin",
                       args.output_dir / "win_dev.manifest.json",
                       train_sha256=manifest["source_train_sha256"])
    print(json.dumps({k: v for k, v in manifest.items() if k != "selected_books"},
                     sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
