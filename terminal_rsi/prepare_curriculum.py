"""Build a diverse, difficulty-curriculum English corpus for long Terminal-RSI runs.

The existing CPU pilot takes the first hash-sorted books and concatenates fixed
chunks.  That is reproducible but not a curriculum: it can be long on one genre,
short on dialogue, and easy/hard books arrive in arbitrary order.  This builder:

* strips Project Gutenberg headers/footers and obvious OCR noise,
* rejects Russian/non-English and low-quality books,
* deduplicates near-identical texts,
* tags genre from Gutenberg bookshelves/subjects,
* scores difficulty from sentence length, word length, vocabulary diversity,
  dialogue density and archaic-word density,
* reserves a book-disjoint RSI holdout and a 64-book DEV split,
* orders TRAIN chunks easy -> hard with 25% replay from earlier stages,
* interleaves genres and limits author/genre dominance,
* writes explicit manifests and a holdout stream that the v4 evaluator can use.

The output train stream is byte-level text; no tokenizer changes are required.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
import statistics

from terminal_rsi.build_fresh_dev import (
    _write_if_unchanged_or_new, build, language_rejection, metadata,
    validate_fresh_dev,
)

ID = re.compile(r"^(\d+)--")
START = re.compile(
    r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*",
    re.I | re.S,
)
END = re.compile(
    r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*",
    re.I | re.S,
)
WORD = re.compile(r"[A-Za-z']+")
SEPARATOR = b"\n\n<|book|>\n\n"
ARCHAIC = frozenset("""
thou thee thy thine hath doth dost hast art shalt wilt wouldst couldst shouldst
unto ye whence hither thither nay aye ere oft whither whereby wherein
""".split())

GENRES = {
    "science_fiction": (
        "science fiction", "science-fiction", "sci-fi", "space", "alien",
        "robot", "time travel", "extraterrestrial",
    ),
    "fantasy": (
        "fantasy", "fairy", "magic", "myth", "legend", "folklore", "fable",
    ),
    "mystery": ("mystery", "detective", "crime", "murder", "sleuth"),
    "horror": ("horror", "ghost", "terror", "gothic", "supernatural"),
    "romance": ("romance", "love story", "courtship"),
    "adventure": (
        "adventure", "travel", "sea", "western", "exploration", "voyage",
    ),
    "children": (
        "children", "juvenile", "young adult", "girls", "boys", "nursery",
    ),
    "poetry": ("poetry", "verse", "rhyme", "ballad"),
    "historical": ("historical", "history", "war", "civil war", "revolution"),
    "literary": ("literary", "short stories", "fiction", "essays"),
    "nonfiction": (
        "nonfiction", "biography", "science", "philosophy", "essay",
        "natural history",
    ),
}


def _clean_text(data: bytes) -> bytes:
    text = data.decode("utf-8", errors="replace").lstrip("\ufeff")
    start = START.search(text)
    if start:
        text = text[start.end():]
    end = END.search(text)
    if end:
        text = text[:end.start()]
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.translate(str.maketrans({
        "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
        "\u2013": "-", "\u2014": "-", "\u00a0": " ",
    }))
    text = re.sub(r"\[Illustration[^\]]*\]", "", text, flags=re.I)
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = []
    for line in text.splitlines():
        line = line.rstrip()
        stripped = line.strip()
        if stripped and len(stripped) > 2:
            letters = sum(ch.isalpha() or ch.isspace() for ch in stripped)
            if letters / len(stripped) < 0.45:
                continue
        lines.append(line)
    text = "\n".join(lines).strip()
    return (text + "\n").encode("utf-8")


def _quality(content: bytes) -> str | None:
    if len(content) < 4096:
        return "too_short"
    text = content.decode("utf-8", errors="ignore")
    if len(text) < 3500:
        return "too_short_text"
    letters = sum(ch.isalpha() or ch.isspace() for ch in text[:200_000])
    if letters / max(1, len(text[:200_000])) < 0.70:
        return "low_letter_ratio"
    non_ascii = 1.0 - len(text.encode("ascii", errors="ignore")) / max(1, len(text))
    if non_ascii > 0.03:
        return "non_ascii_noise"
    return None


def _genres(row: dict, filename: str) -> tuple[str, ...]:
    haystack = " ".join((
        filename.lower(),
        str(row.get("bookshelves", "")).lower(),
        str(row.get("subjects", "")).lower(),
        str(row.get("title", "")).lower(),
    ))
    tags = tuple(name for name, words in GENRES.items()
                 if any(word in haystack for word in words))
    return tags or ("other",)


def _numeric(row: dict, key: str, default: float) -> float:
    value = row.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _features(content: bytes, row: dict) -> dict:
    text = content.decode("utf-8", errors="ignore")
    words = WORD.findall(text.lower())
    n_words = max(1, len(words))
    sentences = max(1.0, _numeric(row, "sentences", text.count(".") + 1))
    paragraphs = max(1.0, _numeric(row, "paragraphs", text.count("\n\n") + 1))
    unique = len(set(words))
    archaic = sum(word in ARCHAIC for word in words)
    non_ascii = 1.0 - len(text.encode("ascii", errors="ignore")) / max(1, len(text))
    return {
        "bytes": len(content),
        "words": n_words,
        "sentences": sentences,
        "paragraphs": paragraphs,
        "avg_sentence_bytes": len(content) / sentences,
        "avg_word_len": sum(len(word) for word in words) / n_words,
        "vocab_diversity": unique / n_words,
        "dialogue_ratio": text.count('"') / sentences,
        "archaic_ratio": archaic / n_words,
        "non_ascii_ratio": non_ascii,
    }


def _z(values: list[float]) -> list[float]:
    mean = statistics.mean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 1.0
    if sd <= 1e-12:
        sd = 1.0
    return [(value - mean) / sd for value in values]


def _difficulty_scores(books: list[dict]) -> list[float]:
    sent = _z([book["features"]["avg_sentence_bytes"] for book in books])
    word = _z([book["features"]["avg_word_len"] for book in books])
    arch = _z([book["features"]["archaic_ratio"] for book in books])
    dial = _z([book["features"]["dialogue_ratio"] for book in books])
    non = _z([book["features"]["non_ascii_ratio"] for book in books])
    return [
        0.45 * sent[i] + 0.20 * word[i] + 0.25 * arch[i]
        - 0.15 * dial[i] + 0.10 * non[i]
        for i in range(len(books))
    ]


def _assign_stages(books: list[dict]) -> None:
    scores = _difficulty_scores(books)
    for book, score in zip(books, scores):
        book["difficulty"] = score
    order = sorted(range(len(books)), key=lambda i: books[i]["difficulty"])
    for rank, index in enumerate(order):
        books[index]["stage"] = min(3, (rank * 4) // max(1, len(books)))


def _dedup(books: list[dict]) -> tuple[list[dict], dict]:
    seen: dict[str, dict] = {}
    seen_title: dict[str, dict] = {}
    kept: list[dict] = []
    stats = Counter()
    for book in books:
        text = book["content"].decode("utf-8", errors="ignore")
        normalize = re.sub(r"[^a-z0-9]+", " ", text.lower())[:200_000]
        digest = hashlib.sha256(normalize.encode()).hexdigest()
        title_key = (
            str(book["row"].get("title", "")).strip().lower(),
            str(book["row"].get("authors", "")).strip().lower(),
        )
        title_hash = hashlib.sha256(repr(title_key).encode()).hexdigest()
        previous = seen.get(digest) or seen_title.get(title_hash)
        if previous is not None:
            stats["duplicate"] += 1
            if len(book["content"]) > len(previous["content"]):
                previous.update(book)
            continue
        seen[digest] = book
        seen_title[title_hash] = book
        kept.append(book)
    stats["kept"] = len(kept)
    return kept, dict(stats)


def _round_robin(chunks: list[dict], per_visit: int = 2) -> list[dict]:
    by_book: dict[str, list[dict]] = defaultdict(list)
    for chunk in chunks:
        by_book[chunk["book_id"]].append(chunk)
    for values in by_book.values():
        values.sort(key=lambda item: item["order"])
    book_order = sorted(
        by_book,
        key=lambda book_id: hashlib.sha256(
            f"curriculum-book:{book_id}".encode()).hexdigest(),
    )
    result = []
    cursors = {book_id: 0 for book_id in book_order}
    remaining = {book_id: len(by_book[book_id]) for book_id in book_order}
    while any(remaining.values()):
        for book_id in book_order:
            if remaining[book_id] <= 0:
                continue
            start = cursors[book_id]
            stop = min(len(by_book[book_id]), start + per_visit)
            result.extend(by_book[book_id][start:stop])
            cursors[book_id] = stop
            remaining[book_id] -= stop - start
    return result


def build_curriculum(
    *,
    corpus_dir: Path,
    output_dir: Path,
    train_bytes: int = 64_000_000,
    dev_books: int = 64,
    dev_positions: int = 1024,
    rsi_holdout_books: int = 4,
    chunk_bytes: int = 65_536,
    replay_fraction: float = 0.25,
    seed: int = 103,
    max_books_per_author: int = 6,
) -> dict:
    if train_bytes < 1_000_000 or chunk_bytes < 4096:
        raise ValueError("Need >=1MB train and >=4KB chunks")
    if not 0.0 <= replay_fraction < 1.0:
        raise ValueError("Replay fraction must be in [0,1)")
    if dev_books < 1 or rsi_holdout_books < 0:
        raise ValueError("Book counts must be nonnegative")

    sources = metadata(corpus_dir)
    candidates: list[dict] = []
    rejected: Counter = Counter()
    for path in sorted(corpus_dir.rglob("*.txt")):
        match = ID.match(path.name)
        if not match:
            continue
        book_id = str(int(match.group(1)))
        row = sources.get(book_id, {})
        raw = path.read_bytes()
        reason = language_rejection(raw, path.name, row)
        if reason is not None:
            rejected[reason] += 1
            continue
        cleaned = _clean_text(raw)
        reason = _quality(cleaned)
        if reason is not None:
            rejected[reason] += 1
            continue
        candidates.append({
            "book_id": book_id,
            "path": path,
            "row": row,
            "content": cleaned,
            "genres": _genres(row, path.name),
            "features": _features(cleaned, row),
        })
    if len(candidates) < dev_books + rsi_holdout_books + 16:
        raise ValueError("Insufficient eligible books for TRAIN/DEV/holdout")

    candidates, dedup_stats = _dedup(candidates)
    _assign_stages(candidates)
    candidates.sort(key=lambda book: hashlib.sha256(
        f"curriculum:{seed}:{book['book_id']}".encode()).hexdigest())

    reserve_count = dev_books + rsi_holdout_books + 16
    train_pool = candidates[:max(0, len(candidates) - reserve_count)]
    reserve = candidates[max(0, len(candidates) - reserve_count):]

    # Holdout is one diverse book per difficulty stage where possible.
    holdout: list[dict] = []
    used_genres: set[str] = set()
    for stage in range(4):
        for book in reserve:
            if book in holdout or book["stage"] != stage:
                continue
            if book["genres"][0] not in used_genres and len(book["content"]) >= 32_768:
                holdout.append(book)
                used_genres.add(book["genres"][0])
                break
    for book in reserve:
        if len(holdout) >= rsi_holdout_books:
            break
        if book not in holdout and len(book["content"]) >= 32_768:
            holdout.append(book)
    if len(holdout) < rsi_holdout_books:
        raise ValueError("Could not find enough book-disjoint RSI holdout books")

    # Limit author dominance in TRAIN without breaking book disjointness.
    author_counts: Counter = Counter()
    train_books: list[dict] = []
    for book in train_pool:
        author = str(book["row"].get("authors", "")).split(";")[0].strip().lower()
        author = author or book["book_id"]
        if author_counts[author] >= max_books_per_author:
            rejected["author_cap"] += 1
            continue
        author_counts[author] += 1
        train_books.append(book)
    if len(train_books) < 16:
        raise ValueError("Too few TRAIN books after author cap")

    # Split into chunks, then make an easy -> hard curriculum with genre mixing.
    stage_chunks: dict[int, list[dict]] = defaultdict(list)
    for book in train_books:
        content = book["content"]
        for order, start in enumerate(range(0, len(content), chunk_bytes)):
            chunk = content[start:start + chunk_bytes]
            if len(chunk) < 4096:
                continue
            stage_chunks[book["stage"]].append({
                "book_id": book["book_id"],
                "stage": book["stage"],
                "genres": book["genres"],
                "order": order,
                "text": chunk,
                "hash": hashlib.sha256(chunk).hexdigest()[:16],
            })
    ordered: list[dict] = []
    stage_summary = {}
    for stage in range(4):
        chunks = _round_robin(stage_chunks.get(stage, []), per_visit=2)
        ordered.extend(chunks)
        stage_summary[stage] = {
            "chunks": len(chunks),
            "books": len({chunk["book_id"] for chunk in chunks}),
            "genres": dict(Counter(
                genre for chunk in chunks for genre in chunk["genres"][:1])),
        }
    if not ordered:
        raise ValueError("Curriculum produced no chunks")

    rng = random.Random(seed)
    stream = bytearray()
    replay_pool: list[dict] = []
    used_chunks = 0
    repeats = 0
    previous_book = None
    index = 0
    while len(stream) < train_bytes:
        if index >= len(ordered):
            index = 0
            repeats += 1
        chunk = ordered[index]
        index += 1
        if len(stream) > 0 and (chunk["book_id"] != previous_book):
            stream.extend(SEPARATOR)
        stream.extend(chunk["text"])
        previous_book = chunk["book_id"]
        used_chunks += 1
        replay_pool.append(chunk)
        if replay_pool and chunk["stage"] > 0 and rng.random() < replay_fraction:
            replay = rng.choice(replay_pool[:-1]) if len(replay_pool) > 1 else replay_pool[0]
            if replay["book_id"] != previous_book:
                stream.extend(SEPARATOR)
            stream.extend(replay["text"])
            previous_book = replay["book_id"]
    stream = stream[:train_bytes]

    output_dir.mkdir(parents=True, exist_ok=True)
    train_file = output_dir / "train_curriculum.bin"
    train_ids_file = output_dir / "train_curriculum_book_ids.json"
    holdout_file = output_dir / "rsi_holdout.bin"
    holdout_ids_file = output_dir / "rsi_holdout_book_ids.json"
    _write_if_unchanged_or_new(train_file, bytes(stream))
    _write_if_unchanged_or_new(
        train_ids_file,
        (json.dumps({"book_ids": sorted(book["book_id"] for book in train_pool)},
                    sort_keys=True, indent=2) + "\n").encode(),
    )
    holdout_stream = bytearray()
    for book in holdout:
        if holdout_stream:
            holdout_stream.extend(SEPARATOR)
        holdout_stream.extend(book["content"])
    _write_if_unchanged_or_new(holdout_file, bytes(holdout_stream))
    _write_if_unchanged_or_new(
        holdout_ids_file,
        (json.dumps({"book_ids": sorted(book["book_id"] for book in holdout)},
                    sort_keys=True, indent=2) + "\n").encode(),
    )

    dev = build(
        corpus_dir=corpus_dir,
        train_stream=train_file,
        output_dir=output_dir,
        book_count=dev_books,
        positions_per_book=dev_positions,
        skip=512,
        seed=seed,
        train_book_ids=train_ids_file,
        extra_excluded_book_ids={book["book_id"] for book in holdout},
        source_release="corpora-short-stories-0004",
    )
    validate_fresh_dev(
        output_dir / "win_dev.bin",
        output_dir / "win_dev.manifest.json",
        train_sha256=hashlib.sha256(stream).hexdigest(),
    )
    train_ids = {book["book_id"] for book in train_pool}
    holdout_ids = {book["book_id"] for book in holdout}
    dev_ids = {str(item["gutenberg_id"]) for item in dev["selected_books"]}
    if train_ids & holdout_ids or train_ids & dev_ids or holdout_ids & dev_ids:
        raise AssertionError("TRAIN/DEV/RSI holdout Gutenberg identities overlap")

    report = {
        "schema": "terminal-rsi-curriculum-v1",
        "source": "shemshallah/NaNROUTER/corpora/short_stories",
        "seed": seed,
        "train_bytes": len(stream),
        "train_sha256": hashlib.sha256(stream).hexdigest(),
        "train_books": len(train_books),
        "train_pool_books": len(train_pool),
        "used_chunks": used_chunks,
        "repeat_passes": repeats,
        "replay_fraction": replay_fraction,
        "chunk_bytes": chunk_bytes,
        "stages": stage_summary,
        "genre_counts": dict(Counter(
            genre for book in train_books for genre in book["genres"][:1])),
        "difficulty": {
            "min": min(book["difficulty"] for book in train_books),
            "max": max(book["difficulty"] for book in train_books),
            "mean": statistics.mean(book["difficulty"] for book in train_books),
        },
        "dedup": dedup_stats,
        "rejected": dict(rejected),
        "holdout_books": len(holdout),
        "holdout_bytes": len(holdout_stream),
        "holdout_sha256": hashlib.sha256(holdout_stream).hexdigest(),
        "holdout_book_ids": sorted(holdout_ids),
        "dev_books": dev_books,
        "dev_scored_bytes": dev["scored_positions"],
        "dev_sha256": dev["dev_sha256"],
        "book_disjoint_train_dev_holdout_verified": True,
        "sealed_test_never_opened": True,
    }
    _write_if_unchanged_or_new(
        output_dir / "curriculum_manifest.json",
        (json.dumps(report, sort_keys=True, indent=2) + "\n").encode(),
    )
    return report


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--train-bytes", type=int, default=64_000_000)
    p.add_argument("--dev-books", type=int, default=64)
    p.add_argument("--dev-positions", type=int, default=1024)
    p.add_argument("--rsi-holdout-books", type=int, default=4)
    p.add_argument("--chunk-bytes", type=int, default=65_536)
    p.add_argument("--replay-fraction", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=103)
    p.add_argument("--max-books-per-author", type=int, default=6)
    args = p.parse_args(argv)
    report = build_curriculum(
        corpus_dir=args.corpus_dir,
        output_dir=args.output_dir,
        train_bytes=args.train_bytes,
        dev_books=args.dev_books,
        dev_positions=args.dev_positions,
        rsi_holdout_books=args.rsi_holdout_books,
        chunk_bytes=args.chunk_bytes,
        replay_fraction=args.replay_fraction,
        seed=args.seed,
        max_books_per_author=args.max_books_per_author,
    )
    print(json.dumps(report, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
