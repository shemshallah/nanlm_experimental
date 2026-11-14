#!/usr/bin/env python3
"""Tune the six-level tokenizer one layer at a time.

Usage examples::

    python z3/tune_layered_tokenizer.py \
        --corpus corpora/train.txt \
        --stage all --max-bytes 2000000 \
        --out runs/layered_tokenizer

    python z3/tune_layered_tokenizer.py \
        --corpus corpora/train.txt --stage word \
        --word-vocab-sizes 16000,32000,64000,128000 \
        --out runs/layered_tokenizer

Each stage has an explicit gate.  The report is written even when a gate
fails, so a failing layer can be retuned without losing the other results.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

try:
    from .layered_tokenizer import (
        BitLayer,
        ByteLayer,
        DEFAULT_SUBWORD_PATH,
        LayeredTokenizer,
        SentenceLayer,
        SubwordLayer,
        WordLayer,
        write_artifact,
    )
except ImportError:  # script execution from z3/
    from layered_tokenizer import (  # type: ignore
        BitLayer,
        ByteLayer,
        DEFAULT_SUBWORD_PATH,
        LayeredTokenizer,
        SentenceLayer,
        SubwordLayer,
        WordLayer,
        write_artifact,
    )


def _read_text(paths: Sequence[str], max_bytes: int) -> tuple[str, int]:
    chunks: list[bytes] = []
    total = 0
    for raw_path in paths:
        path = Path(raw_path).expanduser()
        if not path.is_file():
            raise SystemExit(f"corpus not found: {path}")
        with path.open("rb") as handle:
            while total < max_bytes:
                chunk = handle.read(min(1 << 20, max_bytes - total))
                if not chunk:
                    break
                chunks.append(chunk)
                total += len(chunk)
        if total >= max_bytes:
            break
    raw = b"".join(chunks)
    return raw.decode("utf-8", errors="ignore"), len(raw)


def _split_train_eval(text: str, eval_fraction: float) -> tuple[str, str]:
    if not text:
        raise SystemExit("corpus produced no text")
    eval_chars = int(len(text) * max(0.01, min(0.5, eval_fraction)))
    eval_chars = max(1, min(len(text) - 1, eval_chars))
    # Prefer a whitespace boundary so the eval split starts on a word.
    cut = text.rfind(" ", len(text) - eval_chars - 200, len(text) - eval_chars + 200)
    if cut < 0:
        cut = len(text) - eval_chars
    return text[:cut], text[cut:]


def stage_bit(sample: bytes, report: dict[str, Any]) -> bool:
    results = []
    for order in ("msb_first", "lsb_first"):
        layer = BitLayer(msb_first=(order == "msb_first"))
        bits = layer.encode(sample)
        roundtrip = layer.decode(bits) == sample
        results.append(
            {
                "order": order,
                "bits": len(bits),
                "bits_per_byte": len(bits) / max(1, len(sample)),
                "roundtrip": roundtrip,
            }
        )
    passed = all(item["roundtrip"] for item in results)
    report["bit"] = {
        "selected": "msb_first",
        "candidates": results,
        "gate": {"exact_roundtrip": passed},
    }
    return passed


def stage_byte(sample: bytes, report: dict[str, Any]) -> bool:
    layer = ByteLayer()
    text = layer.decode(sample)
    encoded = layer.encode(text)
    # ``errors=ignore`` can make a truncated UTF-8 cap non-roundtrippable at
    # the byte level; test the actual text instead of the truncated tail.
    exact = layer.decode(encoded) == text and len(encoded) == len(
        text.encode("utf-8")
    )
    report["byte"] = {
        "encoding": "utf-8",
        "bytes": len(sample),
        "roundtrip": exact,
        "gate": {"exact_roundtrip": exact},
    }
    return exact


def train_subword_candidates(
    train_text: str,
    sizes: Sequence[int],
    out: Path,
    specials: Sequence[str],
) -> list[str]:
    """Train additional byte-level BPE candidates on the tuning sample."""
    if not sizes:
        return []
    try:
        from tokenizers import Tokenizer, trainers
        from tokenizers.decoders import ByteLevel as ByteLevelDecoder
        from tokenizers.models import BPE
        from tokenizers.pre_tokenizers import ByteLevel
    except ModuleNotFoundError:
        print("[subword] tokenizers missing; using existing candidates only", flush=True)
        return []
    directory = out / "subword_candidates"
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for size in sorted(set(int(x) for x in sizes if int(x) > 0)):
        tokenizer = Tokenizer(BPE(unk_token=None))
        tokenizer.pre_tokenizer = ByteLevel(
            add_prefix_space=False, use_regex=True
        )
        tokenizer.decoder = ByteLevelDecoder()
        trainer = trainers.BpeTrainer(
            vocab_size=size,
            special_tokens=list(specials),
            min_frequency=2,
            show_progress=False,
        )
        tokenizer.train_from_iterator([train_text], trainer=trainer)
        path = directory / f"bpe_{size}.json"
        tokenizer.save(str(path))
        paths.append(str(path))
        print(f"[subword] trained {size}: {path}", flush=True)
    return paths


def _subword_score(
    bytes_per_token: float, vocab_size: int, objective: str
) -> float:
    if objective == "compression":
        return bytes_per_token
    # Bytes per bit of embedding vocabulary: rewards compression without
    # letting a huge vocabulary win by construction.
    return bytes_per_token / max(1.0, math.log2(max(2, vocab_size)))


def stage_subword(
    text: str,
    candidates: Sequence[str],
    min_bytes_per_token: float,
    report: dict[str, Any],
    objective: str = "efficiency",
    max_vocab: int = 64000,
) -> tuple[str, bool]:
    rows = []
    for candidate in candidates:
        path = Path(candidate).expanduser()
        if not path.is_file():
            rows.append({"path": str(path), "error": "missing"})
            continue
        layer = SubwordLayer(path)
        encoded = layer.encode(text)
        decoded = layer.decode(encoded.ids)
        raw_bytes = len(text.encode("utf-8"))
        bpt = raw_bytes / max(1, len(encoded.ids))
        rows.append(
            {
                "path": str(path),
                "backend": encoded.backend,
                "vocab_size": layer.vocab_size,
                "bytes_per_token": bpt,
                "score": _subword_score(bpt, layer.vocab_size, objective),
                "tokens": len(encoded.ids),
                "roundtrip": decoded == text,
                "byte_coverage": encoded.n_bytes / max(1, raw_bytes),
                "special_tokens": layer.special_tokens,
            }
        )
    valid = [
        row
        for row in rows
        if row.get("roundtrip")
        and row.get("byte_coverage", 0.0) == 1.0
        and row.get("bytes_per_token", 0.0) >= min_bytes_per_token
        and row.get("vocab_size", 0) <= max_vocab
    ]
    chosen = (
        max(valid, key=lambda row: (row["score"], row["bytes_per_token"]))
        if valid
        else max(
            (row for row in rows if "bytes_per_token" in row),
            key=lambda row: row["bytes_per_token"],
            default={"path": str(DEFAULT_SUBWORD_PATH), "bytes_per_token": 0.0},
        )
    )
    passed = bool(valid)
    report["subword"] = {
        "selected": chosen,
        "candidates": rows,
        "gate": {
            "min_bytes_per_token": min_bytes_per_token,
            "max_vocab": max_vocab,
            "objective": objective,
            "passed": passed,
        },
    }
    return str(chosen["path"]), passed


def stage_token(
    tokenizer: LayeredTokenizer,
    text: str,
    report: dict[str, Any],
) -> bool:
    encoded = tokenizer.encode(text[:200_000], add_document_frame=True)
    specials = tokenizer.specials
    unique = len(set(specials)) == len(specials)
    ids_valid = all(0 <= int(token_id) < tokenizer.vocab_size for token_id in encoded.token_ids)
    frame_ok = (
        tokenizer.doc_id is not None
        and tokenizer.eos_id is not None
        and tokenizer.now_id is not None
        and encoded.token_ids[:1] == [tokenizer.doc_id]
        and encoded.token_ids[-1:] == [tokenizer.eos_id]
    )
    no_special_leak = all(
        int(token_id) < tokenizer.vocab_size for token_id in encoded.token_ids
    )
    passed = bool(unique and ids_valid and frame_ok and no_special_leak)
    report["token"] = {
        "doc_id": tokenizer.doc_id,
        "eos_id": tokenizer.eos_id,
        "now_id": tokenizer.now_id,
        "specials": specials,
        "tokens": len(encoded.token_ids),
        "frame": {"doc": frame_ok, "eos": frame_ok},
        "gate": {
            "unique_specials": unique,
            "ids_in_vocab": ids_valid,
            "document_frame": frame_ok,
            "passed": passed,
        },
    }
    return passed


def stage_word(
    train_text: str,
    eval_text: str,
    subword: SubwordLayer,
    sizes: Sequence[int],
    min_coverage: float,
    report: dict[str, Any],
) -> tuple[WordLayer, bool]:
    span_layer = WordLayer()
    spans = span_layer.spans(eval_text)
    counts = Counter(
        piece for piece, _, _, is_word in spans if is_word
    )
    total = max(1, sum(counts.values()))
    unique = sorted(counts)
    subword_cache = {
        word: max(1, len(subword.encode(word).ids)) for word in unique
    }
    rows = []
    for size in sorted(set(int(x) for x in sizes if int(x) > 0)):
        chosen = set(word for word, _ in counts.most_common(size))
        covered = sum(count for word, count in counts.items() if word in chosen)
        fallback_units = sum(
            subword_cache[word] * count
            for word, count in counts.items()
            if word not in chosen
        )
        units = covered + fallback_units
        rows.append(
            {
                "vocab_size": size,
                "coverage": covered / total,
                "fallback_rate": 1.0 - covered / total,
                "units": units,
                "bytes_per_unit": len(eval_text.encode("utf-8")) / max(1, units),
                "types_covered": len(chosen),
                "distinct_words": len(counts),
            }
        )
    passing = [row for row in rows if row["coverage"] >= min_coverage]
    best = (
        min(passing, key=lambda row: (row["vocab_size"], -row["coverage"]))
        if passing
        else max(rows, key=lambda row: row["coverage"], default={"vocab_size": 0})
    )
    selected_size = int(best.get("vocab_size", 0))
    fitted = WordLayer().fit(train_text, selected_size) if selected_size else WordLayer()
    passed = bool(passing)
    report["word"] = {
        "selected": best,
        "fitted_vocab_size": fitted.vocab_size,
        "candidates": rows,
        "gate": {"min_coverage": min_coverage, "passed": passed},
    }
    return fitted, passed


def stage_sentence(
    eval_text: str,
    candidates: Sequence[int],
    target_max_chars: int,
    report: dict[str, Any],
) -> tuple[int, bool]:
    max_chars, detail = SentenceLayer.tune(
        eval_text,
        candidates,
        target_max_chars=target_max_chars,
    )
    report["sentence"] = detail
    return max_chars, bool(detail["gate"]["passed"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", nargs="+", required=True)
    parser.add_argument("--out", default="runs/layered_tokenizer")
    parser.add_argument("--max-bytes", type=int, default=2_000_000)
    parser.add_argument("--eval-fraction", type=float, default=0.1)
    parser.add_argument(
        "--stage",
        choices=("all", "bit", "byte", "subword", "token", "word", "sentence"),
        default="all",
    )
    parser.add_argument(
        "--subword-candidate",
        action="append",
        default=[],
        help="repeatable subword tokenizer JSON path",
    )
    parser.add_argument("--min-bytes-per-token", type=float, default=4.5)
    parser.add_argument(
        "--train-subword-vocabs",
        default="",
        help="optional comma-separated BPE sizes to train on the tuning sample",
    )
    parser.add_argument(
        "--subword-objective",
        choices=("efficiency", "compression"),
        default="efficiency",
    )
    parser.add_argument("--max-subword-vocab", type=int, default=64000)
    parser.add_argument(
        "--word-vocab-sizes", default="16000,32000,64000,128000"
    )
    parser.add_argument("--word-min-coverage", type=float, default=0.90)
    parser.add_argument(
        "--sentence-max-chars", default="256,512,1024,2048"
    )
    parser.add_argument("--sentence-target", type=int, default=512)
    parser.add_argument(
        "--no-native",
        action="store_true",
        help="force the dependency-free BPE backend",
    )
    args = parser.parse_args(argv)

    text, raw_bytes = _read_text(args.corpus, max(1, args.max_bytes))
    train_text, eval_text = _split_train_eval(text, args.eval_fraction)
    run = {
        "version": 1,
        "stage": args.stage,
        "corpus": {
            "paths": [str(Path(p).expanduser()) for p in args.corpus],
            "raw_bytes": raw_bytes,
            "train_chars": len(train_text),
            "eval_chars": len(eval_text),
        },
        "layers": {},
        "gates": {},
    }
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    subword_path = str(DEFAULT_SUBWORD_PATH)
    if args.stage in ("all", "bit"):
        run["gates"]["bit"] = stage_bit(
            train_text.encode("utf-8")[:65536], run["layers"]
        )
    if args.stage in ("all", "byte"):
        run["gates"]["byte"] = stage_byte(
            train_text.encode("utf-8")[:262144], run["layers"]
        )

    subword_candidates = list(args.subword_candidate or [str(DEFAULT_SUBWORD_PATH)])
    if args.stage in ("all", "subword"):
        train_sizes = [
            int(x)
            for x in str(args.train_subword_vocabs).split(",")
            if x.strip()
        ]
        if train_sizes:
            specials = SubwordLayer(
                DEFAULT_SUBWORD_PATH, prefer_native=False
            ).special_tokens
            subword_candidates.extend(
                train_subword_candidates(
                    train_text, train_sizes, out, specials
                )
            )
        subword_path, subword_ok = stage_subword(
            eval_text,
            subword_candidates,
            args.min_bytes_per_token,
            run["layers"],
            objective=args.subword_objective,
            max_vocab=args.max_subword_vocab,
        )
        run["gates"]["subword"] = subword_ok
    else:
        # Use the first available candidate for the higher layers.
        subword_path = next(
            (p for p in subword_candidates if Path(p).expanduser().is_file()),
            str(DEFAULT_SUBWORD_PATH),
        )

    base = LayeredTokenizer(
        subword_path=subword_path,
        prefer_native=not args.no_native,
    )
    if args.stage in ("all", "token"):
        run["gates"]["token"] = stage_token(base, eval_text, run["layers"])

    word_sizes = [
        int(x) for x in str(args.word_vocab_sizes).split(",") if x.strip()
    ]
    word_layer = WordLayer()
    if args.stage in ("all", "word"):
        word_layer, word_ok = stage_word(
            train_text,
            eval_text,
            base.subword,
            word_sizes,
            args.word_min_coverage,
            run["layers"],
        )
        run["gates"]["word"] = word_ok

    sentence_candidates = [
        int(x) for x in str(args.sentence_max_chars).split(",") if x.strip()
    ]
    sentence_max_chars = 512
    if args.stage in ("all", "sentence"):
        sentence_max_chars, sentence_ok = stage_sentence(
            eval_text,
            sentence_candidates,
            args.sentence_target,
            run["layers"],
        )
        run["gates"]["sentence"] = sentence_ok

    final = LayeredTokenizer(
        subword_path=subword_path,
        word_vocab_size=word_layer.vocab_size,
        word_vocab=word_layer.vocab,
        sentence_max_chars=sentence_max_chars,
        prefer_native=not args.no_native,
    )
    artifact_path = write_artifact(final, out / "layered_tokenizer.json")
    run["artifact"] = str(artifact_path)
    run["all_passed"] = all(run["gates"].values()) if run["gates"] else False
    (out / "tuning_report.json").write_text(
        json.dumps(run, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(run, indent=2, ensure_ascii=False), flush=True)
    print(f"artifact: {artifact_path}", flush=True)
    return 0 if run["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
