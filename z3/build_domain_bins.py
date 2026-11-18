#!/usr/bin/env python3
"""Build document-disjoint, document-framed z3 token bins.

NaNROUTER's ``corpora/domains/*`` releases are UTF-8 text with one document per
blank-line-separated block.  This builder:

* hashes each document to a deterministic train/valid/test split;
* encodes each document with explicit ``<doc>`` and ``<eos>`` frames;
* caps a domain by raw bytes and/or training tokens;
* writes ``train.bin``, ``valid.bin``, ``test.bin`` and per-domain test text;
* writes a manifest with counts, hashes, and the split policy.

It is intentionally a one-time data build, not part of the training hot path.

Example::

    python z3/build_domain_bins.py \
      --corpus arxiv=corpora/domains/arxiv.txt \
      --corpus code=corpora/domains/code.txt \
      --corpus math=corpora/domains/math.txt \
      --out data/z3_domains_115m \
      --max-bytes-per-domain 50000000 \
      --train-fraction 0.98 --valid-fraction 0.01

The bin format is the same raw uint16 stream consumed by
``z3/train_v3.py --tokenizer bin``.  The manifest records that the stream is
document-framed so evaluation can use ``eval_v3.py --document-framing``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    from .layered_tokenizer import LayeredTokenizer, load_artifact
    from .text_documents import iter_documents, write_documents
except ImportError:  # script execution from z3/
    import sys

    _HERE = Path(__file__).resolve().parent
    if str(_HERE) not in sys.path:
        sys.path.insert(0, str(_HERE))
    from layered_tokenizer import LayeredTokenizer, load_artifact
    from text_documents import iter_documents, write_documents


def _parse_corpora(specs: Iterable[str]) -> list[tuple[str, Path]]:
    result: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for spec in specs:
        if "=" in spec:
            name, raw_path = spec.split("=", 1)
        else:
            raw_path = spec
            name = Path(spec).stem
        name = name.strip()
        path = Path(raw_path).expanduser()
        if not name:
            raise SystemExit(f"empty corpus name in {spec!r}")
        if name in seen:
            raise SystemExit(f"duplicate corpus name {name!r}")
        if not path.is_file():
            raise SystemExit(f"corpus not found: {path}")
        seen.add(name)
        result.append((name, path))
    if not result:
        raise SystemExit("at least one --corpus name=path is required")
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _doc_fraction(seed: int, domain: str, document: str) -> float:
    digest = hashlib.sha256(
        f"{seed}\0{domain}\0{document}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def _split_for(
    seed: int,
    domain: str,
    document: str,
    train_fraction: float,
    valid_fraction: float,
) -> str:
    value = _doc_fraction(seed, domain, document)
    if value < train_fraction:
        return "train"
    if value < train_fraction + valid_fraction:
        return "valid"
    return "test"


def _concat(parts: list[np.ndarray], dtype: np.dtype) -> np.ndarray:
    if not parts:
        return np.empty(0, dtype=dtype)
    return np.concatenate(parts).astype(dtype, copy=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="repeatable domain=text file",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--layered-model",
        default=str(
            Path(__file__).resolve().with_name("tokenizers")
            / "layered_v1" / "layered_tokenizer.json"
        ),
    )
    parser.add_argument(
        "--max-bytes-per-domain",
        type=int,
        default=0,
        help="raw bytes read per domain; 0 = all (pilot: 50000000)",
    )
    parser.add_argument(
        "--max-tokens-per-domain",
        type=int,
        default=0,
        help="cap training tokens per domain; 0 = no cap",
    )
    parser.add_argument(
        "--max-docs-per-domain",
        type=int,
        default=0,
        help="cap documents read per domain; 0 = no cap",
    )
    parser.add_argument(
        "--dedupe",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="drop exact duplicate documents across domains",
    )
    parser.add_argument("--train-fraction", type=float, default=0.98)
    parser.add_argument("--valid-fraction", type=float, default=0.01)
    parser.add_argument("--min-doc-bytes", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20261113)
    parser.add_argument(
        "--dtype", choices=("uint16", "uint32"), default="uint16"
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if not 0.0 < args.train_fraction < 1.0:
        raise SystemExit("--train-fraction must be in (0, 1)")
    if not 0.0 <= args.valid_fraction < 1.0:
        raise SystemExit("--valid-fraction must be in [0, 1)")
    if args.train_fraction + args.valid_fraction >= 1.0:
        raise SystemExit("train + valid fractions must leave a test split")
    if args.max_bytes_per_domain < 0:
        raise SystemExit("--max-bytes-per-domain must be non-negative")

    corpora = _parse_corpora(args.corpus)
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "test").mkdir(parents=True, exist_ok=True)
    (out / "valid").mkdir(parents=True, exist_ok=True)

    tokenizer: LayeredTokenizer = load_artifact(
        Path(args.layered_model).expanduser()
    )
    dtype = np.dtype(args.dtype)
    if tokenizer.vocab_size - 1 > np.iinfo(dtype).max:
        raise SystemExit(
            f"tokenizer vocab {tokenizer.vocab_size} does not fit in {dtype}"
        )

    split_parts: dict[str, list[np.ndarray]] = {
        "train": [],
        "valid": [],
        "test": [],
    }
    split_docs: dict[str, int] = {"train": 0, "valid": 0, "test": 0}
    split_tokens: dict[str, int] = {"train": 0, "valid": 0, "test": 0}
    split_bytes: dict[str, int] = {"train": 0, "valid": 0, "test": 0}
    domain_manifest: dict[str, dict[str, object]] = {}
    seen_documents: set[str] = set()
    duplicates_removed = 0
    started = time.time()

    for domain, path in corpora:
        train_tokens = 0
        docs_read = 0
        raw_read = 0
        test_documents: list[str] = []
        valid_documents: list[str] = []
        domain_counts = {"train": 0, "valid": 0, "test": 0}
        domain_tokens = {"train": 0, "valid": 0, "test": 0}
        domain_bytes = {"train": 0, "valid": 0, "test": 0}

        for document, raw_bytes in iter_documents(
            path,
            max_bytes=args.max_bytes_per_domain,
            min_bytes=args.min_doc_bytes,
        ):
            docs_read += 1
            raw_read += int(raw_bytes)
            if args.max_docs_per_domain and docs_read > args.max_docs_per_domain:
                break
            if args.dedupe:
                document_hash = hashlib.sha256(
                    document.encode("utf-8", errors="ignore")
                ).hexdigest()
                if document_hash in seen_documents:
                    duplicates_removed += 1
                    continue
                seen_documents.add(document_hash)
            split = _split_for(
                args.seed,
                domain,
                document,
                args.train_fraction,
                args.valid_fraction,
            )
            encoded = tokenizer.encode(document, add_document_frame=True)
            ids = np.asarray(encoded.token_ids, dtype=dtype)
            if ids.size == 0:
                continue
            if split == "train":
                if (
                    args.max_tokens_per_domain
                    and train_tokens + int(ids.size)
                    > args.max_tokens_per_domain
                ):
                    continue
                train_tokens += int(ids.size)
            elif split == "test":
                test_documents.append(document)
            elif split == "valid":
                valid_documents.append(document)
            split_parts[split].append(ids)
            split_docs[split] += 1
            split_tokens[split] += int(ids.size)
            split_bytes[split] += int(raw_bytes)
            domain_counts[split] += 1
            domain_tokens[split] += int(ids.size)
            domain_bytes[split] += int(raw_bytes)

        if test_documents:
            write_documents(out / "test" / f"{domain}.txt", test_documents)
        if valid_documents:
            write_documents(out / "valid" / f"{domain}.txt", valid_documents)
        if domain_counts["valid"] == 0 or domain_counts["test"] == 0:
            print(
                f"[warn] {domain}: valid={domain_counts['valid']} "
                f"test={domain_counts['test']}; the domain sample is too "
                "small for a 3-way document split",
                flush=True,
            )
        test_text_path = (
            out / "test" / f"{domain}.txt" if test_documents else None
        )
        valid_text_path = (
            out / "valid" / f"{domain}.txt" if valid_documents else None
        )
        domain_manifest[domain] = {
            "path": str(path),
            "raw_bytes_read": raw_read,
            "documents_read": docs_read,
            "documents": domain_counts,
            "tokens": domain_tokens,
            "bytes": domain_bytes,
            "test_text": str(test_text_path) if test_text_path else None,
            "test_text_sha256": (
                _sha256(test_text_path) if test_text_path else None
            ),
            "valid_text": str(valid_text_path) if valid_text_path else None,
            "valid_text_sha256": (
                _sha256(valid_text_path) if valid_text_path else None
            ),
        }
        if not args.quiet:
            print(
                f"[domain] {domain}: docs={domain_counts} "
                f"tokens={domain_tokens} raw_bytes={raw_read:,}",
                flush=True,
            )

    bins: dict[str, dict[str, object]] = {}
    for split in ("train", "valid", "test"):
        array = _concat(split_parts[split], dtype)
        target = out / f"{split}.bin"
        array.tofile(target)
        bins[split] = {
            "path": str(target),
            "tokens": int(array.size),
            "bytes": int(target.stat().st_size),
            "sha256": _sha256(target),
            "documents": int(split_docs[split]),
            "source_bytes": int(split_bytes[split]),
        }
        if not args.quiet:
            print(
                f"[bin] {split}: tokens={array.size:,} "
                f"docs={split_docs[split]:,} path={target}",
                flush=True,
            )

    manifest = {
        "version": 1,
        "kind": "z3-document-framed-domain-bins",
        "created_unix": int(time.time()),
        "seed": int(args.seed),
        "layered_model": str(Path(args.layered_model).expanduser().resolve()),
        "layered_model_sha256": _sha256(
            Path(args.layered_model).expanduser()
        ),
        "vocab_size": int(tokenizer.vocab_size),
        "dtype": str(dtype),
        "document_framing": {
            "enabled": True,
            "start_token": int(tokenizer.doc_id) if tokenizer.doc_id is not None else None,
            "end_token": int(tokenizer.eos_id) if tokenizer.eos_id is not None else None,
        },
        "split_policy": {
            "train_fraction": float(args.train_fraction),
            "valid_fraction": float(args.valid_fraction),
            "test_fraction": float(
                1.0 - args.train_fraction - args.valid_fraction
            ),
            "hash": "sha256(seed\\0domain\\0document)[:8] / 2^64",
            "document_disjoint": True,
        },
        "caps": {
            "max_bytes_per_domain": int(args.max_bytes_per_domain),
            "max_tokens_per_domain": int(args.max_tokens_per_domain),
            "max_docs_per_domain": int(args.max_docs_per_domain),
            "min_doc_bytes": int(args.min_doc_bytes),
        },
        "dedupe": {
            "enabled": bool(args.dedupe),
            "duplicates_removed": int(duplicates_removed),
        },
        "bins": bins,
        "domains": domain_manifest,
        "total_seconds": round(time.time() - started, 3),
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest["bins"], indent=2, sort_keys=True), flush=True)
    print(f"[done] wrote {out / 'manifest.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
