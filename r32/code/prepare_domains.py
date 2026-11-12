#!/usr/bin/env python3
"""Build a balanced multi-domain byte corpus from NaNROUTER domain releases.

Each release is one UTF-8 text file with blank-line-separated documents. Full
assets are 0.05–1.5 GB, so this samples evenly spaced byte ranges, discards
partial documents at the cut edges, truncates very long documents, and writes:

  train.bin, valid.bin   uint16 byte+EOS tokens
  meta.json              lengths, domain ids, vocab layout, hashes
  split_manifest.json    per-domain provenance/counts

Domain tokens are not stored in the bins. The trainer prepends token
257+d for domain d, so the model vocabulary is 257 + len(domains).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import requests

REPO = "shemshallah/NaNROUTER"
BASE = f"https://github.com/{REPO}/releases/download"
COLLECTIONS = {
    "wikipedia-en": ("corpora-wikipedia-en", "wikipedia-en.txt", "encyclopedia"),
    "arxiv": ("corpora-arxiv", "arxiv.txt", "STEM papers"),
    "pubmed": ("corpora-pubmed", "pubmed.txt", "biomedicine"),
    "law": ("corpora-law", "law.txt", "law and government"),
    "patents": ("corpora-patents", "patents.txt", "patents"),
    "code": ("corpora-code", "code.txt", "programming"),
    "math": ("corpora-math", "math.txt", "mathematics"),
    "news": ("corpora-news", "news.txt", "news"),
    "recipes": ("corpora-recipes", "recipes.txt", "cooking"),
    "security": ("corpora-security", "security.txt", "cybersecurity"),
    "qa": ("corpora-qa", "qa.txt", "question answering"),
    "multilingual": ("corpora-multilingual", "multilingual.txt", "multilingual"),
    "finance": ("corpora-finance", "finance.txt", "finance"),
    "standards": ("corpora-standards", "standards.txt", "technical standards"),
    "humanities": ("corpora-humanities", "humanities.txt", "humanities"),
    "chemistry": ("corpora-chemistry", "chemistry.txt", "chemistry"),
}
BASE_VOCAB = 257
EOS = 256


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def remote_size(url: str) -> int:
    r = requests.head(url, allow_redirects=True, timeout=60)
    r.raise_for_status()
    return int(r.headers.get("content-length", "0"))


def fetch_sample(url: str, target_bytes: int, chunks: int, cache: Path) -> bytes:
    if cache.exists() and cache.stat().st_size >= min(target_bytes, remote_size(url)):
        return cache.read_bytes()
    size = remote_size(url)
    chunks = max(1, chunks)
    if size <= target_bytes * 1.2:
        with requests.get(url, timeout=300, stream=True) as r:
            r.raise_for_status()
            data = b"".join(c for c in r.iter_content(1 << 20))
    else:
        part = max(1, target_bytes // chunks)
        data = bytearray()
        for i in range(chunks):
            start = int((size - part) * i / max(chunks - 1, 1))
            end = min(size, start + part) - 1
            r = requests.get(url, headers={"Range": f"bytes={start}-{end}"}, timeout=180)
            if r.status_code not in (200, 206):
                r.raise_for_status()
            data.extend(r.content)
    data = bytes(data)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(data)
    return data


def clean_doc(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def split_docs(raw: bytes) -> list[str]:
    text = raw.decode("utf-8", errors="replace")
    return [d for d in (clean_doc(x) for x in text.split("\n\n")) if d]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cache-dir", default="domain_cache")
    ap.add_argument("--domains", default="all")
    ap.add_argument("--per-domain-mb", type=float, default=4.0)
    ap.add_argument("--chunks", type=int, default=4)
    ap.add_argument("--valid-fraction", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--min-doc-bytes", type=int, default=200)
    ap.add_argument("--max-doc-bytes", type=int, default=32000)
    ap.add_argument("--max-docs-per-domain", type=int, default=0)
    args = ap.parse_args()

    if not (0.0 < args.valid_fraction < 0.5):
        raise ValueError("--valid-fraction must be in (0, 0.5)")
    names = list(COLLECTIONS) if args.domains == "all" else \
        [x.strip() for x in args.domains.split(",") if x.strip()]
    bad = [x for x in names if x not in COLLECTIONS]
    if bad:
        raise ValueError(f"unknown domains: {bad}")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cache = Path(args.cache_dir)
    target = int(args.per_domain_mb * 1_000_000)
    domain_id = {name: i for i, name in enumerate(names)}
    train_records, valid_records, manifest_domains = [], [], []

    for name in names:
        tag, filename, domain = COLLECTIONS[name]
        url = f"{BASE}/{tag}/{filename}"
        print(f"[{name}] {url}", flush=True)
        started = time.time()
        raw = fetch_sample(url, target, args.chunks, cache / f"{name}.txt")
        docs = split_docs(raw)
        if len(docs) > 2:
            docs = docs[1:-1]
        kept, seen = [], set()
        for doc in docs:
            b = doc.encode("utf-8")
            if len(b) < args.min_doc_bytes:
                continue
            if len(b) > args.max_doc_bytes:
                b = b[:args.max_doc_bytes]
                doc = b.decode("utf-8", errors="ignore")
                if len(doc.encode("utf-8")) < args.min_doc_bytes:
                    continue
            h = sha256_bytes(doc.encode("utf-8"))
            if h in seen:
                continue
            seen.add(h)
            kept.append((h, doc, len(doc.encode("utf-8"))))
            if args.max_docs_per_domain and len(kept) >= args.max_docs_per_domain:
                break
        kept.sort(key=lambda x: hashlib.sha256((str(args.seed) + ":" + x[0]).encode()).hexdigest())
        if len(kept) < 2:
            print(f"  warning: {len(kept)} usable docs; skipping", file=sys.stderr)
            continue
        nvalid = max(1, min(len(kept) - 1, round(len(kept) * args.valid_fraction)))
        valid, train = kept[:nvalid], kept[nvalid:]
        for h, doc, _ in train:
            train_records.append((domain_id[name], doc))
        for h, doc, _ in valid:
            valid_records.append((domain_id[name], doc))
        manifest_domains.append({
            "collection": name, "release_tag": tag, "file": filename, "domain": domain,
            "url": url, "sampled_bytes": len(raw), "usable_documents": len(kept),
            "train_documents": len(train), "valid_documents": len(valid),
            "train_bytes": sum(len(d.encode("utf-8")) for _, d in train),
            "valid_bytes": sum(len(d.encode("utf-8")) for _, d in valid),
            "elapsed_seconds": round(time.time() - started, 2),
        })
        print(f"  docs train={len(train)} valid={len(valid)}", flush=True)

    if not train_records or not valid_records:
        raise RuntimeError("no usable documents")

    def write_split(records, filename):
        lengths, domains = [], []
        with (out / filename).open("wb") as f:
            for did, doc in records:
                ids = list(doc.encode("utf-8")) + [EOS]
                np.asarray(ids, dtype="<u2").tofile(f)
                lengths.append(len(ids))
                domains.append(did)
        return lengths, domains

    train_lengths, train_domains = write_split(train_records, "train.bin")
    valid_lengths, valid_domains = write_split(valid_records, "valid.bin")
    manifest = {
        "seed": args.seed, "valid_fraction": args.valid_fraction,
        "per_domain_mb": args.per_domain_mb, "chunks": args.chunks,
        "base_vocab": BASE_VOCAB, "eos_id": EOS,
        "domain_token_base": BASE_VOCAB, "domain_names": names,
        "domains": manifest_domains,
        "train_documents": len(train_records), "valid_documents": len(valid_records),
        "train_tokens": int(sum(train_lengths)), "valid_tokens": int(sum(valid_lengths)),
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    (out / "split_manifest.json").write_bytes(manifest_bytes)
    meta = {
        "vocab_size": BASE_VOCAB + len(names), "base_vocab_size": BASE_VOCAB,
        "domain_token_base": BASE_VOCAB, "domain_names": names, "eos_id": EOS,
        "unit": "UTF8_bytes_plus_eos_domain_token_per_window",
        "tokenizer_model": None, "tokenizer_sha256": None, "token_dtype": "<u2",
        "split_unit": "document",
        "train_tokens": int(sum(train_lengths)), "valid_tokens": int(sum(valid_lengths)),
        "train_doc_lengths": train_lengths, "valid_doc_lengths": valid_lengths,
        "train_domains": train_domains, "valid_domains": valid_domains,
        "train_sha256": file_hash(out / "train.bin"), "valid_sha256": file_hash(out / "valid.bin"),
        "bin_train_sha256": file_hash(out / "train.bin"),
        "bin_valid_sha256": file_hash(out / "valid.bin"),
        "split_manifest_sha256": sha256_bytes(manifest_bytes),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps({k: meta[k] for k in ("vocab_size", "train_tokens", "valid_tokens", "split_unit")}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
