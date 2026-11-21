"""Rebuild the exact 94 MB corpus and the 80 MB pilot from broad7h bins."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path("/marimo/nanlm_full2")
DATA = Path("/marimo/broad7h/data")
OUT = Path("/marimo/broad7h/terminal_rsi_corpus_6h")
PILOT = Path("/marimo/broad7h/runs/trsi_80m_pilot")
sys.path.insert(0, str(ROOT))

from terminal_rsi.prepare_cpu_pilot import build_cpu_pilot  # noqa: E402
from z3.layered_tokenizer import LayeredTokenizer  # noqa: E402


def detok() -> None:
    manifest = json.loads((DATA / "manifest.json").read_text())
    tok = LayeredTokenizer(
        str(ROOT / "z3/tokenizers/v4_64k/tokenizer_complete.json"))
    a = np.memmap(DATA / "train.bin", dtype=np.uint16, mode="r")
    parts = manifest["merge"]["parts"]
    caps = manifest["balance"]["caps"]
    src = manifest["balance"]["source_train_tokens"]
    dom = manifest["domains"]
    groups = {}
    for p in parts:
        groups.setdefault(p.rsplit("_", 1)[0], []).append(p)
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*.txt"):
        old.unlink()
    per = 635_000
    offset = 0
    files = 0
    for group, ps in groups.items():
        cap = caps.get(group, src.get(group, 0))
        plen = cap // len(ps)
        for p in ps:
            length = min(dom[p]["tokens"], plen)
            take = min(per, length)
            start = offset + max(0, (length - take) // 2)
            ids = np.asarray(a[start:start + take])
            special = np.flatnonzero((ids == tok.doc_id) | (ids == tok.eos_id))
            pieces = []
            prev = 0
            for b in special:
                if b > prev:
                    pieces.append(tok.decode_token_ids(ids[prev:b].tolist()))
                prev = int(b) + 1
            if prev < len(ids):
                pieces.append(tok.decode_token_ids(ids[prev:].tolist()))
            text = "\n\n".join(x for x in pieces if x.strip())
            size = max(1, len(text) // 4)
            for j in range(4):
                chunk = text[j * size:(j + 1) * size] if j < 3 else text[
                    3 * size:]
                if not chunk.strip():
                    continue
                (OUT / f"book_{files:05d}.txt").write_text(
                    chunk, encoding="utf-8")
                files += 1
            offset += length
    total = sum(p.stat().st_size for p in OUT.glob("book_*.txt"))
    print("DETOK", files, total, offset, flush=True)


def split() -> None:
    files = sorted(OUT.glob("book_*.txt"))
    text = "\n\n".join(p.read_text(encoding="utf-8", errors="ignore")
                       for p in files)
    n = 1600
    size = max(4096, len(text) // n)
    for i in range(n):
        chunk = text[i * size:(i + 1) * size] if i < n - 1 else text[
            (n - 1) * size:]
        (OUT / f"{70000 + i}--sixh_{i:05d}.txt").write_text(
            chunk, encoding="utf-8")
    print("SPLIT", len(text), size, flush=True)


def main() -> int:
    started = time.perf_counter()
    detok()
    split()
    PILOT.mkdir(parents=True, exist_ok=True)
    report = build_cpu_pilot(
        corpus_dir=OUT, output_dir=PILOT, train_bytes=80_000_000,
        dev_books=64, dev_positions=1024, seed=103)
    print("REPORT", json.dumps(report, indent=2), flush=True)
    print("seconds", round(time.perf_counter() - started, 1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
