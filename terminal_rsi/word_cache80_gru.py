#!/usr/bin/env python3
"""Add word-GRU top-100 candidates to the 80 MB cache."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/marimo/nanlm_full2")
PILOT = Path("/marimo/broad7h/runs/trsi_80m_pilot")
CORPUS = Path("/marimo/broad7h/terminal_rsi_corpus_6h")
BASE = Path("/marimo/word_dev_cache80.npz")
GRU_CKPT = Path("/marimo/word_gru80.pt")
OUT = Path("/marimo/word_dev_cache80_gru.npz")
sys.path.insert(0, str(ROOT))
from terminal_rsi.unit_heads import _word_tokens  # noqa: E402
from terminal_rsi.word_gru80 import WordGRU  # noqa: E402

V = 4096
CTX = 32
TOPK = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def top_labels_vocab(train_raw: bytes) -> dict[bytes, int]:
    words = [w for _s, _e, w in _word_tokens(train_raw)]
    counts = Counter(words)
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return {w: i + 1 for i, (w, _c) in enumerate(ordered[:V - 1])}


def main() -> int:
    data = np.load(BASE)
    target = data["target"]
    book = data["book"]
    pos = data["pos"]
    prev = data["prev"]
    n = len(target)
    labels_vocab = top_labels_vocab((PILOT / "train_cpu_pilot.bin").read_bytes())
    ckpt = torch.load(GRU_CKPT, map_location="cpu", weights_only=False)
    gru_vocab = ckpt["vocab"]
    model = WordGRU(V).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    gru_id_to_labels = {}
    for word, gid in gru_vocab.items():
        labels_id = labels_vocab.get(word.encode("utf-8"), 0)
        if labels_id:
            gru_id_to_labels[int(gid)] = int(labels_id)
    print("mapped gru ids", len(gru_id_to_labels), flush=True)

    manifest = json.loads((PILOT / "win_dev.manifest.json").read_text())
    gru_top = np.full((n, TOPK), -1, dtype=np.int32)
    with torch.no_grad():
        for book_idx in range(64):
            rows = np.where(book == book_idx)[0]
            if len(rows) == 0:
                continue
            entry = manifest["selected_books"][book_idx]
            base = int(entry.get("excerpt_start", 0))
            length = int(entry.get("length", 1024))
            excerpt = (CORPUS / entry["relative_path"]).read_bytes()[
                base:base + length]
            toks = _word_tokens(excerpt)
            starts = {int(s): i for i, (s, _e, _w) in enumerate(toks)}
            words = [w.decode("utf-8", "ignore") for _s, _e, w in toks]
            ids = [gru_vocab.get(w, 0) for w in words]
            for row in rows:
                idx = starts.get(int(pos[row]))
                if idx is None:
                    continue
                ctx = ids[max(0, idx - CTX):idx]
                if not ctx:
                    continue
                x = torch.tensor([ctx], dtype=torch.long, device=DEVICE)
                logits = model(x)[0, -1]
                top = torch.topk(logits, TOPK).indices.cpu().tolist()
                mapped = [gru_id_to_labels.get(int(g), 0) for g in top]
                gru_top[row, :len(mapped)] = mapped[:TOPK]
            if book_idx % 8 == 0:
                print("book", book_idx, flush=True)
    np.savez_compressed(
        OUT,
        head=data["head"], bridge=data["bridge"], prior=data["prior"],
        prev=prev, target=target, book=book, pos=pos,
        ppm=data["ppm"], cache=data["cache"], copy=data["copy"],
        gru=gru_top,
    )
    print("wrote", OUT, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
