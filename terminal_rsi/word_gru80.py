#!/usr/bin/env python3
"""Train a non-transformer word-level GRU on the 80 MB pilot and score dev."""
from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path("/marimo/nanlm_full2")
PILOT = Path("/marimo/broad7h/runs/trsi_80m_pilot")
CORPUS = Path("/marimo/broad7h/terminal_rsi_corpus_6h")
OUT = Path("/marimo/word_gru80_results.json")
CKPT = Path("/marimo/word_gru80.pt")
V = 4096
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ = 128
BATCH = 256
EPOCHS = 4
sys.path.insert(0, str(ROOT))
from terminal_rsi.unit_heads import _word_tokens  # noqa: E402


def words(text: bytes) -> list[str]:
    return [w.decode("utf-8", "ignore") for _s, _e, w in _word_tokens(text)]


class WordGRU(nn.Module):
    def __init__(self, vocab=4096, emb=128, hidden=256, layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab, emb)
        self.gru = nn.GRU(emb, hidden, layers, batch_first=True,
                          dropout=0.1)
        self.out = nn.Linear(hidden, vocab)

    def forward(self, x):
        h, _ = self.gru(self.embed(x))
        return self.out(h)


def main() -> int:
    t0 = time.perf_counter()
    train_words = words((PILOT / "train_cpu_pilot.bin").read_bytes())
    counter = Counter(train_words)
    vocab = {w: i + 1 for i, (w, _n) in enumerate(counter.most_common(V - 1))}
    ids = np.array([vocab.get(w, 0) for w in train_words], dtype=np.int32)
    print("train words", len(ids), "vocab", len(vocab), flush=True)
    model = WordGRU(V).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    rng = np.random.default_rng(2026)
    steps = max(1, (len(ids) - SEQ - 1) // (SEQ * BATCH))
    for epoch in range(EPOCHS):
        model.train()
        total = 0.0
        for step in range(steps):
            starts = rng.integers(0, len(ids) - SEQ - 1, size=BATCH)
            seq = np.stack([ids[s:s + SEQ] for s in starts])
            tgt = np.stack([ids[s + 1:s + SEQ + 1] for s in starts])
            x = torch.from_numpy(seq).long().to(DEVICE)
            y = torch.from_numpy(tgt).long().to(DEVICE)
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.detach())
        print("epoch", epoch + 1, "loss", round(total / steps, 4),
              "seconds", round(time.perf_counter() - t0, 1), flush=True)

    manifest = json.loads((PILOT / "win_dev.manifest.json").read_text())
    top1 = top5 = total = 0
    with torch.no_grad():
        model.eval()
        for book in manifest["selected_books"][:32]:
            raw = (CORPUS / book["relative_path"]).read_bytes()
            start = int(book.get("excerpt_start", 0))
            length = int(book.get("length", 1024))
            tokens = words(raw[start:start + length])
            seq = np.array([vocab.get(w, 0) for w in tokens], dtype=np.int32)
            if len(seq) < 2:
                continue
            x = torch.from_numpy(seq[None, :-1]).long().to(DEVICE)
            y = torch.from_numpy(seq[1:]).long().to(DEVICE)
            logits = model(x)[0]
            pred = logits.topk(5, dim=-1).indices
            top1 += int((pred[:, 0] == y).sum())
            top5 += int((pred == y[:, None]).any(-1).sum())
            total += len(y)
    report = {"train_words": len(ids), "vocab": len(vocab),
              "steps": steps, "epochs": EPOCHS,
              "dev_units": total, "top1": top1 / max(total, 1),
              "top5": top5 / max(total, 1),
              "seconds": round(time.perf_counter() - t0, 1)}
    torch.save({"model": model.state_dict(), "vocab": vocab}, CKPT)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
