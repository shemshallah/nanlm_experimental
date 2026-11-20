"""Word-token side branch for Terminal-RSI.

Two pieces live here:

* ``WordTokenBridge`` embeds the previous three word IDs and produces a
  residual over the 4k word logits.  It is zero-initialized, so an existing
  checkpoint is unchanged until it learns.
* ``build_word4_table`` builds a compact hashed 4-gram top-k table from the
  word IDs derived from the byte/subword stream.
"""
from __future__ import annotations

from collections import defaultdict

import torch
from torch import nn

from .unit_heads import WORD_VOCAB_SIZE

WORD4_BUCKETS = 1 << 20
WORD4_TOPK = 4


class WordTokenBridge(nn.Module):
    """Previous-word -> residual next-word logits."""

    def __init__(self, vocab: int = WORD_VOCAB_SIZE, embed_dim: int = 64,
                 hidden: int = 128):
        super().__init__()
        self.vocab = int(vocab)
        self.embed = nn.Embedding(self.vocab, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden, batch_first=True)
        self.out = nn.Linear(hidden, self.vocab, bias=True)
        nn.init.normal_(self.embed.weight, std=0.02)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, prev_ids: torch.Tensor) -> torch.Tensor:
        if prev_ids.ndim != 2 or prev_ids.shape[-1] != 3:
            raise ValueError("WordTokenBridge expects [batch,3] previous word IDs")
        ids = prev_ids.long().clamp(0, self.vocab - 1)
        embedded = self.embed(ids)
        encoded, _ = self.gru(embedded)
        return self.out(encoded[:, -1])

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def word4_bucket(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor,
                 buckets: int = WORD4_BUCKETS) -> torch.Tensor:
    return ((a.long() * 73856093) ^ (b.long() * 19349663)
            ^ (c.long() * 83492791)) & (buckets - 1)


def build_word4_table(ids: list[int], *, buckets: int = WORD4_BUCKETS,
                      topk: int = WORD4_TOPK,
                      alpha: float = 0.05) -> tuple[torch.Tensor, torch.Tensor]:
    """Hashed 4-gram top-k next-word table.

    ``ids`` are positive word vocabulary IDs.  Collisions merge contexts in
    the same bucket; the bigram prior remains the backoff at runtime.
    """
    counts: dict[int, dict[int, int]] = defaultdict(dict)
    for index in range(3, len(ids)):
        a, b, c, nxt = ids[index - 3], ids[index - 2], ids[index - 1], ids[index]
        bucket = ((a * 73856093) ^ (b * 19349663) ^ (c * 83492791)) & (buckets - 1)
        row = counts[bucket]
        row[nxt] = row.get(nxt, 0) + 1
    next_ids = torch.zeros((buckets, topk), dtype=torch.int32)
    logp = torch.zeros((buckets, topk), dtype=torch.float32)
    for bucket, row in counts.items():
        total = sum(row.values()) + alpha * len(row)
        best = sorted(row.items(), key=lambda item: (-item[1], item[0]))[:topk]
        for slot, (word, count) in enumerate(best):
            next_ids[bucket, slot] = int(word)
            logp[bucket, slot] = float(
                torch.log(torch.tensor((count + alpha) / total)))
    return next_ids, logp


@torch.no_grad()
def word_candidates(model, previous: torch.Tensor, prev_word_ids: torch.Tensor,
                    *, top_k: int = 5, ngram_weight: float = 0.5):
    """Top word IDs from the hierarchy's current-word head plus bridge/4-gram."""
    upgrade = model.upgrades(previous)
    logits = upgrade["unit_logits"]["current_word"].float()
    logits = logits + model.word_bridge(prev_word_ids).float()
    prior = model.word_ngram_prior(
        prev_word_ids[:, 0], prev_word_ids[:, 1], prev_word_ids[:, 2])
    logits = logits + float(ngram_weight) * prior
    values, indices = logits.log_softmax(-1).topk(top_k, dim=-1)
    return values, indices


@torch.no_grad()
def generate_words(model, prompt: bytes, steps: int, *,
                   temperature: float = 0.9, top_k: int = 5,
                   device: torch.device | None = None) -> dict:
    """Word-level decoder: bytes -> subword/token/word/sentence states -> words.

    At each word onset the hierarchy's word head, prior-word bridge and 4-gram
    table propose words.  The chosen word's bytes are fed back through the
    byte model so the subword/token/word/sentence states advance.
    """
    import torch

    if steps <= 0:
        raise ValueError("steps must be positive")
    device = device or next(model.parameters()).device
    model.eval()
    history = list(prompt)
    if len(history) < 32:
        history = [0] * (32 - len(history)) + history
    generated: list[str] = []
    records: list[dict] = []
    id_to_word = {v: k for k, v in (model._word_vocab or {}).items()}
    for _ in range(steps):
        context = torch.tensor(history[-256:], dtype=torch.long,
                               device=device)[None]
        # Previous complete words from the observed bytes.
        words = bytes(history).split()
        recent = [int((model._word_vocab or {}).get(word, 0))
                  for word in words[-3:]][::-1]
        prev_ids = torch.tensor((recent + [0, 0, 0])[:3], dtype=torch.long,
                                device=device)[None]
        values, indices = word_candidates(
            model, context, prev_ids, top_k=top_k)
        probs = (values[0] / max(temperature, 1e-4)).softmax(-1)
        pick = int(torch.multinomial(probs, 1))
        word_id = int(indices[0, pick])
        word = id_to_word.get(word_id, b"")
        text = word.decode("utf-8", "replace")
        records.append({"word_id": word_id, "word": text,
                        "logp": float(values[0, pick]),
                        "candidates": [id_to_word.get(int(i), b"").decode(
                            "utf-8", "replace") for i in indices[0]]})
        if not text:
            break
        history.extend(text.encode("utf-8") + b" ")
        generated.append(text)
    return {"words": generated, "records": records,
            "text": " ".join(generated)}
