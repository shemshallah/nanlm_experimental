"""Real subword/token, word and sentence heads for the unified v5 graph.

The v4 hierarchy could only report byte-exactness aggregated into units.  This
module adds actual unit vocabularies and classification heads:

* subword/token: BPE tokens from the frozen v103 merge ranks (heap BPE)
* word: whitespace-delimited words, top-N vocabulary
* sentence: next sentence's first word at sentence boundaries

The labels are built causally: the target at context position ``t`` is the next
unit that starts after byte ``t``.  No future byte enters the input.
"""
from __future__ import annotations

from dataclasses import dataclass
import heapq
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .legacy_v103_reference import read_bpe_ranks

TOKEN_VOCAB_SIZE = 4096
WORD_VOCAB_SIZE = 4096
SENTENCE_VOCAB_SIZE = 1024
UNK_ID = 0
IGNORE_ID = -100
WHITESPACE = {9, 10, 13, 32}
SENTENCE_END = {ord("."), ord("?"), ord("!")}


def bpe_tokens(data: bytes, ranks: dict[tuple[int, int], int]) -> list[tuple[int, int]]:
    """Heap-based BPE; returns ``(byte_start, token_id)`` in order.

    This is O(n log n) instead of the naive O(n * merges) scan used by the
    diagnostic span builder, which matters for a 2-10 MB training stream.
    """
    n = len(data)
    if n == 0:
        return []
    ids = list(data)
    prev = list(range(-1, n - 1))
    nxt = list(range(1, n + 1))
    nxt[-1] = -1
    alive = [True] * n
    heap: list[tuple[int, int, int, int, int]] = []

    def push(index: int) -> None:
        if index < 0:
            return
        right = nxt[index]
        if right < 0:
            return
        rank = ranks.get((ids[index], ids[right]))
        if rank is not None:
            heapq.heappush(
                heap, (rank, index, right, ids[index], ids[right]))

    for index in range(n - 1):
        push(index)
    while heap:
        _rank, left, right, old_left, old_right = heapq.heappop(heap)
        if not (alive[left] and alive[right] and nxt[left] == right
                and ids[left] == old_left and ids[right] == old_right):
            continue
        ids[left] = 256 + _rank
        alive[right] = False
        after = nxt[right]
        nxt[left] = after
        if after >= 0:
            prev[after] = left
        push(prev[left])
        push(left)
    tokens: list[tuple[int, int]] = []
    index = 0
    while index >= 0:
        if alive[index]:
            tokens.append((index, ids[index]))
        index = nxt[index]
    return tokens


def _word_tokens(data: bytes) -> list[tuple[int, int, bytes]]:
    tokens: list[tuple[int, int, bytes]] = []
    start = None
    for index, byte in enumerate(data):
        if byte in WHITESPACE:
            if start is not None:
                tokens.append((start, index, data[start:index]))
                start = None
        elif start is None:
            start = index
    if start is not None:
        tokens.append((start, len(data), data[start:]))
    return tokens


def _sentence_starts(data: bytes) -> list[int]:
    starts = [0]
    index = 0
    while index < len(data):
        if data[index] in SENTENCE_END and (
                index + 1 == len(data) or data[index + 1] in WHITESPACE):
            cursor = index + 1
            while cursor < len(data) and data[cursor] in WHITESPACE:
                cursor += 1
            if cursor < len(data):
                starts.append(cursor)
        index += 1
    return starts


def _top_vocab(items: list[bytes] | list[int], limit: int) -> dict:
    counts: dict = {}
    for item in items:
        counts[item] = counts.get(item, 0) + 1
    ordered = sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))
    return {item: index + 1 for index, (item, _count) in enumerate(ordered[:limit])}


def _next_unit_at(starts: list[int], values: list[int], length: int,
                  default: int = IGNORE_ID) -> list[int]:
    """For every byte position, the next unit at or after the following byte."""
    result = [default] * length
    next_value = default
    start_index = len(starts) - 1
    for position in range(length - 1, -1, -1):
        while start_index >= 0 and starts[start_index] > position:
            next_value = values[start_index]
            start_index -= 1
        result[position] = next_value
    return result


@dataclass
class UnitLabels:
    token_vocab: dict[int, int]
    word_vocab: dict[bytes, int]
    sentence_vocab: dict[int, int]
    next_token: torch.Tensor
    next_word: torch.Tensor
    next_sentence: torch.Tensor
    token_start: torch.Tensor
    word_start: torch.Tensor
    sentence_start: torch.Tensor

    def gather(self, positions: torch.Tensor) -> dict[str, torch.Tensor]:
        positions = positions.long().cpu().clamp(0, len(self.next_word) - 1)
        return {
            "next_token": self.next_token[positions].long(),
            "next_word": self.next_word[positions].long(),
            "next_sentence": self.next_sentence[positions].long(),
            "token_start": self.token_start[positions].bool(),
            "word_start": self.word_start[positions].bool(),
            "sentence_start": self.sentence_start[positions].bool(),
        }


def build_unit_labels(raw: bytes, bpe_merges: Path) -> UnitLabels:
    """Build causal next-unit labels and vocabularies for the whole stream."""
    ranks = read_bpe_ranks(Path(bpe_merges))
    if not ranks:
        raise ValueError("BPE merge ranks required for unit labels")
    length = len(raw)
    token_pairs = bpe_tokens(raw, ranks)
    token_starts = [start for start, _ in token_pairs]
    token_ids_raw = [token for _, token in token_pairs]
    token_vocab = _top_vocab(token_ids_raw, TOKEN_VOCAB_SIZE - 1)
    token_values = [token_vocab.get(token, UNK_ID) for token in token_ids_raw]

    word_pairs = _word_tokens(raw)
    word_starts = [start for start, _end, _word in word_pairs]
    word_vocab = _top_vocab([word for _start, _end, word in word_pairs],
                            WORD_VOCAB_SIZE - 1)
    word_values = [word_vocab.get(word, UNK_ID)
                   for _start, _end, word in word_pairs]

    sentence_starts = _sentence_starts(raw)
    # The sentence head predicts the first word of the next sentence.
    first_word_for_sentence: list[int] = []
    word_cursor = 0
    for start in sentence_starts:
        while (word_cursor < len(word_starts)
               and word_starts[word_cursor] < start):
            word_cursor += 1
        if word_cursor < len(word_values):
            first_word_for_sentence.append(word_values[word_cursor])
        else:
            first_word_for_sentence.append(UNK_ID)
    sentence_vocab = _top_vocab(first_word_for_sentence,
                                SENTENCE_VOCAB_SIZE - 1)
    sentence_values = [sentence_vocab.get(value, UNK_ID)
                       for value in first_word_for_sentence]

    next_token = _next_unit_at(token_starts, token_values, length)
    next_word = _next_unit_at(word_starts, word_values, length)
    next_sentence = _next_unit_at(sentence_starts, sentence_values, length)

    token_start = torch.zeros(length, dtype=torch.bool)
    word_start = torch.zeros(length, dtype=torch.bool)
    sentence_start = torch.zeros(length, dtype=torch.bool)
    for start in token_starts:
        token_start[start] = True
    for start in word_starts:
        word_start[start] = True
    for start in sentence_starts:
        sentence_start[start] = True
    return UnitLabels(
        token_vocab=token_vocab,
        word_vocab=word_vocab,
        sentence_vocab=sentence_vocab,
        next_token=torch.tensor(next_token, dtype=torch.int32),
        next_word=torch.tensor(next_word, dtype=torch.int32),
        next_sentence=torch.tensor(next_sentence, dtype=torch.int32),
        token_start=token_start,
        word_start=word_start,
        sentence_start=sentence_start,
    )


class UnitHeads(nn.Module):
    """Three classification heads over the hierarchy bus states."""

    def __init__(self, bus_width: int):
        super().__init__()
        self.token = nn.Linear(bus_width, TOKEN_VOCAB_SIZE, bias=False)
        self.word = nn.Linear(bus_width, WORD_VOCAB_SIZE, bias=False)
        self.sentence = nn.Linear(bus_width, SENTENCE_VOCAB_SIZE, bias=False)
        for head in (self.token, self.word, self.sentence):
            nn.init.normal_(head.weight, std=0.02)

    def forward(self, subword: torch.Tensor, word: torch.Tensor,
                sentence: torch.Tensor) -> dict[str, torch.Tensor]:
        # [B,T,D] -> last causal position only; unit prediction is one step.
        return {
            "token": self.token(subword[:, -1]),
            "word": self.word(word[:, -1]),
            "sentence": self.sentence(sentence[:, -1]),
        }


def unit_head_loss(logits: dict[str, torch.Tensor],
                   targets: dict[str, torch.Tensor],
                   *, weights: dict[str, float] | None = None) -> tuple[torch.Tensor, dict]:
    weights = weights or {"token": 1.0, "word": 1.0, "sentence": 0.5}
    total = None
    stats = {}
    for name, weight in weights.items():
        target = targets.get(f"next_{name}")
        if target is None or weight <= 0:
            continue
        target = target.to(logits[name].device).long()
        valid = target != IGNORE_ID
        if not bool(valid.any()):
            continue
        loss = F.cross_entropy(logits[name][valid], target[valid])
        total = loss * weight if total is None else total + loss * weight
        with torch.no_grad():
            prediction = logits[name][valid].topk(5, dim=-1).indices
            hits1 = (prediction[:, 0] == target[valid]).float().mean()
            hits5 = (prediction == target[valid, None]).any(-1).float().mean()
        stats[f"{name}_loss"] = float(loss.detach())
        stats[f"{name}_top1"] = float(hits1)
        stats[f"{name}_top5"] = float(hits5)
        stats[f"{name}_units"] = int(valid.sum())
    if total is None:
        total = torch.zeros((), device=next(iter(logits.values())).device,
                            requires_grad=True)
    return total, stats


def unit_parameter_count(bus_width: int) -> int:
    return bus_width * (TOKEN_VOCAB_SIZE + WORD_VOCAB_SIZE + SENTENCE_VOCAB_SIZE)
