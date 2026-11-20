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
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .legacy_v103_reference import read_bpe_ranks

TOKEN_VOCAB_SIZE = 1024
WORD_VOCAB_SIZE = 4096
SENTENCE_VOCAB_SIZE = 512
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
    next_token_start: torch.Tensor
    next_word_start: torch.Tensor
    next_sentence_start: torch.Tensor
    current_token: torch.Tensor
    current_word: torch.Tensor
    token_start: torch.Tensor
    word_start: torch.Tensor
    sentence_start: torch.Tensor

    def gather(self, positions: torch.Tensor,
               target_positions: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        positions = positions.long().cpu().clamp(0, len(self.next_word) - 1)
        if target_positions is None:
            target_positions = (positions + 1).clamp(
                max=len(self.current_word) - 1)
        target_positions = target_positions.long().cpu().clamp(
            0, len(self.current_word) - 1)
        return {
            "next_token": self.next_token[positions].long(),
            "next_word": self.next_word[positions].long(),
            "next_sentence": self.next_sentence[positions].long(),
            "next_token_start": self.next_token_start[positions].bool(),
            "next_word_start": self.next_word_start[positions].bool(),
            "next_sentence_start": self.next_sentence_start[positions].bool(),
            "current_token": self.current_token[target_positions].long(),
            "current_word": self.current_word[target_positions].long(),
            "token_start": self.token_start[positions].bool(),
            "word_start": self.word_start[positions].bool(),
            "sentence_start": self.sentence_start[positions].bool(),
        }

    def gather_window(self, starts: torch.Tensor,
                      context: int) -> dict[str, torch.Tensor]:
        """Per-position labels for every byte in a causal window.

        ``starts`` are the first observed byte positions; every returned
        tensor is ``[batch,context]`` so word/next-word heads can train on
        every boundary instead of only the window's last byte.
        """
        starts = starts.long().cpu()
        positions = starts[:, None] + torch.arange(
            int(context), dtype=torch.long)[None, :]
        positions = positions.clamp(0, len(self.next_word) - 1)
        target_positions = (positions + 1).clamp(
            max=len(self.current_word) - 1)
        return {
            "next_token": self.next_token[positions].long(),
            "next_word": self.next_word[positions].long(),
            "next_sentence": self.next_sentence[positions].long(),
            "next_token_start": self.next_token_start[positions].bool(),
            "next_word_start": self.next_word_start[positions].bool(),
            "next_sentence_start": self.next_sentence_start[positions].bool(),
            "current_token": self.current_token[target_positions].long(),
            "current_word": self.current_word[target_positions].long(),
            "token_start": self.token_start[positions].bool(),
            "word_start": self.word_start[positions].bool(),
            "sentence_start": self.sentence_start[positions].bool(),
        }


def build_unit_labels(raw: bytes, bpe_merges: Path, *,
                      token_vocab: dict[int, int] | None = None,
                      word_vocab: dict[bytes, int] | None = None,
                      sentence_vocab: dict[int, int] | None = None) -> UnitLabels:
    """Build causal next-unit labels; optionally reuse a training vocabulary."""
    ranks = read_bpe_ranks(Path(bpe_merges))
    if not ranks:
        raise ValueError("BPE merge ranks required for unit labels")
    length = len(raw)
    token_pairs = bpe_tokens(raw, ranks)
    token_starts = [start for start, _ in token_pairs]
    token_ids_raw = [token for _, token in token_pairs]
    token_vocab = token_vocab or _top_vocab(
        token_ids_raw, TOKEN_VOCAB_SIZE - 1)
    token_values = [token_vocab.get(token, UNK_ID) for token in token_ids_raw]

    word_pairs = _word_tokens(raw)
    word_starts = [start for start, _end, _word in word_pairs]
    word_vocab = word_vocab or _top_vocab(
        [word for _start, _end, word in word_pairs], WORD_VOCAB_SIZE - 1)
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
    sentence_vocab = sentence_vocab or _top_vocab(
        first_word_for_sentence, SENTENCE_VOCAB_SIZE - 1)
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

    def _next_start(starts: torch.Tensor) -> torch.Tensor:
        result = torch.zeros(length, dtype=torch.bool)
        if length > 1:
            result[:-1] = starts[1:]
        return result

    if token_starts:
        token_lengths = [token_starts[i + 1] - token_starts[i]
                         for i in range(len(token_starts) - 1)]
        token_lengths.append(length - token_starts[-1])
        current_token = torch.repeat_interleave(
            torch.tensor(token_values, dtype=torch.int32),
            torch.tensor(token_lengths, dtype=torch.long))
    else:
        current_token = torch.full((length,), IGNORE_ID, dtype=torch.int32)
    word_values_list: list[int] = []
    word_lengths_list: list[int] = []
    cursor = 0
    for start, end, word in word_pairs:
        if start > cursor:
            word_values_list.append(IGNORE_ID)
            word_lengths_list.append(start - cursor)
        word_values_list.append(word_vocab.get(word, UNK_ID))
        word_lengths_list.append(end - start)
        cursor = end
    if cursor < length:
        word_values_list.append(IGNORE_ID)
        word_lengths_list.append(length - cursor)
    current_word = (torch.repeat_interleave(
        torch.tensor(word_values_list, dtype=torch.int32),
        torch.tensor(word_lengths_list, dtype=torch.long))
        if word_values_list else
        torch.full((length,), IGNORE_ID, dtype=torch.int32))
    if len(current_word) != length:
        raise AssertionError("current-word label length mismatch")

    return UnitLabels(
        token_vocab=token_vocab,
        word_vocab=word_vocab,
        sentence_vocab=sentence_vocab,
        next_token=torch.tensor(next_token, dtype=torch.int32),
        next_word=torch.tensor(next_word, dtype=torch.int32),
        next_sentence=torch.tensor(next_sentence, dtype=torch.int32),
        next_token_start=_next_start(token_start),
        next_word_start=_next_start(word_start),
        next_sentence_start=_next_start(sentence_start),
        current_token=current_token,
        current_word=current_word,
        token_start=token_start,
        word_start=word_start,
        sentence_start=sentence_start,
    )


class UnitHeads(nn.Module):
    """Three classification heads over bus states plus the byte hidden state."""

    def __init__(self, bus_width: int, byte_width: int | None = None):
        super().__init__()
        byte_width = bus_width if byte_width is None else byte_width
        self.byte_proj = nn.Linear(byte_width, bus_width, bias=False)
        width = 2 * bus_width
        self.token = nn.Linear(width, TOKEN_VOCAB_SIZE, bias=False)
        self.word = nn.Linear(width, WORD_VOCAB_SIZE, bias=False)
        self.sentence = nn.Linear(width, SENTENCE_VOCAB_SIZE, bias=False)
        self.current_token = nn.Linear(width, TOKEN_VOCAB_SIZE, bias=False)
        self.current_word = nn.Linear(width, WORD_VOCAB_SIZE, bias=False)
        nn.init.normal_(self.byte_proj.weight, std=0.02)
        for head in (self.token, self.word, self.sentence,
                     self.current_token, self.current_word):
            nn.init.normal_(head.weight, std=0.02)

    def forward(self, subword: torch.Tensor, word: torch.Tensor,
                sentence: torch.Tensor,
                byte_state: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        # [B,T,D] -> last causal position only; unit prediction is one step.
        if byte_state is None:
            byte_proj = torch.zeros_like(subword)
        else:
            byte_proj = self.byte_proj(byte_state)
        token_input = torch.cat((subword[:, -1], byte_proj[:, -1]), dim=-1)
        word_input = torch.cat((word[:, -1], byte_proj[:, -1]), dim=-1)
        sentence_input = torch.cat((sentence[:, -1], byte_proj[:, -1]), dim=-1)
        return {
            "token": self.token(token_input),
            "word": self.word(word_input),
            "sentence": self.sentence(sentence_input),
            "current_token": self.current_token(token_input),
            "current_word": self.current_word(word_input),
        }


def top5_hinge(logits: torch.Tensor, target: torch.Tensor,
               margin: float = 0.5) -> torch.Tensor:
    """Push the target above the fifth-best non-target logit."""
    target_logit = logits.gather(1, target[:, None])
    masked = logits.scatter(1, target[:, None], -1e9)
    fifth = masked.topk(min(5, logits.shape[-1]), dim=-1).values[:, -1:]
    return F.softplus(fifth - target_logit + margin).mean()


def unit_head_loss(logits: dict[str, torch.Tensor],
                   targets: dict[str, torch.Tensor],
                   *, weights: dict[str, float] | None = None,
                   boundary_only: bool = True,
                   top5_weight: float = 0.0) -> tuple[torch.Tensor, dict]:
    weights = weights or {
        "token": 1.0, "word": 1.0, "sentence": 0.5,
        "current_token": 0.5, "current_word": 0.5,
    }
    total = None
    stats = {}
    for name, weight in weights.items():
        if name.startswith("current_"):
            target = targets.get(name)
            boundary = False
        else:
            target = targets.get(f"next_{name}")
            boundary = True
        if target is None or weight <= 0 or name not in logits:
            continue
        target = target.to(logits[name].device).long()
        valid = target != IGNORE_ID
        start_key = f"next_{name}_start"
        if boundary_only and boundary and start_key in targets:
            valid = valid & targets[start_key].to(
                logits[name].device).bool()
        if not bool(valid.any()):
            continue
        loss = F.cross_entropy(logits[name][valid], target[valid])
        if top5_weight > 0:
            loss = loss + top5_weight * top5_hinge(
                logits[name][valid], target[valid])
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


def unit_parameter_count(bus_width: int, byte_width: int | None = None) -> int:
    byte_width = bus_width if byte_width is None else byte_width
    return (byte_width * bus_width + 2 * bus_width * (
        TOKEN_VOCAB_SIZE + WORD_VOCAB_SIZE + SENTENCE_VOCAB_SIZE
        + TOKEN_VOCAB_SIZE + WORD_VOCAB_SIZE))


@torch.no_grad()
def evaluate_unit_head_dev(model, *, corpus_dir: Path, dev_manifest: Path,
                           bpe_merges: Path, context: int, books: int,
                           max_positions: int, device: torch.device,
                           vocabulary: UnitLabels,
                           ngram_weight: float = 0.0,
                           eval_batch: int = 128) -> dict:
    """Score the real token/word/sentence heads on held-out DEV excerpts."""
    eval_batch = max(1, min(int(eval_batch), 64))
    manifest = json.loads(Path(dev_manifest).read_text())
    selected = manifest.get("selected_books") or []
    if not selected:
        raise ValueError("DEV manifest has no selected_books")
    totals = {name: {"units": 0, "top1": 0, "top5": 0}
              for name in ("token", "word", "sentence", "word_observed")}
    used = []
    was_training = model.training
    model.eval()
    try:
        for book in selected[:max(1, books)]:
            path = Path(corpus_dir) / book["relative_path"]
            raw = path.read_bytes()
            start = int(book.get("excerpt_start", 0))
            length = int(book.get("length", 1536))
            excerpt = raw[start:start + length]
            if len(excerpt) < context + 32:
                continue
            labels = build_unit_labels(
                excerpt, bpe_merges,
                token_vocab=vocabulary.token_vocab,
                word_vocab=vocabulary.word_vocab,
                sentence_vocab=vocabulary.sentence_vocab)
            positions = torch.arange(context, len(excerpt), dtype=torch.long)
            if max_positions > 0:
                positions = positions[:max_positions]
            if not len(positions):
                continue
            windows = torch.stack([
                torch.frombuffer(
                    bytearray(excerpt[int(pos) - context:int(pos)]),
                    dtype=torch.uint8).long()
                for pos in positions
            ]).to(device)
            chunked: dict[str, list[torch.Tensor]] = {
                name: [] for name in ("token", "word", "sentence")}
            for start_index in range(0, len(windows), max(1, eval_batch)):
                upgrade = model.upgrades(
                    windows[start_index:start_index + max(1, eval_batch)])
                part = upgrade.get("unit_logits") or {}
                if not part:
                    return {"schema": "terminal-rsi-unit-heads-v1",
                            "levels": {}}
                for name in chunked:
                    chunked[name].append(part[name])
            logits = {name: torch.cat(values, dim=0)
                      for name, values in chunked.items()}
            observed = torch.tensor(
                sorted(set(vocabulary.word_vocab.values())),
                dtype=torch.long, device=device)
            observed_mask = torch.zeros(
                logits["word"].shape[-1], dtype=torch.bool, device=device)
            observed_mask[observed.clamp(0, len(observed_mask) - 1)] = True
            for name in ("token", "word", "sentence"):
                target = getattr(labels, f"next_{name}")[positions - 1].to(device)
                starts = getattr(
                    labels, f"next_{name}_start")[positions - 1].to(device)
                valid = (target != IGNORE_ID) & starts
                if not bool(valid.any()):
                    continue
                head_logits = logits[name]
                if name == "word" and ngram_weight > 0:
                    bigram = getattr(model, "word_bigram_log_probs", None)
                    if bigram is not None and bigram.numel():
                        previous = labels.current_word[positions - 1].to(device)
                        prior = model.word_unigram_log_probs.to(device)[
                            None, :].expand(len(previous), -1)
                        ok = (previous > 0) & (previous < bigram.shape[0])
                        if bool(ok.any()):
                            prior = prior.clone()
                            prior[ok] = bigram.to(device)[previous[ok]]
                        head_logits = head_logits + ngram_weight * prior
                prediction = head_logits[valid].topk(5, dim=-1).indices
                totals[name]["units"] += int(valid.sum())
                totals[name]["top1"] += int(
                    (prediction[:, 0] == target[valid]).sum())
                totals[name]["top5"] += int(
                    (prediction == target[valid, None]).any(-1).sum())
                if name == "word":
                    masked = head_logits[valid].masked_fill(
                        ~observed_mask[None, :], -1e9)
                    obs_pred = masked.topk(5, dim=-1).indices
                    totals["word_observed"]["units"] += int(valid.sum())
                    totals["word_observed"]["top1"] += int(
                        (obs_pred[:, 0] == target[valid]).sum())
                    totals["word_observed"]["top5"] += int(
                        (obs_pred == target[valid, None]).any(-1).sum())
            used.append(book.get("gutenberg_id"))
    finally:
        model.train(was_training)
    levels = {}
    for name, total in totals.items():
        if total["units"]:
            levels[name] = {
                "units": total["units"],
                "top1": total["top1"] / total["units"],
                "top5": total["top5"] / total["units"],
            }
    return {
        "schema": "terminal-rsi-unit-heads-v1",
        "books": used,
        "levels": levels,
        "definition": "next-unit classification head; not byte exactness",
    }


def word_prefix_byte_log_probs(word_logits: torch.Tensor,
                               history: torch.Tensor,
                               word_bytes: torch.Tensor,
                               word_lengths: torch.Tensor,
                               top_k: int = 8) -> torch.Tensor:
    """Top-k current-word candidates consistent with the observed prefix.

    ``word_bytes`` is [vocab,maxlen]; ``history`` is [batch,time]; the last
    observed byte is history[:, -1].  The returned [batch,256] log-probabilities
    are the next-byte prior from words whose prefix matches the bytes after the
    last whitespace.
    """
    if word_logits.ndim != 2:
        raise ValueError("word_logits must be [batch,vocab]")
    vocab = word_logits.shape[-1]
    if word_bytes.shape[0] != vocab:
        if word_bytes.shape[0] < vocab:
            pad = torch.zeros((vocab - word_bytes.shape[0], word_bytes.shape[1]),
                              dtype=word_bytes.dtype, device=word_bytes.device)
            word_bytes = torch.cat((word_bytes, pad), dim=0)
            word_lengths = torch.cat((word_lengths, torch.zeros(
                vocab - word_lengths.shape[0], dtype=word_lengths.dtype,
                device=word_lengths.device)), dim=0)
        else:
            word_bytes = word_bytes[:vocab]
            word_lengths = word_lengths[:vocab]
    device = word_logits.device
    history = history.to(device).long()
    table_all = word_bytes.to(device).long()
    lengths_all = word_lengths.to(device).long()
    batch, time = history.shape
    maxlen = table_all.shape[1]
    whitespace = torch.isin(
        history, torch.tensor([9, 10, 13, 32], device=device))
    positions = torch.arange(time, device=device).expand(batch, -1)
    last_space = torch.where(whitespace, positions,
                             torch.full_like(positions, -1)).cummax(-1).values[:, -1]
    prefix_length = (time - 1 - last_space).clamp(0, maxlen)
    suffix = history.flip(1)[:, :maxlen]
    prefix_mask = torch.arange(maxlen, device=device)[None, :] < prefix_length[:, None]
    logp = torch.log_softmax(
        torch.nan_to_num(word_logits.float(), nan=0.0, posinf=30.0,
                         neginf=-30.0), dim=-1)
    keep = min(int(top_k), logp.shape[-1])
    top_values, top_indices = logp.topk(keep, dim=-1)
    table = table_all[top_indices]              # [B,K,maxlen]
    lengths = lengths_all[top_indices]          # [B,K]
    match = ((table == suffix[:, None, :]) | ~prefix_mask[:, None, :]).all(-1)
    match = match & (prefix_length[:, None] <= lengths)
    next_index = prefix_length[:, None].clamp(0, maxlen - 1)
    next_byte = table.gather(
        2, next_index[..., None].expand(-1, keep, 1)).squeeze(-1)
    valid = match & (prefix_length[:, None] < lengths)
    probabilities = torch.zeros((batch, 256), device=device)
    probabilities.scatter_add_(
        1, next_byte.clamp(0, 255), torch.exp(top_values) * valid)
    # Log a strictly positive tensor: selecting -30 with torch.where while
    # calling log(0) in the other branch makes LogBackward emit 0/0 = NaN
    # even when this prior is gated off.  Clamp first, then floor the result.
    # Make the prior a proper next-byte distribution before it reaches the
    # bit factorization.  Raw scatter counts are unnormalized and the old
    # -30 floor produced extreme logits; a small gate then dominated the
    # parent and cost held-out BPB.  Normalize, then mix a 5% uniform floor
    # so every log-probability is finite and bounded.
    vocab_size = probabilities.shape[-1]
    total = probabilities.sum(-1, keepdim=True)
    normalized = torch.where(
        total > 0,
        probabilities / total.clamp_min(1e-12),
        torch.full_like(probabilities, 1.0 / vocab_size))
    smoothed = 0.95 * normalized + 0.05 / vocab_size
    return smoothed.log()
