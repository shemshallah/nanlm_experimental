"""Hierarchical held-out metrics for the byte-level Terminal-RSI model.

The v103 model predicts bytes and 8-bit paths; it does not have a separate
token head.  These metrics score the observed target byte at every position and
aggregate the same byte decisions into BPE subwords, whitespace words and
punctuation-delimited sentences:

- byte: per-target-byte top-1/top-5, BPB/BPC/loss
- subword/token: every byte in the BPE unit must be top-1/top-5
- word: every byte in the whitespace-delimited word must be top-1/top-5
- sentence: every byte in the sentence must be top-1/top-5

In the v103 byte-BPE tokenizer ``token`` and ``subword`` are the same units;
both are reported explicitly so the distinction is never implied.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from .legacy_v103_reference import (
    PARITY_CONTEXT, build_train_features, read_bpe_ranks,
)

LN2 = math.log(2.0)
WHITESPACE = {9, 10, 13, 32}
SENTENCE_END = {ord("."), ord("?"), ord("!")}


def _causal_contexts(raw: bytes, positions: torch.Tensor, history: int):
    raw_tensor = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
    positions = positions.to("cpu", torch.long).reshape(-1)
    deltas = torch.arange(history, 0, -1, dtype=torch.long)
    indices = positions[:, None] - deltas[None, :]
    causal = indices >= 0
    past = raw_tensor[indices.clamp_min(0)].long()
    past = torch.where(causal, past, torch.zeros_like(past))
    targets = raw_tensor[positions].long()
    return torch.cat((past, targets[:, None]), dim=-1)


def _bpe_spans(data: bytes, ranks: dict[tuple[int, int], int]) -> list[tuple[int, int]]:
    ids = list(data)
    spans = [(i, i + 1) for i in range(len(data))]
    while len(ids) > 1:
        best_rank = None
        best_index = None
        for index in range(len(ids) - 1):
            rank = ranks.get((ids[index], ids[index + 1]))
            if rank is not None and (best_rank is None or rank < best_rank):
                best_rank = rank
                best_index = index
        if best_index is None:
            break
        ids[best_index] = 256 + int(best_rank)
        spans[best_index] = (spans[best_index][0], spans[best_index + 1][1])
        del ids[best_index + 1]
        del spans[best_index + 1]
    return spans


def _word_spans(data: bytes) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = None
    for index, byte in enumerate(data):
        if byte in WHITESPACE:
            if start is not None:
                spans.append((start, index))
                start = None
        elif start is None:
            start = index
    if start is not None:
        spans.append((start, len(data)))
    return spans


def _sentence_spans(data: bytes) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    index = 0
    while index < len(data):
        if data[index] in SENTENCE_END and (
                index + 1 == len(data) or data[index + 1] in WHITESPACE):
            end = index + 1
            if end > start:
                spans.append((start, end))
            start = end
            while start < len(data) and data[start] in WHITESPACE:
                start += 1
        index += 1
    if start < len(data):
        spans.append((start, len(data)))
    return spans


MAX_CANDIDATE_SEQUENCES = 2048


def _safe_eval_batch(eval_batch: int) -> int:
    """Cap 256-way candidate expansion so one forward cannot blow up VRAM."""
    return max(1, min(int(eval_batch), MAX_CANDIDATE_SEQUENCES // 256))


@torch.no_grad()
def _byte_log_probs(model, previous: torch.Tensor, target: torch.Tensor,
                    auxiliary: torch.Tensor, *, eval_batch: int = 8):
    """Exact 256-way byte log-probabilities for each target position."""
    eval_batch = _safe_eval_batch(eval_batch)
    model.eval()
    try:
        batches = []
        for offset in range(0, len(target), max(1, eval_batch)):
            prev = previous[offset:offset + eval_batch]
            tgt = target[offset:offset + eval_batch]
            aux = auxiliary[offset:offset + eval_batch]
            n = prev.shape[0]
            candidates = torch.arange(256, device=prev.device).repeat(n)
            history = prev.repeat_interleave(256, dim=0)
            expanded_aux = aux.repeat_interleave(256, dim=0)
            logits, bits = model(history, candidates, expanded_aux)
            logp = -F.binary_cross_entropy_with_logits(
                logits, bits, reduction="none").sum(-1)
            batches.append(logp.reshape(n, 256))
        return torch.cat(batches, dim=0)
    finally:
        model.train()


def _unit_metrics(parts: list[tuple[bytes, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
                  span_fn) -> dict[str, Any]:
    units = 0
    top1_hits = 0
    top5_hits = 0
    byte_top1 = 0
    byte_top5 = 0
    byte_count = 0
    bits = 0.0
    loss = 0.0
    for text, top1, top5, target_logp, _logp in parts:
        for start, end in span_fn(text):
            if end <= start or end > len(top1):
                continue
            units += 1
            top1_hits += int(bool(top1[start:end].all()))
            top5_hits += int(bool(top5[start:end].all()))
            byte_top1 += int(top1[start:end].sum())
            byte_top5 += int(top5[start:end].sum())
            byte_count += end - start
            bits += float(-target_logp[start:end].sum() / LN2)
            loss += float(-target_logp[start:end].sum())
    if units == 0:
        return {"units": 0}
    return {
        "units": units,
        "top1": top1_hits / units,
        "top5": top5_hits / units,
        "byte_top1": byte_top1 / max(1, byte_count),
        "byte_top5": byte_top5 / max(1, byte_count),
        "bits_per_unit": bits / units,
        "loss_per_unit": loss / units,
    }


def _beam_unit_metrics(parts, span_fn, *, beam: int = 16) -> dict[str, Any]:
    """True unit top-k by beam-ranking the unit's byte sequence.

    At each byte position expand with the model's top-``beam`` bytes, retain
    the best ``beam`` prefixes by summed log-probability, and compare the
    target unit's score to the final beam. This is a proper unit-level top-k
    for a byte model and is distinct from strict all-byte exactness.
    """
    units = 0
    top1_hits = 0
    top5_hits = 0
    bits = 0.0
    loss = 0.0
    for text, _top1, _top5, target_logp, logp in parts:
        for start, end in span_fn(text):
            if end <= start or end > len(target_logp):
                continue
            length = end - start
            beams: list[tuple[list[int], float]] = [([], 0.0)]
            for offset in range(length):
                row = logp[start + offset]
                choices = torch.topk(row, k=min(beam, row.numel())).indices.tolist()
                expanded = []
                for prefix, score in beams:
                    for byte in choices:
                        expanded.append((prefix + [int(byte)], score + float(row[byte])))
                expanded.sort(key=lambda item: item[1], reverse=True)
                beams = expanded[:beam]
            target = list(text[start:end])
            target_score = sum(float(logp[start + i, byte])
                               for i, byte in enumerate(target))
            best_score = beams[0][1] if beams else float("-inf")
            fifth_score = beams[-1][1] if beams else float("-inf")
            units += 1
            top1_hits += int(target_score >= best_score - 1e-7)
            top5_hits += int(target_score >= fifth_score - 1e-7)
            bits += float(-target_logp[start:end].sum() / LN2)
            loss += float(-target_logp[start:end].sum())
    if units == 0:
        return {"units": 0}
    return {
        "units": units,
        "top1": top1_hits / units,
        "top5": top5_hits / units,
        "bits_per_unit": bits / units,
        "loss_per_unit": loss / units,
    }


def evaluate_dev_excerpts(model, *, corpus_dir: Path, dev_manifest: Path,
                          bpe_merges: Path, context: int, books: int,
                          max_positions: int, eval_batch: int,
                          device: torch.device) -> dict[str, Any]:
    """Evaluate contiguous held-out DEV excerpts and aggregate by unit."""
    manifest = json.loads(Path(dev_manifest).read_text())
    selected_books = manifest.get("selected_books") or []
    if not selected_books:
        raise ValueError("DEV manifest has no selected_books")
    ranks = read_bpe_ranks(Path(bpe_merges))
    if not ranks:
        raise ValueError("BPE merge ranks required for hierarchical evaluation")
    corpus_dir = Path(corpus_dir)
    books = min(int(books), len(selected_books))

    parts: list[tuple[bytes, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    used_books: list[Any] = []
    for book in selected_books[:books]:
        path = corpus_dir / book["relative_path"]
        raw = path.read_bytes()
        start = int(book.get("excerpt_start", 0))
        length = int(book.get("length", 1536))
        excerpt = raw[start:start + length]
        if len(excerpt) < context + 32:
            continue
        features, _ = build_train_features(excerpt, ranks)
        aux = torch.from_numpy(
            features[PARITY_CONTEXT:].copy()).to(dtype=torch.int32)
        positions = torch.arange(context, len(excerpt), dtype=torch.long)
        if max_positions > 0:
            positions = positions[:max_positions]
        windows = _causal_contexts(excerpt, positions, context).to(device)
        aux_rows = aux[positions - PARITY_CONTEXT].to(device)
        logp = _byte_log_probs(
            model, windows[:, :-1], windows[:, -1], aux_rows,
            eval_batch=eval_batch)
        targets = windows[:, -1].long()
        target_logp = logp.gather(1, targets[:, None]).squeeze(1)
        top1 = (logp.argmax(dim=-1) == targets).cpu()
        top5 = (logp.topk(5, dim=-1).indices == targets[:, None]).any(dim=-1).cpu()
        scored_text = excerpt[context:context + len(positions)]
        parts.append((scored_text, top1, top5, target_logp.cpu(), logp.cpu()))
        used_books.append(book.get("gutenberg_id"))
    if not parts:
        raise ValueError("No DEV excerpt produced scorable positions")

    all_top1 = torch.cat([part[1] for part in parts])
    all_top5 = torch.cat([part[2] for part in parts])
    all_target_logp = torch.cat([part[3] for part in parts])
    byte_count = len(all_top1)
    characters = sum(len(part[0].decode("utf-8", errors="ignore"))
                     for part in parts)
    total_bits = float(-all_target_logp.sum() / LN2)
    total_loss = float(-all_target_logp.sum())

    bpe_strict = _unit_metrics(parts, lambda text: _bpe_spans(text, ranks))
    bpe_beam = _beam_unit_metrics(
        parts, lambda text: _bpe_spans(text, ranks), beam=16)
    bpe_beam5 = _beam_unit_metrics(
        parts, lambda text: _bpe_spans(text, ranks), beam=5)
    token_metrics = dict(bpe_beam)
    if bpe_beam.get("units"):
        token_metrics.update({
            "exact_top1": bpe_strict.get("top1", 0.0),
            "exact_top5": bpe_strict.get("top5", 0.0),
            "byte_top1": bpe_strict.get("byte_top1", 0.0),
            "byte_top5": bpe_strict.get("byte_top5", 0.0),
            "beam5_top1": bpe_beam5.get("top1", 0.0),
            "beam5_top5": bpe_beam5.get("top5", 0.0),
            "beam_width": 16,
            "definition": "beam-16 BPE token top-k; beam5 fields for comparison",
        })
    word_strict = _unit_metrics(parts, _word_spans)
    word_metrics = _beam_unit_metrics(parts, _word_spans, beam=16)
    word_beam5 = _beam_unit_metrics(parts, _word_spans, beam=5)
    if word_metrics.get("units"):
        word_metrics.update({
            "exact_top1": word_strict.get("top1", 0.0),
            "exact_top5": word_strict.get("top5", 0.0),
            "byte_top1": word_strict.get("byte_top1", 0.0),
            "byte_top5": word_strict.get("byte_top5", 0.0),
            "beam5_top1": word_beam5.get("top1", 0.0),
            "beam5_top5": word_beam5.get("top5", 0.0),
            "beam_width": 16,
            "definition": "beam-16 word top-k; beam5 fields for comparison",
        })
    sentence_mean = _unit_metrics(parts, _sentence_spans)
    sentence_beam = _beam_unit_metrics(parts, _sentence_spans)
    sentence_metrics = dict(sentence_mean)
    if sentence_metrics.get("units"):
        sentence_metrics.update({
            "top1": sentence_mean.get("byte_top1", 0.0),
            "top5": sentence_mean.get("byte_top5", 0.0),
            "beam_top1": sentence_beam.get("top1", 0.0),
            "beam_top5": sentence_beam.get("top5", 0.0),
            "exact_top1": sentence_mean.get("top1", 0.0),
            "exact_top5": sentence_mean.get("top5", 0.0),
            "definition": "mean per-byte top-k; beam_top1/5 are whole-sentence beam ranks",
        })
    result = {
        "schema": "terminal-rsi-hierarchical-eval-v1",
        "books": used_books,
        "positions": byte_count,
        "characters": characters,
        "context": context,
        "note": (
            "v103 is byte-level. subword is strict all-byte BPE exactness; "
            "token is beam-ranked BPE top-k; word is beam-ranked word top-k; "
            "sentence top1/top5 are mean per-byte, with beam_top1/beam_top5 "
            "and exact_top1/exact_top5 also reported. Every level exposes "
            "byte_top1/byte_top5."
        ),
        "byte": {
            "units": byte_count,
            "top1": float(all_top1.float().mean()),
            "top5": float(all_top5.float().mean()),
            "bits_per_unit": total_bits / byte_count,
            "loss_per_unit": total_loss / byte_count,
            "bpb": total_bits / byte_count,
            "bpc": total_bits / max(1, characters),
            "loss": total_loss / byte_count,
        },
        "subword": bpe_strict,
        "token": token_metrics,
        "word": word_metrics,
        "sentence": sentence_metrics,
    }
    return result
