#!/usr/bin/env python3
"""Lexicon-constrained byte-level beam/scorer for Terminal-RSI (W2.4 / W6.1).

WHAT THIS IS
------------
``WordTrie`` stores the 4,096-entry word vocabulary used by
``unit_heads.UnitHeads`` as a byte trie.  Two scoring paths are provided:

``ExactPrefixScorer`` (candidate-prefix path)
    For every trie edge ``(candidate_prefix -> child_byte)`` it runs the real
    ``VerbatimV103RSI`` model once with::

        previous = observed_history + candidate_prefix
        target   = child_byte

    The model returns eight conditional bit logits.  Summing
    ``log sigmoid(bit_logit)`` along the target bit path gives exactly
    ``log P(child_byte | observed_history + candidate_prefix)`` under the
    model's own bit factorization.  Edges at the same trie depth have the same
    ``previous`` length, so all of them are batched into one forward.  A word
    score is the sum of its per-byte conditional log-probabilities:

        log P(w | h) = sum_i log P(w[i] | h + w[:i])

    This is exact for the model; it is *not* a one-shot teacher-forced score.

``BoundaryByteScorer`` (teacher-forced trie fallback)
    Runs a single 256-way enumeration of the next byte at the observed
    boundary, then multiplies those *same* boundary probabilities along every
    trie path.  It is much cheaper (256 model rows per boundary), but it never
    re-conditions on the candidate prefix, so long words are systematically
    under-scored relative to the exact path.

CAUSAL RULE (IMPORTANT)
-----------------------
The target word's bytes are never read as model input.  The observed history
ends immediately before the word start.  At depth ``d`` the model sees only
``history + candidate_prefix`` where ``candidate_prefix`` has ``d`` bytes and
the byte being scored is the ``(d+1)``-th candidate byte.  The teacher-forced
fallback is also target-free: it uses only the boundary history.  No ground
truth future byte enters either path.  The v103 auxiliary features are the
model's causal ``_default_aux`` fallback (last eight observed bytes), not the
six-feature NaNCM bridge used during training; that is the one approximation
in the exact path beyond the 256-byte context truncation.

COST
----
* exact prefix path: one model forward row per trie edge visited.  Cost is
  ``O(sum_d beam_width * branching_d)`` rows per word boundary.  For a 4,096
  word / ~20k node trie, ``--beam 8`` typically visits a few hundred to a few
  thousand rows per boundary; a full exhaustive traversal would be ~20k rows.
  CPU is slow; CUDA is the intended execution profile.
* teacher fallback: 256 model rows per boundary plus O(trie nodes) Python.
* ``WordTrie.score_prefixes`` alone is pure Python and costs O(trie nodes);
  it is useful for caching a boundary distribution once.

These are model-row counts, not wall-clock benchmarks; no real-checkpoint
timing was run in this environment.

EXPECTED RECALL
---------------
The ceiling is in-vocabulary coverage: the beam can only return the words in
the supplied trie, and target id 0 (UNK/OOV) is impossible.  On the repo's
16 MB pilot the measured prior top-1000 coverage is 92.56% (top-100 76.0%)
and union top-100 coverage is 79.75% (see ``WORD_TOP5_CHECKLIST.md``); the
exact beam's recall has NOT been measured here.  Treat it as a candidate generator, not an
oracle: with beam width 8, recall@100 is bounded by the number of distinct
word prefixes the beam keeps, and by the byte model's per-byte top-k mass.
The teacher fallback should be expected to be clearly worse than the exact
path on words longer than ~4 bytes because it cannot condition on the
candidate prefix.  The exact beam also prunes the *first* byte to
``beam_width`` at the root, so words beginning with a rare byte are
unreachable for small beams; use ``beam_width=None`` (CLI ``--beam 0``) or a
larger beam when measuring the vocabulary ceiling.

    The vocabulary must be the *exact training vocabulary* (same bytes -> id
    mapping) used by the checkpoint.  The checkpoint does not embed it;
    ``set_word_vocab`` is driven by this mapping, so shifted ids silently
    misalign the word prior and the head baseline.

USAGE (library)
---------------
>>> trie = WordTrie(vocab, max_len=24)
>>> results = lexicon_beam(model, [history_bytes], trie, mode="prefix",
...                        top_k=100, beam_width=8)
>>> [c.word_id for c in results[0].candidates]

USAGE (CLI, no GPU job is started by this module itself)
--------------------------------------------------------
python -m terminal_rsi.word_lexicon_beam \\
    --checkpoint last.pt --dev-bin win_dev.bin \\
    --dev-manifest win_dev.manifest.json --vocab-json word_vocab.json \\
    --out word_lexicon_beam.json --exact teacher --max-boundaries 8

``--exact auto`` selects the exact path on CUDA and the teacher fallback on
CPU.  ``--exact prefix`` forces the exact path even on CPU.  This module does
not run any training or long GPU job; the CLI only scores the requested dev
boundaries.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import struct
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

WHITESPACE = frozenset((9, 10, 13, 32))
MODEL_MIN_CONTEXT = 32
DEFAULT_CONTEXT_WINDOW = 256
UNK_ID = 0
DEFAULT_MAX_WORD_BYTES = 24
DEFAULT_MAX_VOCAB = 4096
RECALL_KS = (5, 20, 100)


# ---------------------------------------------------------------------------
# Pure-Python helpers
# ---------------------------------------------------------------------------


def word_spans(raw: bytes) -> list[tuple[int, int, bytes]]:
    """Whitespace-delimited word spans, matching ``unit_heads._word_tokens``.

    Whitespace is tab, LF, CR and space.  Punctuation remains attached to the
    word, exactly as in the training vocabulary.
    """
    spans: list[tuple[int, int, bytes]] = []
    start: int | None = None
    for index, byte in enumerate(raw):
        if byte in WHITESPACE:
            if start is not None:
                spans.append((start, index, raw[start:index]))
                start = None
        elif start is None:
            start = index
    if start is not None:
        spans.append((start, len(raw), raw[start:]))
    return spans


def _logsumexp(values: Sequence[float]) -> float:
    if not values:
        return -math.inf
    largest = max(values)
    if largest == -math.inf:
        return -math.inf
    if not math.isfinite(largest):
        raise ValueError("log-sum-exp input contains +inf or NaN")
    return largest + math.log(sum(math.exp(value - largest) for value in values))


def _evenly_spaced(items: Sequence[Any], count: int) -> list[Any]:
    """Deterministic, evenly spread subset of ``items`` (no RNG)."""
    if count <= 0 or not items:
        return []
    if len(items) <= count:
        return list(items)
    if count == 1:
        return [items[len(items) // 2]]
    indices = [(index * (len(items) - 1)) // (count - 1)
               for index in range(count)]
    return [items[index] for index in indices]


def _as_bytes(value: Any) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, (bytearray, memoryview)):
        return bytes(value)
    if isinstance(value, str):
        return value.encode("utf-8")
    if isinstance(value, Sequence):
        return bytes(int(item) for item in value)
    raise TypeError(f"cannot convert {type(value).__name__} to bytes")


def _json_safe(value: Any) -> Any:
    """Recursively replace non-finite floats so JSON output is valid."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def recall_at_k(rankings: Sequence[Sequence[int]], targets: Sequence[int],
                k: int, *, in_vocab_only: bool = False) -> float | None:
    """Fraction of targets present in the first ``k`` ranked IDs."""
    total = 0
    hits = 0
    for ranking, target in zip(rankings, targets):
        target = int(target)
        if in_vocab_only and target == UNK_ID:
            continue
        total += 1
        if target in {int(item) for item in list(ranking)[:k]}:
            hits += 1
    return hits / total if total else None


def unique_hits(rankings: Sequence[Sequence[int]], targets: Sequence[int],
                k: int) -> int:
    """Number of distinct non-UNK target words hit in the first ``k``."""
    hits: set[int] = set()
    for ranking, target in zip(rankings, targets):
        target = int(target)
        if target == UNK_ID:
            continue
        if target in {int(item) for item in list(ranking)[:k]}:
            hits.add(target)
    return len(hits)


def _metric_block(rankings: Sequence[Sequence[int]] | None,
                  targets: Sequence[int], *,
                  in_vocab_only: bool = False) -> dict[str, Any] | None:
    if rankings is None:
        return None
    block: dict[str, Any] = {}
    for k in RECALL_KS:
        block[f"recall@{k}"] = recall_at_k(
            rankings, targets, k, in_vocab_only=in_vocab_only)
    block["unique_hits@5"] = unique_hits(rankings, targets, 5)
    return block


# ---------------------------------------------------------------------------
# Word trie
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PrefixScore:
    node: int
    prefix: bytes
    word_id: int
    score: float
    depth: int
    mean_logprob: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "node": self.node,
            "word": self.prefix.decode("utf-8", "replace"),
            "word_id": self.word_id,
            "score": self.score,
            "depth": self.depth,
            "mean_logprob": self.mean_logprob,
        }


@dataclass(frozen=True)
class WordCandidate:
    word_id: int
    word: bytes
    score: float
    length: int
    mean_logprob: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "word_id": self.word_id,
            "word": self.word.decode("utf-8", "replace"),
            "score": self.score,
            "length": self.length,
            "mean_logprob": self.mean_logprob,
        }


@dataclass
class BeamResult:
    candidates: list[WordCandidate]
    beam_width: int | None
    max_depth: int
    depths: int
    prefixes_expanded: int
    edges_scored: int
    truncated: bool
    stop_reason: str
    seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "beam_width": self.beam_width,
            "max_depth": self.max_depth,
            "depths": self.depths,
            "prefixes_expanded": self.prefixes_expanded,
            "edges_scored": self.edges_scored,
            "truncated": self.truncated,
            "stop_reason": self.stop_reason,
            "seconds": self.seconds,
        }


class WordTrie:
    """Byte trie over ``{word_bytes: word_id}``.

    Node 0 is the root.  ``word_ids[node]`` is the vocabulary id ending at
    that node (0 if the node is only a prefix).  Words longer than ``max_len``
    are skipped, matching ``VerbatimV103RSI.set_word_vocab(maxlen=24)`` which
    truncates the word-head table at 24 bytes.
    """

    def __init__(self, vocabulary: Mapping[bytes, int], *,
                 max_len: int = DEFAULT_MAX_WORD_BYTES,
                 max_vocab: int = DEFAULT_MAX_VOCAB):
        if max_len < 1:
            raise ValueError("max_len must be positive")
        if max_vocab < 2:
            raise ValueError("max_vocab must be >= 2")
        self.max_len = int(max_len)
        self.max_vocab = int(max_vocab)
        self.children: list[dict[int, int]] = [{}]
        self.word_ids: list[int] = [0]
        self.parents: list[int] = [-1]
        self.edge_bytes: list[int] = [-1]
        self.id_to_word: dict[int, bytes] = {}
        self.word_to_id: dict[bytes, int] = {}
        self.skipped = {"empty": 0, "too_long": 0, "unk": 0}
        for raw_word, raw_id in vocabulary.items():
            word = _as_bytes(raw_word)
            try:
                word_id = int(raw_id)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid word id for {word!r}") from exc
            if not word:
                self.skipped["empty"] += 1
                continue
            if word_id <= UNK_ID:
                self.skipped["unk"] += 1
                continue
            if len(word) > self.max_len:
                self.skipped["too_long"] += 1
                continue
            if word_id >= self.max_vocab:
                raise ValueError(
                    f"word id {word_id} >= max_vocab {self.max_vocab}")
            if word_id in self.id_to_word:
                raise ValueError(f"duplicate word id {word_id}")
            if word in self.word_to_id:
                raise ValueError(f"duplicate word {word!r}")
            self.id_to_word[word_id] = word
            self.word_to_id[word] = word_id
            self._insert(word, word_id)
        self.longest_word = max(
            (len(word) for word in self.word_to_id), default=0)

    def _insert(self, word: bytes, word_id: int) -> None:
        node = 0
        for byte in word:
            child = self.children[node].get(byte)
            if child is None:
                child = len(self.children)
                self.children[node][byte] = child
                self.children.append({})
                self.word_ids.append(0)
                self.parents.append(node)
                self.edge_bytes.append(byte)
            node = child
        if self.word_ids[node]:
            raise ValueError(f"word {word!r} already has id {self.word_ids[node]}")
        self.word_ids[node] = word_id

    def __len__(self) -> int:
        return len(self.id_to_word)

    @property
    def node_count(self) -> int:
        return len(self.children)

    def children_of(self, node: int) -> list[tuple[int, int]]:
        """Sorted ``(byte, child_node)`` pairs, for determinism."""
        return sorted(self.children[int(node)].items())

    def prefix_node(self, prefix: bytes) -> int | None:
        node = 0
        for byte in prefix:
            child = self.children[node].get(int(byte))
            if child is None:
                return None
            node = child
        return node

    def node_prefix(self, node: int) -> bytes:
        output = bytearray()
        current = int(node)
        while current > 0:
            output.append(self.edge_bytes[current])
            current = self.parents[current]
        return bytes(reversed(output))

    def words(self) -> list[tuple[bytes, int]]:
        return [(word, word_id)
                for word_id, word in sorted(self.id_to_word.items())]

    def score_prefixes(self, logp256: Sequence[float], *,
                       top_k: int | None = None,
                       terminal_only: bool = True,
                       normalize: bool = False) -> list[PrefixScore]:
        """Score every trie prefix under one boundary byte distribution.

        ``logp256`` is a 256-vector of next-byte log-probabilities at the
        observed boundary.  A prefix score is the sum of ``logp256`` along its
        bytes.  With ``terminal_only=False`` non-terminal prefixes are
        returned as well (``word_id == 0``).  ``normalize=True`` subtracts the
        log mass of ``logp256`` once per byte, turning a slightly unnormalized
        model distribution into a proper one.

        This is the teacher-forced fallback: the same boundary distribution is
        reused at every depth, so candidate prefixes are NOT re-scored.
        """
        values = [float(value) for value in logp256]
        if len(values) != 256:
            raise ValueError("logp256 must contain 256 values")
        for value in values:
            if math.isnan(value) or value == math.inf:
                raise ValueError("logp256 contains NaN or +inf")
        if normalize:
            log_mass = _logsumexp(values)
            if not math.isfinite(log_mass):
                raise ValueError("cannot normalize a zero-mass distribution")
            values = [value - log_mass for value in values]

        results: list[PrefixScore] = []
        stack: list[tuple[int, bytes, float, int]] = [(0, b"", 0.0, 0)]
        while stack:
            node, prefix, score, depth = stack.pop()
            word_id = self.word_ids[node]
            if terminal_only:
                if word_id:
                    results.append(PrefixScore(
                        node=node, prefix=prefix, word_id=word_id,
                        score=score, depth=depth,
                        mean_logprob=score / depth if depth else 0.0))
            elif node != 0:
                results.append(PrefixScore(
                    node=node, prefix=prefix, word_id=word_id,
                    score=score, depth=depth,
                    mean_logprob=score / depth if depth else 0.0))
            if depth >= self.max_len:
                continue
            for byte, child in self.children[node].items():
                stack.append((child, prefix + bytes((byte,)),
                              score + values[byte], depth + 1))
        results.sort(key=lambda item: (-item.score, item.word_id, item.prefix))
        if top_k is not None:
            results = results[:max(0, int(top_k))]
        return results

    def beam_search(self, scorer: Any, context: bytes, *,
                    beam_width: int | None = 8,
                    top_k: int = 100,
                    max_depth: int | None = None,
                    max_edges: int | None = None,
                    progress: Callable[[dict[str, Any]], None] | None = None
                    ) -> BeamResult:
        """Beam over word prefixes using a batched next-byte scorer.

        ``scorer`` must expose ``score_pairs(pairs)`` where each pair is
        ``(context_bytes, [child_byte, ...])`` and the return value is a list
        (aligned with ``pairs``) of lists of ``log P(child | context)``.  A
        plain callable with the same signature is also accepted.

        At depth ``d`` every live prefix has length ``d``, so all
        ``(history + prefix, child)`` rows in one level have the same model
        input length and can be sent through the model as one batch.  Scores
        of returned words are exact under the scorer even when the beam
        prunes other words.  ``beam_width=None`` disables pruning (exhaustive
        trie traversal, guarded by ``max_edges``).
        """
        started = time.perf_counter()
        base = _as_bytes(context)
        depth_limit = self.longest_word if max_depth is None else min(
            int(max_depth), self.longest_word)
        depth_truncated = (max_depth is not None
                           and int(max_depth) < self.longest_word)
        if depth_limit <= 0:
            return BeamResult([], beam_width, 0, 0, 0, 0, False,
                              "empty_trie", time.perf_counter() - started)
        if beam_width is not None and int(beam_width) <= 0:
            raise ValueError("beam_width must be positive or None")
        edge_limit = None if max_edges is None else int(max_edges)
        if edge_limit is not None and edge_limit < 0:
            raise ValueError("max_edges must be non-negative or None")

        beam: list[tuple[bytes, int, float]] = [(b"", 0, 0.0)]
        terminals: dict[int, WordCandidate] = {}
        prefixes_expanded = 0
        edges_scored = 0
        depths = 0
        truncated = False
        stop_reason = "complete"

        for depth in range(depth_limit):
            pairs: list[tuple[bytes, list[int]]] = []
            levels: list[tuple[bytes, int, float, list[tuple[int, int]]]] = []
            for prefix, node, score in beam:
                children = self.children_of(node)
                if not children:
                    continue
                pairs.append((base + prefix, [byte for byte, _ in children]))
                levels.append((prefix, node, score, children))
            if not pairs:
                stop_reason = "no_expansions"
                break
            level_edges = sum(len(children) for _, _, _, children in levels)
            if edge_limit is not None and edges_scored + level_edges > edge_limit:
                truncated = True
                stop_reason = "max_edges"
                break
            scored = _call_scorer(scorer, pairs)
            if len(scored) != len(levels):
                raise ValueError(
                    "scorer returned a different number of rows than pairs")
            next_beam: list[tuple[bytes, int, float]] = []
            for (prefix, _node, score, children), child_scores in zip(
                    levels, scored):
                if len(child_scores) != len(children):
                    raise ValueError(
                        "scorer returned a different number of children")
                for (byte, child), logp in zip(children, child_scores):
                    value = float(logp)
                    if math.isnan(value) or value == -math.inf:
                        value = -1.0e9
                    elif value == math.inf:
                        value = 0.0
                    child_score = score + value
                    child_prefix = prefix + bytes((byte,))
                    word_id = self.word_ids[child]
                    if word_id:
                        candidate = WordCandidate(
                            word_id=word_id,
                            word=self.id_to_word[word_id],
                            score=child_score,
                            length=len(child_prefix),
                            mean_logprob=(child_score / len(child_prefix)
                                          if child_prefix else 0.0))
                        old = terminals.get(word_id)
                        if old is None or candidate.score > old.score:
                            terminals[word_id] = candidate
                    if (depth + 1 < depth_limit
                            and self.children[child]):
                        next_beam.append(
                            (child_prefix, child, child_score))
            prefixes_expanded += len(pairs)
            edges_scored += level_edges
            depths = depth + 1
            if progress is not None:
                progress({
                    "depth": depth + 1,
                    "beam_prefixes": len(next_beam),
                    "edges_scored": edges_scored,
                    "terminals": len(terminals),
                })
            if not next_beam:
                stop_reason = "max_depth" if depth_truncated else "exhausted"
                break
            if beam_width is not None:
                next_beam.sort(key=lambda item: (-item[2], item[1]))
                next_beam = next_beam[:int(beam_width)]
            beam = next_beam

        candidates = sorted(
            terminals.values(), key=lambda item: (-item.score, item.word_id))
        if top_k is not None:
            candidates = candidates[:max(0, int(top_k))]
        return BeamResult(
            candidates=candidates,
            beam_width=beam_width,
            max_depth=depth_limit,
            depths=depths,
            prefixes_expanded=prefixes_expanded,
            edges_scored=edges_scored,
            truncated=truncated or depth_truncated,
            stop_reason=stop_reason,
            seconds=time.perf_counter() - started,
        )


def _call_scorer(scorer: Any,
                 pairs: Sequence[tuple[bytes, Sequence[int]]]
                 ) -> list[list[float]]:
    if hasattr(scorer, "score_pairs"):
        return scorer.score_pairs(list(pairs))
    if callable(scorer):
        return scorer(list(pairs))
    raise TypeError(
        "scorer must expose score_pairs(pairs) or be callable(pairs)")


# ---------------------------------------------------------------------------
# Model-backed scorers (torch imported lazily)
# ---------------------------------------------------------------------------


def _infer_torch_device(model: Any, torch: Any) -> Any:
    try:
        parameter = next(model.parameters())
        return parameter.device
    except (AttributeError, StopIteration):
        pass
    try:
        buffer = next(model.buffers())
        return buffer.device
    except (AttributeError, StopIteration):
        return torch.device("cpu")


def _run_model_logp(model: Any, previous_rows: Sequence[bytes],
                    targets: Sequence[int], *, device: Any
                    ) -> tuple[list[float], int]:
    """One model forward for one batch; returns summed bit log-probs."""
    import torch
    import torch.nn.functional as F

    if len(previous_rows) != len(targets):
        raise ValueError("previous_rows and targets must be aligned")
    if previous_rows:
        lengths = {len(row) for row in previous_rows}
        if len(lengths) != 1:
            raise ValueError("one batch must contain equal-length rows")
    previous = torch.tensor(
        [list(row) for row in previous_rows],
        dtype=torch.long, device=device)
    target = torch.tensor(
        [int(value) for value in targets], dtype=torch.long, device=device)
    with torch.inference_mode():
        output = model(previous, target)
    if isinstance(output, dict):
        logits = output.get("logits")
        bits = output.get("bits")
    elif isinstance(output, (tuple, list)) and output:
        logits = output[0]
        bits = output[1] if len(output) > 1 else None
    else:
        raise TypeError("model must return (logits, bits) or a dict")
    if logits is None or getattr(logits, "ndim", None) != 2:
        raise ValueError("model logits must be [batch, 8]")
    if int(logits.shape[-1]) != 8:
        raise ValueError("model logits must have eight bit columns")
    if bits is None:
        bit_index = torch.arange(8, device=logits.device)
        bits = ((target[:, None] >> (7 - bit_index[None, :])) & 1).float()
    else:
        bits = bits.to(device=logits.device, dtype=logits.dtype)
        if tuple(bits.shape) != tuple(logits.shape):
            raise ValueError("model bits must match logits shape")
    # log P(bit) for the target path; sum over the eight conditional bits.
    logp = F.logsigmoid((2.0 * bits - 1.0) * logits).sum(-1)
    values = logp.detach().float().cpu().tolist()
    nonfinite = 0
    cleaned: list[float] = []
    for value in values:
        value = float(value)
        if math.isnan(value) or value == -math.inf:
            value = -1.0e9
            nonfinite += 1
        elif value == math.inf:
            value = 0.0
            nonfinite += 1
        cleaned.append(value)
    return cleaned, nonfinite


class ExactPrefixScorer:
    """Candidate-prefix scorer: one model row per trie child edge, batched.

    ``score_pairs`` receives ``(context, children)`` pairs.  For every child
    byte it calls the real model with ``previous = context`` and
    ``target = child``.  At a fixed trie depth all contexts have the same
    length, so the rows are grouped by length and batched.
    """

    def __init__(self, model: Any = None, *,
                 runner: Callable[[Sequence[bytes], Sequence[int]],
                                  tuple[list[float], int]] | None = None,
                 context_window: int = DEFAULT_CONTEXT_WINDOW,
                 batch_size: int = 64,
                 device: Any | None = None,
                 min_context: int = MODEL_MIN_CONTEXT,
                 restore_training: bool = True):
        if context_window < min_context:
            raise ValueError("context_window must be >= min_context")
        if min_context < MODEL_MIN_CONTEXT:
            raise ValueError(f"min_context must be >= {MODEL_MIN_CONTEXT}")
        if model is None and runner is None:
            raise ValueError("ExactPrefixScorer needs a model or a runner")
        self.model = model
        self._runner = runner
        self.context_window = int(context_window)
        self.batch_size = max(1, int(batch_size))
        self.min_context = int(min_context)
        self.restore_training = bool(restore_training)
        if runner is not None:
            # Dependency-injected scoring for tests; no torch import.
            self.device = None
            self._was_training = False
        else:
            import torch  # noqa: F401  (lazy dependency)
            self.device = (torch.device(device) if device is not None
                           else _infer_torch_device(model, torch))
            self._was_training = bool(getattr(model, "training", False))
            if hasattr(model, "eval"):
                model.eval()
        self.model_calls = 0
        self.forward_rows = 0
        self.edges_scored = 0
        self.nonfinite = 0

    def __enter__(self) -> "ExactPrefixScorer":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def close(self) -> None:
        if (self.restore_training and self._was_training
                and hasattr(self.model, "train")):
            self.model.train()

    def window(self, context: bytes) -> bytes:
        data = _as_bytes(context)
        if len(data) > self.context_window:
            data = data[-self.context_window:]
        if len(data) < self.min_context:
            data = b"\x00" * (self.min_context - len(data)) + data
        return data

    def score_pairs(self, pairs: Sequence[tuple[bytes, Sequence[int]]]
                    ) -> list[list[float]]:
        windows: list[bytes] = []
        flat: list[tuple[int, int]] = []
        spans: list[tuple[int, int]] = []
        for context, children in pairs:
            windowed = self.window(context)
            child_list = [int(child) for child in children]
            if any(child < 0 or child > 255 for child in child_list):
                raise ValueError("child bytes must be in [0, 255]")
            start = len(flat)
            spans.append((start, start + len(child_list)))
            flat.extend((len(windows), child) for child in child_list)
            windows.append(windowed)
        values = [0.0] * len(flat)
        groups: dict[int, list[int]] = defaultdict(list)
        for index, (window_index, _child) in enumerate(flat):
            groups[len(windows[window_index])].append(index)
        for length, indices in sorted(groups.items()):
            for start in range(0, len(indices), self.batch_size):
                chunk = indices[start:start + self.batch_size]
                previous_rows = [windows[flat[index][0]] for index in chunk]
                targets = [flat[index][1] for index in chunk]
                if self._runner is not None:
                    logp, nonfinite = self._runner(previous_rows, targets)
                else:
                    logp, nonfinite = _run_model_logp(
                        self.model, previous_rows, targets,
                        device=self.device)
                if len(logp) != len(chunk):
                    raise ValueError("model returned the wrong batch size")
                self.model_calls += 1
                self.forward_rows += len(chunk)
                self.nonfinite += nonfinite
                for index, value in zip(chunk, logp):
                    values[index] = value
        self.edges_scored += len(flat)
        result: list[list[float]] = []
        for start, end in spans:
            result.append(values[start:end])
        return result

    def stats(self) -> dict[str, Any]:
        return {
            "teacher_forced": False,
            "model_calls": self.model_calls,
            "forward_rows": self.forward_rows,
            "edges_scored": self.edges_scored,
            "nonfinite": self.nonfinite,
            "context_window": self.context_window,
        }


class BoundaryByteScorer:
    """Teacher-forced trie fallback: one 256-way boundary distribution.

    The distribution is estimated once at the observed boundary and reused at
    every depth.  ``score_pairs`` ignores the supplied context on purpose;
    this is the documented approximation.  ``mass_error`` reports
    ``exp(logsumexp(logp256)) - 1`` so callers can see whether the model's bit
    factorization is normalized (the legacy ``byte_metrics`` performs the same
    diagnostic).
    """

    def __init__(self, model: Any = None, base_context: bytes = b"", *,
                 runner: Callable[[Sequence[bytes], Sequence[int]],
                                  tuple[list[float], int]] | None = None,
                 context_window: int = DEFAULT_CONTEXT_WINDOW,
                 batch_size: int = 256,
                 device: Any | None = None,
                 min_context: int = MODEL_MIN_CONTEXT,
                 normalize: bool = False,
                 restore_training: bool = True):
        self.base_context = _as_bytes(base_context)
        self._prefix = ExactPrefixScorer(
            model, runner=runner, context_window=context_window,
            batch_size=max(1, int(batch_size)), device=device,
            min_context=min_context,
            restore_training=restore_training)
        values = self._prefix.score_pairs(
            [(self.base_context, list(range(256)))])[0]
        self.log_mass = _logsumexp(values)
        self.mass_error = (math.exp(self.log_mass) - 1.0
                           if math.isfinite(self.log_mass) else math.nan)
        self.normalized = bool(normalize)
        if self.normalized:
            if not math.isfinite(self.log_mass):
                raise ValueError("cannot normalize a zero-mass distribution")
            self.logp = [value - self.log_mass for value in values]
        else:
            self.logp = list(values)

    def __enter__(self) -> "BoundaryByteScorer":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def close(self) -> None:
        self._prefix.close()

    def score_pairs(self, pairs: Sequence[tuple[bytes, Sequence[int]]]
                    ) -> list[list[float]]:
        return [[float(self.logp[int(child)]) for child in children]
                for _context, children in pairs]

    def stats(self) -> dict[str, Any]:
        stats = self._prefix.stats()
        stats.update({
            "teacher_forced": True,
            "mass_error": self.mass_error,
            "normalized": self.normalized,
        })
        return stats


# ---------------------------------------------------------------------------
# Public beam entry point
# ---------------------------------------------------------------------------


@dataclass
class LexiconResult:
    candidates: list[WordCandidate]
    requested_mode: str
    mode: str
    context_bytes: int
    truncated: bool
    stop_reason: str
    fallback_reason: str | None
    cost: dict[str, Any]
    notes: list[str] = field(default_factory=list)

    def to_dict(self, *, max_candidates: int | None = None) -> dict[str, Any]:
        candidates = self.candidates
        if max_candidates is not None:
            candidates = candidates[:max(0, int(max_candidates))]
        return {
            "candidates": [candidate.to_dict() for candidate in candidates],
            "requested_mode": self.requested_mode,
            "mode": self.mode,
            "context_bytes": self.context_bytes,
            "truncated": self.truncated,
            "stop_reason": self.stop_reason,
            "fallback_reason": self.fallback_reason,
            "cost": _json_safe(self.cost),
            "notes": list(self.notes),
        }


def _normalise_histories(histories: Any) -> list[bytes]:
    if isinstance(histories, (bytes, bytearray, memoryview, str)):
        return [_as_bytes(histories)]
    try:
        items = list(histories)
    except TypeError as exc:
        raise TypeError("histories must be bytes or a sequence of bytes") from exc
    return [_as_bytes(item) for item in items]


def _torch_on_cuda(model: Any, device: Any | None) -> bool:
    try:
        import torch
    except ImportError:
        return False
    if device is not None:
        return torch.device(device).type == "cuda"
    return _infer_torch_device(model, torch).type == "cuda"


_COUNTER_STAT_KEYS = ("model_calls", "forward_rows", "edges_scored",
                      "nonfinite")


def _stats_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """Difference of cumulative scorer counters; configuration is copied."""
    delta: dict[str, Any] = {}
    for key, value in after.items():
        if (key in _COUNTER_STAT_KEYS
                and isinstance(value, (int, float))
                and not isinstance(value, bool)):
            delta[key] = value - float(before.get(key, 0.0))
        else:
            delta[key] = value
    return delta


def lexicon_beam(model: Any, histories: Any, trie: WordTrie, *,
                 mode: str = "auto",
                 scorer: Any | None = None,
                 runner: Callable[[Sequence[bytes], Sequence[int]],
                                  tuple[list[float], int]] | None = None,
                 top_k: int = 100,
                 beam_width: int | None = 8,
                 context_window: int = DEFAULT_CONTEXT_WINDOW,
                 batch_size: int = 64,
                 max_edges: int | None = 20000,
                 device: Any | None = None,
                 fallback_on_error: bool = True,
                 progress: Callable[[dict[str, Any]], None] | None = None
                 ) -> list[LexiconResult]:
    """Score the next word for one or more histories against ``trie``.

    Parameters are documented in the module docstring.  ``scorer`` may be a
    custom object implementing ``score_pairs``; when supplied, the exact
    candidate-prefix path is used and ``model`` may be ``None`` (useful for
    pure-Python tests).  ``runner`` is the lower-level dependency-injection
    hook used by ``ExactPrefixScorer``/``BoundaryByteScorer`` for tests.
    ``mode='auto'`` selects the exact path on CUDA and the teacher-forced trie
    fallback on CPU.
    """
    requested = str(mode).lower()
    if requested not in ("auto", "prefix", "teacher"):
        raise ValueError("mode must be auto, prefix or teacher")
    if trie is None or len(trie) == 0:
        raise ValueError("trie must contain at least one word")
    items = _normalise_histories(histories)
    if scorer is not None and requested == "teacher":
        raise ValueError("a custom scorer is only valid for prefix mode")
    if model is None and scorer is None and runner is None:
        raise ValueError("model, scorer or runner is required")
    if scorer is not None:
        effective_mode = "prefix"
    elif requested == "auto":
        effective_mode = ("prefix" if _torch_on_cuda(model, device)
                          else "teacher")
    else:
        effective_mode = requested

    prefix_scorer = scorer
    owned_prefix_scorer = False
    if effective_mode == "prefix" and prefix_scorer is None:
        prefix_scorer = ExactPrefixScorer(
            model, runner=runner, context_window=context_window,
            batch_size=batch_size, device=device)
        owned_prefix_scorer = True
    if effective_mode == "prefix" and prefix_scorer is None:
        raise ValueError("prefix mode requires a model or scorer")

    notes: list[str] = []
    model_vocab = getattr(model, "_word_vocab", None) if model is not None else None
    if model_vocab is not None and set(model_vocab) != set(trie.word_to_id):
        notes.append(
            "trie vocabulary differs from model._word_vocab; the model's "
            "word-prefix prior uses the model vocabulary, the beam uses the trie")

    results: list[LexiconResult] = []
    try:
        for history in items:
            base = bytes(history)
            started_mode = effective_mode
            fallback_reason: str | None = None
            before = (prefix_scorer.stats()
                      if prefix_scorer is not None
                      and hasattr(prefix_scorer, "stats") else {})
            started = time.perf_counter()
            candidates: list[WordCandidate]
            truncated = False
            stop_reason = "complete"
            used_mode = started_mode

            if started_mode == "prefix":
                try:
                    beam = trie.beam_search(
                        prefix_scorer, base, beam_width=beam_width,
                        top_k=top_k, max_edges=max_edges, progress=progress)
                    candidates = beam.candidates
                    truncated = beam.truncated
                    stop_reason = beam.stop_reason
                except Exception as exc:  # noqa: BLE001
                    if (requested != "auto" or not fallback_on_error
                            or model is None):
                        raise
                    fallback_reason = f"{type(exc).__name__}: {exc}"
                    used_mode = "teacher"
                    candidates = []
                    truncated = False
                    stop_reason = "fallback"
            else:
                candidates = []

            if used_mode == "teacher":
                with BoundaryByteScorer(
                        model, base, runner=runner,
                        context_window=context_window,
                        batch_size=max(1, int(batch_size)),
                        device=device) as teacher:
                    prefix_scores = trie.score_prefixes(
                        teacher.logp, top_k=top_k, terminal_only=True)
                    candidates = [
                        WordCandidate(
                            word_id=score.word_id,
                            word=score.prefix,
                            score=score.score,
                            length=score.depth,
                            mean_logprob=score.mean_logprob)
                        for score in prefix_scores if score.word_id]
                    stats = teacher.stats()
                truncated = False
                stop_reason = "all_trie_nodes"
                if fallback_reason is None:
                    notes_for_result = [
                        "teacher-forced boundary distribution: candidate "
                        "prefixes are not re-scored"]
                else:
                    notes_for_result = [
                        "teacher-forced fallback after prefix error: "
                        "candidate prefixes are not re-scored"]
                stats["trie_nodes"] = trie.node_count
                stats["trie_edges"] = trie.node_count - 1
                cost = dict(stats)
            else:
                after = (prefix_scorer.stats()
                         if hasattr(prefix_scorer, "stats") else {})
                cost = _stats_delta(before, after) if after else {}
                cost["trie_nodes"] = trie.node_count
                cost["trie_edges"] = int(cost.get("edges_scored", 0))
                notes_for_result = [
                    "candidate-prefix re-scoring: one model row per trie "
                    "edge, batched by depth; target bytes are never inputs"]
            cost["seconds"] = time.perf_counter() - started
            cost["candidates"] = len(candidates)
            results.append(LexiconResult(
                candidates=candidates,
                requested_mode=requested,
                mode=used_mode,
                context_bytes=len(base),
                truncated=truncated,
                stop_reason=stop_reason,
                fallback_reason=fallback_reason,
                cost=cost,
                notes=notes + notes_for_result,
            ))
    finally:
        if owned_prefix_scorer and prefix_scorer is not None:
            prefix_scorer.close()
    return results


# ---------------------------------------------------------------------------
# Dev-bin / vocabulary / baseline helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Boundary:
    book: int
    word_index: int
    position: int
    word: bytes
    target_id: int


def read_dev_bin(path: str | Path) -> tuple[list[bytes], int, str, bytes]:
    """Read the NaNCM ``win_dev.bin`` layout.

    Layout: ``<II>`` = (book_count, skip), then per book ``<I>`` length and
    the excerpt bytes.  If the header does not parse cleanly, the whole file
    is returned as one raw excerpt and ``format='raw'`` (the manifest SHA
    check still catches accidental swaps).
    """
    payload = Path(path).read_bytes()
    if len(payload) >= 8:
        books, skip = struct.unpack_from("<II", payload, 0)
        if 0 < books <= 100_000:
            offset = 8
            excerpts: list[bytes] = []
            valid = True
            for _ in range(books):
                if offset + 4 > len(payload):
                    valid = False
                    break
                length = struct.unpack_from("<I", payload, offset)[0]
                offset += 4
                if offset + length > len(payload):
                    valid = False
                    break
                excerpts.append(payload[offset:offset + length])
                offset += length
            if valid and offset == len(payload):
                return excerpts, int(skip), "nanro", payload
    return [payload], 0, "raw", payload


def load_vocab_json(path: str | Path, *,
                    max_vocab: int = DEFAULT_MAX_VOCAB) -> dict[bytes, int]:
    """Load a word vocabulary from JSON.

    Accepted shapes:
    * ``{"word": id, ...}``
    * ``{"vocab": {...}}`` / ``{"word_vocab": {...}}`` / ``{"words": [...]}``
    * ``["word", ...]`` (ids 1..N)
    * ``[{"word": "word", "id": 7}, ...]``
    * a word string of the form ``"hex:00ff"`` is decoded as raw bytes,
      which is useful for a byte vocabulary that is not valid UTF-8
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        for key in ("vocab", "word_vocab", "words"):
            candidate = data.get(key)
            if isinstance(candidate, (dict, list)):
                data = candidate
                break
    entries: list[tuple[Any, Any]]
    if isinstance(data, dict):
        entries = list(data.items())
    elif isinstance(data, list):
        if all(isinstance(item, str) for item in data):
            entries = [(item, index + 1) for index, item in enumerate(data)]
        elif all(isinstance(item, dict) for item in data):
            entries = []
            for item in data:
                word = item.get("word", item.get("token"))
                word_id = item.get("id", item.get("index"))
                if word is None or word_id is None:
                    raise ValueError("word entries need word/token and id/index")
                entries.append((word, word_id))
        else:
            raise ValueError("JSON list must contain strings or word objects")
    else:
        raise ValueError("vocabulary JSON must be an object or a list")

    vocabulary: dict[bytes, int] = {}
    seen_ids: set[int] = set()
    for raw_word, raw_id in entries:
        if isinstance(raw_word, str) and raw_word.startswith("hex:"):
            try:
                word = bytes.fromhex(raw_word[4:])
            except ValueError as exc:
                raise ValueError(f"invalid hex word {raw_word!r}") from exc
        else:
            word = _as_bytes(raw_word)
        if not word:
            continue
        if isinstance(raw_id, bool):
            raise ValueError("boolean word ids are not allowed")
        word_id = int(raw_id)
        if word_id <= UNK_ID:
            continue
        if word_id >= max_vocab:
            raise ValueError(f"word id {word_id} >= max_vocab {max_vocab}")
        if word_id in seen_ids:
            raise ValueError(f"duplicate word id {word_id}")
        if word in vocabulary:
            raise ValueError(f"duplicate word {word!r}")
        seen_ids.add(word_id)
        vocabulary[word] = word_id
    return vocabulary


def enumerate_boundaries(
        excerpts: Sequence[bytes], vocabulary: Mapping[bytes, int], *,
        min_context: int = MODEL_MIN_CONTEXT,
        per_book: int = 1,
        max_boundaries: int = 8
) -> tuple[list[Boundary], list[list[bytes]], list[list[int]]]:
    """Enumerate causal word-start boundaries.

    Returns ``(selected, words_by_book, ids_by_book)``.  ``words_by_book`` and
    ``ids_by_book`` cover every word in each excerpt, so a PPM baseline can
    reconstruct the prefix before any selected boundary.
    """
    if min_context < MODEL_MIN_CONTEXT:
        raise ValueError(f"min_context must be >= {MODEL_MIN_CONTEXT}")
    words_by_book: list[list[bytes]] = []
    ids_by_book: list[list[int]] = []
    per_book_boundaries: list[list[Boundary]] = []
    for book_index, raw in enumerate(excerpts):
        spans = word_spans(raw)
        words = [word for _start, _end, word in spans]
        ids = [int(vocabulary.get(word, UNK_ID)) for word in words]
        words_by_book.append(words)
        ids_by_book.append(ids)
        boundaries = [
            Boundary(book=book_index, word_index=index,
                     position=start, word=word, target_id=ids[index])
            for index, (start, _end, word) in enumerate(spans)
            if start >= min_context
        ]
        if per_book > 0:
            boundaries = _evenly_spaced(boundaries, per_book)
        per_book_boundaries.append(boundaries)
    selected = [boundary for group in per_book_boundaries
                for boundary in group]
    if max_boundaries > 0 and len(selected) > max_boundaries:
        selected = _evenly_spaced(selected, max_boundaries)
    return selected, words_by_book, ids_by_book


def build_ppm(word_ids: Sequence[int], *, max_order: int = 8
              ) -> list[dict[tuple[int, ...], Counter] | None]:
    """PPM count tables: ``counts[k][context][next_id] += 1``."""
    if max_order < 1:
        raise ValueError("max_order must be positive")
    counts: list[dict[tuple[int, ...], Counter] | None] = [
        None] + [defaultdict(Counter) for _ in range(max_order)]
    for index, word_id in enumerate(word_ids):
        for order in range(1, max_order + 1):
            if index >= order:
                key = tuple(int(value) for value in word_ids[index - order:index])
                counts[order][key][int(word_id)] += 1
    return counts


def ppm_ranking(prefix_ids: Sequence[int],
                counts: Sequence[dict[tuple[int, ...], Counter] | None],
                unigram_top: Sequence[int], *,
                max_order: int = 8, top_k: int = 100) -> list[int]:
    """Longest-match PPM ranking with unigram fallback (causal prefix only)."""
    index = len(prefix_ids)
    ranking: list[int] | None = None
    for order in range(min(max_order, index), 0, -1):
        row = counts[order].get(tuple(int(v) for v in
                                      prefix_ids[index - order:index]))
        if row and len(row) >= 2:
            ranking = [int(word) for word, _ in row.most_common(top_k)]
            break
    if ranking is None:
        ranking = [int(word) for word in unigram_top]
    seen = set(ranking)
    for word in unigram_top:
        word = int(word)
        if word not in seen:
            ranking.append(word)
            seen.add(word)
    return ranking[:max(0, int(top_k))]


def word_head_topk(model: Any, windows: Sequence[bytes], kmax: int, *,
                   device: Any | None = None,
                   batch_size: int = 16) -> list[list[int]] | None:
    """Next-word head top-k baseline at the same causal boundaries.

    Uses the raw ``unit_logits['word']`` head, matching the cached ``head``
    array in ``word_dev_cache.npz``.  Returns ``None`` if the model does not
    expose a word head.
    """
    import torch

    if not hasattr(model, "upgrades") and not hasattr(model, "unit_heads"):
        return None
    if device is None:
        device = _infer_torch_device(model, torch)
    groups: dict[int, list[int]] = defaultdict(list)
    for index, window in enumerate(windows):
        groups[len(window)].append(index)
    output: list[list[int]] = [[] for _ in windows]
    module = getattr(model, "upgrades", model)
    mode_name = getattr(getattr(model, "cfg", None), "name", None)
    with torch.inference_mode():
        for length, indices in sorted(groups.items()):
            for start in range(0, len(indices), max(1, int(batch_size))):
                chunk = indices[start:start + max(1, int(batch_size))]
                batch = torch.tensor(
                    [list(windows[index]) for index in chunk],
                    dtype=torch.long, device=device)
                # Match VerbatimV103RSI's Blackwell upgrade autocast for the
                # head baseline; on CPU this is an ordinary fp32 forward.
                autocast = (mode_name == "blackwell" and batch.is_cuda)
                context = (torch.autocast("cuda", dtype=torch.bfloat16)
                           if autocast else contextlib.nullcontext())
                with context:
                    result = module(batch)
                if not isinstance(result, dict):
                    return None
                unit_logits = result.get("unit_logits") or {}
                logits = unit_logits.get("word")
                if logits is None:
                    return None
                values, ids = logits.float().topk(
                    min(int(kmax), logits.shape[-1]), dim=-1)
                del values
                for row, index in enumerate(chunk):
                    output[index] = [int(item) for item in ids[row].tolist()]
    return output


def load_cached_baseline(path: str | Path, boundaries: Sequence[Boundary], *,
                         kmax: int = 100) -> dict[str, Any]:
    """Load the cached head/PPM rankings from ``word_dev_cache.npz``.

    Rows are matched by ``(book, pos)`` when available.  Targets are compared
    against the loaded vocabulary; a mismatch means the cache was built with a
    different vocabulary and the comparison is refused rather than reported as
    a misleading baseline.
    """
    import numpy as np

    result: dict[str, Any] = {
        "available": False,
        "reason": None,
        "matched": 0,
        "target_agreement": None,
        "head": None,
        "ppm": None,
    }
    data = np.load(Path(path), allow_pickle=False)
    keys = set(data.files)
    rows: list[int | None]
    if "book" in keys and "pos" in keys:
        lookup: dict[tuple[int, int], int] = {}
        for index, (book, position) in enumerate(zip(
                data["book"].tolist(), data["pos"].tolist())):
            lookup[(int(book), int(position))] = index
        rows = [lookup.get((boundary.book, boundary.position))
                for boundary in boundaries]
        matched = sum(row is not None for row in rows)
        result["matched"] = matched
        if matched < len(boundaries):
            result["reason"] = (
                f"only {matched}/{len(boundaries)} boundaries matched the "
                "cache (book,pos); unmatched rows are counted as misses")
    elif "target" in keys and len(data["target"]) == len(boundaries):
        if [int(value) for value in data["target"]] != [
                boundary.target_id for boundary in boundaries]:
            result["reason"] = (
                "cache target sequence differs from the scored boundaries "
                "(vocabulary or boundary mismatch)")
            return result
        rows = list(range(len(boundaries)))
        result["matched"] = len(boundaries)
    else:
        result["reason"] = (
            "cache has no (book,pos) columns and its target length does not "
            "match the scored boundaries")
        return result

    if "target" in keys:
        comparable = [(row, boundary) for row, boundary in zip(rows, boundaries)
                      if row is not None]
        if comparable:
            agreement = sum(
                int(data["target"][row]) == boundary.target_id
                for row, boundary in comparable) / len(comparable)
            result["target_agreement"] = agreement
            if agreement < 0.99:
                result["reason"] = (
                    "cache target IDs disagree with the loaded vocabulary "
                    f"(agreement={agreement:.3f})")
                return result

    def _rank_head(row: int) -> list[int]:
        values = np.asarray(data["head"][row], dtype=np.float32)
        return [int(item) for item in np.argsort(-values)[:kmax]]

    def _rank_ppm(row: int) -> list[int]:
        return [int(item) for item in data["ppm"][row].tolist()
                if int(item) >= 0][:kmax]

    if "head" in keys:
        head: list[list[int]] = []
        for row in rows:
            if row is None:
                head.append([])
                continue
            try:
                head.append(_rank_head(row))
            except (IndexError, ValueError):
                result["reason"] = "cache head rows are shorter than the vocab"
                return result
        result["head"] = head
    if "ppm" in keys:
        ppm: list[list[int]] = []
        for row in rows:
            ppm.append([] if row is None else _rank_ppm(row))
        result["ppm"] = ppm
    result["available"] = bool(
        result["matched"] > 0
        and (result["head"] is not None or result["ppm"] is not None))
    if result["matched"] == 0 and result["reason"] is None:
        result["reason"] = "no cache rows matched the scored boundaries"
    if not result["available"] and result["reason"] is None:
        result["reason"] = "cache contains neither head nor ppm"
    return result


def _manifest_check(path: str | Path | None, excerpts: Sequence[bytes],
                    skip: int, payload: bytes) -> tuple[dict | None, list[str]]:
    if path is None:
        return None, ["no --dev-manifest supplied; dev SHA was not checked"]
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    notes: list[str] = []
    expected = manifest.get("dev_sha256")
    if expected:
        actual = hashlib.sha256(payload).hexdigest()
        if actual != expected:
            raise ValueError(
                f"dev bin SHA-256 {actual} != manifest {expected}")
    if manifest.get("books") is not None and int(manifest["books"]) != len(excerpts):
        raise ValueError("manifest book count differs from dev bin")
    if manifest.get("skip") is not None and int(manifest["skip"]) != int(skip):
        notes.append(
            "dev-bin header skip differs from manifest skip; header value used")
    selected = manifest.get("selected_books") or []
    if selected and len(selected) != len(excerpts):
        raise ValueError("manifest selected_books length differs from dev bin")
    return manifest, notes


def _book_labels(manifest: dict | None, count: int) -> list[str]:
    labels = []
    selected = (manifest or {}).get("selected_books") or []
    for index in range(count):
        if index < len(selected):
            item = selected[index]
            labels.append(str(item.get("relative_path", f"book{index}")))
        else:
            labels.append(f"book{index}")
    return labels


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _load_model(checkpoint: str | Path, mode: str, device: Any | None
                ) -> tuple[Any, Any]:
    import torch
    try:
        from .verbatim_v103 import VerbatimV103RSI
        from .unified import device_for
    except ImportError:  # direct script execution
        from terminal_rsi.verbatim_v103 import VerbatimV103RSI
        from terminal_rsi.unified import device_for
    state = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model = VerbatimV103RSI(mode)
    model.load_state_dict(state, strict=True)
    target_device = torch.device(device) if device is not None else device_for(mode)
    model.to(target_device)
    model.eval()
    return model, target_device


def _build_report(args: argparse.Namespace, *, boundaries: Sequence[Boundary],
                  words_by_book: Sequence[Sequence[bytes]],
                  ids_by_book: Sequence[Sequence[int]],
                  trie: WordTrie, vocabulary: Mapping[bytes, int],
                  manifest: dict | None, dev_format: str, skip: int,
                  excerpts: Sequence[bytes], model: Any | None,
                  device: Any | None, notes: Sequence[str]) -> dict[str, Any]:
    targets = [boundary.target_id for boundary in boundaries]
    oov = sum(target == UNK_ID for target in targets)
    report: dict[str, Any] = {
        "schema": "terminal-rsi-word-lexicon-beam-v1",
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "dev_bin": str(args.dev_bin),
        "dev_manifest": str(args.dev_manifest) if args.dev_manifest else None,
        "vocab_json": str(args.vocab_json),
        "dev": {
            "format": dev_format,
            "books": len(excerpts),
            "skip": int(skip),
            "book_labels": _book_labels(manifest, len(excerpts)),
        },
        "vocab": {
            "size": len(vocabulary),
            "trie_words": len(trie),
            "trie_nodes": trie.node_count,
            "longest_word": trie.longest_word,
            "skipped": dict(trie.skipped),
        },
        "boundaries": {
            "units": len(boundaries),
            "in_vocab_units": len(boundaries) - oov,
            "oov_units": oov,
            "oov_rate": oov / len(boundaries) if boundaries else None,
            "min_context": args.min_context,
            "per_book": args.per_book,
            "max_boundaries": args.max_boundaries,
        },
        "notes": list(notes),
    }

    if model is None:
        report["mode"] = {"requested": args.exact, "used": "boundaries-only"}
        report["recall"] = None
        report["in_vocab_recall"] = None
        report["details"] = [
            {"book": boundary.book, "pos": boundary.position,
             "target": boundary.target_id,
             "target_word": boundary.word.decode("utf-8", "replace")}
            for boundary in boundaries
        ] if args.details else None
        return report

    kmax = max([int(args.topk)] + list(RECALL_KS))
    report["notes"].extend([
        "recall fields are fractions in [0,1]; unique_hits is a count",
        "the vocabulary JSON must be the exact training bytes -> id mapping; "
        "shifted ids misalign the word prior and the head baseline",
        "beam scores are raw summed byte log-probs; no EOS term and no "
        "length normalization (short words are favored)",
        "the lexicon beam cannot return OOV/UNK id 0; overall recall is "
        "bounded by in-vocabulary coverage",
        "exact prefix mode re-runs the model on candidate prefixes; target "
        "word bytes are never model inputs",
        "teacher mode reuses one boundary distribution and does not "
        "re-score candidate prefixes",
        "a finite beam prunes the first word byte at the root; words with a "
        "rare first byte are unreachable unless --beam is larger or 0",
        "v103 auxiliary features use model._default_aux (last eight observed "
        "bytes), not the six-feature training bridge",
    ])
    context_window = min(int(args.context_window),
                         int(getattr(getattr(model, "cfg", None),
                                     "context", DEFAULT_CONTEXT_WINDOW)))
    context_window = max(MODEL_MIN_CONTEXT, context_window)
    windows = [
        excerpts[boundary.book][max(0, boundary.position - context_window):
                                boundary.position]
        for boundary in boundaries]
    head_rankings = word_head_topk(
        model, windows, kmax, device=device, batch_size=args.head_batch)

    ppm_rankings: list[list[int]] | None = None
    if args.train_bin is not None:
        train_raw = Path(args.train_bin).read_bytes()
        train_words = [word for _start, _end, word in word_spans(train_raw)]
        train_ids = [int(vocabulary.get(word, UNK_ID)) for word in train_words]
        if args.ppm_max_train_words > 0:
            train_ids = train_ids[:int(args.ppm_max_train_words)]
        counts = build_ppm(train_ids, max_order=args.ppm_max_order)
        unigram_top = [int(word) for word, _ in
                       Counter(train_ids).most_common(kmax)]
        ppm_rankings = []
        for boundary in boundaries:
            prefix_ids = list(ids_by_book[boundary.book][:boundary.word_index])
            ppm_rankings.append(ppm_ranking(
                prefix_ids, counts, unigram_top,
                max_order=args.ppm_max_order, top_k=kmax))
        report["ppm_baseline"] = {
            "train_bin": str(args.train_bin),
            "train_words": len(train_ids),
            "max_order": args.ppm_max_order,
            "max_train_words": args.ppm_max_train_words,
        }
    else:
        report["ppm_baseline"] = None

    cache = None
    if args.baseline_cache is not None:
        cache = load_cached_baseline(
            args.baseline_cache, boundaries, kmax=kmax)
        report["cached_baseline"] = {
            "path": str(args.baseline_cache),
            "available": cache["available"],
            "reason": cache["reason"],
            "matched": cache["matched"],
            "target_agreement": cache["target_agreement"],
        }
    else:
        report["cached_baseline"] = None

    requested_mode = args.exact
    beam_mode = {"auto": "auto", "prefix": "prefix",
                 "teacher": "teacher"}[requested_mode]
    beam_width = None if int(args.beam) == 0 else int(args.beam)
    histories = [excerpts[boundary.book][:boundary.position]
                 for boundary in boundaries]
    progress = None
    if args.progress:
        def progress(info: dict[str, Any]) -> None:
            print("beam " + json.dumps(info, sort_keys=True), file=sys.stderr,
                  flush=True)
    started = time.perf_counter()
    results = lexicon_beam(
        model, histories, trie, mode=beam_mode, top_k=kmax,
        beam_width=beam_width, context_window=context_window,
        batch_size=args.batch_size, max_edges=args.max_edges,
        device=device, fallback_on_error=True, progress=progress)
    elapsed = time.perf_counter() - started
    beam_rankings = [[candidate.word_id for candidate in result.candidates]
                     for result in results]
    used_modes = sorted({result.mode for result in results})

    report["mode"] = {
        "requested": requested_mode,
        "used": used_modes,
        "beam_width": beam_width,
        "context_window": context_window,
        "top_k": kmax,
    }
    if requested_mode == "prefix" and getattr(device, "type", None) == "cpu":
        report["notes"].append(
            "exact prefix mode is running on CPU; each boundary can require "
            "hundreds to thousands of model forwards")
    report["recall"] = {
        "beam": _metric_block(beam_rankings, targets),
        "head": _metric_block(head_rankings, targets),
        "ppm": _metric_block(ppm_rankings, targets),
        "cached_head": _metric_block(
            cache["head"] if cache else None, targets),
        "cached_ppm": _metric_block(
            cache["ppm"] if cache else None, targets),
    }
    report["in_vocab_recall"] = {
        "beam": _metric_block(beam_rankings, targets, in_vocab_only=True),
        "head": _metric_block(head_rankings, targets, in_vocab_only=True),
        "ppm": _metric_block(ppm_rankings, targets, in_vocab_only=True),
        "cached_head": _metric_block(
            cache["head"] if cache else None, targets, in_vocab_only=True),
        "cached_ppm": _metric_block(
            cache["ppm"] if cache else None, targets, in_vocab_only=True),
    }
    report["cost"] = {
        "seconds_total": elapsed,
        "seconds_beam": sum(result.cost.get("seconds", 0.0) for result in results),
        "model_calls": sum(int(result.cost.get("model_calls", 0))
                           for result in results),
        "forward_rows": sum(int(result.cost.get("forward_rows", 0))
                            for result in results),
        "edges_scored": sum(int(result.cost.get("edges_scored", 0))
                            for result in results),
        "trie_edges_scored": sum(int(result.cost.get("trie_edges", 0))
                                 for result in results),
        "boundary_mass_error_max": (
            max(abs(float(result.cost["mass_error"]))
                for result in results if "mass_error" in result.cost)
            if any("mass_error" in result.cost for result in results)
            else None),
        "boundaries": len(boundaries),
        "truncated_boundaries": sum(result.truncated for result in results),
        "forward_rows_per_boundary": (
            sum(int(result.cost.get("forward_rows", 0)) for result in results)
            / len(boundaries) if boundaries else None),
    }
    report["fallbacks"] = [
        {"index": index, "reason": result.fallback_reason}
        for index, result in enumerate(results) if result.fallback_reason]
    if args.details:
        report["details"] = [
            {
                "book": boundary.book,
                "pos": boundary.position,
                "target": boundary.target_id,
                "target_word": boundary.word.decode("utf-8", "replace"),
                "beam": results[index].to_dict(max_candidates=kmax),
                "head": (head_rankings[index][:kmax]
                         if head_rankings is not None else None),
                "ppm": (ppm_rankings[index][:kmax]
                        if ppm_rankings is not None else None),
                "cached_head": (cache["head"][index][:kmax]
                                if cache and cache["head"] is not None else None),
                "cached_ppm": (cache["ppm"][index][:kmax]
                               if cache and cache["ppm"] is not None else None),
            }
            for index, boundary in enumerate(boundaries)
        ]
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Lexicon-constrained byte beam for Terminal-RSI words")
    parser.add_argument("--checkpoint", type=Path,
                        help="VerbatimV103RSI checkpoint (not needed with "
                             "--boundaries-only)")
    parser.add_argument("--dev-bin", type=Path, required=True,
                        help="NaNCM win_dev.bin or a raw byte excerpt")
    parser.add_argument("--dev-manifest", type=Path,
                        help="win_dev.manifest.json; its dev SHA is checked")
    parser.add_argument("--vocab-json", type=Path, required=True,
                        help="exact training word vocabulary: {word: id}, "
                             "list, or wrapper; hex:... keys accepted")
    parser.add_argument("--out", type=Path, required=True,
                        help="JSON report path")
    parser.add_argument("--mode", choices=("cpu", "t4", "blackwell"),
                        default="cpu", help="VerbatimV103RSI execution profile")
    parser.add_argument("--device", default=None,
                        help="Override torch device, e.g. cuda or cpu")
    parser.add_argument("--exact", choices=("auto", "prefix", "teacher"),
                        default="auto",
                        help="auto=prefix on CUDA/teacher on CPU; prefix="
                             "candidate-prefix re-scoring; teacher=one "
                             "boundary distribution")
    parser.add_argument("--beam", type=int, default=8,
                        help="prefix beam width; 0=exhaustive (costly), "
                             "negative is rejected")
    parser.add_argument("--topk", type=int, default=100,
                        help="candidate list length (recall@100 needs >=100)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="model batch rows for prefix scoring")
    parser.add_argument("--head-batch", type=int, default=16,
                        help="model batch rows for the word-head baseline")
    parser.add_argument("--context-window", type=int,
                        default=DEFAULT_CONTEXT_WINDOW,
                        help="observed+prefix context bytes (256 for v5)")
    parser.add_argument("--min-context", type=int,
                        default=DEFAULT_CONTEXT_WINDOW,
                        help="minimum observed bytes before a boundary")
    parser.add_argument("--max-word-bytes", type=int,
                        default=DEFAULT_MAX_WORD_BYTES,
                        help="skip words longer than this (head table cap)")
    parser.add_argument("--max-vocab", type=int, default=DEFAULT_MAX_VOCAB,
                        help="word id upper bound (4096 for the head)")
    parser.add_argument("--per-book", type=int, default=1,
                        help="evenly spaced boundaries per dev book"
                             " (0=all, then capped)")
    parser.add_argument("--max-boundaries", type=int, default=8,
                        help="total scored boundaries; keep this small")
    parser.add_argument("--max-edges", type=int, default=20000,
                        help="prefix-mode trie-edge budget per boundary; "
                             "0 disables expansion")
    parser.add_argument("--train-bin", type=Path,
                        help="optional training stream for an in-run PPM "
                             "baseline")
    parser.add_argument("--ppm-max-order", type=int, default=8)
    parser.add_argument("--ppm-max-train-words", type=int, default=0,
                        help="0=all training words; a positive value truncates"
                             " (documented baseline change)")
    parser.add_argument("--baseline-cache", type=Path,
                        help="optional word_dev_cache.npz with head/ppm")
    parser.add_argument("--boundaries-only", action="store_true",
                        help="enumerate/validate boundaries without a model")
    parser.add_argument("--details", action="store_true",
                        help="include per-boundary candidate lists")
    parser.add_argument("--progress", action="store_true",
                        help="print per-depth beam progress to stderr")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.min_context < MODEL_MIN_CONTEXT:
        parser.error(f"--min-context must be >= {MODEL_MIN_CONTEXT}")
    if args.context_window < MODEL_MIN_CONTEXT:
        parser.error(f"--context-window must be >= {MODEL_MIN_CONTEXT}")
    if args.beam < 0:
        parser.error("--beam must be non-negative (0=exhaustive)")
    if not args.boundaries_only and args.checkpoint is None:
        parser.error("--checkpoint is required unless --boundaries-only")

    vocabulary = load_vocab_json(args.vocab_json, max_vocab=args.max_vocab)
    trie = WordTrie(vocabulary, max_len=args.max_word_bytes,
                    max_vocab=args.max_vocab)
    excerpts, skip, dev_format, payload = read_dev_bin(args.dev_bin)
    manifest, notes = _manifest_check(
        args.dev_manifest, excerpts, skip, payload)
    boundaries, words_by_book, ids_by_book = enumerate_boundaries(
        excerpts, vocabulary, min_context=args.min_context,
        per_book=args.per_book, max_boundaries=args.max_boundaries)
    notes = list(notes)
    if dev_format == "raw":
        notes.append("dev bin header did not parse; treated the file as one "
                     "raw excerpt")
    if len(excerpts) > 1 and not any(
            boundary.book for boundary in boundaries):
        notes.append("all selected boundaries came from book 0; increase "
                     "--per-book or --max-boundaries for book coverage")

    model = None
    device = None
    if not args.boundaries_only:
        try:
            model, device = _load_model(args.checkpoint, args.mode, args.device)
        except ImportError as exc:
            print(f"torch/terminal_rsi is required for scoring: {exc}",
                  file=sys.stderr)
            return 2
        model.set_word_vocab(vocabulary)
        model.to(device)

    report = _build_report(
        args, boundaries=boundaries,
        words_by_book=words_by_book, ids_by_book=ids_by_book,
        trie=trie, vocabulary=vocabulary, manifest=manifest,
        dev_format=dev_format, skip=skip, excerpts=excerpts,
        model=model, device=device, notes=notes)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    summary = {
        "units": len(boundaries),
        "in_vocab": len(boundaries) - sum(
            boundary.target_id == UNK_ID for boundary in boundaries),
        "mode": report.get("mode"),
        "beam": (report.get("recall") or {}).get("beam"),
        "head": (report.get("recall") or {}).get("head"),
        "ppm": (report.get("recall") or {}).get("ppm"),
        "cost": report.get("cost"),
        "out": str(args.out),
    }
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
