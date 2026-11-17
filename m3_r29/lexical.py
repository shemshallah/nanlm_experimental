"""Causal word-conditioned byte successors on top of frozen M3 R26.

The lexical memory is book-local, bounded by transition count, and read-only
at prediction time. It indexes *observed* completed words, the current partial
word, and the separator between them; it never uses a target in its own score.
Eight zero-initialized readout coefficients preserve R26 bit for bit.
"""
from __future__ import annotations

from collections import Counter, deque

import numpy as np

from m3_default.predictor import M3DefaultPredictor, R26_THETA
from m3_r26.hybrid import M3R26Predictor

LEXICAL_CHANNELS = 4
NEW_WEIGHTS = 2 * LEXICAL_CHANNELS
TOTAL_WEIGHTS = len(R26_THETA) + NEW_WEIGHTS
LEXICAL_SMOOTHING = (8., 10., 6., 16.)
NONSPACE_MASK = np.ones(256, dtype=np.float64)
NONSPACE_MASK[32] = 0.


def _letter(byte: int) -> bool:
    return 65 <= byte <= 90 or 97 <= byte <= 122


class _BoundedCounts:
    def __init__(self, horizon: int):
        if horizon <= 0:
            raise ValueError("positive horizon required")
        self.horizon = horizon
        self.rows: dict[tuple[bytes, ...], Counter] = {}
        self.entries: deque[tuple[tuple[bytes, ...], int]] = deque()

    def observe(self, key: tuple[bytes, ...] | None, target: int) -> None:
        if key is None:
            return
        self.rows.setdefault(key, Counter())[target] += 1
        self.entries.append((key, target))
        if len(self.entries) > self.horizon:
            old_key, old_target = self.entries.popleft()
            row = self.rows[old_key]
            row[old_target] -= 1
            if not row[old_target]:
                del row[old_target]
            if not row:
                del self.rows[old_key]


class WordSuccessorMemory:
    """Previous-word + current-prefix counts with controlled recency.

    Word identity is ASCII case-folded only in the lookup key; the *target*
    remains the original exact byte. Non-ASCII bytes are separators, avoiding
    accidental splitting of a UTF-8 codepoint into fabricated ASCII tokens.
    """

    def __init__(self, near_horizon: int = 1024, long_horizon: int = 32768):
        if not 0 < near_horizon <= long_horizon:
            raise ValueError("invalid lexical memory horizons")
        self.previous: bytes = b""
        self.second_previous: bytes = b""
        self.prefix: bytes = b""
        self.separator: bytes = b""
        self.tables = (
            _BoundedCounts(long_horizon),  # previous word + current partial word
            _BoundedCounts(long_horizon),  # previous two words + current partial word
            _BoundedCounts(near_horizon),  # recent previous word + partial word
            _BoundedCounts(long_horizon),  # current partial word independent of predecessor
        )
        self.observed = 0

    def keys(self) -> tuple[tuple[bytes, ...] | None, ...]:
        # Gap distinguishes ordinary word joins from punctuation and paragraphs.
        # Use only the first three separator bytes, not the last three, so
        # arbitrary-length whitespace cannot masquerade as a normal word join.
        gap = self.separator
        word = ((self.previous, gap, self.prefix)
                if len(self.previous) >= 2 else None)
        pair = ((self.second_previous, self.previous, gap, self.prefix)
                if len(self.second_previous) >= 2 and word is not None else None)
        spelling = ((self.prefix,) if len(self.prefix) >= 2 else None)
        return word, pair, word, spelling

    def observe(self, byte: int) -> None:
        if not isinstance(byte, (int, np.integer)) or not 0 <= int(byte) < 256:
            raise ValueError("byte outside 0..255")
        byte = int(byte)
        for table, key in zip(self.tables, self.keys()):
            table.observe(key, byte)
        if _letter(byte):
            self.prefix = (self.prefix + bytes((byte | 32,)))[:20]
        else:
            if self.prefix:
                self.second_previous = self.previous
                self.previous = self.prefix
                self.prefix = b""
                self.separator = bytes((byte,))
            else:
                self.separator = (self.separator + bytes((byte,)))[:3]
        self.observed += 1

    def features(self, base: np.ndarray, *, wrong_values: bool = False,
                 query: "WordSuccessorMemory | None" = None) -> np.ndarray:
        if base.shape != (256,) or not np.isfinite(base).all():
            raise ValueError("256 finite log probabilities required")
        keys = self.keys() if query is None else query.keys()
        prior = np.exp(base)
        output = np.zeros((LEXICAL_CHANNELS, 256), dtype=np.float64)
        for i, (table, key, alpha) in enumerate(zip(self.tables, keys, LEXICAL_SMOOTHING)):
            row = table.rows.get(key) if key is not None else None
            if not row:
                continue
            total = sum(row.values())
            prediction = alpha * prior
            for target, count in row.items():
                prediction[(target + 37) % 256 if wrong_values else target] += count
            prediction /= total + alpha
            confidence = total / (total + alpha)
            output[i] = confidence * (np.log(prediction) - base)
        return output


def pack_lexical(features: np.ndarray) -> np.ndarray:
    if features.shape != (LEXICAL_CHANNELS, 256) or not np.isfinite(features).all():
        raise ValueError("expected four finite lexical distributions")
    return np.concatenate((features, features * NONSPACE_MASK[None, :]))


def lexical_readout(base: np.ndarray, features: np.ndarray, theta: np.ndarray) -> np.ndarray:
    if base.shape != (256,) or features.shape != (NEW_WEIGHTS, 256) or theta.shape != (NEW_WEIGHTS,):
        raise ValueError("incorrect R29 readout dimensions")
    if not np.isfinite(base).all() or not np.isfinite(theta).all():
        raise ValueError("nonfinite R29 log probabilities or weights")
    if not np.any(theta):
        return base  # exact frozen R26 migration parity
    logits = base + theta @ features
    logits -= float(logits.max())
    return logits - np.log(np.exp(logits).sum())


class M3R29Predictor(M3R26Predictor):
    """One inherited R26 state plus a zero-initialized lexical specialist."""

    def __init__(self, prior: np.ndarray, theta: np.ndarray | None = None):
        self.lexical_theta = (np.zeros(NEW_WEIGHTS, dtype=np.float64)
                              if theta is None else np.asarray(theta, dtype=np.float64).copy())
        if self.lexical_theta.shape != (NEW_WEIGHTS,) or not np.isfinite(self.lexical_theta).all():
            raise ValueError("expected eight finite new coefficients")
        super().__init__(prior, theta=R26_THETA)

    def reset(self) -> None:
        super().reset()
        self.lexical = WordSuccessorMemory()

    def observe(self, byte: int) -> None:
        super().observe(byte)
        self.lexical.observe(byte)

    def base_and_lexical_features(self) -> tuple[np.ndarray, np.ndarray]:
        base = super().predict_logp("r26")
        return base, pack_lexical(self.lexical.features(base))

    def predict_logp(self, mode: str | None = None) -> np.ndarray:
        if mode is not None and mode != "r29":
            if mode not in ("r26", "r25", "r24"):
                return M3DefaultPredictor.predict_logp(self, mode)
            return super().predict_logp(mode)
        base, features = self.base_and_lexical_features()
        return lexical_readout(base, features, self.lexical_theta)
