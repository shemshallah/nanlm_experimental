"""Trainable temporal-memory feedback layered *on* the accepted M3 R24.

R24's original wave, signed mirrors, triangles and frozen prefix-copy path are
unchanged. Three independently trainable residuals read the same observed
history at short, near and long timescales. Predict never updates any state.
Setting all three coefficients to zero returns R24 bit-for-bit.
"""
from __future__ import annotations

from collections import Counter, deque

import numpy as np

from m3_default.predictor import M3DefaultPredictor

LENGTHS = (2, 4, 8, 16, 32)
SHORT = (2,)
COPY_LENGTHS = (4, 8, 16, 32)
NEAR_HORIZON = 256
LONG_HORIZON = 32768
FEATURE_COUNT = 3


class TemporalSuccessors:
    """Sparse exact-prefix counts and independently expiring recent counts."""

    def __init__(self, near: int = NEAR_HORIZON, horizon: int = LONG_HORIZON):
        if not 0 < near <= horizon:
            raise ValueError("expected 0 < near <= horizon")
        self.near = near
        self.horizon = horizon
        self.context: deque[int] = deque(maxlen=max(LENGTHS))
        self.all_counts = {n: {} for n in LENGTHS}
        self.near_counts = {n: {} for n in LENGTHS}
        self.all_entries = {n: deque() for n in LENGTHS}
        self.near_entries = {n: deque() for n in LENGTHS}
        self.observations = 0

    @staticmethod
    def _add(table: dict, key: bytes, byte: int) -> None:
        table.setdefault(key, Counter())[byte] += 1

    @staticmethod
    def _drop(table: dict, key: bytes, byte: int) -> None:
        counter = table[key]
        counter[byte] -= 1
        if not counter[byte]:
            del counter[byte]
        if not counter:
            del table[key]

    def observe(self, byte: int) -> None:
        if not isinstance(byte, (int, np.integer)) or not 0 <= int(byte) <= 255:
            raise ValueError("byte outside 0..255")
        byte = int(byte)
        context = bytes(self.context)
        for n in LENGTHS:
            if len(context) < n:
                continue
            key = context[-n:]
            for table, entries, limit in ((self.all_counts[n], self.all_entries[n], self.horizon),
                                          (self.near_counts[n], self.near_entries[n], self.near)):
                self._add(table, key, byte)
                entries.append((key, byte))
                if len(entries) > limit:
                    old_key, old_byte = entries.popleft()
                    self._drop(table, old_key, old_byte)
        self.context.append(byte)
        self.observations += 1

    def _residual(self, base_logp: np.ndarray, lengths: tuple[int, ...],
                  near: bool, wrong_values: bool) -> np.ndarray:
        context = bytes(self.context)
        tables = self.near_counts if near else self.all_counts
        for n in reversed(lengths):
            if len(context) < n:
                continue
            counts = tables[n].get(context[-n:])
            if not counts:
                continue
            total = sum(counts.values())
            # Short contexts are frequent and noisier: require independent support.
            if n == 2 and total < 3:
                continue
            alpha = 8.0 if n == 2 else 2.0
            distribution = np.exp(base_logp)
            prediction = alpha * distribution
            for target, count in counts.items():
                prediction[(target + 37) % 256 if wrong_values else target] += count
            prediction /= total + alpha
            confidence = total / (total + alpha)
            return confidence * (np.log(prediction) - base_logp)
        return np.zeros(256, dtype=np.float64)

    def features(self, base_logp: np.ndarray, wrong_values: bool = False) -> np.ndarray:
        if base_logp.shape != (256,) or not np.isfinite(base_logp).all():
            raise ValueError("expected 256 finite base log probabilities")
        return np.stack((
            self._residual(base_logp, SHORT, False, wrong_values),
            self._residual(base_logp, COPY_LENGTHS, True, wrong_values),
            self._residual(base_logp, COPY_LENGTHS, False, wrong_values),
        ))


def readout(base: np.ndarray, features: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Normalized residual feedback; zero weights are an exact R24 identity."""
    if base.shape != (256,) or features.shape != (FEATURE_COUNT, 256):
        raise ValueError("wrong probability or feature shape")
    if weights.shape != (FEATURE_COUNT,) or not np.isfinite(weights).all():
        raise ValueError("wrong learned scalar shape")
    if not np.any(weights):
        return base
    logits = base + weights @ features
    logits -= float(np.max(logits))
    return logits - np.log(np.exp(logits).sum())


class M3R25Predictor(M3DefaultPredictor):
    """One inherited R24 state, one auxiliary temporal memory, no other model."""

    def __init__(self, prior: np.ndarray, weights: np.ndarray | None = None):
        self.weights = (np.zeros(FEATURE_COUNT, dtype=np.float64) if weights is None
                        else np.asarray(weights, dtype=np.float64).copy())
        if self.weights.shape != (FEATURE_COUNT,) or not np.isfinite(self.weights).all():
            raise ValueError("three finite R25 weights required")
        super().__init__(prior, mode="r24")

    def reset(self) -> None:
        super().reset()
        self.temporal = TemporalSuccessors()

    def observe(self, byte: int) -> None:
        super().observe(byte)
        self.temporal.observe(int(byte))

    def base_and_features(self, wrong_values: bool = False) -> tuple[np.ndarray, np.ndarray]:
        base = super().predict_logp("r24")
        return base, self.temporal.features(base, wrong_values=wrong_values)

    def predict_logp(self, mode: str | None = None) -> np.ndarray:
        if mode is not None and mode != "r25":
            return super().predict_logp(mode)
        base, features = self.base_and_features()
        return readout(base, features, self.weights)
