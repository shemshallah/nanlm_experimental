"""Adaptive hierarchical byte-context feedback on the SAME M3 R24/R25 state.

The R25 trained three-channel head is inherited unchanged. An independent
causal context tree learns byte continuation statistics for orders 1/2/3/4/6/8.
Its four residuals and class-specific non-space corrections expand the readout
to fourteen scalars. Zero new weights reproduces R25 exactly. Zero all weights
reproduces accepted R24, without disturbing the waves, mirrors, or triangles.
"""
from __future__ import annotations

from collections import Counter, deque

import numpy as np

from m3_r25.temporal import M3R25Predictor, TemporalSuccessors, readout as r25_readout

ORDERS = (1, 2, 3, 4, 6, 8)
SMOOTHING = {1: 12., 2: 6., 3: 3., 4: 2., 6: 1.5, 8: 1.}
CHANNELS = 4
WEIGHTS = 14  # inherited 3 + context tree 4 + nonspace correction 3+4
NONSPACE = np.ones(256, dtype=np.float64)
NONSPACE[32] = 0.


class ByteContextTree:
    """Book-local sparse byte n-grams with separate recent and lifetime histories."""

    def __init__(self, near_horizon: int = 512, long_horizon: int = 32768):
        if not 0 < near_horizon <= long_horizon:
            raise ValueError("invalid context-tree horizons")
        self.context = deque(maxlen=max(ORDERS))
        self.horizons = {"all": long_horizon, "near": near_horizon}
        self.counts = {kind: {n: {} for n in ORDERS} for kind in self.horizons}
        self.entries = {kind: {n: deque() for n in ORDERS} for kind in self.horizons}
        self.observed = 0

    def observe(self, byte: int) -> None:
        if not isinstance(byte, (int, np.integer)) or not 0 <= byte <= 255:
            raise ValueError("byte outside 0..255")
        byte = int(byte)
        ctx = bytes(self.context)
        for n in ORDERS:
            if len(ctx) < n:
                continue
            key = ctx[-n:]
            for kind, limit in self.horizons.items():
                table = self.counts[kind][n]
                rows = table.setdefault(key, Counter())
                rows[byte] += 1
                queue = self.entries[kind][n]
                queue.append((key, byte))
                if len(queue) > limit:
                    old_key, old_byte = queue.popleft()
                    row = table[old_key]
                    row[old_byte] -= 1
                    if not row[old_byte]:
                        del row[old_byte]
                    if not row:
                        del table[old_key]
        self.context.append(byte)
        self.observed += 1

    def _levels(self, base: np.ndarray, kind: str, wrong_values: bool) -> dict[int, np.ndarray]:
        q = np.exp(base)
        ctx = bytes(self.context)
        outputs: dict[int, np.ndarray] = {}
        for n in ORDERS:
            if len(ctx) >= n:
                row = self.counts[kind][n].get(ctx[-n:])
                if row:
                    alpha = SMOOTHING[n]
                    counts = alpha * q
                    total = 0
                    for target, value in row.items():
                        counts[(target + 37) % 256 if wrong_values else target] += value
                        total += value
                    q = counts / (total + alpha)
            outputs[n] = q
        return outputs

    def features(self, base: np.ndarray, wrong_values: bool = False) -> np.ndarray:
        if base.shape != (256,) or not np.isfinite(base).all():
            raise ValueError("256 finite base logits required")
        all_levels = self._levels(base, "all", wrong_values)
        near_levels = self._levels(base, "near", wrong_values)
        # Nested PPM-style residuals: short syntax, medium motifs, long exact
        # continuation, and recent-vs-lifetime distributional drift.
        short = np.log(all_levels[3]) - base
        medium = np.log(all_levels[6]) - np.log(all_levels[3])
        long = np.log(all_levels[8]) - np.log(all_levels[6])
        drift = np.log(near_levels[4]) - np.log(all_levels[4])
        return np.stack((short, medium, long, drift))


def pack_features(r25_features: np.ndarray, tree_features: np.ndarray) -> np.ndarray:
    if r25_features.shape != (3, 256) or tree_features.shape != (CHANNELS, 256):
        raise ValueError("expected three R25 and four context-tree features")
    joint = np.concatenate((r25_features, tree_features))
    return np.concatenate((joint, joint * NONSPACE[None, :]))


def hybrid_readout(base: np.ndarray, features: np.ndarray, weights: np.ndarray) -> np.ndarray:
    if base.shape != (256,) or features.shape != (WEIGHTS, 256) or weights.shape != (WEIGHTS,):
        raise ValueError("invalid R26 distribution / feedback dimensions")
    if not np.isfinite(weights).all():
        raise ValueError("nonfinite R26 weights")
    if not np.any(weights[3:]):
        # Exact R25 parity during optimizer/checkpoint migration.
        return r25_readout(base, features[:3], weights[:3])
    logits = base + weights @ features
    logits -= float(np.max(logits))
    return logits - np.log(np.exp(logits).sum())


class M3R26Predictor(M3R25Predictor):
    def __init__(self, prior: np.ndarray, theta: np.ndarray):
        theta = np.asarray(theta, dtype=np.float64)
        if theta.shape != (WEIGHTS,) or not np.isfinite(theta).all():
            raise ValueError("R26 requires fourteen finite weights")
        self.theta = theta.copy()
        super().__init__(prior, weights=theta[:3])

    def reset(self) -> None:
        super().reset()
        self.tree = ByteContextTree()

    def observe(self, byte: int) -> None:
        super().observe(byte)
        self.tree.observe(int(byte))

    def base_and_hybrid_features(self, wrong_values: bool = False) -> tuple[np.ndarray, np.ndarray]:
        base, r25 = super().base_and_features(wrong_values=wrong_values)
        return base, pack_features(r25, self.tree.features(base, wrong_values=wrong_values))

    def predict_logp(self, mode: str | None = None) -> np.ndarray:
        if mode in ("r24", "r25"):
            return super().predict_logp(mode)
        if mode not in (None, "r26"):
            raise ValueError("unrecognized M3 R26 mode")
        base, features = self.base_and_hybrid_features()
        return hybrid_readout(base, features, self.theta)
