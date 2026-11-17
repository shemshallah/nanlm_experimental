"""Causal recurrent 256-head apex coupled to the inherited M3 R20 predictor.

This is an actual dynamical field: head amplitudes and velocities persist
through observations and interact via a 16x16 toroidal Laplacian. Observed
oriented triangle circulation injects feedback *after* the true byte arrives.
A small trainable residual readout and causal 4/8/16-byte suffix retrieval
share the SAME R21 predictor state. Zero readout is exact frozen R21.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from m3_default.predictor import M3DefaultPredictor

SIDE = 16
HEADS = 256
SUFFIX_LENGTHS = (4, 8, 16)
HORIZON = 8192
FEATURES = 5


class SuffixApexMemory:
    """Observed successors indexed by exact earlier byte prefixes, not future."""

    def __init__(self, horizon: int = HORIZON):
        if horizon < 1:
            raise ValueError("horizon must be positive")
        self.horizon = horizon
        self.context = deque(maxlen=max(SUFFIX_LENGTHS))
        self.counts = {n: {} for n in SUFFIX_LENGTHS}
        self.entries = {n: deque() for n in SUFFIX_LENGTHS}

    def observe(self, target: int) -> None:
        for n in SUFFIX_LENGTHS:
            if len(self.context) < n:
                continue
            key = bytes(tuple(self.context)[-n:])
            rows = self.counts[n]
            if key not in rows:
                rows[key] = np.zeros(HEADS, dtype=np.int32)
            rows[key][target] += 1
            queue = self.entries[n]
            queue.append((key, target))
            if len(queue) > self.horizon:
                old_key, old_target = queue.popleft()
                row = rows[old_key]
                row[old_target] -= 1
                if not np.any(row):
                    del rows[old_key]
        self.context.append(target)

    def residual(self, frozen_logp: np.ndarray, wrong_values: bool = False) -> np.ndarray:
        """Longest available earlier match, smoothed against frozen R21."""
        ctx = tuple(self.context)
        for n in reversed(SUFFIX_LENGTHS):
            if len(ctx) < n:
                continue
            counts = self.counts[n].get(bytes(ctx[-n:]))
            if counts is None:
                continue
            total = int(counts.sum())
            if total == 0:
                continue
            observed = np.roll(counts, 37) if wrong_values else counts
            frozen = np.exp(frozen_logp)
            predictive = (observed + 2.0 * frozen) / (total + 2.0)
            confidence = total / (total + 2.0)
            return confidence * (np.log(predictive) - frozen_logp)
        return np.zeros(HEADS, dtype=np.float64)


class RecurrentTriangleApex:
    """256-head Hamiltonian-like wave with bounded damping and triangle drive."""

    def __init__(self, geometry: str = "torus"):
        if geometry not in ("torus", "shuffled"):
            raise ValueError("geometry must be torus or shuffled")
        self.mapping = (np.arange(HEADS) if geometry == "torus"
                        else np.random.default_rng(20261117).permutation(HEADS))
        self.u = np.zeros(HEADS, dtype=np.float64)
        self.v = np.zeros(HEADS, dtype=np.float64)

    @staticmethod
    def _neighbor_mean(values: np.ndarray) -> np.ndarray:
        lattice = values.reshape(SIDE, SIDE)
        return (np.roll(lattice, 1, 0) + np.roll(lattice, -1, 0)
                + np.roll(lattice, 1, 1) + np.roll(lattice, -1, 1)).ravel() / 4

    def _area(self, p: int, a: int) -> np.ndarray:
        # The signed oriented area on the torus changes with head placement.
        pp, aa = int(self.mapping[p]), int(self.mapping[a])
        points = self.mapping
        dx = ((aa // SIDE - pp // SIDE + SIDE // 2) % SIDE) - SIDE // 2
        dy = ((aa % SIDE - pp % SIDE + SIDE // 2) % SIDE) - SIDE // 2
        bx = ((points // SIDE - aa // SIDE + SIDE // 2) % SIDE) - SIDE // 2
        by = ((points % SIDE - aa % SIDE + SIDE // 2) % SIDE) - SIDE // 2
        return np.tanh((dx * by - dy * bx) / 32.0)

    def observe(self, target: int, previous: int | None,
                prevprev: int | None, signed_pair_field: np.ndarray) -> None:
        # Integrate first; the newly revealed target drives only FUTURE states.
        near = self._neighbor_mean(self.u)
        self.v = .92 * self.v + .17 * (near - self.u)
        self.u = .995 * self.u + .23 * self.v
        idx = int(self.mapping[target])
        self.v[idx] += .35
        if previous is not None and prevprev is not None:
            p, a, b = int(prevprev), int(previous), int(target)
            circulation = (signed_pair_field[p, a] + signed_pair_field[a, b]
                           + signed_pair_field[b, p])
            area = float(self._area(p, a)[b])
            impulse = .16 * np.tanh(circulation) * area
            self.v[idx] += impulse
            self.v[int(self.mapping[a])] -= .5 * impulse
            self.v[int(self.mapping[p])] -= .5 * impulse

    def features(self, prevprev: int | None, previous: int | None) -> np.ndarray:
        points = self.mapping
        near = self._neighbor_mean(self.u)
        amplitude = np.tanh(self.u[points])
        velocity = np.tanh(self.v[points])
        contrast = np.tanh(near[points] - self.u[points])
        if prevprev is None or previous is None:
            curvature = np.zeros(HEADS)
        else:
            curvature = self._area(prevprev, previous) * velocity
        return np.stack((amplitude, velocity, contrast, curvature))


class M3R23Predictor(M3DefaultPredictor):
    """Single R21 state plus trainable 5-scalar apex/copy residual."""

    def __init__(self, prior: np.ndarray, geometry: str = "torus",
                 theta: np.ndarray | None = None):
        self.geometry = geometry
        self.theta = np.zeros(FEATURES) if theta is None else np.asarray(theta, dtype=np.float64).copy()
        if self.theta.shape != (FEATURES,):
            raise ValueError("R23 readout requires five weights")
        super().__init__(prior)

    def reset(self) -> None:
        super().reset()
        self.apex = RecurrentTriangleApex(self.geometry)
        self.suffix = SuffixApexMemory()

    def observe(self, observed_byte: int) -> None:
        prev, p = self.last, self.previous_previous
        super().observe(observed_byte)
        self.suffix.observe(int(observed_byte))
        self.apex.observe(int(observed_byte), prev, p,
                          self.waves["spatial_wave"].pair_u[1])

    def base_and_features(self, wrong_values: bool = False) -> tuple[np.ndarray, np.ndarray]:
        baseline = super().predict_logp("r20")
        copy = self.suffix.residual(baseline, wrong_values=wrong_values)
        geometry = self.apex.features(self.previous_previous, self.last)
        return baseline, np.concatenate((copy[None, :], geometry), axis=0)

    @staticmethod
    def readout(base: np.ndarray, features: np.ndarray, theta: np.ndarray) -> np.ndarray:
        if not np.any(theta):
            return base
        logits = base + theta @ features
        peak = float(logits.max())
        logits = logits - peak
        return logits - np.log(np.exp(logits).sum())

    def predict_logp(self, mode: str | None = None) -> np.ndarray:
        if mode is not None and mode != "r23":
            return super().predict_logp(mode)
        base, features = self.base_and_features()
        return self.readout(base, features, self.theta)


class AdamReadout:
    """Causal, restartable 5-parameter Adam optimizer; frozen parent model."""

    def __init__(self, theta: np.ndarray | None = None, lr: float = 0.002):
        self.theta = np.zeros(FEATURES) if theta is None else np.asarray(theta).copy()
        self.m = np.zeros(FEATURES)
        self.v = np.zeros(FEATURES)
        self.steps = 0
        self.lr = lr

    def update(self, gradient: np.ndarray) -> None:
        self.steps += 1
        grad = np.clip(gradient, -2.0, 2.0)
        self.m = .9 * self.m + .1 * grad
        self.v = .999 * self.v + .001 * grad * grad
        self.theta -= self.lr * (self.m / (1 - .9 ** self.steps)) / (
            np.sqrt(self.v / (1 - .999 ** self.steps)) + 1e-8)
