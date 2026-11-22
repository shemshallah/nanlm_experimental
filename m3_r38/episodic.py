"""Causal, bounded, multiscale document-local successor memory for M3 R38.

Stores observed preceding-suffix -> observed next-byte counts. Every target is
inserted only after its forecast has been emitted and scored. No state leaks
between books. All stored keys/targets are genuine observed bytes.
"""
from __future__ import annotations
from collections import OrderedDict
import numpy as np

VOCAB = 256
ORDERS = (2, 4, 8, 16)
GROUPS = 3 * len(ORDERS)


class CausalSuccessors:
    def __init__(self, orders=ORDERS, capacity=32768, max_history=32):
        if not orders or tuple(sorted(set(orders))) != tuple(orders):
            raise ValueError('orders must be increasing and distinct')
        if any(not isinstance(n, int) or n < 1 for n in orders):
            raise ValueError('invalid suffix order')
        if capacity < len(orders) or max_history < max(orders):
            raise ValueError('insufficient capacity or history')
        self.orders = tuple(orders)
        self.capacity = int(capacity)
        self.max_history = int(max_history)
        self.reset()

    def reset(self):
        self.history = bytearray()
        self.tables = OrderedDict()  # (order, immutable suffix) -> [counts, total]
        self.observed = 0

    def observe(self, byte):
        if not isinstance(byte, (int, np.integer)) or not 0 <= int(byte) < 256:
            raise ValueError('expected one observed byte')
        b = int(byte)
        for order in self.orders:
            if len(self.history) < order:
                continue
            key = (order, bytes(self.history[-order:]))
            row = self.tables.get(key)
            if row is None:
                row = [np.zeros(VOCAB, dtype=np.int32), 0]
                self.tables[key] = row
                if len(self.tables) > self.capacity:
                    self.tables.popitem(last=False)
            else:
                self.tables.move_to_end(key)
            # Saturate counts, rather than permit signed integer overflow.
            if row[1] < 2**31 - 2:
                row[0][b] += 1
                row[1] += 1
        self.history.append(b)
        if len(self.history) > self.max_history:
            del self.history[:len(self.history) - self.max_history]
        self.observed += 1

    def features(self, parent_logp):
        """Return (12,256) finite features from *past* observations only.

        Three terms per order: residual evidence, across-scale disagreement,
        and entropy-conditioned residual evidence. The base expectation is
        subtracted for numerical conditioning; softmax is invariant to it.
        """
        logp = np.asarray(parent_logp, dtype=np.float64)
        if logp.shape != (VOCAB,) or not np.isfinite(logp).all():
            raise ValueError('expected a finite 256-way parent distribution')
        p = np.exp(logp - np.logaddexp.reduce(logp))
        logp = np.log(np.maximum(p, 1e-300))
        entropy = float(-p @ logp) / np.log(VOCAB)
        features = np.zeros((GROUPS, VOCAB), dtype=np.float64)
        previous_q = None
        for j, order in enumerate(self.orders):
            if len(self.history) < order:
                continue
            row = self.tables.get((order, bytes(self.history[-order:])))
            if row is None:
                continue
            counts, total = row
            if not total:
                continue
            # Explicitly smooth with the parent's causal probabilities.
            smoothing = 8.0 + 0.5 * order
            q = (counts.astype(np.float64) + smoothing * p) / (total + smoothing)
            reliability = total / (total + smoothing)
            # Limit the influence of extremely improbable isolated candidates.
            residual = np.clip(np.log(np.maximum(q, 1e-300)) - logp, -10.0, 10.0)
            scale = reliability * (1.0 - 0.5 * entropy)
            features[3*j] = scale * (residual - float(p @ residual))
            if previous_q is not None:
                delta = np.clip(np.log(np.maximum(q, 1e-300)) -
                                np.log(np.maximum(previous_q, 1e-300)), -6., 6.)
                features[3*j+1] = reliability * (delta - float(p @ delta))
            features[3*j+2] = reliability * entropy * (residual - float(p @ residual))
            previous_q = q
        if not np.isfinite(features).all():
            raise FloatingPointError('nonfinite suffix features')
        return features


def corrected_logp(parent_logp, features, theta):
    """Exact frozen-parent distribution at zero theta, normalized otherwise."""
    base = np.asarray(parent_logp, dtype=np.float64)
    f = np.asarray(features, dtype=np.float64)
    t = np.asarray(theta, dtype=np.float64)
    if base.shape != (256,) or f.shape != (GROUPS, 256) or t.shape != (GROUPS,):
        raise ValueError('R38 readout shape mismatch')
    if not all(np.isfinite(v).all() for v in (base, f, t)):
        raise ValueError('nonfinite readout')
    if not np.any(t) or not np.any(f):
        return base.copy()  # exact parent for migration regression
    z = base + t @ f
    return z - np.logaddexp.reduce(z)


def nll_grad(logp, features, target):
    if not isinstance(target, (int, np.integer)) or not 0 <= int(target) < 256:
        raise ValueError('invalid byte target')
    p = np.exp(logp)
    return features @ p - features[:, int(target)]


class M3R38Predictor:
    """Frozen parent plus causal learned episodic residual; no separate model.

    The parent must expose observe(), reset() and predict_logp(). Its weights
    are never changed here. A R37 parent can be installed after its checkpoint
    and source are independently recovered and hash verified.
    """
    def __init__(self, parent, theta=None, capacity=32768):
        self.parent = parent
        self.theta = (np.zeros(GROUPS, dtype=np.float64) if theta is None
                      else np.asarray(theta, dtype=np.float64).copy())
        if self.theta.shape != (GROUPS,) or not np.isfinite(self.theta).all():
            raise ValueError('invalid R38 parameters')
        self.memory = CausalSuccessors(capacity=capacity)

    def predict_parts(self):
        base = np.asarray(self.parent.predict_logp(), dtype=np.float64)
        features = self.memory.features(base)
        return base, features

    def predict_logp(self):
        return corrected_logp(*self.predict_parts(), self.theta)

    def predict_proba(self):
        return np.exp(self.predict_logp())

    def observe(self, target):
        self.parent.observe(target)
        self.memory.observe(target)

    def reset(self):
        self.parent.reset()
        self.memory.reset()