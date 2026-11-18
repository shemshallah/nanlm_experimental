"""Three-parameter selective R29 lexical feedback with exact migration parity.

The frozen R26 and R29 readouts are *not* retrained. R30 adds three small
corrective logit channels: prefix spelling, word-conditioned context, and
support-weighted word context. Zero weights reproduce R29 bit for bit.
"""
from __future__ import annotations

import numpy as np

from m3_default.predictor import M3DefaultPredictor, R29_THETA
from m3_r29.lexical import M3R29Predictor, WordSuccessorMemory, pack_lexical, lexical_readout

ROUTER_FEATURES = 3


def support(memory: WordSuccessorMemory, query: WordSuccessorMemory | None = None) -> float:
    """Causal local count support; never uses the pending target byte."""
    key = (memory if query is None else query).keys()[0]
    if key is None:
        return 0.
    row = memory.tables[0].rows.get(key)
    if not row:
        return 0.
    return min(1., np.log1p(sum(row.values())) / np.log1p(8.))


def router_features(lexical: np.ndarray, evidence: float) -> np.ndarray:
    if lexical.shape != (8, 256) or not np.isfinite(lexical).all():
        raise ValueError("expected eight causal R29 lexical features")
    if not np.isfinite(evidence) or evidence < 0 or evidence > 1:
        raise ValueError("invalid support confidence")
    prefix = R29_THETA[3] * lexical[3] + R29_THETA[7] * lexical[7]
    context = R29_THETA @ lexical - prefix
    return np.stack((prefix, context, evidence * context))


def router_readout(base: np.ndarray, features: np.ndarray, delta: np.ndarray) -> np.ndarray:
    if base.shape != (256,) or features.shape != (ROUTER_FEATURES, 256):
        raise ValueError("R30 router feature dimensions mismatch")
    if delta.shape != (ROUTER_FEATURES,) or not np.isfinite(delta).all():
        raise ValueError("three finite router gains required")
    if not np.any(delta):
        return base  # exact R29 default distribution
    logits = base + delta @ features
    logits -= float(np.max(logits))
    return logits - np.log(np.exp(logits).sum())


class M3R30Predictor(M3R29Predictor):
    """The *same* R29 state, with only three experimental trainable gains."""

    def __init__(self, prior: np.ndarray, delta: np.ndarray | None = None):
        self.delta = (np.zeros(ROUTER_FEATURES, dtype=np.float64)
                      if delta is None else np.asarray(delta, dtype=np.float64).copy())
        if self.delta.shape != (ROUTER_FEATURES,) or not np.isfinite(self.delta).all():
            raise ValueError("invalid R30 coefficients")
        super().__init__(prior, theta=R29_THETA)

    def r29_base_and_router_features(self) -> tuple[np.ndarray, np.ndarray]:
        r26, lexical = self.base_and_lexical_features()
        r29 = lexical_readout(r26, lexical, R29_THETA)
        return r29, router_features(lexical, support(self.lexical))

    def predict_logp(self, mode: str | None = None) -> np.ndarray:
        if mode is not None and mode != "r30":
            if mode == "r29":
                return M3R29Predictor.predict_logp(self, "r29")
            return M3R29Predictor.predict_logp(self, mode)
        base, features = self.r29_base_and_router_features()
        return router_readout(base, features, self.delta)
