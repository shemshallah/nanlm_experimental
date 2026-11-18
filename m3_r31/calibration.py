"""Three-scalar class-conditional correction over frozen M3 R29 and R30.

R29 is the accepted directed default. R30's complete trained 25-group state
remains available and untouched, but its learned correction is split into a
space-specific logit, a nonspace logit vector and a space intercept. Newborn
weights start at zero and therefore reproduce the R29 default *bitwise*.
"""
from __future__ import annotations

import numpy as np

from m3_default.predictor import R29_THETA
from m3_r29.lexical import lexical_readout
from m3_r30.router import M3R30Predictor, router_features, support

R30_DELTA = np.array((
    -0.0583874445684363, -0.3092941951187244, -0.257904760223574,
), dtype=np.float64)
CALIBRATION_WEIGHTS = 3
SPACE = np.zeros(256, dtype=np.float64)
SPACE[32] = 1.
NONSPACE = 1. - SPACE


def class_features(lexical: np.ndarray, evidence: float) -> np.ndarray:
    if lexical.shape != (8, 256) or not np.isfinite(lexical).all():
        raise ValueError("need eight finite inherited lexical channels")
    r30_correction = R30_DELTA @ router_features(lexical, evidence)
    return np.stack((r30_correction * NONSPACE,
                     r30_correction * SPACE, SPACE.copy()))


def calibrated_readout(base: np.ndarray, features: np.ndarray,
                       theta: np.ndarray) -> np.ndarray:
    if base.shape != (256,) or features.shape != (CALIBRATION_WEIGHTS, 256):
        raise ValueError("invalid R31 class-gated dimensions")
    if theta.shape != (CALIBRATION_WEIGHTS,) or not np.isfinite(theta).all():
        raise ValueError("invalid R31 calibration coefficients")
    if not np.any(theta):
        return base
    logits = base + theta @ features
    logits -= float(logits.max())
    return logits - np.log(np.exp(logits).sum())


class M3R31Predictor(M3R30Predictor):
    """One inherited R29 state, trained R30 specialist, and calibrated readout."""

    def __init__(self, prior: np.ndarray, theta: np.ndarray | None = None):
        self.calibration = (np.zeros(CALIBRATION_WEIGHTS, dtype=np.float64)
                            if theta is None else np.asarray(theta, dtype=np.float64).copy())
        if self.calibration.shape != (CALIBRATION_WEIGHTS,) or not np.isfinite(self.calibration).all():
            raise ValueError("three finite calibration gains required")
        super().__init__(prior, delta=R30_DELTA)

    def r29_base_and_calibration(self) -> tuple[np.ndarray, np.ndarray]:
        r26, lexical = self.base_and_lexical_features()
        r29 = lexical_readout(r26, lexical, R29_THETA)
        return r29, class_features(lexical, support(self.lexical))

    def predict_logp(self, mode: str | None = None) -> np.ndarray:
        if mode not in (None, "r31"):
            return super().predict_logp(mode)
        base, features = self.r29_base_and_calibration()
        return calibrated_readout(base, features, self.calibration)
