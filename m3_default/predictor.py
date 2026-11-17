"""Canonical online M3 predictor: independently certified R24 causal copy.

The original R21 signed pair/triangle model remains M3R21Predictor exactly.
R24 adds only the frozen TRAIN/GATE-selected prefix-copy readout, verified on
24 previously untouched books. The failed recurrent geometric apex is NOT
enabled. Every state update occurs after the target byte is observed.
"""
from __future__ import annotations

import numpy as np

from m3_r13.mirror_wave import M3MirrorWave
from m3_r14.predictive_probe import ExactWindowMemory
from m3_r15.learned_wave_readout import ARM_CONFIGS, STRIDE, wave_features
from m3_r17.locked_test import frozen_predictions
from m3_r19.class_conditional_wave import residual_scores, scaled_predictions
from m3_r20.signed_mirror_memory import SignedMirrorSuccessorMemory
from m3_r20.successor_dev import (
    combined_logp, R19_SPACE_SCALE, R19_NONSPACE_SCALE, SHIFT,
)

DEFAULT_MODE = "r24"
R20_PAIR_SCALE = 0.5
R20_TRIANGLE_SCALE = 0.5
LEGACY_MODES = ("r20", "r19", "oscillator", "wrong_address", "wrong_context", "triangle_only")
MODES = ("r24",) + LEGACY_MODES
R24_COPY_GAIN = 1.0962571833587516  # exact R23 frozen TRAIN-only scalar


class M3R21Predictor:
    """Frozen R21 reference: exactly the earlier signed-pair/triangle predictor."""

    def __init__(self, original_train_prior: np.ndarray, mode: str = "r20"):
        if mode not in LEGACY_MODES:
            raise ValueError("unrecognized M3 mode")
        if original_train_prior.shape != (256,) or np.any(original_train_prior <= 0):
            raise ValueError("expected strictly positive original TRAIN prior")
        self.prior = np.asarray(original_train_prior, dtype=np.float64).copy()
        self.prior /= self.prior.sum()
        self.mode = mode
        self.reset()

    def reset(self) -> None:
        """Reset document-local causal memory; no weights or configuration lost."""
        self.memory = ExactWindowMemory(self.prior)
        self.signed = SignedMirrorSuccessorMemory(self.prior)
        self.waves = {
            name: M3MirrorWave(ARM_CONFIGS[name])
            for name in ("uncoupled", "spatial_wave")
        }
        self.last: int | None = None
        self.previous_previous: int | None = None
        self.seen_bytes = 0

    def observe(self, observed_byte: int) -> None:
        """Only the observed byte can enter memory, after any prior prediction."""
        if not isinstance(observed_byte, (int, np.integer)) or not 0 <= observed_byte < 256:
            raise ValueError("observed byte must be an integer between 0 and 255")
        b = int(observed_byte)
        if self.last is not None:
            a = self.last
            p = self.previous_previous
            pos = self.seen_bytes  # matches R15 collect's 1-based transition position
            self.memory.observe(a, b)
            self.signed.observe(p, a, b)
            for wave in self.waves.values():
                wave.inject_transition(a, b, impulse=.2)
                if pos % STRIDE == 0:
                    wave.step()
            self.previous_previous = a
        self.last = b
        self.seen_bytes += 1

    def predict_all(self) -> dict[str, np.ndarray]:
        """Read-only 256-way normalized log probabilities; never changes state."""
        if self.last is None:
            raise ValueError("observe at least one byte before predicting")
        previous = self.last
        base_logp = np.log(self.memory.predict(previous))[None, :]
        data = {
            "base_logp": base_logp,
            "features": {
                name: wave_features(self.waves[name], previous).astype(np.float32)[None, ...]
                for name in self.waves
            },
        }
        oscillator, _ = frozen_predictions(data)
        r19 = scaled_predictions(oscillator, residual_scores(data),
                                 R19_SPACE_SCALE, R19_NONSPACE_SCALE)
        pair, triangle = self.signed.predict(self.previous_previous, previous)
        pair_delta = (np.log(pair) - base_logp[0])[None, :]
        tri_delta = (np.log(triangle) - np.log(pair))[None, :]
        r20 = combined_logp(r19, pair_delta, tri_delta,
                            R20_PAIR_SCALE, R20_TRIANGLE_SCALE)
        wrong_pair = np.roll(pair_delta, SHIFT, axis=1)
        wrong_triangle = np.roll(tri_delta, SHIFT, axis=1)
        return {
            "r20": r20[0],
            "r19": r19[0],
            "oscillator": oscillator[0],
            "wrong_address": combined_logp(
                r19, wrong_pair, wrong_triangle,
                R20_PAIR_SCALE, R20_TRIANGLE_SCALE
            )[0],
            "wrong_context": combined_logp(
                r19, pair_delta, wrong_triangle,
                R20_PAIR_SCALE, R20_TRIANGLE_SCALE
            )[0],
            "triangle_only": combined_logp(r19, pair_delta, tri_delta, 0.0, 1.0)[0],
        }

    def predict_logp(self, mode: str | None = None) -> np.ndarray:
        chosen = self.mode if mode is None else mode
        if chosen not in LEGACY_MODES:
            raise ValueError("invalid M3 prediction mode")
        return self.predict_all()[chosen]

    def predict_proba(self, mode: str | None = None) -> np.ndarray:
        return np.exp(self.predict_logp(mode))


class M3DefaultPredictor(M3R21Predictor):
    """Accepted R24 default with inherited R21 available as exact ablation.

    The two paths use the same byte history, oscillator waves, pair memory and
    triangle memory. R24 adds ONE suffix memory and ONE frozen scalar readout.
    """

    def __init__(self, original_train_prior: np.ndarray, mode: str = DEFAULT_MODE):
        if mode not in MODES:
            raise ValueError("unrecognized M3 mode")
        super().__init__(original_train_prior,mode="r20")
        self.mode = mode

    def reset(self) -> None:
        super().reset()
        # Imported lazily to keep the historical R23 research predictor, which
        # imports M3R21Predictor, independent of the promoted default.
        from m3_r23.apex import SuffixApexMemory
        self.prefix_copy = SuffixApexMemory()

    def observe(self, observed_byte: int) -> None:
        super().observe(observed_byte)
        self.prefix_copy.observe(int(observed_byte))

    def predict_all(self) -> dict[str, np.ndarray]:
        predictions = super().predict_all()
        base = predictions["r20"]
        copy = self.prefix_copy.residual(base)
        logits = base + R24_COPY_GAIN * copy
        logits -= float(logits.max())
        predictions["r24"] = logits - np.log(np.exp(logits).sum())
        return predictions

    def predict_logp(self, mode: str | None = None) -> np.ndarray:
        chosen = self.mode if mode is None else mode
        if chosen not in MODES:
            raise ValueError("invalid M3 prediction mode")
        return self.predict_all()[chosen]
