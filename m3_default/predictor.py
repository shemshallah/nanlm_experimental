"""Canonical online M3 predictor: R32 with exact older-model fallbacks.

The original R21 signed pair/triangle model remains M3R21Predictor exactly.
R24 adds the previously accepted frozen prefix-copy readout. The R26 candidate
adds one causal multiscale successor memory and one hierarchical byte-context
tree to this SAME online state, with its exact independently tested readout.
R29 adds a frozen trained lexical successor feedback path. Its promotion is
user-directed, despite failing the unrelated-history DEV control, and is not
an independent TEST certification. The rejected R23 toroidal apex is NOT
enabled. R31 adds trained, class-selective lexical calibration. R32 adds an
immutable TRAIN-only lexical anchor trained on disjoint books, with all 32
optimizer groups retained in its recoverable checkpoint. R32 is selected by
user direction; an independent TEST result is tracked separately.
All state updates are causal.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

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

DEFAULT_MODE = "r32"
R20_PAIR_SCALE = 0.5
R20_TRIANGLE_SCALE = 0.5
LEGACY_MODES = ("r20", "r19", "oscillator", "wrong_address", "wrong_context", "triangle_only")
MODES = ("r32", "r31", "r30", "r29", "r26", "r24") + LEGACY_MODES
R24_COPY_GAIN = 1.0962571833587516  # exact R23 frozen TRAIN-only scalar
R26_THETA = np.array((
    0.20522936927790486, 0.5247820788932681, 0.014546217566135089,
    0.41110746025816125, 0.3068762355174945, 0.31118690110333225,
    -0.012420522830071154, -0.5031923913862479, -0.17807131974908555,
    -0.5457650152177792, -0.24970163692227434, -0.06057178164629776,
    0.12880799567382484, 0.09004538510784975,
), dtype=np.float64)
R26_TRAINING_CHECKPOINT_SHA256 = "deb112d4d9b25cb57bd68f689660550a315032bea42c5e9376dbcdeb4a5c97a9"
R29_THETA = np.array((
    -0.10714695493929974, -0.25978899415081763,
    -0.05510184952538218, 0.9506351396956768,
    -0.11474513231763826, -0.1366084670486249,
    -0.038336626963906786, 0.7541392402081305,
), dtype=np.float64)
R29_TRAINING_CHECKPOINT_SHA256 = "02c340e82ec77ae2c6e1d20ddd808ca4f63e3c242fe6fc94ff8f2235d8ab9386"
R30_DELTA = np.array((
    -0.0583874445684363, -0.3092941951187244, -0.257904760223574,
), dtype=np.float64)
R30_TRAINING_CHECKPOINT_SHA256 = "a757433f6b72b4b444763de2f3158eea960a530c94fe613af58de247647b3421"
R31_THETA = np.array((
    1.2195544888726837, 0.03536605045213109, -0.8597706306738135,
), dtype=np.float64)
R31_TRAINING_CHECKPOINT_SHA256 = "dd8fbaa93a17e5568bf2dc774b5599e5761824dee74db8f71242039f743a12ff"
R32_TRAINING_CHECKPOINT_SHA256 = "67e7417fb38c8d914714cee243fb080da6e63cf5f9e9c9737e28a5b168c1da1a"
R32_THETA = np.array((
    0.5927959113137101, -1.5, -0.20722426183204995,
    0.294518336989888,
), dtype=np.float64)
R32_CHECKPOINT = Path(__file__).with_name("r32_accepted_checkpoint.npz")


def load_accepted_r32_anchor():
    """Verify and recover the frozen trained R32 state and optimizer history."""
    from m3_r32.anchor import load_r32_checkpoint
    if hashlib.sha256(R32_CHECKPOINT.read_bytes()).hexdigest() != R32_TRAINING_CHECKPOINT_SHA256:
        raise ValueError("accepted R32 checkpoint SHA-256 differs from frozen candidate")
    trained = load_r32_checkpoint(R32_CHECKPOINT)
    if not np.array_equal(trained.anchor_theta, R32_THETA):
        raise ValueError("R32 frozen gains differ from checkpoint")
    return trained


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
    """One unified R32/R31/R30/R29/R26/R24/R21 predictor with fallbacks.

    The frozen R24 parent is always present. New R26 context memories are
    instantiated only for the promoted default mode, so R25/R26 research
    subclasses requesting mode=r24 preserve their exact historical behavior.
    """

    def __init__(self, original_train_prior: np.ndarray, mode: str = DEFAULT_MODE):
        if mode not in MODES:
            raise ValueError("unrecognized M3 mode")
        # M3R21Predictor.__init__ calls reset by dynamic dispatch. Set this
        # flag first so both initialization and later resets agree.
        self._r26_enabled = mode in ("r32", "r31", "r30", "r29", "r26")
        self._r29_enabled = mode in ("r32", "r31", "r30", "r29")
        super().__init__(original_train_prior,mode="r20")
        self.mode = mode
        if mode == "r32":
            trained = load_accepted_r32_anchor()
            self._r32_anchor = trained.anchor
            self._r32_theta = trained.anchor_theta.copy()

    def reset(self) -> None:
        super().reset()
        # Imported lazily to keep the historical R23 research predictor, which
        # imports M3R21Predictor, independent of the promoted default.
        from m3_r23.apex import SuffixApexMemory
        self.prefix_copy = SuffixApexMemory()
        if self._r26_enabled:
            from m3_r25.temporal import TemporalSuccessors
            from m3_r26.hybrid import ByteContextTree
            self.r26_temporal = TemporalSuccessors()
            self.r26_tree = ByteContextTree()
        if self._r29_enabled:
            from m3_r29.lexical import WordSuccessorMemory
            self.r29_lexical = WordSuccessorMemory()

    def observe(self, observed_byte: int) -> None:
        super().observe(observed_byte)
        self.prefix_copy.observe(int(observed_byte))
        if self._r26_enabled:
            self.r26_temporal.observe(int(observed_byte))
            self.r26_tree.observe(int(observed_byte))
        if self._r29_enabled:
            self.r29_lexical.observe(int(observed_byte))

    def predict_all(self) -> dict[str, np.ndarray]:
        predictions = super().predict_all()
        base = predictions["r20"]
        copy = self.prefix_copy.residual(base)
        logits = base + R24_COPY_GAIN * copy
        logits -= float(logits.max())
        predictions["r24"] = logits - np.log(np.exp(logits).sum())
        if self._r26_enabled:
            from m3_r26.hybrid import hybrid_readout, pack_features
            frozen_base = predictions["r24"]
            features = pack_features(self.r26_temporal.features(frozen_base),
                                     self.r26_tree.features(frozen_base))
            predictions["r26"] = hybrid_readout(frozen_base, features, R26_THETA)
        if self._r29_enabled:
            from m3_r29.lexical import lexical_readout, pack_lexical
            frozen_base = predictions["r26"]
            features = pack_lexical(self.r29_lexical.features(frozen_base))
            predictions["r29"] = lexical_readout(frozen_base, features, R29_THETA)
            if self.mode in ("r32", "r30", "r31"):
                from m3_r30.router import router_features, router_readout, support
                evidence = support(self.r29_lexical)
                predictions["r30"] = router_readout(
                    predictions["r29"], router_features(features, evidence), R30_DELTA)
            if self.mode in ("r32", "r31"):
                from m3_r31.calibration import class_features, calibrated_readout
                predictions["r31"] = calibrated_readout(
                    predictions["r29"], class_features(features, evidence), R31_THETA)
            if self.mode == "r32":
                from m3_r32.anchor import anchor_features, anchored_readout
                anchor_input = anchor_features(
                    predictions["r26"], features, self.r29_lexical, self._r32_anchor)
                predictions["r32"] = anchored_readout(
                    predictions["r31"], anchor_input, self._r32_theta)
        return predictions

    def predict_logp(self, mode: str | None = None) -> np.ndarray:
        chosen = self.mode if mode is None else mode
        if chosen not in MODES:
            raise ValueError("invalid M3 prediction mode")
        if chosen == "r26" and not self._r26_enabled:
            raise ValueError("R26 mode requires an R26-enabled predictor")
        if chosen == "r29" and not self._r29_enabled:
            raise ValueError("R29 mode requires an R29-enabled predictor")
        if chosen == "r30" and self.mode not in ("r32", "r30", "r31"):
            raise ValueError("R30 mode requires an R30-enabled predictor")
        if chosen == "r31" and self.mode not in ("r32", "r31"):
            raise ValueError("R31 mode requires an R31-enabled predictor")
        if chosen == "r32" and self.mode != "r32":
            raise ValueError("R32 mode requires a TRAIN-anchor-enabled predictor")
        return self.predict_all()[chosen]
