"""Causal R32 cross-document lexical anchor trained only on separate TRAIN books.

Unlike the R31 null diagnostic, the anchor contains no DEV or TEST document.
The anchor is immutable; prediction keys come exclusively from the current
document's *already observed* bytes. Zero gains preserve trained R31 exactly.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from m3_default.predictor import R26_THETA, R29_THETA, R30_DELTA, R31_THETA
from m3_r29.lexical import WordSuccessorMemory, lexical_readout, pack_lexical
from m3_r30.router import support
from m3_r31.calibration import M3R31Predictor, class_features, calibrated_readout

GAIN_COUNT = 4
SPACE_BIAS = np.zeros(256, dtype=np.float64)
SPACE_BIAS[32] = 1.


def make_anchor(books: list[bytes]) -> WordSuccessorMemory:
    if not books or any(not isinstance(book, bytes) or not book for book in books):
        raise ValueError("anchor requires nonempty separate TRAIN books")
    # Preserve every training transition in the anchor's long table. A bounded
    # near table is a separate recency channel, as in the inherited R29 memory.
    memory = WordSuccessorMemory(long_horizon=max(32768, sum(map(len, books))))
    for book in books:
        # Keep learned transition counts while resetting query syntax at each
        # document boundary. Do not synthesize cross-document word successors.
        memory.previous = memory.second_previous = memory.prefix = memory.separator = b""
        for b in book:
            memory.observe(b)
    return memory


def anchor_features(r26: np.ndarray, local: np.ndarray,
                    current: WordSuccessorMemory, anchor: WordSuccessorMemory,
                    wrong_values: bool = False) -> np.ndarray:
    """Read the frozen background at causal current-document query keys."""
    if r26.shape != (256,) or local.shape != (8,256):
        raise ValueError("R32 requires R26 probability and R29 lexical channels")
    global_lex = pack_lexical(anchor.features(r26, query=current, wrong_values=wrong_values))
    prefix = R29_THETA[3] * global_lex[3] + R29_THETA[7] * global_lex[7]
    word = R29_THETA @ global_lex - prefix
    discrepancy = R29_THETA @ (global_lex - local)
    confidence = support(anchor, query=current)
    return np.stack((prefix, word, confidence * discrepancy, SPACE_BIAS.copy()))


def anchored_readout(base: np.ndarray, features: np.ndarray,
                     theta: np.ndarray) -> np.ndarray:
    if base.shape != (256,) or features.shape != (GAIN_COUNT,256):
        raise ValueError("R32 anchored readout dimensions mismatch")
    if theta.shape != (GAIN_COUNT,) or not np.isfinite(theta).all():
        raise ValueError("four finite anchor gains required")
    if not np.any(theta):
        return base
    logits = base + theta @ features
    logits -= float(logits.max())
    return logits - np.log(np.exp(logits).sum())


class M3R32Predictor(M3R31Predictor):
    """One inherited causal M3 state and a read-only TRAIN-derived anchor."""

    def __init__(self, prior: np.ndarray, anchor: WordSuccessorMemory,
                 theta: np.ndarray | None = None):
        if not isinstance(anchor, WordSuccessorMemory) or not anchor.observed:
            raise ValueError("immutable prefit lexical anchor required")
        self.anchor = anchor
        self.anchor_theta = (np.zeros(GAIN_COUNT, dtype=np.float64)
                             if theta is None else np.asarray(theta,dtype=np.float64).copy())
        if self.anchor_theta.shape != (GAIN_COUNT,) or not np.isfinite(self.anchor_theta).all():
            raise ValueError("invalid anchor weights")
        super().__init__(prior, theta=R31_THETA)

    def r31_base_and_anchor_features(self, wrong_values: bool = False) -> tuple[np.ndarray,np.ndarray]:
        r26, lexical = self.base_and_lexical_features()
        r29 = lexical_readout(r26,lexical,R29_THETA)
        r31 = calibrated_readout(r29,class_features(lexical,support(self.lexical)),R31_THETA)
        return r31, anchor_features(r26,lexical,self.lexical,self.anchor,wrong_values)

    def predict_logp(self, mode: str | None = None) -> np.ndarray:
        if mode not in (None, "r32"):
            return super().predict_logp(mode)
        base, feat = self.r31_base_and_anchor_features()
        return anchored_readout(base,feat,self.anchor_theta)


def load_r32_checkpoint(path: Path) -> M3R32Predictor:
    """Restart inference from the self-contained 32-group NPZ, no corpus needed.

    The embedded TRAIN-only anchor is hash-checked and the whole recovered
    optimizer state remains in the same checkpoint for further training.
    """
    with np.load(path,allow_pickle=False) as x:
        theta=np.asarray(x["theta"],dtype=np.float64)
        moments=np.asarray(x["adam_m"],dtype=np.float64)
        variance=np.asarray(x["adam_v"],dtype=np.float64)
        ages=np.asarray(x["adam_steps"],dtype=np.int64)
        prior=np.asarray(x["prior"],dtype=np.float64)
        payload=np.asarray(x["background_bytes"],dtype=np.uint8)
        lengths=np.asarray(x["background_lengths"],dtype=np.int64)
        hashes=np.asarray(x["background_sha256"])
        background_indices=np.asarray(x["background_indices"],dtype=np.int64)
    if (theta.shape!=(32,) or moments.shape!=(32,) or variance.shape!=(32,)
            or ages.shape!=(32,) or prior.shape!=(256,)
            or not np.isfinite(theta).all() or not np.isfinite(moments).all()
            or not np.isfinite(variance).all() or np.any(variance<0)
            or np.any(prior<=0) or not np.isfinite(prior).all()
            or not np.array_equal(theta[:14],R26_THETA)
            or not np.array_equal(theta[14:22],R29_THETA)
            or not np.array_equal(theta[22:25],R30_DELTA)
            or not np.array_equal(theta[25:28],R31_THETA)
            or ages.tolist()!=[1536]*3+[768]*29
            or not np.array_equal(background_indices,[468,484])
            or lengths.shape!=(16,) or np.any(lengths<=0)
            or int(lengths.sum())!=len(payload) or hashes.shape!=(16,)):
        raise ValueError("invalid or incomplete 32-group R32 checkpoint")
    background=[]
    raw=payload.tobytes()
    offset=0
    for n,expected in zip(lengths,hashes):
        book=raw[offset:offset+int(n)]
        offset+=int(n)
        if hashlib.sha256(book).hexdigest()!=str(expected):
            raise ValueError("R32 embedded anchor hash mismatch")
        background.append(book)
    return M3R32Predictor(prior,make_anchor(background),theta=theta[-GAIN_COUNT:])
