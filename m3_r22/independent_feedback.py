"""M3 R22: independent pair/triangle/antisymmetric-feedback candidate.

One causal successor history. The accepted R21 online R20 distribution is the
zero-residual fallback; all R22 coefficients are selected on disjoint TRAIN
documents, never the preopened R21 TEST or the fresh R22 DEV.
"""
from __future__ import annotations

from collections import deque
import numpy as np

from m3_r20.signed_mirror_memory import (
    SignedMirrorSuccessorMemory, PAIR_PSEUDOCOUNT, TRIANGLE_PSEUDOCOUNT,
    HEADS, HISTORY,
)


class ConventionalCountNgram:
    """Capacity/pseudocount-matched conventional bigram/trigram reference.

    Deliberately uses ordinary directed count tables and tuple keys, without
    mirror-address indexing, signed rails or geometric propagation.
    """

    def __init__(self, prior: np.ndarray, horizon: int = HISTORY):
        self.prior = np.asarray(prior, dtype=np.float64).copy()
        self.prior /= self.prior.sum()
        self.horizon = horizon
        self.bigrams = np.zeros((HEADS, HEADS), dtype=np.int32)
        self.trigrams: dict[tuple[int, int], np.ndarray] = {}
        self.edges: deque[tuple[int, int]] = deque()
        self.triples: deque[tuple[int, int, int]] = deque()

    def predict(self, prevprev: int | None, prev: int) -> tuple[np.ndarray, np.ndarray]:
        row = self.bigrams[prev].astype(np.float64)
        pair = (row + PAIR_PSEUDOCOUNT * self.prior) / (
            row.sum() + PAIR_PSEUDOCOUNT
        )
        counts = self.trigrams.get((prevprev, prev)) if prevprev is not None else None
        if counts is None:
            return pair, pair.copy()
        tri = (counts + TRIANGLE_PSEUDOCOUNT * pair) / (
            counts.sum() + TRIANGLE_PSEUDOCOUNT
        )
        return pair, tri

    def observe(self, prevprev: int | None, prev: int, target: int) -> None:
        self.bigrams[prev, target] += 1
        self.edges.append((prev, target))
        if len(self.edges) > self.horizon:
            a, b = self.edges.popleft()
            self.bigrams[a, b] -= 1
        if prevprev is not None:
            key = (prevprev, prev)
            if key not in self.trigrams:
                self.trigrams[key] = np.zeros(HEADS, dtype=np.int32)
            self.trigrams[key][target] += 1
            self.triples.append((prevprev, prev, target))
            if len(self.triples) > self.horizon:
                p, a, b = self.triples.popleft()
                counts = self.trigrams[(p, a)]
                counts[b] -= 1
                if not np.any(counts):
                    del self.trigrams[(p, a)]


def signed_flux(memory: SignedMirrorSuccessorMemory, previous: int) -> np.ndarray:
    """Antisymmetric directed-current flux at a 256-head prediction site.

    J(a,b)=(N(a,b)-N(b,a))/(N(a,b)+N(b,a)+1).
    Unlike pair/trigram probabilities, J explicitly compares both temporal rails.
    """
    indices = memory.pair_index_table[previous]
    candidates = np.arange(HEADS)
    reverse = np.empty(HEADS, dtype=np.float64)
    mask_up = candidates > previous
    mask_down = candidates < previous
    reverse[mask_up] = memory.s_minus[indices[mask_up]]
    reverse[mask_down] = memory.s_plus[indices[mask_down]]
    reverse[previous] = memory.diagonal[previous]
    forward = memory.directed_row(previous)
    flux = (forward - reverse) / (forward + reverse + 1.0)
    flux[previous] = 0.0
    return flux


def feedback_at_prefix(memory: SignedMirrorSuccessorMemory,
                       prevprev: int | None, prev: int,
                       base_logp: np.ndarray) -> dict:
    """Feature extraction before target observation; no memory mutation."""
    pair, triangle = memory.predict(prevprev, prev)
    n_pair = float(memory.directed_row(prev).sum())
    key = (prevprev, prev) if prevprev is not None else None
    tri_row = memory.ordered_triangles.get(key) if key is not None else None
    n_tri = float(tri_row.sum()) if tri_row is not None else 0.0
    return {
        "pair": pair,
        "triangle": triangle,
        "pair_residual": (np.log(pair) - base_logp) * n_pair / (n_pair + 16.0),
        "triangle_residual": (np.log(triangle) - np.log(pair)) * n_tri / (n_tri + 8.0),
        "flux": signed_flux(memory, prev),
        "pair_count": n_pair, "triangle_count": n_tri,
    }


def normalized_logp(logits: np.ndarray) -> np.ndarray:
    peak = np.max(logits, axis=-1, keepdims=True)
    centered = logits - peak
    return centered - np.log(np.exp(centered).sum(axis=-1, keepdims=True))


def candidate_logp(frozen_r20: np.ndarray, pair_residual: np.ndarray,
                   triangle_residual: np.ndarray, flux: np.ndarray,
                   scales: tuple[float, float, float]) -> np.ndarray:
    """Selected independent feedback; scales=(0,0,0) is EXACT frozen R20."""
    if not any(scales):
        return frozen_r20
    return normalized_logp(frozen_r20 + scales[0] * pair_residual
                           + scales[1] * triangle_residual + scales[2] * flux)
