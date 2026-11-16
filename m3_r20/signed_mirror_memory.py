"""M3 R20: causal successor memory on 256 heads, 32640 signed mirror pairs.

Each off-diagonal unordered pair has two oriented S+/S- rails.
Sparse ordered triangles (previous-previous, previous, candidate) feed the
same next-byte distribution. State updates strictly after each prediction.
This is a candidate head/pair/triangle architecture, not an R12 recovery.
"""
from __future__ import annotations

from collections import deque
import numpy as np

HEADS = 256
MIRROR_PAIRS = HEADS * (HEADS - 1) // 2
HISTORY = 8192
PAIR_PSEUDOCOUNT = 8.0
TRIANGLE_PSEUDOCOUNT = 4.0


def pair_index(a: int, b: int) -> int:
    """Unique C(256,2) address for an unordered off-diagonal byte pair."""
    if not (0 <= a < HEADS and 0 <= b < HEADS) or a == b:
        raise ValueError("mirror-pair address requires distinct byte heads")
    lo, hi = (a, b) if a < b else (b, a)
    return lo * (2 * HEADS - lo - 1) // 2 + (hi - lo - 1)


class SignedMirrorSuccessorMemory:
    """One causal history, addressable in two directed rails and sparse triads."""

    def __init__(self, prior: np.ndarray, horizon: int = HISTORY):
        if prior.shape != (HEADS,) or np.any(prior <= 0) or horizon < 1:
            raise ValueError("invalid prior or retention horizon")
        self.prior = np.asarray(prior, dtype=np.float64) / prior.sum()
        self.horizon = horizon
        self.s_plus = np.zeros(MIRROR_PAIRS, dtype=np.int32)
        self.s_minus = np.zeros(MIRROR_PAIRS, dtype=np.int32)
        self.diagonal = np.zeros(HEADS, dtype=np.int32)
        self.ordered_triangles: dict[tuple[int, int], np.ndarray] = {}
        self.edges: deque[tuple[int, int]] = deque()
        self.triangles: deque[tuple[int, int, int]] = deque()
        self.pair_index_table = np.full((HEADS, HEADS), -1, dtype=np.int32)
        for a in range(HEADS):
            for b in range(a + 1, HEADS):
                index = pair_index(a, b)
                self.pair_index_table[a, b] = index
                self.pair_index_table[b, a] = index
        if int(self.pair_index_table.max()) != MIRROR_PAIRS - 1:
            raise AssertionError("signed-pair topology incomplete")

    def _edge_increment(self, a: int, b: int, direction: int) -> None:
        if a == b:
            self.diagonal[a] += direction
        else:
            index = int(self.pair_index_table[a, b])
            if a < b:
                self.s_plus[index] += direction
            else:
                self.s_minus[index] += direction

    def directed_row(self, previous: int) -> np.ndarray:
        candidates = np.arange(HEADS)
        indices = self.pair_index_table[previous]
        counts = np.empty(HEADS, dtype=np.float64)
        counts[candidates > previous] = self.s_plus[indices[candidates > previous]]
        counts[candidates < previous] = self.s_minus[indices[candidates < previous]]
        counts[previous] = self.diagonal[previous]
        return counts

    def predict(self, previous_previous: int | None,
                previous: int) -> tuple[np.ndarray, np.ndarray]:
        """Return pair and ordered-triangle distributions before observing target."""
        counts = self.directed_row(previous)
        pair = (counts + PAIR_PSEUDOCOUNT * self.prior) / (
            counts.sum() + PAIR_PSEUDOCOUNT
        )
        if previous_previous is None:
            return pair, pair.copy()
        triangle_counts = self.ordered_triangles.get((previous_previous, previous))
        if triangle_counts is None:
            return pair, pair.copy()
        tri = (triangle_counts + TRIANGLE_PSEUDOCOUNT * pair) / (
            triangle_counts.sum() + TRIANGLE_PSEUDOCOUNT
        )
        return pair, tri

    def observe(self, previous_previous: int | None, previous: int, target: int) -> None:
        """Causal feedback: only observed transitions enter future memory."""
        if not (0 <= previous < HEADS and 0 <= target < HEADS):
            raise ValueError("byte outside 256-head universe")
        self._edge_increment(previous, target, 1)
        self.edges.append((previous, target))
        if len(self.edges) > self.horizon:
            a, b = self.edges.popleft()
            self._edge_increment(a, b, -1)

        if previous_previous is not None:
            if not 0 <= previous_previous < HEADS:
                raise ValueError("invalid previous-previous head")
            key = (previous_previous, previous)
            if key not in self.ordered_triangles:
                self.ordered_triangles[key] = np.zeros(HEADS, dtype=np.int32)
            self.ordered_triangles[key][target] += 1
            self.triangles.append((previous_previous, previous, target))
            if len(self.triangles) > self.horizon:
                p, a, b = self.triangles.popleft()
                row = self.ordered_triangles[(p, a)]
                row[b] -= 1
                if not np.any(row):
                    del self.ordered_triangles[(p, a)]

    def invariants(self) -> dict:
        """Exact conservation for regression tests and replay diagnostics."""
        pair_total = int(self.s_plus.sum() + self.s_minus.sum() + self.diagonal.sum())
        triangle_total = sum(int(v.sum()) for v in self.ordered_triangles.values())
        return {
            "heads": HEADS, "mirror_addresses": MIRROR_PAIRS,
            "signed_rails": 2, "edge_mass": pair_total,
            "retained_edges": len(self.edges),
            "triangle_mass": triangle_total,
            "retained_triangles": len(self.triangles),
            "pair_nonnegative": bool(np.all(self.s_plus >= 0) and
                                    np.all(self.s_minus >= 0) and
                                    np.all(self.diagonal >= 0)),
        }


def causal_successor_deltas(book: bytes, prior: np.ndarray,
                            base_logp: np.ndarray,
                            warmup: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """Aligned to M3 R15 collect: score first, then inject observed transition."""
    memory = SignedMirrorSuccessorMemory(prior)
    pair_deltas, triangle_deltas = [], []
    start = warmup
    while start < len(book) and (book[start] & 0xC0) == 0x80:
        start += 1
    for pos in range(1, len(book)):
        previous, target = book[pos - 1], book[pos]
        context = book[pos - 2] if pos >= 2 else None
        if pos >= start:
            pair, triangle = memory.predict(context, previous)
            row = len(pair_deltas)
            pair_deltas.append(np.log(pair) - base_logp[row])
            triangle_deltas.append(np.log(triangle) - np.log(pair))
        memory.observe(context, previous, target)
    if len(pair_deltas) != len(base_logp):
        raise ValueError("M3 wave and signed-memory scored positions misaligned")
    return (
        np.asarray(pair_deltas, dtype=np.float64),
        np.asarray(triangle_deltas, dtype=np.float64),
        memory.invariants(),
    )
