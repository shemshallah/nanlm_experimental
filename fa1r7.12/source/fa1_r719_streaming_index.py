#!/usr/bin/env python3
"""FA1 R7.19: causal streaming 32K..128K weighted suffix index.

Experimental *read-only* accelerator for the frozen R7.18 1..4-gram far
reader.  This does not change accepted FA1 weights, adapter parameters,
optimizers, tokenization, or retrieval semantics.  Results are only valid
against the reference R7.18 implementation after numerical parity checks.

Streaming step: O(orders * log U) amortized, where U is the number of
unique next-token outcomes for the affected n-gram. Query: O(orders * log U)
(amortized heap cleanup), independent of the number of repeated matches.
Build: O(orders * horizon). Memory: O(orders * horizon), with heap compaction.
Random seeks are intentionally not promised to be fast: a large seek resets
and rebuilds.  Never index future tokens, other documents, or the near band.
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
import heapq
import math
import numpy as np

HORIZON = 131072
EXCLUSION = 32768
ORDERS = 4
FAR_LEN = HORIZON - EXCLUSION
ALPHA = 2.0 / (FAR_LEN - 1)


def _slog(w):
    return w * math.log(w) if w > 0.0 else 0.0


@dataclass
class _Value:
    weight: float
    at: int
    count: int
    version: int


class _Bucket:
    __slots__ = ('when', 'weight', 'weighted_log', 'events', 'values', 'heap', 'version')

    def __init__(self, when):
        self.when = when
        self.weight = 0.0
        self.weighted_log = 0.0
        self.events = deque()
        self.values = {}
        self.heap = []
        self.version = 0

    def _advance(self, now):
        dt = now - self.when
        if dt < 0:
            raise ValueError('nonmonotone index')
        if dt:
            factor = math.exp(-ALPHA * dt)
            old = self.weight
            self.weight *= factor
            self.weighted_log = factor * (self.weighted_log - ALPHA * dt * old)
            self.when = now

    @staticmethod
    def _current(value, now):
        return value.weight * math.exp(-ALPHA * (now - value.at))

    def modify(self, now, position, outcome, signed_weight, adding):
        self._advance(now)
        old = self.values.get(outcome)
        old_w = self._current(old, now) if old is not None else 0.0
        old_n = old.count if old is not None else 0
        new_n = old_n + (1 if adding else -1)
        if new_n < 0:
            raise AssertionError('unpaired index removal')
        new_w = max(0.0, old_w + signed_weight) if new_n else 0.0
        self.weight = max(0.0, self.weight + new_w - old_w)
        self.weighted_log += _slog(new_w) - _slog(old_w)
        self.version += 1
        if adding:
            self.events.append(position)
        else:
            if not self.events or self.events[0] != position:
                raise AssertionError('out-of-order or noncausal index removal')
            self.events.popleft()
        if new_n:
            self.values[outcome] = _Value(new_w, now, new_n, self.version)
            heapq.heappush(self.heap, (-(math.log(new_w) + ALPHA * now), outcome, self.version))
        elif old is not None:
            del self.values[outcome]
        # Bound stale heap entries even for high-churn n-grams.
        if len(self.heap) > max(64, 4 * len(self.values)):
            self.heap = [(-(math.log(self._current(v, now)) + ALPHA * now), k, v.version)
                         for k, v in self.values.items()]
            heapq.heapify(self.heap)

    def record(self, now, start, target):
        self._advance(now)
        if not self.events:
            return dict(n=0, weight_total=0.0, peak=0.0, entropy=0.0,
                        recent=0.0, modal=1, target_probability=0.0)
        while self.heap:
            _, value, version = self.heap[0]
            obj = self.values.get(value)
            if obj is not None and obj.version == version:
                break
            heapq.heappop(self.heap)
        assert self.heap and self.weight > 0
        modal = self.heap[0][1]
        top_w = self._current(self.values[modal], now)
        target_state = self.values.get(target)
        target_w = self._current(target_state, now) if target_state else 0.0
        ent = (math.log(self.weight) - self.weighted_log / self.weight) / math.log1p(len(self.values))
        return dict(n=len(self.events), weight_total=self.weight,
                    peak=top_w / self.weight, entropy=max(0.0, ent),
                    recent=(self.events[-1] - start) / FAR_LEN,
                    modal=modal, target_probability=target_w / self.weight)


class StreamingFarIndex:
    """Read-only ordered token sequence; `position` is next-token index.

    `document_start` prevents cross-document indexing. For concatenated books,
    create/reset one index per book and do not advance across a boundary.
    """
    def __init__(self, sequence, position, document_start=0):
        self.sequence = np.asarray(sequence, dtype=np.int64)
        self.document_start = int(document_start)
        self.buckets = [{} for _ in range(ORDERS)]
        self.position = int(position)
        if not (self.document_start <= self.position <= len(self.sequence)):
            raise ValueError('invalid causal position/document start')
        self._build()

    def _transition(self, order, p):
        if p - order + 1 < self.document_start or p + 1 >= len(self.sequence):
            return None
        key = tuple(map(int, self.sequence[p-order+1:p+1]))
        outcome = int(self.sequence[p+1])
        if outcome <= 1 or any(x <= 1 for x in key):
            return None
        return key, outcome

    def _weight(self, p, when):
        idx = p - (when - HORIZON)
        return math.exp(ALPHA * (idx - (FAR_LEN - 1)))

    def _apply(self, order, p, when, adding):
        pair = self._transition(order, p)
        if pair is None:
            return
        key, outcome = pair
        d = self.buckets[order-1]
        bucket = d.get(key)
        if bucket is None:
            if not adding:
                raise AssertionError('missing removed key')
            bucket = d[key] = _Bucket(when)
        weight = self._weight(p, when) * (1.0 if adding else -1.0)
        bucket.modify(when, p, outcome, weight, adding)
        if not bucket.events:
            del d[key]

    def _build(self):
        t = self.position
        lo = max(self.document_start, t - HORIZON)
        hi = min(t - EXCLUSION - 2, len(self.sequence) - 2)
        for p in range(lo, hi+1):
            for order in range(1, min(ORDERS, p-lo+1)+1):
                self._apply(order, p, t, True)

    def advance(self, position):
        new = int(position)
        if new < self.position or new > len(self.sequence):
            raise ValueError('index cannot advance backwards or beyond sequence')
        while self.position < new:
            t = self.position
            old_start = max(self.document_start, t-HORIZON)
            old_first = t-HORIZON
            old_last_candidate = t-EXCLUSION-2
            if old_first >= self.document_start:
                for order in range(1, ORDERS+1):
                    p = old_first + order - 1
                    if p <= old_last_candidate:
                        self._apply(order, p, t+1, False)
            incoming = t - EXCLUSION - 1
            if incoming >= self.document_start and incoming+1 < len(self.sequence):
                new_start = max(self.document_start, t+1-HORIZON)
                for order in range(1, min(ORDERS, incoming-new_start+1)+1):
                    self._apply(order, incoming, t+1, True)
            self.position += 1
        return self

    def query(self, context, target=None):
        """Return R7.18-compatible four records; target only used for loss.

        For inference pass target=None: no target label enters features/modal.
        The `target_probability` field must not be interpreted as a feature.
        """
        q = np.asarray(context, dtype=np.int64)[-ORDERS:]
        if len(q) != ORDERS:
            raise ValueError('expected at least 4 context words')
        start = self.position-HORIZON
        out = []
        for order in range(1, ORDERS+1):
            key = tuple(map(int, q[-order:]))
            b = self.buckets[order-1].get(key) if all(x>1 for x in key) else None
            out.append(b.record(self.position, start, target) if b is not None else
                       dict(n=0, weight_total=0.0, peak=0.0, entropy=0.0,
                            recent=0.0, modal=1, target_probability=0.0))
        return out

    def stats(self):
        return dict(position=self.position, tokens_indexed=sum(len(x.events) for d in self.buckets for x in d.values()),
                    keys=sum(map(len, self.buckets)), heap_entries=sum(len(x.heap) for d in self.buckets for x in d.values()))


def far_reference(sequence, position, context, target, document_start=0):
    """Reference helper reproducing Corpus.get left padding without exposing target."""
    from fa1_r718_far_residual import far_only
    start = max(document_start, position - HORIZON)
    h = np.asarray(sequence[start:position], dtype=np.int64)
    if len(h) < HORIZON:
        h = np.pad(h, (HORIZON-len(h),0), constant_values=1)
    return far_only(h, context, target)