#!/usr/bin/env python3
"""Opt-in book-balanced causal sampling and independent successor-history view.

The R7.18/R7.21 historical sampler remains the default for all prior rounds.
"""
from __future__ import annotations

import numpy as np
import torch
from r7_core import Corpus


def balanced_causal_positions(corpus, n, seed):
    """Sample every eligible short/long book before repeating books.

    Requires at least one actual preceding word per sampled target. Index
    boundaries are in the original split's word-position coordinate system.
    """
    if corpus.doc_start is None:
        raise ValueError("R7.53 requires independently identified books")
    if n <= 0:
        raise ValueError("nonpositive sample size")
    starts = np.unique(np.asarray(corpus.doc_start, dtype=np.int64))
    ends = np.r_[starts[1:], len(corpus.seq)]
    books = [(int(lo + 1), int(hi - 1))
             for lo, hi in zip(starts, ends) if hi - lo >= 2]
    if not books:
        raise ValueError("no books have causal target positions")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(books))
    selected = []
    while len(selected) < n:
        chosen = order[:min(n - len(selected), len(order))]
        for i in chosen:
            lo, hi = books[int(i)]
            selected.append(int(rng.integers(lo, hi + 1)))
        order = rng.permutation(len(books))
    return np.asarray(selected, dtype=np.int64)


def with_successor_history(data, root, split, corpus_dir, context):
    """Attach a second history; never replace inherited R7.49 backoff inputs."""
    if context < 2 or context > 32768:
        raise ValueError("R7.53 context must be 2..32768 words")
    corpus = Corpus(root, split, corpus_dir=corpus_dir)
    positions = np.asarray(data["positions"], dtype=np.int64)
    if len(positions) != len(data["target"]):
        raise ValueError("history and supervised examples mismatch")
    hist, indices = corpus.get(positions, context=context)
    boundary = np.asarray(corpus.doc_start[positions], dtype=np.int64)
    mask = ((indices >= boundary[:, None]) & (indices < positions[:, None])
            & (hist != 1))
    out = dict(data)
    out["r753_successor_history_ids"] = torch.from_numpy(hist.copy()).long()
    out["r753_successor_history_mask"] = torch.from_numpy(mask.copy())
    return out
