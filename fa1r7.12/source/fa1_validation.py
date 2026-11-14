#!/usr/bin/env python3
"""Shared FA1 validation: literal-byte uncertainty and artifact integrity.

Each observation represents one target word, with a mean loss difference per
literal UTF-8 byte and a count of scored bytes.  Resample whole books and
recompute the ratio of total loss difference to total scored bytes.  This is
the same estimator reported by conditional spelling metrics, unlike a mean
of word-averaged losses (which overweights short spellings).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def book_byte_ratio_bootstrap(per_word_delta, byte_counts, book_ids, seed,
                              n_resamples=1024):
    """Cluster-bootstrap the pooled conditional NLL difference per byte.

    `per_word_delta` is the mean literal-byte NLL *difference* for each word;
    it may be negative.  EOW is not included in `byte_counts`.  Zero-byte
    targets contribute neither numerator nor denominator.  A book is one
    resampling unit, regardless of how many target positions it contributes.
    """
    if any(not isinstance(x, torch.Tensor) or x.ndim != 1
           for x in (per_word_delta, byte_counts, book_ids)):
        raise ValueError('byte bootstrap expects aligned 1D tensors')
    if (per_word_delta.shape != byte_counts.shape or
            byte_counts.shape != book_ids.shape):
        raise ValueError('byte bootstrap needs identical position shapes')
    if not isinstance(n_resamples, int) or n_resamples <= 0:
        raise ValueError('byte bootstrap needs a positive integer resample count')
    deltas = per_word_delta.detach().cpu().double().numpy()
    counts = byte_counts.detach().cpu().double().numpy()
    books = book_ids.detach().cpu().numpy()
    # Empty targets are excluded altogether, including their unused losses.
    # A NaN placeholder on an empty target must not change the estimand.
    if not np.all(np.isfinite(counts)):
        raise ValueError('byte bootstrap requires finite byte counts')
    if np.any(counts < 0) or not np.any(counts > 0):
        raise ValueError('byte bootstrap requires nonnegative, nonempty bytes')
    if np.any(counts != np.floor(counts)):
        raise ValueError('literal byte counts must be integers')
    if not np.all(np.isfinite(deltas[counts > 0])):
        raise ValueError('byte bootstrap requires finite scored observations')
    if not np.all(np.isfinite(books)):
        raise ValueError('byte bootstrap requires finite book identifiers')
    book_numerators = []
    book_denominators = []
    for book in np.unique(books[counts > 0]):
        selected = (books == book) & (counts > 0)
        book_numerators.append(float(np.sum(deltas[selected] * counts[selected])))
        book_denominators.append(float(np.sum(counts[selected])))
    numerators = np.asarray(book_numerators)
    denominators = np.asarray(book_denominators)
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(numerators),
                           size=(n_resamples, len(numerators)))
    ratios = numerators[sampled].sum(axis=1) / denominators[sampled].sum(axis=1)
    return {
        'books': len(numerators),
        'book_average': float(np.mean(numerators / denominators)),
        'book_pooled_estimate': float(numerators.sum() / denominators.sum()),
        'book_bootstrap_ci95': list(map(float, np.percentile(ratios, [2.5, 97.5]))),
        'resampling_unit': 'book',
        'estimand': 'pooled_literal_byte_nll_difference',
    }


def validate_artifact_paths(checkpoint, output, candidate_checkpoint=''):
    """Reject aliasing between the inherited checkpoint and all outputs.

    Both resolved paths and existing hard links are checked before any model
    loading or updates. Saving a candidate is additionally create-only in
    fa1_r727_checkpoint.save_unpromoted, which closes the overwrite race.
    """
    paths = {'source checkpoint': Path(checkpoint), 'report': Path(output)}
    if candidate_checkpoint:
        paths['candidate checkpoint'] = Path(candidate_checkpoint)
    entries = list(paths.items())
    for i, (left_name, left) in enumerate(entries):
        for right_name, right in entries[i+1:]:
            same = left.resolve() == right.resolve()
            if left.exists() and right.exists() and left.samefile(right):
                same = True
            if same:
                raise ValueError('FA1 protected paths alias: ' +
                                 left_name + ' and ' + right_name)
