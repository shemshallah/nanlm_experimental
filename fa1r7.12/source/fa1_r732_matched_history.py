#!/usr/bin/env python3
"""R7.32: causal, recent-context-matched geometry controls for the R7.31 model.

There are NO new model parameters or optimizer groups. A foreign-book donor
replaces only older Diamond evidence. Recent local evidence, inherited hidden
state, targets, selected vertices and the near-memory features stay fixed.
Comparing (geometry-on minus geometry-off) for the two histories isolates the
history-specific contribution of geometry from generic language prediction.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F

from fa1_r725_controls import different_book_donors


def matched_history_context(data, seed=732, recent_words=8, recent_bytes=32,
                            recent_sentences=2):
    """Replace older context by a *different-book* sample, retaining recent cues.

    Recent tokens/bytes are on the right. Only the old 32 far-memory channels
    and four far modal tokens are replaced; the 60 near channels and eight near
    modal tokens remain intact. All model targets and auxiliary byte labels
    remain unchanged. No gold target is ever consulted for donor selection.
    """
    donors = different_book_donors(data['book_ids'], seed)
    if len(donors) != len(data['target']):
        raise ValueError('book IDs and word targets differ in length')
    widths = {'word_ids': (32, recent_words),
              'byte_ids': (128, recent_bytes),
              'sentence_ids': (8, recent_sentences)}
    result = dict(data)
    for key, (width, keep) in widths.items():
        x = data[key]
        if x.ndim != 2 or x.shape != (len(donors), width):
            raise ValueError('wrong matched-history input shape: ' + key)
        if not 0 <= keep <= width:
            raise ValueError('invalid recent-context window: ' + key)
        copied = x.clone()
        if width > keep:
            copied[:, :width-keep] = x[donors.to(x.device), :width-keep]
        result[key] = copied
    # Recent vertices come from the same last local words as the unchanged
    # R7 hidden state; do NOT swap or reconstruct them from a foreign book.
    memory = data.get('diamond_memory_features')
    if memory is None:
        memory = torch.cat((data['near_features'], data['far_features']), dim=-1)
    if memory.shape != (len(donors), 92):
        raise ValueError('expected 60 near + 32 far memory features')
    result['diamond_memory_features'] = memory.clone()
    result['diamond_memory_features'][:, 60:] = memory[donors.to(memory.device), 60:]
    for output, fallback in (('diamond_modal_ids', 'memory_modal_ids'),
                             ('diamond_modal_mask', 'memory_modal_mask')):
        source = data.get(output, data.get(fallback))
        if source is None or source.shape != (len(donors), 12):
            raise ValueError('expected 8 near + 4 far modal fields: ' + output)
        result[output] = source.clone()
        result[output][:, 8:] = source[donors.to(source.device), 8:]
    return result, donors


def geometry_specific_history_gain(correct_on, foreign_on, correct_off, foreign_off):
    """Positive iff geometry benefits true older history more than foreign.

    Works for per-target branch NLL or per-word conditional byte NLL. The off
    tensors are fixed no-grad controls during training; gradients flow through
    BOTH live predictions. Shapes must align to prevent accidental broadcasts.
    """
    if any(x.shape != correct_on.shape for x in
           (foreign_on, correct_off, foreign_off)):
        raise ValueError('history comparisons require identical target positions')
    return (correct_off - correct_on) - (foreign_off - foreign_on)


def context_specific_margin(gain, margin=0.003):
    """Penalize absent context-specific gain with stable per-position gradients."""
    if gain.numel() == 0 or not 0 <= margin < 1:
        raise ValueError('invalid context-specific contrast configuration')
    return (F.softplus(8. * (margin-gain)) / 8.).mean()
