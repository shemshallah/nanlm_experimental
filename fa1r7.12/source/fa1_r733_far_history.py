#!/usr/bin/env python3
"""R7.33 multi-horizon curriculum on the inherited R7.32 matched control.

No duplicate counterfactual implementation: delegate to R7.32's rigorously
tested causal donor swap, varying only the protected *recent suffix* length.
The target, inherited hidden state, near memory and vertices stay fixed.
"""
from __future__ import annotations

from fa1_r732_matched_history import matched_history_context

# Progressively preserve more recent evidence. Every configuration still
# swaps the 32 older/far memory channels and the four far modal tokens.
HORIZONS = ((8, 32, 2), (16, 64, 4), (24, 96, 6))

def matched_far_history(data, seed=733, horizon=0):
    """Use R7.32's exact causal intervention at one of three recency scales."""
    if not isinstance(horizon,int) or not 0 <= horizon < len(HORIZONS):
        raise ValueError('unknown R7.33 history horizon')
    words, bytes_, sentences = HORIZONS[horizon]
    return matched_history_context(data, seed, recent_words=words,
                                   recent_bytes=bytes_, recent_sentences=sentences)
