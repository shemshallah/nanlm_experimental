#!/usr/bin/env python3
"""R7.38: bounded, causal reliability modulation of inherited triangle writes.

This is an in-place extension of the R7.35 far-history writer.  A zero gain
preserves the R7.37 computation bit-for-bit, including masked rows.  The new
path sees only the query and observed, pre-target far modal values; it cannot
look at the target token or move the apex/point readout around the decoder.
"""
from __future__ import annotations

import contextlib

import torch
from torch import nn
from torch.nn import functional as F

from fa1_r735_history_values import FarHistoryValueWriter

ROUND = 'FA1_R7_38_CAUSAL_RELIABILITY_V1'


class TriangleReliability(nn.Module):
    """Per-written-triangle quality estimate from causal values and coverage."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4, 16), nn.GELU(),
                                 nn.Linear(16, 1), nn.Tanh())
        self.gain = nn.Parameter(torch.zeros(()))

    def forward(self, keys, query, values, mask):
        batch, count, width = keys.shape
        if (query.shape != (batch, width) or
                values.shape != (batch, 4, width) or
                mask.shape != (batch, 4)):
            raise ValueError('triangle reliability requires causal [B,K,D] and [B,4,D]')
        valid = mask.bool().unsqueeze(-1).to(values.dtype)
        coverage = valid.sum(1).clamp_min(1.)
        centroid = (valid * values).sum(1) / coverage
        dispersion = ((values - centroid.unsqueeze(1)).square() * valid).sum((1, 2))
        dispersion = dispersion / (coverage.squeeze(-1) * width)
        key_match = (F.normalize(keys, dim=-1) *
                     F.normalize(centroid, dim=-1).unsqueeze(1)).sum(-1)
        query_match = (F.normalize(query, dim=-1) *
                       F.normalize(centroid, dim=-1)).sum(-1)
        features = torch.stack((key_match,
                                query_match.unsqueeze(1).expand(-1, count),
                                (coverage / 4.).expand(-1, count),
                                torch.tanh(dispersion).unsqueeze(1).expand(-1, count)), -1)
        quality = self.net(features)  # bounded in [-1,+1]
        return 1. + torch.tanh(self.gain) * quality


class ReliabilityHistoryValueWriter(FarHistoryValueWriter):
    """Uses the original writer weights, attention and sparse apex path."""

    def forward(self, selected_keys, context, far_values, far_mask):
        inherited = super().forward(selected_keys, context, far_values, far_mask)
        multiplier = self.reliability(selected_keys, context, far_values, far_mask)
        return inherited * multiplier


def attach_reliability(model, optimizer, learning_rate=1.5e-4):
    """Add one AdamW group; never replace a parent parameter or moment."""
    hierarchy = getattr(model.diamond, 'hierarchy', None)
    writer = getattr(model.diamond.geometry, 'history_writer', None)
    if hasattr(writer, 'reliability'):
        raise ValueError('R7.38 reliability already attached')
    if (type(writer) is not FarHistoryValueWriter or
            not hasattr(model.diamond.geometry, 'history_read_mix') or
            hierarchy is None or not hasattr(hierarchy, 'apex_reentry')):
        raise TypeError('R7.38 requires the entire R7.37 causal writer stack')
    inherited = {id(p) for p in model.parameters()}
    writer.__class__ = ReliabilityHistoryValueWriter
    writer.reliability = TriangleReliability()
    added = [(name, p) for name, p in model.named_parameters()
             if id(p) not in inherited]
    if not added or not all(name.startswith('diamond.geometry.history_writer.reliability.')
                            for name, _ in added):
        raise AssertionError('unexpected reliability parameters')
    optimizer._name_by_id.update({id(p): name for name, p in added})
    optimizer._group_names.append({'algorithm': 'adamw',
                                   'names': [name for name, _ in added]})
    optimizer._adamw.add_param_group({'params': [p for _, p in added],
                                     'lr': learning_rate, 'weight_decay': 0.})
    optimizer._source_group_counts[-1] += 1
    optimizer.param_groups = optimizer._legacy_groups + optimizer._adamw.param_groups
    actual = [id(p) for group in optimizer.param_groups for p in group['params']]
    if (len(actual) != len(set(actual)) or
            set(actual) != {id(p) for p in model.parameters()} or
            not inherited.issubset(actual)):
        raise AssertionError('R7.38 detached or duplicated inherited optimizer parameters')
    return model, optimizer


@contextlib.contextmanager
def reliability_disabled(model):
    """Ablate only the new quality modulation; restore even on exceptions."""
    writer = getattr(model.diamond.geometry, 'history_writer', None)
    if not hasattr(writer, 'reliability'):
        yield
        return
    gain = writer.reliability.gain
    saved = gain.detach().clone()
    with torch.no_grad():
        gain.zero_()
    try:
        yield
    finally:
        with torch.no_grad():
            gain.copy_(saved)
