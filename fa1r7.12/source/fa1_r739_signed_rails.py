#!/usr/bin/env python3
"""R7.39: signed S+/S- causal cross-rail interaction on sparse triangle writes.

The existing word/byte trunk, triangle addresses, far-value writer, apex
feedback, and optimizer moments are retained.  The new path can only modulate
values already written by the causal far-history writer; it cannot invent an
older observation or bypass the inherited apex -> pair -> point descent.
"""
from __future__ import annotations

import contextlib

import torch
from torch import nn

from fa1_r730_factorized_geometry import FactorizedContextualGeometry
from fa1_r738_reliability import ReliabilityHistoryValueWriter

ROUND = 'FA1_R7_39_SIGNED_CROSS_RAIL_V1'


def antisymmetric_cross(positive, negative):
    """Orientation-sensitive quarter-turn interaction; swaps change sign."""
    if positive.shape != negative.shape or positive.ndim != 3:
        raise ValueError('cross rails require matching [B,K,D] tensors')
    return (positive * torch.roll(negative, 1, -1) -
            negative * torch.roll(positive, 1, -1))


class SignedRailCross(nn.Module):
    """Two directional responses to the SAME observed causal far history."""

    def __init__(self, width=32):
        super().__init__()
        self.plus = nn.Linear(width, width, bias=False)
        self.minus = nn.Linear(width, width, bias=False)
        self.project = nn.Linear(width, width, bias=False)
        self.gain = nn.Parameter(torch.zeros(()))

    def forward(self, values, mask, plus_codes, minus_codes):
        if (values.ndim != 3 or values.shape[1] != 4 or
                mask.shape != values.shape[:2] or
                plus_codes.shape != minus_codes.shape or
                plus_codes.shape[0] != values.shape[0]):
            raise ValueError('signed cross expects [B,4,D] causal values, masks, and rails')
        valid = mask.bool().unsqueeze(-1).to(values.dtype)
        observed = (values * valid).sum(1) / valid.sum(1).clamp_min(1.)
        positive = (torch.tanh(plus_codes).unsqueeze(-1) *
                    torch.tanh(self.plus(observed)).unsqueeze(1))
        negative = (torch.tanh(minus_codes).unsqueeze(-1) *
                    torch.tanh(self.minus(observed)).unsqueeze(1))
        mixing = torch.tanh(self.project(antisymmetric_cross(positive, negative)))
        return torch.tanh(self.gain) * mixing


class CrossRailGeometry(FactorizedContextualGeometry):
    """Modulates only already-written triangle values at their exact addresses."""

    def triangles_up(self, vertex_ids, tri_ids, tri, state):
        tri, state = super().triangles_up(vertex_ids, tri_ids, tri, state)
        if 'history_triangle_values' not in state:
            return tri, state
        indices = state['history_triangle_indices']
        plus = self.face_plus(tri_ids).squeeze(-1).gather(1, indices)
        minus = self.face_minus(tri_ids).squeeze(-1).gather(1, indices)
        full = state['history_triangle_values']
        selector = indices.unsqueeze(-1).expand(-1, -1, full.size(-1))
        selected = full.gather(1, selector)
        modulation = self.rail_cross(state['far_modal_values'],
                                     state['far_modal_mask'], plus, minus)
        delta = torch.zeros_like(full).scatter(1, selector, selected * modulation)
        state['history_triangle_values'] = full + delta
        state['metrics']['r739_cross_rail_gain'] = self.rail_cross.gain.detach().clone()
        return tri, state


def attach_signed_cross_rail(model, optimizer, learning_rate=1.5e-4):
    """In-place one-group migration with inherited parameters/moments intact."""
    geometry = model.diamond.geometry
    if hasattr(geometry, 'rail_cross'):
        raise ValueError('R7.39 cross-rail interaction already attached')
    if (type(geometry) is not FactorizedContextualGeometry or
            not isinstance(getattr(geometry, 'history_writer', None),
                           ReliabilityHistoryValueWriter) or
            not hasattr(geometry, 'history_read_mix') or
            not hasattr(model.diamond.hierarchy, 'apex_reentry')):
        raise TypeError('R7.39 requires the complete R7.38 causal writer stack')
    old = {id(p) for p in model.parameters()}
    geometry.__class__ = CrossRailGeometry
    geometry.rail_cross = SignedRailCross(geometry.factor_key.embedding_dim)
    added = [(n, p) for n, p in model.named_parameters() if id(p) not in old]
    if not added or not all(n.startswith('diamond.geometry.rail_cross.') for n, _ in added):
        raise AssertionError('unexpected signed rail parameter registration')
    optimizer._name_by_id.update({id(p): n for n, p in added})
    optimizer._group_names.append({'algorithm': 'adamw', 'names': [n for n, _ in added]})
    optimizer._adamw.add_param_group({'params': [p for _, p in added],
                                     'lr': learning_rate, 'weight_decay': 0.})
    optimizer._source_group_counts[-1] += 1
    optimizer.param_groups = optimizer._legacy_groups + optimizer._adamw.param_groups
    actual = [id(p) for group in optimizer.param_groups for p in group['params']]
    if (len(actual) != len(set(actual)) or set(actual) !=
            {id(p) for p in model.parameters()} or not old.issubset(actual)):
        raise AssertionError('R7.39 changed inherited optimizer parameter identity')
    return model, optimizer


@contextlib.contextmanager
def signed_cross_disabled(model):
    rail = getattr(model.diamond.geometry, 'rail_cross', None)
    if rail is None:
        yield
        return
    previous = rail.gain.detach().clone()
    with torch.no_grad():
        rail.gain.zero_()
    try:
        yield
    finally:
        with torch.no_grad():
            rail.gain.copy_(previous)
