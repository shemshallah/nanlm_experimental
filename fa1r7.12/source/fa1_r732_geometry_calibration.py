#!/usr/bin/env python3
"""R7.32: confidence-calibrated readout on the SAME layered FA1 Diamond.

This adds only two tiny, causal per-example geometric trust routers. The
routers start at exactly one, so all inherited R7.31 probabilities, optimizer
states and sparse geometric addresses are preserved during migration.
"""
from __future__ import annotations

import torch
from torch import nn

from fa1_r731_layered_decoder import (LayeredDiamondDecoder, CausalApexFeedback,
                                      attach_layered)
from fa1_r730_factorized_geometry import FactorizedContextualGeometry, attach_factorized
from fa1_r729_contextual_geometry import ContextualGeometry, attach_contextual

ROUND = 'FA1_R7_32_CALIBRATED_DIAMOND_V1'
APEX_FEEDBACK_ROUND = 'FA1_R7_UNIFIED_APEX_FEEDBACK_V1'


class GeometricTrust(nn.Module):
    """Bounded 0..2 multiplier, initialized to exactly 1 for all inputs."""

    def __init__(self, width):
        super().__init__()
        self.norm = nn.LayerNorm(3 * width)
        self.net = nn.Sequential(nn.Linear(3 * width, width), nn.GELU(),
                                 nn.Linear(width, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, fused):
        return self.net(self.norm(fused))


def attach_calibrated(model, optimizer, learning_rate=1.5e-4):
    """Append one optimizer group; never replace any trained Parameter."""
    h = getattr(model.diamond, 'hierarchy', None)
    if not isinstance(h, LayeredDiamondDecoder):
        raise TypeError('R7.32 requires the existing R7.31 layered decoder')
    if hasattr(h, 'byte_geometry_router') or hasattr(h, 'word_geometry_router'):
        raise ValueError('geometric trust routers are already attached')
    prior = {id(p) for p in model.parameters()}
    h.byte_geometry_router = GeometricTrust(h.width)
    h.word_geometry_router = GeometricTrust(h.width)
    added = [(n, p) for n, p in model.named_parameters() if id(p) not in prior]
    if not added or not all(n.startswith('diamond.hierarchy.') and
                            '_geometry_router.' in n for n, _ in added):
        raise AssertionError('calibration unexpectedly replaced inherited parameters')
    optimizer._name_by_id.update({id(p): name for name, p in added})
    optimizer._group_names.append({'algorithm': 'adamw', 'names': [n for n, _ in added]})
    optimizer._adamw.add_param_group({'params': [p for _, p in added],
                                     'lr': learning_rate, 'weight_decay': 0.})
    optimizer._source_group_counts[-1] += 1
    optimizer.param_groups = optimizer._legacy_groups + optimizer._adamw.param_groups
    if {id(p) for g in optimizer.param_groups for p in g['params']} != {
            id(p) for p in model.parameters()}:
        raise AssertionError('calibration detached parameters from unified optimizer')
    return model, optimizer


def attach_apex_feedback(model, optimizer, learning_rate=1.5e-4):
    """Append one causal feedback loop to the SAME calibrated FA1 model.

    A zero gate leaves inherited logits bitwise unchanged. All existing
    Parameter identities and Adam/legacy optimizer states remain attached.
    """
    h = getattr(model.diamond, 'hierarchy', None)
    if not isinstance(h, LayeredDiamondDecoder) or not hasattr(h, 'word_geometry_router'):
        raise TypeError('apex feedback requires existing calibrated FA1 hierarchy')
    if hasattr(h, 'apex_feedback') or hasattr(h, 'apex_feedback_gain'):
        raise ValueError('FA1 apex feedback is already attached')
    previous = {id(p) for p in model.parameters()}
    h.apex_feedback = CausalApexFeedback(h.width)
    h.apex_feedback_gain = nn.Parameter(torch.zeros(()))
    added = [(n, p) for n, p in model.named_parameters() if id(p) not in previous]
    if not added or not all(n.startswith('diamond.hierarchy.apex_feedback') for n, _ in added):
        raise AssertionError('unexpected FA1 feedback parameter registration')
    optimizer._name_by_id.update({id(p): n for n, p in added})
    optimizer._group_names.append({'algorithm': 'adamw', 'names': [n for n, _ in added]})
    optimizer._adamw.add_param_group({'params': [p for _, p in added],
                                     'lr': learning_rate, 'weight_decay': 0.})
    optimizer._source_group_counts[-1] += 1
    optimizer.param_groups = optimizer._legacy_groups + optimizer._adamw.param_groups
    if {id(p) for g in optimizer.param_groups for p in g['params']} != {
            id(p) for p in model.parameters()}:
        raise AssertionError('feedback detached inherited optimizer parameters')
    return model, optimizer


def load_calibrated(path, root, threads=2, with_apex_feedback=False):
    """Restore or migrate one FA1 checkpoint without reinitializing history."""
    from fa1_r727_checkpoint import load_unified
    model, optimizer, base, comp, meta = load_unified(path, root, threads)
    if meta.get('architecture_round') in (ROUND, APEX_FEEDBACK_ROUND,
                                          'FA1_R7_35_CAUSAL_HISTORY_VALUES_V1',
                                          'FA1_R7_36_DEEP_WIRING_V1',
                                          'FA1_R7_37_CLOSED_FEEDBACK_V1'):
        if not hasattr(model.diamond.hierarchy, 'byte_geometry_router'):
            raise AssertionError('saved R7.32 trust router was not restored')
        if with_apex_feedback and not hasattr(model.diamond.hierarchy, 'apex_feedback'):
            attach_apex_feedback(model, optimizer)
        return model, optimizer, base, comp, meta
    if type(model.diamond.geometry) is not FactorizedContextualGeometry:
        if type(model.diamond.geometry) is not ContextualGeometry:
            attach_contextual(model, optimizer)
        attach_factorized(model, optimizer)
    if not hasattr(model.diamond, 'hierarchy'):
        attach_layered(model, optimizer)
    attach_calibrated(model, optimizer)
    if with_apex_feedback:
        attach_apex_feedback(model, optimizer)
    return model, optimizer, base, comp, meta
