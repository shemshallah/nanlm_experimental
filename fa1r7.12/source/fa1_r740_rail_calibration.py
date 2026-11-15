#!/usr/bin/env python3
"""R7.40: zero-gated, scale-normalized continuation of saved R7.39 rails.

The inherited R7.39 output and every Adam moment remain exact when the new
calibration gate is zero. Scale normalization operates only on the eight
already-selected S+/S- face coefficients and masked pre-target far values.
"""
from __future__ import annotations

import contextlib

import torch
from torch import nn
from torch.nn import functional as F

from fa1_r739_signed_rails import SignedRailCross, antisymmetric_cross

ROUND = 'FA1_R7_40_NORMALIZED_SIGNED_RAILS_V1'


class NormalizedSignedRailCross(SignedRailCross):
    """Adds an independently ablatable, magnitude-corrected signed response."""

    def forward(self, values, mask, plus_codes, minus_codes):
        inherited = super().forward(values, mask, plus_codes, minus_codes)
        valid = mask.bool().unsqueeze(-1).to(values.dtype)
        observed = (values * valid).sum(1) / valid.sum(1).clamp_min(1.)
        width = observed.size(-1)
        # Both rails retain signs, but a small trained scalar code can now
        # matter. The denominator is computed only from selected addresses.
        plus_rms = plus_codes.square().mean(1, keepdim=True).sqrt().clamp_min(.005)
        minus_rms = minus_codes.square().mean(1, keepdim=True).sqrt().clamp_min(.005)
        plus = (torch.tanh(plus_codes / plus_rms).unsqueeze(-1) *
                torch.tanh(F.layer_norm(self.plus(observed), (width,))).unsqueeze(1))
        minus = (torch.tanh(minus_codes / minus_rms).unsqueeze(-1) *
                 torch.tanh(F.layer_norm(self.minus(observed), (width,))).unsqueeze(1))
        normalized = torch.tanh(self.project(antisymmetric_cross(plus, minus)))
        calibrated = (torch.tanh(self.gain) *
                      torch.tanh(self.calibration_gain) * normalized)
        return inherited + calibrated


def attach_rail_calibration(model, optimizer, learning_rate=1.5e-4):
    """Append group 24, keeping the 23-group checkpoint and all old tensors."""
    rail = getattr(model.diamond.geometry, 'rail_cross', None)
    if not isinstance(rail, SignedRailCross):
        raise TypeError('R7.40 requires an intact R7.39 signed cross-rail module')
    if hasattr(rail, 'calibration_gain'):
        raise ValueError('R7.40 normalized rails already attached')
    old = {id(p) for p in model.parameters()}
    rail.__class__ = NormalizedSignedRailCross
    rail.calibration_gain = nn.Parameter(torch.zeros(()))
    added = [(n,p) for n,p in model.named_parameters() if id(p) not in old]
    expected = {'diamond.geometry.rail_cross.calibration_gain'}
    if {n for n,_ in added} != expected:
        raise AssertionError('unexpected R7.40 parameter attachment')
    optimizer._name_by_id.update({id(p):n for n,p in added})
    optimizer._group_names.append({'algorithm':'adamw','names':[n for n,_ in added]})
    optimizer._adamw.add_param_group({'params':[p for _,p in added],
                                     'lr':learning_rate,'weight_decay':0.})
    optimizer._source_group_counts[-1] += 1
    optimizer.param_groups=optimizer._legacy_groups+optimizer._adamw.param_groups
    actual=[id(p) for group in optimizer.param_groups for p in group['params']]
    if (len(actual)!=len(set(actual)) or set(actual)!={id(p) for p in model.parameters()}
            or not old.issubset(actual)):
        raise AssertionError('R7.40 optimizer did not preserve inherited parameters')
    return model,optimizer


@contextlib.contextmanager
def calibrated_rails_disabled(model):
    rail = getattr(model.diamond.geometry,'rail_cross',None)
    if rail is None or not hasattr(rail,'calibration_gain'):
        yield
        return
    saved=rail.calibration_gain.detach().clone()
    with torch.no_grad():
        rail.calibration_gain.zero_()
    try:
        yield
    finally:
        with torch.no_grad():
            rail.calibration_gain.copy_(saved)
