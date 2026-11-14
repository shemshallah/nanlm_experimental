#!/usr/bin/env python3
"""R7.37: one finite apex->pair->point->apex->pair->point feedback cycle.

The original R7 decoder and optimizer are not replaced. An identity-initialized
gain closes the previously absent return edge after the first descent. The
second descent still uses the existing learned pair and point decoders.
"""
from __future__ import annotations

import torch
from torch import nn

ROUND = 'FA1_R7_37_CLOSED_FEEDBACK_V1'


class ApexReentry(nn.Module):
    """A content-dependent return message, requiring both prior directions."""

    def __init__(self, width=32):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(2 * width),
                                 nn.Linear(2 * width, 2 * width), nn.GELU(),
                                 nn.Linear(2 * width, width, bias=False))

    def forward(self, apex, descended_point):
        if apex.shape != descended_point.shape or apex.ndim != 2:
            raise ValueError('reentry requires matching [B,D] apex and point')
        # Nothing reaches the apex through this return edge when either its
        # first upward signal or the inherited pair->point route is removed.
        route = (torch.tanh(apex.norm(dim=-1, keepdim=True)) *
                 torch.tanh(descended_point.norm(dim=-1, keepdim=True)))
        return route * self.net(torch.cat((apex, descended_point), -1))


def attach_feedback_closure(model, optimizer, learning_rate=1.5e-4):
    """Attach the R7.37 cycle without resetting any inherited Adam moments."""
    from fa1_r731_layered_decoder import LayeredDiamondDecoder
    h = getattr(model.diamond, 'hierarchy', None)
    geo = model.diamond.geometry
    if (not isinstance(h, LayeredDiamondDecoder) or
            not hasattr(geo, 'history_read_mix') or
            not hasattr(h, 'apex_feedback_center_gain')):
        raise TypeError('closed feedback requires the complete R7.36 stack')
    if hasattr(h, 'apex_reentry') or hasattr(h, 'apex_reentry_gain'):
        raise ValueError('R7.37 feedback closure already attached')
    previous = {id(p) for p in model.parameters()}
    h.apex_reentry = ApexReentry(h.width)
    h.apex_reentry_gain = nn.Parameter(torch.zeros(()))
    added = [(n, p) for n, p in model.named_parameters() if id(p) not in previous]
    if not added or not all(n.startswith('diamond.hierarchy.apex_reentry') for n, _ in added):
        raise AssertionError('unexpected feedback closure parameters')
    optimizer._name_by_id.update({id(p): n for n, p in added})
    optimizer._group_names.append({'algorithm': 'adamw', 'names': [n for n, _ in added]})
    optimizer._adamw.add_param_group({'params': [p for _, p in added],
                                     'lr': learning_rate, 'weight_decay': 0.})
    optimizer._source_group_counts[-1] += 1
    optimizer.param_groups = optimizer._legacy_groups + optimizer._adamw.param_groups
    all_group_ids = [id(p) for g in optimizer.param_groups for p in g['params']]
    if len(all_group_ids) != len(set(all_group_ids)) or set(all_group_ids) != {
            id(p) for p in model.parameters()}:
        raise AssertionError('closed feedback detached or duplicated parameters')
    return model, optimizer
