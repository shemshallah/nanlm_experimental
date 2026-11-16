#!/usr/bin/env python3
"""R7.45: target-free geometric attention over observed causal word values.

Unlike R7.44's global geometric confidence, the last DESCENDED Diamond point
state selects *which* previously observed word values get copy probability.
It remains one normalized word distribution, inside the same LM/optimizer.
The added read is zero-gated and exactly preserves the 25-group R7.44 parent.
"""
from __future__ import annotations

import contextlib

import torch
from torch import nn
from torch.nn import functional as F

from fa1_r744_causal_word_backoff import CausalWordBackoff, UnifiedR744Backoff


ROUND = 'FA1_R7_45_GEOMETRIC_VALUE_READ_V1'


class GeometricWordBackoff(CausalWordBackoff):
    def forward(self, data, trace, targets, embeddings=None):
        alpha, inherited_q, telemetry = super().forward(data,trace,targets,embeddings)
        if embeddings is None or embeddings.ndim != 2:
            raise ValueError('geometric candidate values require inherited word embeddings')
        point = trace.get('descended_point_summary')
        if point is None or point.ndim != 2 or point.size(0) != targets.size(0):
            raise ValueError('geometric word value read requires final descended point state')
        ids = telemetry['backoff_candidate_ids']
        weights = telemetry['backoff_candidate_weights']
        if ids.numel() and int(ids.max()) >= embeddings.size(0):
            raise ValueError('history candidate exceeds inherited vocabulary')
        # The only content keys are actual observed pre-target word IDs. Each
        # key inherits the parent token embedding; no target or future token
        # is ever available to the geometric query.
        keys = F.normalize(F.embedding(ids,embeddings),dim=-1)
        query = F.normalize(self.geometric_query(point),dim=-1)
        scores = (keys * query.unsqueeze(1)).sum(-1) * 8.
        scores = scores + weights.clamp_min(1.e-30).log()
        eligible = weights > 0
        scores = scores.masked_fill(~eligible, -1.e4)
        attention = F.softmax(scores,dim=1) * eligible.float()
        attention = attention / attention.sum(1,keepdim=True).clamp_min(1.e-20)
        attended_target = ((ids==targets[:,None]).float()*attention).sum(1)
        beta = (.6 * torch.tanh(self.geometric_gain).clamp_min(0.) *
                (telemetry['history_mass']>0).float())
        q = (1. - beta) * inherited_q + beta * attended_target
        telemetry.update({'r745_geometric_beta':beta,
                          'r745_attended_target_probability':attended_target,
                          'r745_geometric_candidate_attention':attention})
        return alpha,q,telemetry


def attach_geometric_value_read(model,optimizer,learning_rate=2.e-4):
    """Keep every R7.44 Parameter/moment, append one group (25 -> 26)."""
    if type(model) is not UnifiedR744Backoff or type(model.word_backoff) is not CausalWordBackoff:
        raise TypeError('R7.45 requires the intact R7.44 word-backoff model')
    old_ids={id(p) for p in model.parameters()}
    module=model.word_backoff
    module.__class__=GeometricWordBackoff
    module.geometric_query=nn.Linear(32,model.core.trunk.E.size(1),bias=False)
    module.geometric_gain=nn.Parameter(torch.zeros(()))
    added=[(name,p) for name,p in model.named_parameters() if id(p) not in old_ids]
    if {name for name,p in added}!={'word_backoff.geometric_gain',
                                   'word_backoff.geometric_query.weight'}:
        raise AssertionError('R7.45 attached unexpected parameters')
    optimizer._name_by_id.update({id(p):name for name,p in added})
    optimizer._group_names.append({'algorithm':'adamw','names':[name for name,p in added]})
    optimizer._adamw.add_param_group({'params':[p for name,p in added],
                                     'lr':learning_rate,'weight_decay':0.})
    optimizer._source_group_counts[-1]+=1
    optimizer.param_groups=optimizer._legacy_groups+optimizer._adamw.param_groups
    actual=[id(p) for group in optimizer.param_groups for p in group['params']]
    if (len(actual)!=len(set(actual)) or set(actual)!={id(p) for p in model.parameters()}
            or not old_ids.issubset(actual)):
        raise AssertionError('geometric read broke unified optimizer continuity')
    return model,optimizer


@contextlib.contextmanager
def geometric_value_read_disabled(model):
    read=model.word_backoff
    if not hasattr(read,'geometric_gain'):
        yield
        return
    gain=read.geometric_gain
    before=gain.detach().clone()
    with torch.no_grad():gain.zero_()
    try:
        yield
    finally:
        with torch.no_grad():gain.copy_(before)
