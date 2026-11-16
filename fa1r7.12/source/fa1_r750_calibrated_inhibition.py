#!/usr/bin/env python3
"""R7.50 bounded, trainable inhibitory likelihood evidence at exact C(256,2).

This extends the actual R7.49 29-group FA1 checkpoint, NOT a new model.
Canonical-address exclusion is learned from TRAIN labels only. Every routing
feature is available before the target; the existing copy distribution stays
normalized and target-independent until its final gold-probability lookup.
"""
from __future__ import annotations

import contextlib

import torch
from torch import nn
from torch.nn import functional as F

from fa1_r744_causal_word_backoff import UnifiedR744Backoff
from fa1_r748_mirror_pair_values import PAIR_COUNT, pair_indices
from fa1_r749_address_reliability import ReliableAddressedMirrorBackoff


ROUND = 'FA1_R7_50_CALIBRATED_PAIR_INHIBITION_V1'


class CalibratedPairInhibitionBackoff(ReliableAddressedMirrorBackoff):
    def pair_logits(self, data, trace, candidate_ids):
        logits, info = super().pair_logits(data, trace, candidate_ids)
        heads = self.word_head_lookup[candidate_ids.long()]
        candidate_pair, valid = pair_indices(heads[..., 0], heads[..., 1])
        active = data['vertex_ids'].long()
        edges = torch.triu_indices(8, 8, 1, device=active.device)
        active_pairs, _ = pair_indices(active[:, edges[0]], active[:, edges[1]])
        exact = (candidate_pair[..., None] == active_pairs[:, None, :]).any(-1) & valid
        mode = getattr(self, '_r750_control', 'full')
        if mode not in ('full', 'disabled', 'permute_inhibitory_values', 'constant_reliability'):
            raise ValueError('invalid R7.50 control')
        pair_key = candidate_pair
        if mode == 'permute_inhibitory_values':
            pair_key = (pair_key + 7919) % PAIR_COUNT
        # Sparse *value* memory is not a synonym for exact membership: the
        # address chooses a learned inhibitory magnitude, not merely a flag.
        magnitude = F.softplus(self.mirror_inhibition_table(pair_key).squeeze(-1) +
                               self.mirror_inhibition_bias)
        summary = trace['descended_point_summary']
        if summary.shape != (len(active), 32):
            raise ValueError('causal apex descent is required for inhibition')
        trust_logit = self.mirror_inhibition_context(summary).squeeze(-1)
        trust = torch.sigmoid(trust_logit)
        if mode == 'constant_reliability':
            trust = torch.full_like(trust, .5)
        gain = (.5 * torch.tanh(self.mirror_inhibition_gain).clamp_min(0.))
        if mode == 'disabled':
            gain = gain * 0.
        penalty = gain * magnitude * trust[:, None] * exact.float()
        answer = (logits - penalty).masked_fill(~valid, 0.)
        info.update({'r750_exact_hit_fraction':exact.float().mean(1).detach(),
                     'r750_inhibition_mean':penalty.mean(1).detach(),
                     'r750_inhibition_trust':trust.detach(),
                     'r750_inhibition_gain':gain.detach()})
        return answer, info


def attach_calibrated_inhibition(model, optimizer, learning_rate=2.e-4):
    if (type(model) is not UnifiedR744Backoff or
            type(model.word_backoff) is not ReliableAddressedMirrorBackoff or
            len(optimizer.param_groups) != 29):
        raise TypeError('R7.50 requires exact unified 29-group R7.49 lineage')
    inherited_ids = {id(p) for p in model.parameters()}
    m = model.word_backoff
    m.__class__ = CalibratedPairInhibitionBackoff
    m.mirror_inhibition_table = nn.Embedding(PAIR_COUNT, 1)
    m.mirror_inhibition_context = nn.Linear(32, 1)
    m.mirror_inhibition_bias = nn.Parameter(torch.tensor(-1.))
    m.mirror_inhibition_gain = nn.Parameter(torch.zeros(()))
    nn.init.zeros_(m.mirror_inhibition_table.weight)
    nn.init.zeros_(m.mirror_inhibition_context.weight)
    nn.init.zeros_(m.mirror_inhibition_context.bias)
    new = [(name, p) for name, p in model.named_parameters() if id(p) not in inherited_ids]
    expected = {'word_backoff.mirror_inhibition_table.weight',
                'word_backoff.mirror_inhibition_context.weight',
                'word_backoff.mirror_inhibition_context.bias',
                'word_backoff.mirror_inhibition_bias',
                'word_backoff.mirror_inhibition_gain'}
    if {name for name, _ in new} != expected:
        raise AssertionError('unexpected R7.50 parameter topology')
    optimizer._name_by_id.update({id(p): name for name, p in new})
    optimizer._group_names.append({'algorithm': 'adamw', 'names': [name for name, _ in new]})
    optimizer._adamw.add_param_group({'params': [p for _, p in new],
                                     'lr':learning_rate, 'weight_decay':0.})
    optimizer._source_group_counts[-1] += 1
    optimizer.param_groups = optimizer._legacy_groups + optimizer._adamw.param_groups
    ids = [id(p) for group in optimizer.param_groups for p in group['params']]
    if (len(ids) != len(set(ids)) or len(optimizer.param_groups) != 30 or
            set(ids) != {id(p) for p in model.parameters()}):
        raise AssertionError('R7.50 must retain every inherited optimizer parameter')
    return model, optimizer


@contextlib.contextmanager
def inhibition_control(model, mode='disabled'):
    if mode not in ('disabled', 'permute_inhibitory_values', 'constant_reliability'):
        raise ValueError('unrecognized inhibition control')
    m = model.word_backoff
    before = getattr(m, '_r750_control', 'full')
    m._r750_control = mode
    try:
        yield
    finally:
        m._r750_control = before
