#!/usr/bin/env python3
"""R7.48: trainable C(256,2) value memory in the inherited FA1 predictor.

The 32,640 pair states are *values*, not a new language model. A causal
eight-point/28-pair read produces a wave used to score observed prior words
at their accepted pair addresses. All target-dependent gathering is confined
to the final word-probability lookup. Zero gates exactly recover R7.47.
"""
from __future__ import annotations

import contextlib
import itertools

import torch
from torch import nn
from torch.nn import functional as F

from fa1_r744_causal_word_backoff import UnifiedR744Backoff
from fa1_r747_head_address_copy import HeadAddressWordBackoff


ROUND = 'FA1_R7_48_TRAINABLE_MIRROR_PAIR_VALUES_V1'
PAIR_COUNT = 32640
VALUE_DIM = 16
ACTIVE_EDGES = torch.tensor(list(itertools.combinations(range(8), 2)), dtype=torch.long)


def pair_indices(a, b):
    """Lexicographic undirected pair IDs; repeated heads are marked invalid."""
    if a.shape != b.shape:
        raise ValueError('mismatched pair endpoint shapes')
    if bool(((a < 0) | (a >= 256) | (b < 0) | (b >= 256)).any()):
        raise ValueError('pair endpoints must be in 0..255')
    lo = torch.minimum(a, b)
    hi = torch.maximum(a, b)
    return (lo * (511 - lo) // 2 + hi - lo - 1).clamp_min(0).long(), a != b


class MirrorPairValueBackoff(HeadAddressWordBackoff):
    """Differentiable pair-value read with exact interventions for attribution."""

    def pair_logits(self, data, trace, candidate_ids):
        active = data['vertex_ids'].long()
        if active.ndim != 2 or active.shape[1] != 8:
            raise ValueError('expected eight active causal points')
        if bool((active.sort(1).values[:, 1:] == active.sort(1).values[:, :-1]).any()):
            raise ValueError('active points must be unique')
        if candidate_ids.ndim != 2 or candidate_ids.shape[0] != active.shape[0]:
            raise ValueError('candidate batch mismatch')
        heads = self.word_head_lookup[candidate_ids.long()]
        ids, valid = pair_indices(heads[..., 0], heads[..., 1])
        edges = ACTIVE_EDGES.to(active.device)
        p, q = active[:, edges[:, 0]], active[:, edges[:, 1]]
        active_ids, _ = pair_indices(p, q)
        mode = getattr(self, '_r748_control', 'full')
        if mode == 'scramble_pair_values':
            # Permute values, not the input vertices or any inherited state.
            # Nonzero permutation fixes no pair and preserves memory norms.
            ids = (ids + 7919) % PAIR_COUNT
            active_ids = (active_ids + 7919) % PAIR_COUNT
        if mode not in ('full', 'scramble_pair_values', 'without_pair_values',
                        'without_signed_orientation'):
            raise ValueError('unknown R7.48 pair-memory control')

        candidate = self.mirror_pair_values(ids)
        active_value = self.mirror_pair_values(active_ids)
        if mode == 'without_pair_values':
            candidate = torch.zeros_like(candidate)
            active_value = torch.zeros_like(active_value)
        # Shared endpoints generalize to pairs not seen in the minibatch.
        candidate = candidate + .5 * (self.mirror_head_values(heads[..., 0]) +
                                      self.mirror_head_values(heads[..., 1]))
        active_value = active_value + .5 * (self.mirror_head_values(p) +
                                            self.mirror_head_values(q))
        sign_active = torch.where(p < q, 1., -1.).unsqueeze(-1)
        sign_candidate = torch.where(heads[..., 0] < heads[..., 1], 1., -1.)
        if mode == 'without_signed_orientation':
            sign_active = torch.zeros_like(sign_active)
            sign_candidate = torch.zeros_like(sign_candidate)
        # The symmetric rail remains present if the orientation is ablated.
        symmetric = active_value.mean(dim=1)
        antisymmetric = (active_value * sign_active).mean(dim=1)
        descended = trace.get('descended_point_summary')
        if descended is None or descended.shape != (len(active), 32):
            raise ValueError('pair read needs the causal apex->pair->point trace')
        query = self.mirror_pair_query(descended)
        wave = F.normalize(symmetric + query, dim=-1)
        signed_wave = F.normalize(antisymmetric + .25 * query, dim=-1)
        candidate_unit = F.normalize(candidate, dim=-1)
        logits = (2. * (candidate_unit * wave[:, None, :]).sum(-1) +
                  self.mirror_signed_weight * sign_candidate *
                  (candidate_unit * signed_wave[:, None, :]).sum(-1))
        return logits.masked_fill(~valid, 0.), {
            'r748_pair_candidate_ids': ids.detach(),
            'r748_pair_active_ids': active_ids.detach(),
            'r748_valid_pair_fraction': valid.float().mean(dim=1),
            'r748_wave_norm': symmetric.norm(dim=-1),
        }

    def forward(self, data, trace, targets, embeddings=None):
        alpha, q, telemetry = super().forward(data, trace, targets, embeddings)
        ids = telemetry['backoff_candidate_ids']
        weights = telemetry['backoff_candidate_weights']
        logits, pair_info = self.pair_logits(data, trace, ids)
        # Only observed causal words receive copy mass; masked softmax
        # reweighting cannot invent a word not in the history.
        valid = weights > 0
        inherited_attn = telemetry['r747_head_attention']
        scores = logits.clamp(-8., 8.)
        unnormalized = inherited_attn * scores.exp() * valid.float()
        pair_attn = unnormalized / unnormalized.sum(1, keepdim=True).clamp_min(1.e-20)
        target_probability = ((ids == targets[:, None]).float() * pair_attn).sum(1)
        has_history = (telemetry['history_mass'] > 0).float()
        beta = .50 * torch.tanh(self.mirror_pair_gain).clamp_min(0.) * has_history
        extra = .125 * torch.tanh(self.mirror_pair_copy_gain).clamp_min(0.) * has_history
        # Both operands are normalized target probabilities, so this is a
        # proper mixture. Preserve EXACT parent values at zero migration.
        q_new = torch.where(beta == 0, q, (1. - beta) * q + beta * target_probability)
        alpha_new = torch.where(extra == 0, alpha, alpha + (1. - alpha) * extra)
        telemetry.update(pair_info)
        telemetry.update({'r748_pair_target_probability': target_probability,
                          'r748_pair_attention': pair_attn,
                          'r748_pair_mixture': beta,
                          'r748_pair_copy_mass': extra})
        return alpha_new, q_new, telemetry


def attach_mirror_pair_values(model, optimizer, learning_rate=2.e-4):
    """Append group 28; do not replace weights, optimizer, or Adam moments."""
    if type(model) is not UnifiedR744Backoff or type(model.word_backoff) is not HeadAddressWordBackoff:
        raise TypeError('R7.48 must directly extend the complete R7.47 model')
    if len(optimizer.param_groups) != 27:
        raise ValueError('R7.48 requires exactly 27 inherited optimizer groups')
    prior = {id(p) for p in model.parameters()}
    module = model.word_backoff
    module.__class__ = MirrorPairValueBackoff
    module.mirror_pair_values = nn.Embedding(PAIR_COUNT, VALUE_DIM)
    module.mirror_head_values = nn.Embedding(256, VALUE_DIM)
    module.mirror_pair_query = nn.Linear(32, VALUE_DIM, bias=False)
    module.mirror_pair_gain = nn.Parameter(torch.zeros(()))
    module.mirror_pair_copy_gain = nn.Parameter(torch.zeros(()))
    module.mirror_signed_weight = nn.Parameter(torch.tensor(.25))
    nn.init.normal_(module.mirror_pair_values.weight, std=.02)
    nn.init.normal_(module.mirror_head_values.weight, std=.02)
    nn.init.normal_(module.mirror_pair_query.weight, std=.02)
    new = [(name, p) for name, p in model.named_parameters() if id(p) not in prior]
    expected = {'word_backoff.mirror_pair_values.weight',
                'word_backoff.mirror_head_values.weight',
                'word_backoff.mirror_pair_query.weight',
                'word_backoff.mirror_pair_gain', 'word_backoff.mirror_pair_copy_gain',
                'word_backoff.mirror_signed_weight'}
    if {name for name, _ in new} != expected:
        raise AssertionError('unexpected pair-memory parameters')
    optimizer._name_by_id.update({id(p): name for name, p in new})
    optimizer._group_names.append({'algorithm': 'adamw', 'names': [name for name, _ in new]})
    optimizer._adamw.add_param_group({'params': [p for _, p in new],
                                     'lr': learning_rate, 'weight_decay': 0.})
    optimizer._source_group_counts[-1] += 1
    optimizer.param_groups = optimizer._legacy_groups + optimizer._adamw.param_groups
    ids = [id(p) for group in optimizer.param_groups for p in group['params']]
    if len(ids) != len(set(ids)) or set(ids) != {id(p) for p in model.parameters()} or len(optimizer.param_groups) != 28:
        raise AssertionError('R7.48 lost unified optimizer continuity')
    return model, optimizer


@contextlib.contextmanager
def pair_value_control(model, mode='disabled'):
    module = model.word_backoff
    if mode == 'disabled':
        old = [module.mirror_pair_gain.detach().clone(),
               module.mirror_pair_copy_gain.detach().clone()]
        with torch.no_grad():
            module.mirror_pair_gain.zero_()
            module.mirror_pair_copy_gain.zero_()
        try:
            yield
        finally:
            with torch.no_grad():
                module.mirror_pair_gain.copy_(old[0])
                module.mirror_pair_copy_gain.copy_(old[1])
    else:
        before = getattr(module, '_r748_control', 'full')
        module._r748_control = mode
        try:
            yield
        finally:
            module._r748_control = before
