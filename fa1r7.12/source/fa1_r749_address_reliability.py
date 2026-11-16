#!/usr/bin/env python3
"""R7.49: causal address supervision and signed-rail reliability on R7.48.

No replacement backbone, no new corpus dependence and no gold-label features.
All added gains start at zero: migration is bitwise identical to R7.48.
"""
from __future__ import annotations

import contextlib

import torch
from torch import nn

from fa1_r744_causal_word_backoff import UnifiedR744Backoff
from fa1_r748_mirror_pair_values import MirrorPairValueBackoff, pair_indices


ROUND = 'FA1_R7_49_CAUSAL_PAIR_ADDRESS_RELIABILITY_V1'


class ReliableAddressedMirrorBackoff(MirrorPairValueBackoff):
    def pair_logits(self, data, trace, candidate_ids):
        original, info = super().pair_logits(data, trace, candidate_ids)
        active = data['vertex_ids'].long()
        heads = self.word_head_lookup[candidate_ids.long()]
        matched = (heads[..., None] == active[:, None, None, :]).any(-1)
        hits = matched.sum(-1).float()
        candidate_pair, valid = pair_indices(heads[..., 0], heads[..., 1])
        # This is an exact C(256,2) membership check. It is *not* a proxy
        # based on a learned value similarity or a gold target label.
        v = torch.triu_indices(8, 8, 1, device=active.device)
        active_pairs, _ = pair_indices(active[:, v[0]], active[:, v[1]])
        exact = (candidate_pair[:, :, None] == active_pairs[:, None, :]).any(-1)
        partial = (hits == 1) & valid
        # Orientation changes on a point-order reversal; pair ID does not.
        pos = (heads[..., None] == active[:, None, None, :]).long().argmax(-1)
        ordered = torch.where(pos[..., 0] < pos[..., 1], 1., -1.)
        signed_match = ordered * torch.where(heads[..., 0] < heads[..., 1], 1., -1.)
        address_term = (self.mirror_exact_address_gain * exact.float() +
                        self.mirror_partial_address_gain * partial.float() +
                        self.mirror_orientation_address_gain * signed_match * exact.float())
        mode = getattr(self, '_r748_control', 'full')
        if mode != 'without_signed_orientation':
            # Isolate ONLY the inherited antisymmetric rail by an identical
            # target-independent second read. The reliability gate never
            # modifies the symmetric rail or the sparse address term.
            self._r748_control = 'without_signed_orientation'
            try:
                neutral, _ = super().pair_logits(data, trace, candidate_ids)
            finally:
                self._r748_control = mode
            overlap = (hits / 2.).clamp(0., 1.)
            trust = torch.sigmoid(self.mirror_reliability_bias +
                                  self.mirror_reliability_overlap * overlap)
            suppression = torch.tanh(self.mirror_signed_suppression).clamp(0., 1.) * trust
            signed_adjustment = -suppression * (original - neutral)
        else:
            trust = torch.zeros_like(hits)
            signed_adjustment = torch.zeros_like(original)
        result = (original + signed_adjustment + address_term).masked_fill(~valid, 0.)
        info.update({'r749_exact_active_pair_fraction': exact.float().mean(1).detach(),
                     'r749_any_endpoint_fraction': (hits > 0).float().mean(1).detach(),
                     'r749_signed_reliability': trust.mean(1).detach(),
                     'r749_signed_suppression': torch.tanh(self.mirror_signed_suppression).detach()})
        return result, info


def attach_address_reliability(model, optimizer, learning_rate=2.e-4):
    """Append 29th group, retaining 28 old groups and their state unchanged."""
    if (type(model) is not UnifiedR744Backoff or
            type(model.word_backoff) is not MirrorPairValueBackoff or
            len(optimizer.param_groups) != 28):
        raise TypeError('R7.49 requires the complete 28-group R7.48 checkpoint')
    previous = {id(p) for p in model.parameters()}
    module = model.word_backoff
    module.__class__ = ReliableAddressedMirrorBackoff
    module.mirror_signed_suppression = nn.Parameter(torch.zeros(()))
    module.mirror_reliability_bias = nn.Parameter(torch.tensor(0.))
    module.mirror_reliability_overlap = nn.Parameter(torch.zeros(()))
    module.mirror_exact_address_gain = nn.Parameter(torch.zeros(()))
    module.mirror_partial_address_gain = nn.Parameter(torch.zeros(()))
    module.mirror_orientation_address_gain = nn.Parameter(torch.zeros(()))
    new = [(name, p) for name, p in model.named_parameters() if id(p) not in previous]
    expected = {'word_backoff.mirror_signed_suppression',
                'word_backoff.mirror_reliability_bias',
                'word_backoff.mirror_reliability_overlap',
                'word_backoff.mirror_exact_address_gain',
                'word_backoff.mirror_partial_address_gain',
                'word_backoff.mirror_orientation_address_gain'}
    if {name for name, _ in new} != expected:
        raise AssertionError('unexpected R7.49 trainable state')
    optimizer._name_by_id.update({id(p): name for name, p in new})
    optimizer._group_names.append({'algorithm': 'adamw', 'names': [name for name, _ in new]})
    optimizer._adamw.add_param_group({'params': [p for _, p in new],
                                     'lr': learning_rate, 'weight_decay': 0.})
    optimizer._source_group_counts[-1] += 1
    optimizer.param_groups = optimizer._legacy_groups + optimizer._adamw.param_groups
    ids = [id(p) for group in optimizer.param_groups for p in group['params']]
    if (len(ids) != len(set(ids)) or set(ids) != {id(p) for p in model.parameters()}
            or len(optimizer.param_groups) != 29):
        raise AssertionError('R7.49 broke one unified optimizer')
    return model, optimizer


@contextlib.contextmanager
def address_reliability_control(model, mode='disabled'):
    """Reversible counterfactuals; do not modify inherited Adam moments."""
    m = model.word_backoff
    if mode not in ('disabled', 'without_new_address', 'without_new_signed_gate'):
        raise ValueError('unknown R7.49 control')
    address_names = ('mirror_exact_address_gain', 'mirror_partial_address_gain',
                     'mirror_orientation_address_gain')
    sign_names = ('mirror_signed_suppression',)
    names = (address_names + sign_names if mode == 'disabled' else
             address_names if mode == 'without_new_address' else sign_names)
    original = {name: getattr(m, name).detach().clone() for name in names}
    with torch.no_grad():
        for name in names:
            getattr(m, name).zero_()
    try:
        yield
    finally:
        with torch.no_grad():
            for name in names:
                getattr(m, name).copy_(original[name])
