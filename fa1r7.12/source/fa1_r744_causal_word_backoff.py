#!/usr/bin/env python3
"""FA1 R7.44: bounded, target-free copy distribution over causal word history.

An experimental continuation of the *unchanged* R7.40 model.  The added
probability distribution is normalized across vocabulary words because it is
a weighted empirical distribution over observed previous word IDs.  Only its
probability at the supervised target is gathered for cross-entropy; the target
does not enter the history weights, geometric trust, or mixture gate.

This is distinct from conditional spelling BPB and cannot imply raw-text BPB.
"""
from __future__ import annotations

import contextlib

import torch
from torch import nn
from torch.nn import functional as F

from fa1_r723_diamond import UnifiedR723Diamond


ROUND = 'FA1_R7_44_CAUSAL_WORD_BACKOFF_V1'


class CausalWordBackoff(nn.Module):
    """Predict a small normalized copy mixture using pre-target evidence only."""

    def __init__(self, max_weight=.125):
        super().__init__()
        if not 0 < max_weight <= .25:
            raise ValueError('backoff must be bounded by 25 percent')
        self.max_weight = float(max_weight)
        self.gain = nn.Parameter(torch.zeros(()))
        self.route = nn.Sequential(nn.Linear(7, 16), nn.Tanh(), nn.Linear(16, 1))
        nn.init.zeros_(self.route[-1].weight)
        nn.init.constant_(self.route[-1].bias, -1.5)

    def forward(self, data, trace, targets, embeddings=None):
        recent = data.get('diamond_backoff_recent_ids', data['word_ids'])
        recent_mask = data.get('diamond_backoff_recent_mask')
        if recent_mask is None:
            # 1 is the recorded BOS sentinel. Genuine OOV (0) remains legal.
            recent_mask = recent != 1
        modal = data.get('diamond_backoff_modal_ids',
                         data.get('diamond_modal_ids', data.get('memory_modal_ids')))
        modal_mask = data.get('diamond_backoff_modal_mask',
                              data.get('diamond_modal_mask',data.get('memory_modal_mask')))
        if recent.ndim != 2 or targets.shape != (recent.size(0),):
            raise ValueError('expected causal [batch, prior words] and [batch] targets')
        if recent_mask.shape != recent.shape:
            raise ValueError('recent-word mask shape mismatch')
        if modal is None or modal_mask is None:
            modal = recent.new_zeros((recent.size(0), 0))
            modal_mask = recent_mask.new_zeros((recent.size(0), 0))
        if modal.ndim != 2 or modal.size(0) != recent.size(0) or modal_mask.shape != modal.shape:
            raise ValueError('modal history/mask shape mismatch')
        if bool((recent < 0).any()) or bool((modal < 0).any()):
            raise ValueError('history IDs must be nonnegative')

        batch, length = recent.shape
        near_valid = recent_mask.to(torch.bool)
        far_valid = modal_mask.to(torch.bool)
        # These weights are fixed and target-independent. Later past tokens
        # receive more weight, while genuine far retrieval adds sparse mass.
        near_weights = (torch.linspace(.25, 1., length,device=recent.device)
                        .unsqueeze(0).expand(batch,-1) * near_valid.float())
        counts = data.get('diamond_backoff_source_counts')
        if counts is not None and counts.shape == modal.shape:
            source_counts = counts
        else:
            source_counts = None
        if source_counts is None:
            near_counts = data.get('diamond_backoff_near_count', data.get('near_count'))
            far_counts = data.get('diamond_backoff_far_count', data.get('far_count'))
            if (near_counts is not None and far_counts is not None and
                    near_counts.size(0) == batch and far_counts.size(0) == batch and
                    near_counts.size(1) + far_counts.size(1) == modal.size(1)):
                source_counts = torch.cat((near_counts, far_counts), dim=1)
        if source_counts is None:
            source_counts = torch.ones_like(modal, dtype=torch.float)
        far_weights = (.5 * torch.log1p(source_counts.float().clamp_min(0))
                       .clamp(.1, 4.) * far_valid.float())
        all_ids = torch.cat((recent.long(),modal.long()),dim=1)
        all_weights = torch.cat((near_weights,far_weights),dim=1)
        mass = all_weights.sum(dim=1)
        # q(word) is a normalized empirical distribution; the scalar lookup
        # here is the usual target probability for loss, never a gate feature.
        target_mass = (all_weights * (all_ids == targets[:,None]).float()).sum(1)
        q_target = target_mass / mass.clamp_min(1.e-20)

        near_coverage = near_valid.float().mean(1)
        far_coverage = (far_valid.float().mean(1) if modal.size(1)
                        else near_coverage.new_zeros(batch))
        # Similarity of the two *observed* histories, independently of label.
        agreement = ((recent[:,:,None] == modal[:,None,:]) &
                     near_valid[:,:,None] & far_valid[:,None,:]).float().sum((1,2))
        agreement = agreement / (near_valid.float().sum(1) *
                                 far_valid.float().sum(1)).clamp_min(1.)
        diversity = (1. - ((recent[:,:,None] == recent[:,None,:]) &
                           near_valid[:,:,None] & near_valid[:,None,:]).float().sum((1,2)) /
                     near_valid.float().sum(1).square().clamp_min(1.))
        attention = trace['triangle_attention'].detach()
        entropy = -(attention * attention.clamp_min(1.e-12).log()).sum(1)
        entropy = entropy / max(1., float(attention.size(1)))**.5
        geometry_trust = torch.sigmoid(trace['trust_logit'].detach())
        apex_activity = torch.tanh(trace['apex_std'].detach()).expand(batch)
        features = torch.stack((near_coverage, far_coverage, agreement,
                                diversity, entropy, geometry_trust, apex_activity),1)
        # Closed gain and no-history gate recover the inherited result exactly.
        confidence = torch.sigmoid(self.route(features).squeeze(1))
        alpha = self.max_weight * torch.tanh(self.gain).clamp_min(0.) * confidence
        alpha = alpha * (mass > 0).float()
        return alpha, q_target, {'history_mass':mass, 'history_coverage':near_coverage,
                                  'modal_coverage':far_coverage,
                                  'geometric_trust':geometry_trust,
                                  'backoff_candidate_ids':all_ids,
                                  'backoff_candidate_weights':all_weights}


class UnifiedR744Backoff(UnifiedR723Diamond):
    def joint_nll(self, data, near_enabled=True, far_enabled=True,
                  semantic_enabled=True, diamond_enabled=True,
                  two_way=True, return_extras=False, **ablations):
        if not diamond_enabled:
            return super().joint_nll(data,near_enabled,far_enabled,
                                     semantic_enabled,diamond_enabled,two_way,
                                     return_extras,**ablations)
        parent, close, extras = super().joint_nll(data,near_enabled,far_enabled,
                                                  semantic_enabled,diamond_enabled,two_way,
                                                  True,**ablations)
        alpha, q_target, telemetry = self.word_backoff(data,extras['trace'],data['target'],
                                                       self.core.trunk.E.detach())
        # log((1-a)P_parent(y) + a Q_history(y)), stable even when Q(y)=0.
        alternative = alpha * q_target
        correction = -torch.logaddexp(torch.log1p(-alpha) - parent,
                                     alternative.clamp_min(1e-38).log())
        corrected = torch.where(alpha == 0,parent,correction)
        if return_extras:
            extras.update({'backoff_alpha':alpha, 'backoff_target_probability':q_target,
                           'backoff_parent_nll':parent,**telemetry})
            return corrected, close, extras
        return corrected, close


def attach_causal_word_backoff(model, optimizer, learning_rate=1.5e-4):
    """One in-place model and one optimizer: 24 inherited groups, new group 25."""
    if type(model) is not UnifiedR723Diamond or not hasattr(model.diamond.geometry.rail_cross,
                                                             'calibration_gain'):
        raise TypeError('R7.44 requires the complete R7.40 model')
    if hasattr(model,'word_backoff'):
        raise ValueError('causal word backoff already attached')
    old = {id(p) for p in model.parameters()}
    model.__class__ = UnifiedR744Backoff
    model.word_backoff = CausalWordBackoff()
    added = [(n,p) for n,p in model.named_parameters() if id(p) not in old]
    if not added or not all(n.startswith('word_backoff.') for n,p in added):
        raise AssertionError('unexpected backoff parameter registration')
    optimizer._name_by_id.update({id(p):n for n,p in added})
    optimizer._group_names.append({'algorithm':'adamw','names':[n for n,p in added]})
    optimizer._adamw.add_param_group({'params':[p for n,p in added],
                                     'lr':learning_rate,'weight_decay':0.})
    optimizer._source_group_counts[-1] += 1
    optimizer.param_groups = optimizer._legacy_groups + optimizer._adamw.param_groups
    ids=[id(p) for group in optimizer.param_groups for p in group['params']]
    if len(ids)!=len(set(ids)) or set(ids)!={id(p) for p in model.parameters()} or not old.issubset(ids):
        raise AssertionError('R7.44 must preserve every parent parameter and optimizer group')
    return model,optimizer


@contextlib.contextmanager
def causal_word_backoff_disabled(model):
    if not hasattr(model,'word_backoff'):
        yield
        return
    gain=model.word_backoff.gain
    before=gain.detach().clone()
    with torch.no_grad():
        gain.zero_()
    try:
        yield
    finally:
        with torch.no_grad():
            gain.copy_(before)
