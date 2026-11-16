#!/usr/bin/env python3
"""R7.47 exact-address lexical copy through the inherited 256 NaN heads.

One additive continuation of R7.45, not a replacement model. A prior word's
two immutable R7 head addresses are compared with the 8 active causal points.
Both S+/S- orientations are explicit. The extra probability is a normalized,
target-free word-history distribution; zero gates reproduce R7.45 exactly.
"""
from __future__ import annotations

import contextlib

import torch
from torch import nn

from fa1_r744_causal_word_backoff import UnifiedR744Backoff
from fa1_r745_geometric_value_read import GeometricWordBackoff


ROUND='FA1_R7_47_EXACT_HEAD_ADDRESS_COPY_V1'


def head_address_features(candidate_ids, active_points, lookup):
    """Five target-independent head/pair/signed-rail membership indicators."""
    if (candidate_ids.ndim!=2 or active_points.ndim!=2 or
            candidate_ids.size(0)!=active_points.size(0) or
            active_points.size(1)!=8 or lookup.ndim!=2 or lookup.size(1)!=2):
        raise ValueError('expected [B,K] candidates, [B,8] points, [V,2] heads')
    if (bool((candidate_ids<0).any()) or bool((candidate_ids>=lookup.size(0)).any()) or
            bool((active_points<0).any()) or bool((active_points>=256).any())):
        raise ValueError('invalid head or word address')
    if bool((active_points.sort(1).values[:,1:]==active_points.sort(1).values[:,:-1]).any()):
        raise ValueError('active heads must be unique')
    heads=lookup[candidate_ids.long()]
    equal=heads[:,:,:,None]==active_points[:,None,None,:]
    matched=equal.any(-1)
    first=matched[:,:,0]
    second=matched[:,:,1]
    both=first&second
    pos=equal.long().argmax(-1)
    aligned=both&(pos[:,:,0]<pos[:,:,1])
    crossed=both&(pos[:,:,0]>pos[:,:,1])
    return torch.stack((first,second,both,aligned,crossed),dim=-1).float()


class HeadAddressWordBackoff(GeometricWordBackoff):
    def forward(self, data, trace, targets, embeddings=None):
        alpha,q,telemetry=super().forward(data,trace,targets,embeddings)
        if 'vertex_ids' not in data:
            raise ValueError('exact head read requires causal active point addresses')
        ids=telemetry['backoff_candidate_ids']
        weights=telemetry['backoff_candidate_weights']
        feature=head_address_features(ids,data['vertex_ids'],self.word_head_lookup)
        # Every raw candidate word was actually observed before the target.
        # Orientations are those of its inherited H/P/M heads in the real
        # active point ordering, never inferred from gold label membership.
        bias=(feature*self.address_coefficients).sum(-1).clamp(-8.,8.)
        original=telemetry['r745_geometric_candidate_attention']
        valid=weights>0
        weighted=original*bias.exp()*valid.float()
        changed=weighted/weighted.sum(1,keepdim=True).clamp_min(1.e-20)
        head_target=((ids==targets[:,None]).float()*changed).sum(1)
        # Its own word distribution is normalized because `changed` sums to
        # exactly one whenever any causal candidate exists. The inherited
        # R7.45 distribution is also normalized, so their convex blend is.
        gamma=(.50*torch.tanh(self.address_gain).clamp_min(0.)*
               (telemetry['history_mass']>0).float())
        q_new=torch.where(gamma==0,q,(1.-gamma)*q+gamma*head_target)
        # A second, bounded copy mass can be learned without altering the
        # complete inherited R7.45 mixture or its 26 optimizer histories.
        # This gate uses only the prevalence of observed head membership.
        head_membership=(feature[:,:,:2].amax(-1)*weights).sum(1)/(
            telemetry['history_mass'].clamp_min(1.e-20))
        trust=torch.sigmoid(self.address_route_bias+self.address_route_weight*head_membership)
        extra_alpha=(.125*torch.tanh(self.address_copy_gain).clamp_min(0.)*trust*
                     (telemetry['history_mass']>0).float())
        alpha_new=torch.where(extra_alpha==0,alpha,alpha+extra_alpha)
        telemetry.update({'r747_head_features':feature,
                          'r747_attended_target_probability':head_target,
                          'r747_head_attention':changed,
                          'r747_head_mixture':gamma,
                          'r747_extra_copy_mass':extra_alpha,
                          'r747_head_membership':head_membership})
        return alpha_new,q_new,telemetry


def attach_exact_head_copy(model,optimizer,base,learning_rate=2.e-4):
    """Append exactly one group 27, preserving all 26 existing Adam moments."""
    if (type(model) is not UnifiedR744Backoff or
            type(model.word_backoff) is not GeometricWordBackoff):
        raise TypeError('R7.47 must extend the intact R7.45 unified model')
    old_ids={id(p) for p in model.parameters()}
    module=model.word_backoff
    module.__class__=HeadAddressWordBackoff
    word_heads=torch.as_tensor(base.heads,dtype=torch.long).clone()
    if (word_heads.shape[1]!=2 or bool((word_heads<0).any()) or
            bool((word_heads>=256).any())):
        raise ValueError('invalid accepted R7 vocabulary head mapping')
    module.register_buffer('word_head_lookup',word_heads)
    module.address_coefficients=nn.Parameter(torch.zeros(5))
    module.address_gain=nn.Parameter(torch.zeros(()))
    module.address_copy_gain=nn.Parameter(torch.zeros(()))
    module.address_route_bias=nn.Parameter(torch.tensor(-1.5))
    module.address_route_weight=nn.Parameter(torch.zeros(()))
    new=[(name,p) for name,p in model.named_parameters() if id(p) not in old_ids]
    expected={'word_backoff.address_coefficients','word_backoff.address_gain',
              'word_backoff.address_copy_gain','word_backoff.address_route_bias',
              'word_backoff.address_route_weight'}
    if {name for name,p in new}!=expected:
        raise AssertionError('R7.47 added unexpected trainable state')
    optimizer._name_by_id.update({id(p):name for name,p in new})
    optimizer._group_names.append({'algorithm':'adamw','names':[name for name,p in new]})
    optimizer._adamw.add_param_group({'params':[p for name,p in new],
                                     'lr':learning_rate,'weight_decay':0.})
    optimizer._source_group_counts[-1]+=1
    optimizer.param_groups=optimizer._legacy_groups+optimizer._adamw.param_groups
    actual=[id(p) for group in optimizer.param_groups for p in group['params']]
    if (len(actual)!=len(set(actual)) or set(actual)!={id(p) for p in model.parameters()}
            or not old_ids.issubset(actual) or len(optimizer.param_groups)!=27):
        raise AssertionError('R7.47 broke the unified optimizer lineage')
    return model,optimizer


@contextlib.contextmanager
def exact_head_copy_disabled(model):
    """Disable ONLY R7.47, keeping inherited R7.45 geometry intact."""
    module=model.word_backoff
    if not hasattr(module,'address_gain'):
        yield
        return
    names=('address_gain','address_copy_gain')
    old=[getattr(module,name).detach().clone() for name in names]
    with torch.no_grad():
        for name in names:getattr(module,name).zero_()
    try:
        yield
    finally:
        with torch.no_grad():
            for name,value in zip(names,old):getattr(module,name).copy_(value)
