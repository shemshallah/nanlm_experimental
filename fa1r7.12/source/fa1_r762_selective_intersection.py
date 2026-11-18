#!/usr/bin/env python3
"""R7.62: trainable, normalized selection over the two signed head rails.

Unlike the support-weighted average in R7.61, every rail has an independent
Dirichlet-style lexical shrinkage and a learned selector.  The fourth selector
output is a true lexical-only null expert.  The exact two-rail intersection
is a distinct expert, not counted twice through marginal support.
"""
from __future__ import annotations
import contextlib
import math
import torch
from torch import nn
from fa1_r757_geometric_transition_router import VOCAB
from fa1_r758_contextual_geometry import sparse_probability
from fa1_r759_geometric_prior import PriorBackoff,conditioned_address
from fa1_r760_geometric_calibration import CalibratedGeometricPrior
from fa1_r761_two_rail_geometry import BANK_FIELDS,HEADS,build_two_rail_bank

ROUND='FA1_R7_62_SELECTIVE_GEOMETRIC_INTERSECTION_V1'


class SelectiveIntersectionPrior(CalibratedGeometricPrior):
    """One model, one trained continuation; no parallel replacement predictor."""
    def forward(self,older,recent,target,heads,summary,exact_bank,word_bank,
                *,control='full',valid=None,leave_one_out=False):
        parent_q,g,ec,geo_n,wc,k=super().forward(
            older,recent,target,heads,summary,exact_bank,word_bank,
            control=control,valid=valid,leave_one_out=leave_one_out)
        if not getattr(self,'_r762_enabled',True):
            return parent_q,g,ec,geo_n,wc,k
        if valid is None:
            valid=torch.ones_like(recent,dtype=torch.bool)
        loo=valid if leave_one_out and control not in (
            'wrong_older','wrong_geometry_values') else False
        older_geo=(older+7919)%VOCAB if control=='wrong_older' else older
        corrupted=(target+7919)%VOCAB if control=='wrong_geometry_values' else target
        h0=heads[older_geo.long(),0].long()
        h1=heads[older_geo.long(),1].long()
        a0=recent.long()*HEADS+h0
        a1=conditioned_address(older_geo,recent,heads)
        ac=recent.long()*(HEADS*HEADS)+h0*HEADS+h1
        q0,n0=sparse_probability(self.rail0_keys,self.rail0_counts,a0,corrupted,
            self.rail0_totals,total_keys=self.rail0_addresses,leave_one_out=loo)
        q1,n1=sparse_probability(self.conditional_keys,self.conditional_counts,
            a1,corrupted,self.conditional_totals,
            total_keys=self.conditional_addresses,leave_one_out=loo)
        qc,nc=sparse_probability(self.cross_keys,self.cross_counts,ac,corrupted,
            self.cross_totals,total_keys=self.cross_addresses,leave_one_out=loo)
        word_q,_=word_bank.lookup(word_bank.word_keys,word_bank.word_counts,
            word_bank.word_totals,recent.long(),target,
            leave_one_out=leave_one_out)
        exact_addr=older.long()*VOCAB+recent.long()
        exact_q,_=sparse_probability(exact_bank.exact_keys,exact_bank.exact_counts,
            exact_addr,target,exact_bank.exact_totals,
            total_keys=exact_bank.exact_addresses,
            leave_one_out=valid if leave_one_out else False)
        counts=torch.stack((n0.float(),n1.float(),nc.float()),dim=-1)
        probabilities=torch.stack((q0,q1,qc),dim=-1)
        tau=self.r762_log_tau.exp().clamp(1.,512.)
        shrunk=(counts*probabilities+tau*word_q.unsqueeze(-1))/(counts+tau)
        features=torch.stack((n0.float().log1p(),n1.float().log1p(),
            nc.float().log1p(),ec.float().log1p(),wc.float().log1p()),dim=-1)
        selectors=torch.softmax(self.r762_selector(features),dim=-1)
        prior=(selectors[:,:3]*shrunk).sum(-1)+selectors[:,3]*word_q
        if control=='lexical_matched':
            prior=word_q
        smoothed=(exact_q*ec.float()+k*prior)/(ec.float()+k)
        strength=torch.tanh(self.r762_strength).clamp_min(0.)
        mix=strength*valid.float()
        if control=='disabled':
            mix=mix*0.
        return parent_q+mix*(smoothed-parent_q),g,ec,geo_n,wc,k


def attach_selective_intersection(model,optimizer,banks,learning_rate=.002):
    backoff=model.word_backoff
    if (type(backoff) is not PriorBackoff or
            type(backoff.r759_prior) is not CalibratedGeometricPrior or
            len(optimizer.param_groups)!=34):
        raise TypeError('R7.62 requires intact R7.60, exactly 34 groups')
    inherited={id(p) for p in model.parameters()}
    prior=backoff.r759_prior
    prior.__class__=SelectiveIntersectionPrior
    for name in BANK_FIELDS:
        prior.register_buffer(name,banks[name].clone().long())
    prior.r762_strength=nn.Parameter(torch.zeros(()))
    prior.r762_log_tau=nn.Parameter(torch.tensor(
        [math.log(32.),math.log(32.),math.log(64.)],dtype=torch.float32))
    prior.r762_selector=nn.Linear(5,4)
    nn.init.zeros_(prior.r762_selector.weight)
    with torch.no_grad():
        prior.r762_selector.bias.copy_(torch.tensor([0.,0.,0.,2.]))
    added=[(n,p) for n,p in model.named_parameters() if id(p) not in inherited]
    expected={'word_backoff.r759_prior.r762_strength',
              'word_backoff.r759_prior.r762_log_tau',
              'word_backoff.r759_prior.r762_selector.weight',
              'word_backoff.r759_prior.r762_selector.bias'}
    if {n for n,_ in added}!=expected:
        raise AssertionError('R7.62 must append four new tensors')
    optimizer._name_by_id.update({id(p):n for n,p in added})
    optimizer._group_names.append({'algorithm':'adamw','names':[n for n,_ in added]})
    optimizer._adamw.add_param_group({'params':[p for _,p in added],
                                      'lr':learning_rate,'weight_decay':0.})
    optimizer._source_group_counts[-1]+=1
    optimizer.param_groups=optimizer._legacy_groups+optimizer._adamw.param_groups
    if len(optimizer.param_groups)!=35:
        raise AssertionError('R7.62 must have 35 total optimizer groups')
    return model,optimizer


@contextlib.contextmanager
def intersection_control(model,enabled):
    prior=model.word_backoff.r759_prior
    old=getattr(prior,'_r762_enabled',True)
    prior._r762_enabled=enabled
    try:
        yield
    finally:
        prior._r762_enabled=old
