#!/usr/bin/env python3
"""R7.60 adaptive shrinkage of R7.59 geometric evidence toward its lexical match.

The inherited R7.59 conditional bank and every earlier optimizer state stay
unchanged. The calibrator may suppress *only* the geometric conditional; it
cannot make the unsmoothed geometric prior more extreme. Its features are
causal supports plus the inherited 32D apex, never the target identity.
"""
from __future__ import annotations

import contextlib
import torch
from torch import nn
from fa1_r759_geometric_prior import (
    GeometricPrior, PriorBackoff, ROUND as PARENT, CONTROLS,
)

ROUND='FA1_R7_60_SUPPORT_CALIBRATED_GEOMETRY_V1'


class CalibratedGeometricPrior(GeometricPrior):
    def forward(self,older,recent,target,heads,summary,exact_bank,word_bank,
                *,control='full',valid=None,leave_one_out=False):
        old_q,g,exact_n,geo_n,word_n,k=super().forward(
            older,recent,target,heads,summary,exact_bank,word_bank,
            control=control,valid=valid,leave_one_out=leave_one_out)
        if (control=='lexical_matched' or not getattr(self,'_r760_enabled',True)
                or (not torch.is_grad_enabled() and self.r760_shrinkage.item()==0.)):
            # For the uncalibrated arm, reproduce the full R7.59 forward.
            return old_q,g,exact_n,geo_n,word_n,k
        word_q,_=word_bank.lookup(
            word_bank.word_keys,word_bank.word_counts,word_bank.word_totals,
            recent.long(),target,leave_one_out=leave_one_out)
        from fa1_r758_contextual_geometry import sparse_probability
        if valid is None: valid=torch.ones_like(recent,dtype=torch.bool)
        # Reproduce exact transition probabilities using inherited R7.58 bank.
        from fa1_r757_geometric_transition_router import VOCAB
        exact_addr=older.long()*VOCAB+recent.long()
        exact_q,_=sparse_probability(
            exact_bank.exact_keys,exact_bank.exact_counts,exact_addr,target,
            exact_bank.exact_totals,total_keys=exact_bank.exact_addresses,
            leave_one_out=valid if leave_one_out else False)
        lexical=(exact_q*exact_n.float()+k*word_q)/(exact_n.float()+k)
        features=torch.cat((summary.detach(),
            exact_n.float().log1p()[:,None],geo_n.float().log1p()[:,None],
            word_n.float().log1p()[:,None],
            (geo_n.float()/(word_n.float()+1.)).log1p()[:,None],
            (geo_n.float()/(exact_n.float()+1.)).log1p()[:,None]),1)
        if control=='no_apex':features=features.clone();features[:,:32]=0.
        selection=torch.sigmoid(self.r760_selector(features)).squeeze(-1)
        mix=(1.-self.r760_shrinkage*selection).clamp(0.,1.)
        calibrated=lexical+mix*(old_q-lexical)
        return calibrated,g,exact_n,geo_n,word_n,k


def attach_geometric_calibration(model,optimizer,learning_rate=.002):
    backoff=model.word_backoff
    if (type(backoff) is not PriorBackoff or
            type(backoff.r759_prior) is not GeometricPrior or
            len(optimizer.param_groups)!=33):
        raise TypeError('R7.60 needs actual R7.59 checkpoint and 33 groups')
    original={id(p) for p in model.parameters()}
    prior=backoff.r759_prior
    prior.__class__=CalibratedGeometricPrior
    prior.r760_shrinkage=nn.Parameter(torch.zeros(()))
    prior.r760_selector=nn.Linear(37,1)
    nn.init.zeros_(prior.r760_selector.weight)
    nn.init.zeros_(prior.r760_selector.bias)
    added=[(n,p) for n,p in model.named_parameters() if id(p) not in original]
    expected={'word_backoff.r759_prior.r760_shrinkage',
              'word_backoff.r759_prior.r760_selector.weight',
              'word_backoff.r759_prior.r760_selector.bias'}
    if {n for n,_ in added}!=expected:raise AssertionError('wrong R7.60 topology')
    optimizer._name_by_id.update({id(p):n for n,p in added})
    optimizer._group_names.append({'algorithm':'adamw','names':[n for n,_ in added]})
    optimizer._adamw.add_param_group({'params':[p for _,p in added],
                                       'lr':learning_rate,'weight_decay':0.})
    optimizer._source_group_counts[-1]+=1
    optimizer.param_groups=optimizer._legacy_groups+optimizer._adamw.param_groups
    if len(optimizer.param_groups)!=34:raise AssertionError('34 groups required')
    return model,optimizer


@contextlib.contextmanager
def calibration_control(model,enabled):
    prior=model.word_backoff.r759_prior
    old=getattr(prior,'_r760_enabled',True)
    prior._r760_enabled=enabled
    try:yield
    finally:prior._r760_enabled=old
