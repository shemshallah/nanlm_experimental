#!/usr/bin/env python3
"""FA1 R7.59: exact-word-conditioned temporal geometric prior.

Bayesian exact-bigram smoothing uses either a geometric or a matched lexical
prior at identical pseudocount strength. The inherited FA1 R7.58 distribution
and its 32 optimizer groups are left intact.
"""
from __future__ import annotations
import contextlib
import numpy as np
import torch
from torch import nn
from r7_core import Corpus
from fa1_r744_causal_word_backoff import UnifiedR744Backoff
from fa1_r753_causal_successor import compose_inherited_distribution
from fa1_r757_geometric_transition_router import VOCAB
from fa1_r758_contextual_geometry import (
    ContextualGeometryBackoff, sparse_probability)

ROUND='FA1_R7_59_WORD_CONDITIONED_GEOMETRIC_PRIOR_V1'
HEADS=256
BANK_FIELDS=('conditional_keys','conditional_counts',
             'conditional_addresses','conditional_totals')
CONTROLS=('full','disabled','lexical_matched','wrong_older',
          'wrong_geometry_values','no_apex')

def conditioned_address(older,recent,heads):
    # Retain *exact recent word*; pool only earlier-word head geometry.
    return recent.long()*HEADS+heads[older.long(),1].long()

def build_conditioned_bank(root,corpus_dir,heads):
    c=Corpus(root,'train',corpus_dir=corpus_dir)
    seq=np.asarray(c.seq,dtype=np.int64)
    starts=np.asarray(c.doc_start)
    valid=((starts[:-2]==starts[1:-1])&(starts[1:-1]==starts[2:])&
           (seq[:-2]!=1)&(seq[1:-1]!=1)&(seq[2:]!=1))
    old,now,y=seq[:-2][valid],seq[1:-1][valid],seq[2:][valid]
    if not len(old):raise ValueError('TRAIN causal trigrams are missing')
    lookup=heads.detach().cpu().numpy()
    address=now*HEADS+lookup[old,1].astype(np.int64)
    keys,counts=np.unique(address*VOCAB+y,return_counts=True)
    addresses,totals=np.unique(address,return_counts=True)
    if counts.sum()!=totals.sum() or counts.sum()!=len(y):
        raise AssertionError('conditional geometric probability mass incomplete')
    return {name:torch.from_numpy(np.asarray(value,dtype=np.int64).copy())
            for name,value in dict(conditional_keys=keys,
                conditional_counts=counts,conditional_addresses=addresses,
                conditional_totals=totals).items()}

class GeometricPrior(nn.Module):
    def __init__(self,banks):
        super().__init__()
        for name in BANK_FIELDS:
            self.register_buffer(name,banks[name].clone().long())
        self.gain=nn.Parameter(torch.zeros(()))
        self.log_pseudocount=nn.Parameter(torch.tensor(np.log(4.),dtype=torch.float32))
        self.route=nn.Linear(36,1)
        nn.init.zeros_(self.route.weight)
        nn.init.zeros_(self.route.bias)

    def forward(self,older,recent,target,heads,summary,exact_bank,word_bank,
                *,control='full',valid=None,leave_one_out=False):
        if control not in CONTROLS:raise ValueError('unknown geometric prior control')
        if summary.shape!=(len(recent),32):raise ValueError('requires inherited 32D apex')
        if valid is None:valid=torch.ones_like(recent,dtype=torch.bool)
        loo=valid if leave_one_out else False
        exact_addr=older.long()*VOCAB+recent.long()
        exact_q,exact_n=sparse_probability(
            exact_bank.exact_keys,exact_bank.exact_counts,exact_addr,target,
            exact_bank.exact_totals,total_keys=exact_bank.exact_addresses,
            leave_one_out=loo)
        word_q,word_n=word_bank.lookup(
            word_bank.word_keys,word_bank.word_counts,word_bank.word_totals,
            recent.long(),target,leave_one_out=leave_one_out)
        older_geom=((older+7919)%VOCAB if control=='wrong_older' else older)
        cond_addr=conditioned_address(older_geom,recent,heads)
        geo_target=((target+7919)%VOCAB if control=='wrong_geometry_values'
                    else target)
        geo_q,geo_n=sparse_probability(
            self.conditional_keys,self.conditional_counts,cond_addr,geo_target,
            self.conditional_totals,total_keys=self.conditional_addresses,
            leave_one_out=(loo if control not in
                           ('wrong_older','wrong_geometry_values') else False))
        # Absent geometric support falls back to the same lexical conditional.
        usable=valid&(word_n>0)
        geo_prior=torch.where(geo_n>0,geo_q,word_q)
        if control=='lexical_matched':geo_prior=word_q
        k=self.log_pseudocount.exp().clamp(.5,32.)
        # Each prior is a normalized conditional distribution; exact counts
        # plus k times that prior therefore remain exactly normalized.
        smoothed=(exact_q*exact_n.float()+k*geo_prior)/(exact_n.float()+k)
        if control=='no_apex':summary=torch.zeros_like(summary)
        f=torch.cat((summary.detach(),exact_n.float().log1p()[:,None],
                     geo_n.float().log1p()[:,None],word_n.float().log1p()[:,None],
                     (geo_n.float()/exact_n.float().clamp_min(1)).log1p()[:,None]),1)
        trust=torch.sigmoid(self.route(f).squeeze(-1))
        weight=.18*torch.tanh(self.gain).clamp_min(0)*trust*usable.float()
        if control=='disabled':weight=weight*0.
        return smoothed,weight,exact_n,geo_n,word_n,k

class PriorBackoff(ContextualGeometryBackoff):
    def forward(self,data,trace,targets,embeddings=None):
        old_alpha,old_q,telemetry=super().forward(data,trace,targets,embeddings)
        history=data.get('r753_successor_history_ids',data['word_ids'])
        mask=data.get('r753_successor_history_mask')
        if mask is None:mask=history!=1
        valid=(mask[:,-2]&mask[:,-1]&(history[:,-2]!=1)&(history[:,-1]!=1))
        q,g,ec,gc,wc,k=self.r759_prior(
            history[:,-2].long(),history[:,-1].long(),targets,
            self.word_head_lookup,trace['descended_point_summary'],
            self.r758_context,self.r757_router,
            control=getattr(self,'_r759_control','full'),valid=valid,
            leave_one_out=bool(data.get('r759_leave_one_out',False)))
        a,p=compose_inherited_distribution(old_alpha,old_q,q,torch.zeros_like(q),
                                           g,torch.zeros_like(g))
        telemetry.update({'r759_geometry_gain':g.detach(),
                          'r759_exact_support':ec.detach(),
                          'r759_geometry_support':gc.detach(),
                          'r759_word_support':wc.detach(),
                          'r759_pseudocount':k.detach()})
        return a,p,telemetry

def attach_geometric_prior(model,optimizer,banks,learning_rate=.002):
    if (type(model) is not UnifiedR744Backoff or
        type(model.word_backoff) is not ContextualGeometryBackoff or
        len(optimizer.param_groups)!=32):
        raise TypeError('R7.59 requires the full R7.58 32-group model')
    original={id(p) for p in model.parameters()}
    backoff=model.word_backoff
    backoff.__class__=PriorBackoff
    backoff.r759_prior=GeometricPrior(banks)
    added=[(n,p) for n,p in model.named_parameters() if id(p) not in original]
    expected={'word_backoff.r759_prior.gain',
              'word_backoff.r759_prior.log_pseudocount',
              'word_backoff.r759_prior.route.weight',
              'word_backoff.r759_prior.route.bias'}
    if {n for n,_ in added}!=expected:raise AssertionError('R7.59 topology mismatch')
    optimizer._name_by_id.update({id(p):n for n,p in added})
    optimizer._group_names.append({'algorithm':'adamw','names':[n for n,_ in added]})
    optimizer._adamw.add_param_group({'params':[p for _,p in added],
                                      'lr':learning_rate,'weight_decay':0.})
    optimizer._source_group_counts[-1]+=1
    optimizer.param_groups=optimizer._legacy_groups+optimizer._adamw.param_groups
    ids=[id(p) for g in optimizer.param_groups for p in g['params']]
    if len(ids)!=len(set(ids)) or len(optimizer.param_groups)!=33:
        raise AssertionError('R7.59 optimizer inheritance broken')
    return model,optimizer

@contextlib.contextmanager
def prior_control(model,mode):
    if mode not in CONTROLS:raise ValueError('unknown R7.59 control')
    backoff=model.word_backoff
    old=getattr(backoff,'_r759_control','full')
    backoff._r759_control=mode
    try:yield
    finally:backoff._r759_control=old
