#!/usr/bin/env python3
"""FA1 R7.61: causal hierarchical old-word geometry from both 256-head rails.

At each exact recent-word address, estimate next-word counts conditioned on
old head0, old head1, and their intersection. Hierarchical backoff to the
matched lexical prior avoids asserting certainty on unsupported addresses.
All new tables are TRAIN-only; no DEV or TEST tokens are stored.
"""
from __future__ import annotations
import contextlib
import numpy as np
import torch
from torch import nn
from r7_core import Corpus
from fa1_r757_geometric_transition_router import VOCAB
from fa1_r758_contextual_geometry import sparse_probability
from fa1_r760_geometric_calibration import CalibratedGeometricPrior
from fa1_r759_geometric_prior import conditioned_address, PriorBackoff

ROUND='FA1_R7_61_TWO_RAIL_HIERARCHICAL_GEOMETRY_V1'
HEADS=256
BANK_FIELDS=('rail0_keys','rail0_counts','rail0_addresses','rail0_totals',
             'cross_keys','cross_counts','cross_addresses','cross_totals')


def build_two_rail_bank(root,corpus_dir,heads):
    corpus=Corpus(root,'train',corpus_dir=corpus_dir)
    seq=np.asarray(corpus.seq,dtype=np.int64)
    starts=np.asarray(corpus.doc_start)
    valid=(starts[:-2]==starts[1:-1])&(starts[1:-1]==starts[2:])&\
          (seq[:-2]!=1)&(seq[1:-1]!=1)&(seq[2:]!=1)
    old,now,y=seq[:-2][valid],seq[1:-1][valid],seq[2:][valid]
    if len(old)==0:raise ValueError('no causal TRAIN trigrams')
    head=heads.detach().cpu().numpy()
    first=head[old,0].astype(np.int64)
    second=head[old,1].astype(np.int64)
    addresses={'rail0':now*HEADS+first,
               'cross':now*(HEADS*HEADS)+first*HEADS+second}
    result={}
    for prefix,addr in addresses.items():
        keys,counts=np.unique(addr*VOCAB+y,return_counts=True)
        unique,totals=np.unique(addr,return_counts=True)
        if int(counts.sum())!=len(y) or int(totals.sum())!=len(y):
            raise AssertionError('two-rail probability mass incomplete')
        result.update({prefix+'_keys':torch.from_numpy(keys.copy()),
            prefix+'_counts':torch.from_numpy(counts.astype(np.int64).copy()),
            prefix+'_addresses':torch.from_numpy(unique.copy()),
            prefix+'_totals':torch.from_numpy(totals.astype(np.int64).copy())})
    return result


class TwoRailGeometricPrior(CalibratedGeometricPrior):
    def forward(self,older,recent,target,heads,summary,exact_bank,word_bank,
                *,control='full',valid=None,leave_one_out=False):
        parent_q,g,ec,geo_n,wc,k=super().forward(
            older,recent,target,heads,summary,exact_bank,word_bank,
            control=control,valid=valid,leave_one_out=leave_one_out)
        if (not getattr(self,'_r761_enabled',True) or
                (not torch.is_grad_enabled() and self.r761_strength.item()==0.)):
            return parent_q,g,ec,geo_n,wc,k
        if valid is None:valid=torch.ones_like(recent,dtype=torch.bool)
        use_loo=(valid if leave_one_out and control not in
                 ('wrong_older','wrong_geometry_values') else False)
        older_geo=((older+7919)%VOCAB if control=='wrong_older' else older)
        value=((target+7919)%VOCAB if control=='wrong_geometry_values' else target)
        a0=recent.long()*HEADS+heads[older_geo.long(),0].long()
        a1=conditioned_address(older_geo,recent,heads)
        ac=recent.long()*(HEADS*HEADS)+heads[older_geo.long(),0].long()*HEADS+\
           heads[older_geo.long(),1].long()
        q0,n0=sparse_probability(self.rail0_keys,self.rail0_counts,a0,value,
            self.rail0_totals,total_keys=self.rail0_addresses,
            leave_one_out=use_loo)
        q1,n1=sparse_probability(self.conditional_keys,self.conditional_counts,
            a1,value,self.conditional_totals,
            total_keys=self.conditional_addresses,leave_one_out=use_loo)
        qc,nc=sparse_probability(self.cross_keys,self.cross_counts,ac,value,
            self.cross_totals,total_keys=self.cross_addresses,
            leave_one_out=use_loo)
        word_q,_=word_bank.lookup(word_bank.word_keys,word_bank.word_counts,
            word_bank.word_totals,recent.long(),target,
            leave_one_out=leave_one_out)
        exact_addr=older.long()*VOCAB+recent.long()
        exact_q,_=sparse_probability(exact_bank.exact_keys,exact_bank.exact_counts,
            exact_addr,target,exact_bank.exact_totals,
            total_keys=exact_bank.exact_addresses,
            leave_one_out=valid if leave_one_out else False)
        support=n0.float()+n1.float()+2.*nc.float()
        pooled=(n0.float()*q0+n1.float()*q1+2.*nc.float()*qc)/support.clamp_min(1.)
        tau=self.r761_log_tau.exp().clamp(1.,128.)
        reliability=support/(support+tau)
        prior=word_q+reliability*(pooled-word_q)
        if control=='lexical_matched':prior=word_q
        smooth=(exact_q*ec.float()+k*prior)/(ec.float()+k)
        features=torch.stack((n0.float().log1p(),n1.float().log1p(),
            nc.float().log1p(),ec.float().log1p(),wc.float().log1p()),1)
        select=torch.sigmoid(self.r761_selector(features)).squeeze(-1)
        mix=torch.tanh(self.r761_strength).clamp_min(0.)*select*valid.float()
        if control=='disabled':mix=mix*0.
        return parent_q+mix*(smooth-parent_q),g,ec,geo_n,wc,k


def attach_two_rail_geometry(model,optimizer,banks,learning_rate=.002):
    backoff=model.word_backoff
    if (type(backoff) is not PriorBackoff or
            type(backoff.r759_prior) is not CalibratedGeometricPrior or
            len(optimizer.param_groups)!=34):
        raise TypeError('R7.61 requires intact 34-group R7.60 checkpoint')
    original={id(p) for p in model.parameters()}
    prior=backoff.r759_prior
    prior.__class__=TwoRailGeometricPrior
    for name in BANK_FIELDS:
        prior.register_buffer(name,banks[name].clone().long())
    prior.r761_strength=nn.Parameter(torch.zeros(()))
    prior.r761_log_tau=nn.Parameter(torch.tensor(np.log(32.),dtype=torch.float32))
    prior.r761_selector=nn.Linear(5,1)
    nn.init.zeros_(prior.r761_selector.weight)
    nn.init.zeros_(prior.r761_selector.bias)
    added=[(n,p) for n,p in model.named_parameters() if id(p) not in original]
    expected={'word_backoff.r759_prior.r761_strength',
        'word_backoff.r759_prior.r761_log_tau',
        'word_backoff.r759_prior.r761_selector.weight',
        'word_backoff.r759_prior.r761_selector.bias'}
    if {n for n,_ in added}!=expected:raise AssertionError('four new tensors required')
    optimizer._name_by_id.update({id(p):n for n,p in added})
    optimizer._group_names.append({'algorithm':'adamw','names':[n for n,_ in added]})
    optimizer._adamw.add_param_group({'params':[p for _,p in added],
                                      'lr':learning_rate,'weight_decay':0.})
    optimizer._source_group_counts[-1]+=1
    optimizer.param_groups=optimizer._legacy_groups+optimizer._adamw.param_groups
    if len(optimizer.param_groups)!=35:raise AssertionError('35 groups required')
    return model,optimizer


@contextlib.contextmanager
def hierarchy_control(model,enabled):
    prior=model.word_backoff.r759_prior
    old=getattr(prior,'_r761_enabled',True)
    prior._r761_enabled=enabled
    try:yield
    finally:prior._r761_enabled=old
