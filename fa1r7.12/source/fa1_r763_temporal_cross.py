#!/usr/bin/env python3
"""FA1 R7.63: oriented temporal crossings of the two 256-point head rails.

The previous geometry summarized one older word.  This module composes
head-0 of t-3 with head-1 of t-2, and the reversed crossing, while retaining
the exact word at t-1.  A learned null expert keeps lexical fallback available.
"""
from __future__ import annotations
import contextlib
import math
import numpy as np
import torch
from torch import nn
from r7_core import Corpus
from fa1_r753_causal_successor import compose_inherited_distribution
from fa1_r757_geometric_transition_router import VOCAB
from fa1_r758_contextual_geometry import sparse_probability
from fa1_r759_geometric_prior import PriorBackoff
from fa1_r762_selective_intersection import SelectiveIntersectionPrior

ROUND='FA1_R7_63_TEMPORAL_CROSS_RAIL_GEOMETRY_V1'
HEADS=256
BANK_FIELDS=('forward_keys','forward_counts','forward_addresses','forward_totals',
             'reverse_keys','reverse_counts','reverse_addresses','reverse_totals')


def build_temporal_banks(root,corpus_dir,heads):
    corpus=Corpus(root,'train',corpus_dir=corpus_dir)
    seq=np.asarray(corpus.seq,dtype=np.int64)
    start=np.asarray(corpus.doc_start)
    valid=((start[:-3]==start[1:-2]) & (start[1:-2]==start[2:-1]) &
           (start[2:-1]==start[3:]) & (seq[:-3]!=1) &
           (seq[1:-2]!=1) & (seq[2:-1]!=1) & (seq[3:]!=1))
    third,older,recent,target=(seq[:-3][valid],seq[1:-2][valid],
                               seq[2:-1][valid],seq[3:][valid])
    if len(target)==0:raise ValueError('No causal TRAIN four-word windows')
    h=heads.detach().cpu().numpy()
    forward=recent*(HEADS*HEADS)+h[third,0].astype(np.int64)*HEADS+h[older,1]
    reverse=recent*(HEADS*HEADS)+h[third,1].astype(np.int64)*HEADS+h[older,0]
    result={}
    for prefix,address in (('forward',forward),('reverse',reverse)):
        keys,counts=np.unique(address*VOCAB+target,return_counts=True)
        addresses,totals=np.unique(address,return_counts=True)
        if int(counts.sum())!=len(target) or int(totals.sum())!=len(target):
            raise AssertionError('Temporal geometric count mass lost')
        result.update({prefix+'_keys':torch.from_numpy(keys.astype(np.int64).copy()),
            prefix+'_counts':torch.from_numpy(counts.astype(np.int64).copy()),
            prefix+'_addresses':torch.from_numpy(addresses.astype(np.int64).copy()),
            prefix+'_totals':torch.from_numpy(totals.astype(np.int64).copy())})
    return result


class TemporalCrossPrior(SelectiveIntersectionPrior):
    def forward(self,older,recent,target,heads,summary,exact_bank,word_bank,
                *,control='full',valid=None,leave_one_out=False,
                third_older=None,valid_third=None):
        parent_q,g,ec,geo_n,wc,k=super().forward(
            older,recent,target,heads,summary,exact_bank,word_bank,
            control=control,valid=valid,leave_one_out=leave_one_out)
        if not getattr(self,'_r763_enabled',True):
            return parent_q,g,ec,geo_n,wc,k
        if third_older is None:
            raise ValueError('Temporal geometry requires t-3 history')
        if valid is None:valid=torch.ones_like(recent,dtype=torch.bool)
        if valid_third is None:valid_third=valid
        eligible=valid & valid_third
        use_loo=(eligible if leave_one_out and control not in
                 ('wrong_older','wrong_geometry_values') else False)
        older_geo=(older+7919)%VOCAB if control=='wrong_older' else older
        third_geo=(third_older+7919)%VOCAB if control=='wrong_older' else third_older
        value=(target+7919)%VOCAB if control=='wrong_geometry_values' else target
        h30=heads[third_geo.long(),0].long()
        h31=heads[third_geo.long(),1].long()
        h20=heads[older_geo.long(),0].long()
        h21=heads[older_geo.long(),1].long()
        base=recent.long()*(HEADS*HEADS)
        af=base+h30*HEADS+h21
        ar=base+h31*HEADS+h20
        qf,nf=sparse_probability(self.forward_keys,self.forward_counts,af,value,
            self.forward_totals,total_keys=self.forward_addresses,
            leave_one_out=use_loo)
        qr,nr=sparse_probability(self.reverse_keys,self.reverse_counts,ar,value,
            self.reverse_totals,total_keys=self.reverse_addresses,
            leave_one_out=use_loo)
        word_q,_=word_bank.lookup(word_bank.word_keys,word_bank.word_counts,
            word_bank.word_totals,recent.long(),target,
            leave_one_out=leave_one_out)
        exact_addr=older.long()*VOCAB+recent.long()
        exact_q,_=sparse_probability(exact_bank.exact_keys,exact_bank.exact_counts,
            exact_addr,target,exact_bank.exact_totals,
            total_keys=exact_bank.exact_addresses,
            leave_one_out=valid if leave_one_out else False)
        counts=torch.stack((nf.float(),nr.float()),-1)
        probs=torch.stack((qf,qr),-1)
        tau=self.r763_log_tau.exp().clamp(1.,512.)
        shrunk=(counts*probs+tau*word_q.unsqueeze(-1))/(counts+tau)
        features=torch.stack((nf.float().log1p(),nr.float().log1p(),
            ec.float().log1p(),wc.float().log1p(),(h30==h21).float(),
            (h31==h20).float()),-1)
        selector=torch.softmax(self.r763_selector(features),-1)
        prior=(selector[:,:2]*shrunk).sum(-1)+selector[:,2]*word_q
        if control=='lexical_matched':prior=word_q
        smoothed=(exact_q*ec.float()+k*prior)/(ec.float()+k)
        mix=torch.tanh(self.r763_strength).clamp_min(0.)*eligible.float()
        if control=='disabled':mix=mix*0.
        return parent_q+mix*(smoothed-parent_q),g,ec,geo_n,wc,k


class TemporalPriorBackoff(PriorBackoff):
    def forward(self,data,trace,targets,embeddings=None):
        old_alpha,old_q,telemetry=super(PriorBackoff,self).forward(
            data,trace,targets,embeddings)
        history=data.get('r753_successor_history_ids',data['word_ids'])
        mask=data.get('r753_successor_history_mask')
        if mask is None:mask=history!=1
        valid2=(mask[:,-2]&mask[:,-1]&
                (history[:,-2]!=1)&(history[:,-1]!=1))
        valid3=(valid2&mask[:,-3]&(history[:,-3]!=1))
        q,g,ec,gc,wc,k=self.r759_prior(
            history[:,-2].long(),history[:,-1].long(),targets,
            self.word_head_lookup,trace['descended_point_summary'],
            self.r758_context,self.r757_router,
            control=getattr(self,'_r759_control','full'),valid=valid2,
            third_older=history[:,-3].long(),valid_third=valid3,
            leave_one_out=bool(data.get('r759_leave_one_out',False)))
        a,p=compose_inherited_distribution(old_alpha,old_q,q,
            torch.zeros_like(q),g,torch.zeros_like(g))
        telemetry.update({'r759_geometry_gain':g.detach(),
            'r759_exact_support':ec.detach(),'r759_geometry_support':gc.detach(),
            'r759_word_support':wc.detach(),'r759_pseudocount':k.detach()})
        return a,p,telemetry


def attach_temporal_cross(model,optimizer,banks,learning_rate=.002):
    if (type(model.word_backoff) is not PriorBackoff or
        type(model.word_backoff.r759_prior) is not SelectiveIntersectionPrior or
        len(optimizer.param_groups)!=35):
        raise TypeError('R7.63 requires the full R7.62 35-group checkpoint')
    old={id(p) for p in model.parameters()}
    model.word_backoff.__class__=TemporalPriorBackoff
    prior=model.word_backoff.r759_prior
    prior.__class__=TemporalCrossPrior
    for name in BANK_FIELDS:
        prior.register_buffer(name,banks[name].clone().long())
    prior.r763_strength=nn.Parameter(torch.zeros(()))
    prior.r763_log_tau=nn.Parameter(torch.tensor(
        [math.log(64.),math.log(64.)],dtype=torch.float32))
    prior.r763_selector=nn.Linear(6,3)
    nn.init.zeros_(prior.r763_selector.weight)
    with torch.no_grad():
        prior.r763_selector.bias.copy_(torch.tensor([0.,0.,2.]))
    added=[(n,p) for n,p in model.named_parameters() if id(p) not in old]
    expected={'word_backoff.r759_prior.r763_strength',
              'word_backoff.r759_prior.r763_log_tau',
              'word_backoff.r759_prior.r763_selector.weight',
              'word_backoff.r759_prior.r763_selector.bias'}
    if {n for n,_ in added}!=expected:
        raise AssertionError('R7.63 must append four trainable tensors')
    optimizer._name_by_id.update({id(p):n for n,p in added})
    optimizer._group_names.append({'algorithm':'adamw','names':[n for n,_ in added]})
    optimizer._adamw.add_param_group({'params':[p for _,p in added],
                                      'lr':learning_rate,'weight_decay':0.})
    optimizer._source_group_counts[-1]+=1
    optimizer.param_groups=optimizer._legacy_groups+optimizer._adamw.param_groups
    if len(optimizer.param_groups)!=36:
        raise AssertionError('R7.63 must have 36 optimizer groups')
    return model,optimizer


@contextlib.contextmanager
def temporal_control(model,enabled):
    prior=model.word_backoff.r759_prior
    old=getattr(prior,'_r763_enabled',True)
    prior._r763_enabled=enabled
    try:yield
    finally:prior._r763_enabled=old
