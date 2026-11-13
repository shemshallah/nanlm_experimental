#!/usr/bin/env python3
"""FA1 R7.22: one R7 model, one checkpoint, five inherited optimizers.

R7.22 extends R7.21's single unified architecture in-place by adding a
zero-contribution causal embedding-associative memory reader. All original
parameters and all five Adam histories are loaded, not reinitialized. The
new memory submodule uses one added group inside the existing fifth optimizer.
No R7 Git branches, no separate trained model, no automatic promotion.
"""
from __future__ import annotations
import argparse, copy, json, pathlib
import numpy as np
import torch
from torch import nn
from fa1_r721_unified import (UnifiedR7, blank, load_candidate as load_r721,
                                  _safe_save, PARENT, STEP)
from fa1_r722_semantic import CausalSemanticIndex, SemanticGate, EXCLUSION
from fa1_r720_runtime import FA1R720Runtime
from r710_checkpoint_integrity import deep_equal, sha256
from r75_retrieval import OPT_NAMES

FORMAT='FA1_R7_22_SINGLE_UNIFIED_CONTEXTUAL_MEMORY_V1'
PARENT_R721='7d2ff0c17430075e3fc6ddeb02f61bf447c07eed027ae77afac0506737734b1f'

class UnifiedR722(UnifiedR7):
    """Extend the SAME R7 model with the zero-init embedding memory head."""
    def __init__(self,core,calibration,near,far,semantic=None):
        super().__init__(core,calibration,near,far)
        self.semantic=semantic if semantic is not None else SemanticGate()
        self.long_semantic=SemanticGate()

    def joint_nll(self,data,near_enabled=True,far_enabled=True,semantic_enabled=True):
        final,close=super().joint_nll(data,near_enabled,far_enabled)
        if semantic_enabled and 'semantic_features' in data:
            final,_=self.semantic(final,data['semantic_features'],
                         data['semantic_prob'],data['semantic_count'])
        if semantic_enabled and 'long_semantic_features' in data:
            final,_=self.long_semantic(final,data['long_semantic_features'],
                         data['long_semantic_prob'],data['long_semantic_count'])
        return final,close

    def runtime(self,root,base,comp,split='test',corpus_dir=None):
        return R722Runtime(self,root,base,comp,split,corpus_dir)

class R722Runtime:
    """One integrated R7 inference interface; semantic path uses same causal corpus."""
    def __init__(self,model,root,base,comp,split='test',corpus_dir=None):
        self.model=model
        self.exact=FA1R720Runtime(root,model.core,base,comp,model.near,model.far,
                                  split=split,corpus_dir=corpus_dir)
        self.corpus=self.exact.corpus
        self.emb=base.E
        self.semantic_index=None
        self.long_index=None
        self.semantic_resets=0
        self.long_resets=0

    @torch.no_grad()
    def score_positions(self,positions,mode='auto',batch_size=16,return_details=False,probe_memory=False):
        positions=np.asarray(positions,dtype=np.int64)
        base=self.exact.score_positions(positions,mode=mode,batch_size=batch_size,
                                         return_details=return_details)
        short_on=probe_memory or float(self.model.semantic.open_fraction.detach())>0.
        long_on=probe_memory or float(self.model.long_semantic.open_fraction.detach())>0.
        # Pay no 128K/1M semantic indexing cost until a train-selected gate is
        # actually open. Probes opt in explicitly; inference stays CPU-cheap.
        if not short_on and not long_on:
            z=[0.]*len(positions)
            base.update({'semantic_nll':base['far_nll'].copy(),
                         'long_semantic_nll':base['far_nll'].copy(),
                         'semantic_gate':z.copy(),'long_semantic_gate':z.copy(),
                         'semantic_eligible_positions':0,'long_semantic_eligible_positions':0,
                         'semantic_resets':0,'long_semantic_resets':0,
                         'r7_architecture':'R7.22 single unified',
                         'semantic_indices_skipped_in_closed_gate':True})
            return base
        features=[];probs=[];counts=[];details=[]
        longfeatures=[];longprobs=[];longcounts=[]
        for t in positions:
            t=int(t)
            doc=0 if self.corpus.doc_start is None else int(self.corpus.doc_start[t])
            if short_on:
                if (self.semantic_index is None or self.semantic_index.document_start!=doc
                    or t<self.semantic_index.position or t-self.semantic_index.position>1024):
                    self.semantic_index=CausalSemanticIndex(self.corpus.seq,self.emb,t,document_start=doc)
                    self.semantic_resets+=1
                else:self.semantic_index.advance(t)
                rec=self.semantic_index.query(target=int(self.corpus.seq[t]))
            else:rec={'feature':np.zeros(16,dtype=np.float32),'n':0,
                      'target_probability':0.,'max_cosine':0.}
            features.append(rec['feature']);probs.append(rec['target_probability']);counts.append(rec['n'])
            if long_on:
                if (self.long_index is None or self.long_index.document_start!=doc
                    or t<self.long_index.position or t-self.long_index.position>1024):
                    self.long_index=CausalSemanticIndex(self.corpus.seq,self.emb,t,
                           document_start=doc,stride=32,horizon=1048576,exclusion=131072)
                    self.long_resets+=1
                else:self.long_index.advance(t)
                longrec=self.long_index.query(target=int(self.corpus.seq[t]))
            else:longrec={'feature':np.zeros(16,dtype=np.float32),'n':0,
                          'target_probability':0.,'max_cosine':0.}
            longfeatures.append(longrec['feature']);longprobs.append(longrec['target_probability']);longcounts.append(longrec['n'])
            if return_details:details.append({'position':t,'semantic_count':rec['n'],
                                'semantic_max_cosine':rec['max_cosine'],
                                'long_semantic_count':longrec['n']})
        f=torch.from_numpy(np.asarray(features,dtype=np.float32))
        p=torch.tensor(probs,dtype=torch.float32)
        n=torch.tensor(counts,dtype=torch.float32)
        prior=torch.tensor(base['far_nll'],dtype=torch.float32)
        semantic,weights=self.model.semantic(prior,f,p,n)
        long_prior=semantic
        longsem,longweights=self.model.long_semantic(long_prior,
              torch.from_numpy(np.asarray(longfeatures,dtype=np.float32)),
              torch.tensor(longprobs,dtype=torch.float32),
              torch.tensor(longcounts,dtype=torch.float32))
        base.update({'semantic_nll':semantic.tolist(),'semantic_gate':weights.tolist(),
                     'long_semantic_nll':longsem.tolist(),'long_semantic_gate':longweights.tolist(),
                     'long_semantic_eligible_positions':int(np.count_nonzero(np.asarray(longcounts)>0)),
                     'long_semantic_resets':self.long_resets,
                     'semantic_eligible_positions':int(np.count_nonzero(np.asarray(counts)>0)),
                     'semantic_resets':self.semantic_resets,'r7_architecture':'R7.22 single unified'})
        if return_details:base['semantic_details']=details
        return base


def _assert_lineage(parent_blob):
    if parent_blob['candidate_joint_adapter_updates']!=0:
        raise ValueError('R7.22 requires selected identity migration; no pilot weights')
    if parent_blob.get('promotion') or parent_blob['accepted_joint_steps']!=STEP:
        raise ValueError('accepted R7 ancestry cannot be changed')


def extend_r721(r721_path,root,threads=2):
    r721_path=pathlib.Path(r721_path)
    if sha256(r721_path)!=PARENT_R721:raise ValueError('R7.21 seed SHA mismatch')
    old,opts,base,comp,blob=load_r721(r721_path,root,threads)
    _assert_lineage(blob)
    oldweights={k:v.detach().clone() for k,v in old.state_dict().items()}
    oldopts=[copy.deepcopy(opt.state_dict()) for opt in opts]
    upgraded=UnifiedR722(old.core,old.calibration,old.near,old.far)
    opts[4].add_param_group({'params':list(upgraded.semantic.parameters())+list(upgraded.long_semantic.parameters()),
                             'lr':5e-4,'weight_decay':0.0})
    for name,v in oldweights.items():
        assert torch.equal(upgraded.state_dict()[name],v),name
    for j in range(4):assert deep_equal(oldopts[j],opts[j].state_dict())
    current=opts[4].state_dict()
    assert len(current['param_groups'])==len(oldopts[4]['param_groups'])+1
    assert deep_equal(current['param_groups'][:-1],oldopts[4]['param_groups'])
    assert deep_equal(current['state'],oldopts[4]['state'])
    upgraded.eval()
    return upgraded,opts,base,comp,blob


def save_unified(path,model,opts,parent_path,train_updates=0,selection=None):
    if len(opts)!=5 or len(opts[4].param_groups)!=7:raise ValueError('R7 unified optimizer topology changed')
    # A one-time R7.21 -> R7.22 identity migration may use the verified
    # ancestor. All later R7.22 checkpoints MUST be able to continue from
    # their latest complete R7.22 checkpoint alone, without R7.21 on disk.
    direct_parent_digest=sha256(parent_path)
    if direct_parent_digest!=PARENT_R721:
        predecessor=torch.load(parent_path,map_location='cpu',weights_only=False)
        if (predecessor.get('format')!=FORMAT or
            predecessor.get('accepted_parent_sha256')!=PARENT or
            predecessor.get('r721_ancestor_sha256')!=PARENT_R721 or
            predecessor.get('optimizer_count')!=5 or predecessor.get('promotion')):
            raise ValueError('untrusted R7.22 continuation parent')
    state={
      'format':FORMAT,'lineage':'FA1 R7 one model on master',
      'accepted_parent_sha256':PARENT,'accepted_step':STEP,
      'r721_ancestor_sha256':PARENT_R721,
      'direct_parent_sha256':direct_parent_digest,
      'model':model.state_dict(), 'optimizer_count':5,
      'optimizer_names':list(OPT_NAMES),
      'optimizer_states':[o.state_dict() for o in opts],
      'semantic_updates':int(train_updates),'promotion':False,
      'selection':selection or {'status':'identity migration','new_books':False},
      'semantic_index':{'horizon':131072,'exclusion':EXCLUSION,'stride':8,
                        'hash_bits':10,'causal':True,'target_free_routing':True,
                        'long_horizon':1048576,'long_exclusion':131072,'long_stride':32},
      'rng':{'torch':torch.get_rng_state(),'numpy':np.random.get_state()},
      'architecture':'R7 96D trunk + directed rank64 + near 8K/32K + packed far 128K + 128K embedding memory + 1M embedding memory (zero-gated)'}
    return {'sha256':_safe_save(path,state),'bytes':pathlib.Path(path).stat().st_size,
            'parameter_tensors':len(state['model']),'optimizer_histories':5,
            'optimizer_fifth_groups':len(opts[4].param_groups),'promoted':False}


def load_unified(path,root,threads=2):
    obj=torch.load(path,map_location='cpu',weights_only=False)
    if obj.get('format')!=FORMAT or obj.get('r721_ancestor_sha256')!=PARENT_R721:
        raise ValueError('wrong R7.22 lineage')
    if obj.get('promotion') or obj.get('accepted_step')!=STEP:raise ValueError('unexpected promotion')
    if obj.get('optimizer_count')!=5 or obj.get('optimizer_names')!=list(OPT_NAMES):
        raise ValueError('R7 optimizer count/order changed')
    old,opts,base,comp=blank(root,threads,embedded_state=obj['model'])
    model=UnifiedR722(old.core,old.calibration,old.near,old.far)
    opts[4].add_param_group({'params':list(model.semantic.parameters())+list(model.long_semantic.parameters()),
                             'lr':5e-4,'weight_decay':0.0})
    model.load_state_dict(obj['model'],strict=True)
    for opt,state in zip(opts,obj['optimizer_states']):opt.load_state_dict(state)
    if len(opts)!=5 or len(opts[-1].param_groups)!=7:raise AssertionError('R7 optimizer ancestry failed')
    model.eval()
    return model,opts,base,comp,obj


def memory_trainable(model,semantic_only=True):
    for p in model.core.parameters():p.requires_grad_(False)
    for p in model.calibration.parameters():p.requires_grad_(False)
    for p in model.near.parameters():p.requires_grad_(not semantic_only)
    for p in model.far.parameters():p.requires_grad_(not semantic_only)
    for p in model.semantic.parameters():p.requires_grad_(True)
    for p in model.long_semantic.parameters():p.requires_grad_(True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',default='/mnt/data/fa1_full')
    p.add_argument('--ancestor',default='/mnt/data/fa1r721/checkpoint/r7_unified_candidate.pt')
    p.add_argument('--output',default='/mnt/data/fa1r722/checkpoint/r7_unified_candidate.pt')
    p.add_argument('--reload',action='store_true')
    a=p.parse_args()
    if a.reload:
        model,opts,_,_,obj=load_unified(a.output,a.root)
        report={'reloaded':True,'parameter_tensors':len(model.state_dict()),
                'optimizer_histories':len(opts),'semantic_updates':obj['semantic_updates']}
    else:
        model,opts,_,_,blob=extend_r721(a.ancestor,a.root)
        report=save_unified(a.output,model,opts,a.ancestor)
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()