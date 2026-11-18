#!/usr/bin/env python3
"""TRAIN-only R71 continuation from R69. Keep one model and all 198 AdamW states."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fa2_active import load_active_fa2
from poincare_frontier import file_hash
from train_r55_pair_successor import checked_corpus
from train_r69_future_book import cache_future_book


def continue_r69(model,optimizer,records,updates=64,lr=.001,
                 weight2=.35,anchor=.01):
    if not records or updates<1 or not 0<lr<1 or not 0<=weight2<=1 or anchor<0:
        raise ValueError('Invalid R71 training configuration')
    p=model.pair_successor.future_weight
    if p is None or p.shape!=(2,4) or len(optimizer.state)!=198:
        raise ValueError('Expected R69 future tensor and complete optimizer')
    original=p.detach().clone()
    group=optimizer.param_groups[0]
    old_lr,old_decay=group['lr'],group['weight_decay']
    for q in model.parameters(): q.requires_grad_(q is p)
    rng=np.random.default_rng(71)
    order=np.empty(0,dtype=int)
    cursor=0
    losses=[]
    try:
        group['lr'],group['weight_decay']=lr,0.
        for _ in range(updates):
            if cursor==len(order):
                order=rng.permutation(len(records))
                cursor=0
            a,b,h,lo,hi,y2,y3=records[int(order[cursor])]
            cursor+=1
            optimizer.zero_grad(set_to_none=True)
            z2,z3=model.pair_successor.future_forward((a,b),h)
            ce2=F.cross_entropy(z2[:,lo:hi].flatten(0,1),y2)
            ce3=F.cross_entropy(z3[:,lo:hi].flatten(0,1),y3)
            loss=weight2*ce2+ce3+anchor*(p-original).square().sum()
            if not torch.isfinite(loss): raise FloatingPointError('R71 nonfinite loss')
            loss.backward()
            if p.grad is None or not torch.isfinite(p.grad).all():
                raise FloatingPointError('R71 nonfinite gradient')
            torch.nn.utils.clip_grad_norm_([p],1.)
            optimizer.step()
            losses.append((float(ce2.detach()),float(ce3.detach())))
    finally:
        group['lr'],group['weight_decay']=old_lr,old_decay
        optimizer.zero_grad(set_to_none=True)
        for q in model.parameters(): q.requires_grad_(True)
    return {'updates':updates,'loss2_first_last':[losses[0][0],losses[-1][0]],
            'loss3_first_last':[losses[0][1],losses[-1][1]],
            'future_before':original.tolist(),'future_after':p.detach().tolist(),
            'distinct_positions':sum(len(x[-1]) for x in records),
            'cached_chunks':len(records),'weight2':weight2,'anchor':anchor}


def audit(saved,model,optimizer):
    old,new=saved['model'],model.state_dict()
    if set(old)!=set(new): raise AssertionError('Changed model structure')
    for name,t in old.items():
        if name!='pair_successor.future_weight' and not torch.equal(t,new[name]):
            raise AssertionError('Mutated inherited tensor '+name)
    before,after=saved['optimizer'],optimizer.state_dict()
    if (len(before['state'])!=198 or len(after['state'])!=198 or
            before['param_groups'][0]['params']!=after['param_groups'][0]['params']):
        raise AssertionError('Optimizer lineage mismatch')
    last=before['param_groups'][0]['params'][-1]
    for pid,state in before['state'].items():
        if pid==last: continue
        for key,value in state.items():
            newer=after['state'][pid][key]
            match=torch.equal(value,newer) if torch.is_tensor(value) else value==newer
            if not match: raise AssertionError('Inherited Adam state changed')
    return {'inherited_neural_tensors_unchanged':len(old)-1,
            'inherited_adam_states_unchanged':197,'all_adam_states':198}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','data-dir','candidate','report'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--train-books',type=int,default=8)
    p.add_argument('--positions',type=int,default=192)
    p.add_argument('--updates',type=int,default=64)
    p.add_argument('--lr',type=float,default=.001)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--weight2',type=float,default=.35)
    p.add_argument('--anchor',type=float,default=.01)
    a=p.parse_args()
    if a.candidate.exists() or a.report.exists():
        p.error('Refuse to overwrite checkpoint or provenance')
    if min(a.train_books,a.positions,a.threads)<1: p.error('Positive controls required')
    torch.set_num_threads(a.threads)
    model,optimizer,saved=load_active_fa2(a.checkpoint,include_optimizer=True)
    if saved['step']!=4465 or not model.cfg.pair_successor_future:
        raise AssertionError('Requires canonical FA2 R69 step-4465 checkpoint')
    meta,arrays=checked_corpus(a.data_dir,saved)
    t=perf_counter()
    records=[]
    offset=0
    for i,length in enumerate(meta['train_doc_lengths'][:a.train_books]):
        length=int(length)
        take=min(length,a.positions+3)
        tokens=torch.from_numpy(np.asarray(arrays['train'][offset:offset+take],
                                           dtype=np.int64).copy())
        records.extend(cache_future_book(model,tokens,f'R71_TRAIN_{i}',
                                         positions=a.positions))
        offset+=length
    feature_seconds=perf_counter()-t
    t=perf_counter()
    learning=continue_r69(model,optimizer,records,a.updates,a.lr,a.weight2,a.anchor)
    optimizer_seconds=perf_counter()-t
    inherited=audit(saved,model,optimizer)
    candidate=dict(saved)
    candidate.update(model=model.state_dict(),optimizer=optimizer.state_dict(),
                     config=asdict(model.cfg),step=4465+a.updates,
                     fa2_r71_training=learning)
    a.candidate.parent.mkdir(parents=True,exist_ok=True)
    a.report.parent.mkdir(parents=True,exist_ok=True)
    torch.save(candidate,a.candidate)
    report={'round':'FA2_R71','status':'TRAIN-only, not promoted',
            'parent_step':4465,'candidate_step':candidate['step'],
            'parent_sha256':file_hash(a.checkpoint),
            'candidate_sha256':file_hash(a.candidate),
            'train_sha256':meta['train_sha256'],
            'books':min(a.train_books,len(meta['train_doc_lengths'])),
            'feature_seconds':feature_seconds,
            'optimizer_seconds':optimizer_seconds,'learning':learning,
            'inheritance':inherited,'test_opened':False,
            'dev_opened':False,'promoted':False}
    a.report.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__': main()
