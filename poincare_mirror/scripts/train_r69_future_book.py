#!/usr/bin/env python3
"""Train the SAME FA2 model's +2/+3 causal-memory trust from active R68.

This R69 candidate adds eight zero-initialized trust values, keeps 197
inherited Adam slots byte-for-byte, and never reads DEV/TEST during updates.
"""
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
from book_successor_stream import CausalLagBookSuccessor
from fa2_active import load_active_fa2,ACTIVE_MANIFEST
from poincare_frontier import file_hash
from train_r55_pair_successor import checked_corpus


@torch.no_grad()
def cache_future_book(model,tokens,book_id,*,positions=512,context=64,
                      chunk=16,history=32768,key_mode='hierarchical'):
    if tokens.ndim!=1 or tokens.dtype!=torch.long or context>model.cfg.context:
        raise ValueError('Invalid future TRAIN book or context')
    total=min(len(tokens)-3,positions)
    if total<=0: raise ValueError('Book lacks +2/+3 future targets')
    streams=[CausalLagBookSuccessor(lag=lag,horizon=history,
                 key_mode=key_mode) for lag in (2,3)]
    for s in streams: s.begin_book(book_id)
    records=[]
    for start in range(0,total,chunk):
        stop=min(total,start+chunk)
        left=max(0,stop-context)
        ids=tokens[left:stop].unsqueeze(0)
        _,hidden,_=model(ids)
        pairs=model._r54_geometry().r54_last_pairs.detach().clone()
        future2,future3=model.future_logits(hidden)
        evidence=[]
        for stream in streams:
            hist=torch.zeros_like(future2)
            cov=torch.zeros_like(ids,dtype=torch.float32)
            hits=torch.zeros_like(ids,dtype=torch.float32)
            for t in range(start,stop):
                value,coverage,count=stream.step(book_id,int(tokens[t]),
                    pairs[0,t-left].tolist())
                hist[0,t-left]=value
                cov[0,t-left]=coverage
                hits[0,t-left]=count
            evidence.extend((hist,cov,hits))
        records.append((future2.detach(),future3.detach(),tuple(evidence),
            start-left,stop-left,tokens[start+2:stop+2].clone(),
            tokens[start+3:stop+3].clone()))
    if sum(len(record[-1]) for record in records)!=total:
        raise AssertionError('Future TRAIN positions must be distinct')
    return records


def train_future_only(model,optimizer,records,*,updates=256,lr=.004,seed=69):
    if not records or updates<1 or not 0<lr<1:
        raise ValueError('Invalid R69 training job')
    future=model.pair_successor.future_weight
    if future is None: raise ValueError('R69 future trust not enabled')
    group=optimizer.param_groups[0]
    old_lr,old_decay=group['lr'],group['weight_decay']
    for p in model.parameters(): p.requires_grad_(p is future)
    rng=np.random.default_rng(seed)
    order=np.empty(0,dtype=np.int64)
    cursor=0
    losses=[]
    try:
        group['lr'],group['weight_decay']=lr,0.
        for _ in range(updates):
            if cursor==len(order):
                order=rng.permutation(len(records))
                cursor=0
            p2,p3,evidence,lo,hi,y2,y3=records[int(order[cursor])]
            cursor+=1
            optimizer.zero_grad(set_to_none=True)
            prediction=model.pair_successor.future_forward((p2,p3),evidence)
            loss=.5*(F.cross_entropy(prediction[0][:,lo:hi].flatten(0,1),y2)+
                     F.cross_entropy(prediction[1][:,lo:hi].flatten(0,1),y3))
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite future loss')
            loss.backward()
            if future.grad is None or not torch.isfinite(future.grad).all():
                raise FloatingPointError('Invalid future gradient')
            torch.nn.utils.clip_grad_norm_([future],1.)
            optimizer.step()
            losses.append(float(loss.detach()))
    finally:
        group['lr'],group['weight_decay']=old_lr,old_decay
        optimizer.zero_grad(set_to_none=True)
        for p in model.parameters(): p.requires_grad_(True)
    return {'updates':updates,'first_ce':losses[0],'last_ce':losses[-1],
            'mean_ce':float(np.mean(losses)),
            'future_weights':future.detach().tolist(),
            'cached_chunks':len(records),
            'distinct_training_positions':sum(len(x[-1]) for x in records)}


def assert_inheritance(saved,model,optimizer):
    for name,prior in saved['model'].items():
        if not torch.equal(prior,model.state_dict()[name]):
            raise AssertionError('Inherited neural tensor mutated: '+name)
    before,after=saved['optimizer'],optimizer.state_dict()
    if (len(before['param_groups'])!=1 or len(after['param_groups'])!=1 or
            after['param_groups'][0]['params'][:-1]!=before['param_groups'][0]['params']):
        raise AssertionError('R69 changed optimizer parameter lineage')
    for pid,state in before['state'].items():
        for key,value in state.items():
            new=after['state'][pid][key]
            equal=(torch.equal(value,new) if torch.is_tensor(value) else value==new)
            if not equal: raise AssertionError('Inherited Adam changed: '+str(pid))
    return {'preserved_inherited_model_tensors':len(saved['model']),
            'preserved_inherited_adam_states':len(before['state']),
            'total_adam_states_after_train':len(after['state'])}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--manifest',type=Path,default=ACTIVE_MANIFEST)
    p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True)
    p.add_argument('--positions',type=int,default=512)
    p.add_argument('--books',type=int,default=12)
    p.add_argument('--updates',type=int,default=256)
    p.add_argument('--lr',type=float,default=.004)
    p.add_argument('--threads',type=int,default=2)
    args=p.parse_args()
    if args.candidate.exists() or args.report.exists():
        p.error('Refuse to overwrite candidate/report')
    torch.set_num_threads(args.threads)
    model,optimizer,saved=load_active_fa2(args.checkpoint,
         manifest_path=args.manifest,include_optimizer=True)
    if saved['step']!=4209 or len(optimizer.state)!=197:
        raise AssertionError('R69 requires verified R68 active lineage')
    meta,arrays=checked_corpus(args.data_dir,saved)
    parameter=model.enable_future_pair_successor(optimizer)
    if tuple(parameter.shape)!=(2,4) or not torch.count_nonzero(parameter).item()==0:
        raise AssertionError('R69 must start with an exactly neutral future rail')
    start=perf_counter()
    offset=0
    records=[]
    books=min(args.books,len(meta['train_doc_lengths']))
    for idx,length in enumerate(meta['train_doc_lengths'][:books]):
        length=int(length)
        take=min(length,args.positions+3)
        tokens=torch.from_numpy(np.asarray(arrays['train'][offset:offset+take],
                                  dtype=np.int64).copy())
        records.extend(cache_future_book(model,tokens,'TRAIN:'+str(idx),
                                         positions=args.positions))
        offset+=length
    feature_seconds=perf_counter()-start
    start=perf_counter()
    result=train_future_only(model,optimizer,records,updates=args.updates,
                             lr=args.lr)
    opt_seconds=perf_counter()-start
    inheritance=assert_inheritance(saved,model,optimizer)
    payload=dict(saved)
    payload.update(model=model.state_dict(),optimizer=optimizer.state_dict(),
                   config=asdict(model.cfg),step=4209+args.updates,
                   fa2_r69_training=result)
    args.candidate.parent.mkdir(parents=True,exist_ok=True)
    args.report.parent.mkdir(parents=True,exist_ok=True)
    torch.save(payload,args.candidate)
    report={'round':'FA2_R69_CAUSAL_FUTURE','parent_step':4209,
            'candidate_step':payload['step'],
            'parent_sha256':file_hash(args.checkpoint),
            'candidate_sha256':file_hash(args.candidate),
            'train_books':books,'train_sha256':meta['train_sha256'],
            'feature_seconds':feature_seconds,
            'optimization_seconds':opt_seconds,'learning':result,
            'inheritance':inheritance,'test_opened':False,
            'dev_opened':False,'promoted':False}
    args.report.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
