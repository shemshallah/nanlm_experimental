#!/usr/bin/env python3
"""R74 CPU pilot: train a four-scalar factual-vs-control rail on TRAIN books.

The complete R69 model and 198 AdamW states are inherited. No DEV or TEST
target participates in training. Model weights are not silently replaced.
"""
import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from book_successor_stream import CausalBookSuccessor
from fa2_active import load_active_fa2
from poincare_frontier import file_hash
from rank_safe_fusion import protect_parent_space_rank
from train_r55_pair_successor import checked_corpus


@torch.no_grad()
def cache_book(model,tokens,identity,*,warmup=128,positions=128,chunk=16,
               context=64,history=32768):
    if len(tokens)<warmup+positions+2:
        raise ValueError('R74 TRAIN book too short')
    exact=CausalBookSuccessor(horizon=history,key_mode='hierarchical')
    negative=CausalBookSuccessor(horizon=history,key_mode='hierarchical',
                                  value_control='rotated')
    for stream in (exact,negative): stream.begin_book(identity)
    records=[]
    for start in range(0,warmup+positions,chunk):
        stop=min(warmup+positions,start+chunk)
        left=max(0,stop-context)
        observed=tokens[left:stop].unsqueeze(0)
        _,hidden,_=model(observed)
        pairs=model._r54_geometry().r54_last_pairs.detach()
        raw=model.head(hidden).detach()
        joint=[]
        for stream in (exact,negative):
            hist=torch.zeros_like(raw)
            cov=torch.zeros_like(observed,dtype=torch.float32)
            hits=torch.zeros_like(cov)
            for t in range(start,stop):
                h,c,p=stream.step(identity,int(tokens[t]),pairs[0,t-left].tolist())
                hist[0,t-left]=h
                cov[0,t-left]=c
                hits[0,t-left]=p
            joint.append((hist,cov,hits))
        if stop<=warmup: continue
        neutral,_=model.pair_successor(raw,pairs,observed,
            joint_memory_override=(torch.zeros_like(joint[0][0]),
            torch.zeros_like(joint[0][1]),torch.zeros_like(joint[0][2])))
        factual,_=model.pair_successor(raw,pairs,observed,
                                        joint_memory_override=joint[0])
        factual=protect_parent_space_rank(neutral,factual)
        lo=max(start,warmup)-left
        hi=stop-left
        index=max(start,warmup)
        records.append((factual[:,lo:hi].detach(),
            tuple(v[:,lo:hi].detach() for v in joint[0]),
            tuple(v[:,lo:hi].detach() for v in joint[1]),
            tokens[index+1:stop+1].clone()))
    assert sum(len(x[-1]) for x in records)==positions
    return records


@torch.no_grad()
def metrics(model,records):
    sums={'parent_ce':0.,'candidate_ce':0.,'parent_top1':0.,
          'candidate_top1':0.,'parent_top5':0.,'candidate_top5':0.}
    n=0
    for base,exact,negative,y in records:
        candidate=model.pair_successor.counterfactual_forward(base,exact,negative)
        candidate=protect_parent_space_rank(base,candidate)
        for name,logits in (('parent',base),('candidate',candidate)):
            logits=logits.flatten(0,1)
            sums[name+'_ce']+=float(F.cross_entropy(logits,y,reduction='sum'))
            sums[name+'_top1']+=int(logits.argmax(-1).eq(y).sum())
            sums[name+'_top5']+=int(logits.topk(5,-1).indices.eq(y[:,None]).any(-1).sum())
        n+=len(y)
    return {'positions':n,**{key:val/n for key,val in sums.items()},
            'parent_bpb':sums['parent_ce']/(n*np.log(2)),
            'candidate_bpb':sums['candidate_ce']/(n*np.log(2))}


def train(model,optimizer,records,*,updates=96,lr=.003,seed=74):
    if updates<1 or not 0<lr<.1 or not records:
        raise ValueError('Invalid R74 training budget')
    target=model.pair_successor.counterfactual_weight
    old_lr=optimizer.param_groups[0]['lr']
    old_decay=optimizer.param_groups[0]['weight_decay']
    for p in model.parameters(): p.requires_grad_(p is target)
    rng=np.random.default_rng(seed)
    order=np.empty(0,dtype=np.int64)
    cursor=0
    losses=[]
    try:
        optimizer.param_groups[0]['lr']=lr
        optimizer.param_groups[0]['weight_decay']=0.
        for _ in range(updates):
            if cursor==len(order):
                order=rng.permutation(len(records))
                cursor=0
            base,exact,negative,y=records[int(order[cursor])]
            cursor+=1
            optimizer.zero_grad(set_to_none=True)
            pred=model.pair_successor.counterfactual_forward(base,exact,negative)
            pred=protect_parent_space_rank(base,pred)
            loss=F.cross_entropy(pred.flatten(0,1),y)
            if not torch.isfinite(loss):
                raise FloatingPointError('R74 train CE became nonfinite')
            loss.backward()
            if target.grad is None or not torch.isfinite(target.grad).all():
                raise FloatingPointError('R74 train gradient invalid')
            torch.nn.utils.clip_grad_norm_([target],1.)
            optimizer.step()
            losses.append(float(loss.detach()))
    finally:
        optimizer.param_groups[0]['lr']=old_lr
        optimizer.param_groups[0]['weight_decay']=old_decay
        optimizer.zero_grad(set_to_none=True)
        for p in model.parameters(): p.requires_grad_(True)
    return {'updates':updates,'first_batch_ce':losses[0],
            'last_batch_ce':losses[-1],
            'weights':target.detach().tolist()}


def run(checkpoint,data_dir,report,*,fit_books=8,confirm_books=4,
        positions=128,warmup=128,updates=96,lr=.003,threads=2):
    torch.set_num_threads(threads)
    model,optimizer,saved=load_active_fa2(checkpoint,include_optimizer=True)
    if saved['step']!=4465 or len(optimizer.state)!=198:
        raise AssertionError('R74 must extend active R69 with 198 Adam states')
    meta,arrays=checked_corpus(data_dir,saved)
    param=model.pair_successor.enable_counterfactual_content()
    optimizer.param_groups[0]['params'].append(param)
    if param.detach().count_nonzero().item():
        raise AssertionError('R74 not neutral at initialization')
    offset=0
    fit=[]
    confirm=[]
    started=perf_counter()
    for i,length in enumerate(meta['train_doc_lengths'][:fit_books+confirm_books]):
        n=warmup+positions+2
        tokens=torch.from_numpy(np.asarray(arrays['train'][offset:offset+n],
                              dtype=np.int64).copy())
        offset+=int(length)
        row=cache_book(model,tokens,f'TRAIN:{i}',warmup=warmup,positions=positions)
        (fit if i<fit_books else confirm).extend(row)
        print(f'R74 cached TRAIN book {i+1}/{fit_books+confirm_books}',flush=True)
    neutral=metrics(model,confirm)
    learning=train(model,optimizer,fit,updates=updates,lr=lr)
    after=metrics(model,confirm)
    for key in ('parent_ce','parent_top1','parent_top5'):
        if abs(neutral[key]-after[key])>1e-8:
            raise AssertionError('R74 changed frozen inherited baseline')
    for key,value in saved['model'].items():
        if not torch.equal(value,model.state_dict()[key]):
            raise AssertionError('R74 changed inherited parameter: '+key)
    old,new=saved['optimizer'],optimizer.state_dict()
    if len(new['state'])!=199 or new['param_groups'][0]['params'][:-1]!=old['param_groups'][0]['params']:
        raise AssertionError('R74 optimizer lineage not preserved')
    for pid,state in old['state'].items():
        for key,value in state.items():
            updated=new['state'][pid][key]
            if not (torch.equal(value,updated) if torch.is_tensor(value)
                    else value==updated):
                raise AssertionError('R74 modified inherited Adam state')
    result={'round':'FA2_R74_COUNTERFACTUAL','status':'TRAIN confirmation, not promoted',
            'parent_sha256':file_hash(checkpoint),'parent_step':4465,
            'train_sha256':meta['train_sha256'],'fit_books':fit_books,
            'confirmation_books':confirm_books,'fit_positions':fit_books*positions,
            'confirmation_positions':confirm_books*positions,
            'warmup':warmup,'seconds':perf_counter()-started,
            'learning':learning,'before':neutral,'after':after,
            'ce_gain':neutral['parent_ce']-after['candidate_ce'],
            'top1_gain':after['candidate_top1']-neutral['parent_top1'],
            'top5_gain':after['candidate_top5']-neutral['parent_top5'],
            'preserved_adam_states':198,'total_adam_states':199,
            'dev_opened':False,'test_opened':False,'promoted':False}
    report.parent.mkdir(parents=True,exist_ok=True)
    report.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2),flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True,type=Path)
    p.add_argument('--data-dir',required=True,type=Path)
    p.add_argument('--report',required=True,type=Path)
    p.add_argument('--fit-books',type=int,default=8)
    p.add_argument('--confirm-books',type=int,default=4)
    p.add_argument('--positions',type=int,default=128)
    p.add_argument('--warmup',type=int,default=128)
    p.add_argument('--updates',type=int,default=96)
    p.add_argument('--lr',type=float,default=.003)
    p.add_argument('--threads',type=int,default=2)
    a=p.parse_args()
    run(a.checkpoint,a.data_dir,a.report,fit_books=a.fit_books,
        confirm_books=a.confirm_books,positions=a.positions,warmup=a.warmup,
        updates=a.updates,lr=a.lr,threads=a.threads)


if __name__=='__main__': main()
