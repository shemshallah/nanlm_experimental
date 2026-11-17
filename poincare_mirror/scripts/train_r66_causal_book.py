#!/usr/bin/env python3
"""Train FA2's existing joint-memory rail with persistent causal TRAIN books.

Only four new joint weights receive gradients; all R58 neural tensors and
196 optimizer-state entries remain intact. DEV/TEST are never used for updates.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from book_successor_stream import CausalBookSuccessor
from poincare_frontier import (Config, MirrorLM, content_balanced_ce,
                               file_hash, load_v25_compatible)
from train_r55_pair_successor import checked_corpus, audit_inheritance


@torch.no_grad()
def cache_causal_book(model,tokens,book_id,*,context=64,chunk=16,history=32768,
                      warmup=128,positions=256,key_mode="hierarchical"):
    if (tokens.ndim!=1 or tokens.dtype!=torch.long or tokens.device.type!="cpu" or
            warmup<0 or positions<1 or context<4 or context>model.cfg.context or
            not 1<=chunk<=context or history<1):
        raise ValueError("Invalid causal TRAIN book/window")
    total=min(len(tokens)-1,warmup+positions)
    if total<=warmup:
        raise ValueError("Book is shorter than warmup plus one target")
    stream=CausalBookSuccessor(horizon=history,key_mode=key_mode)
    stream.begin_book(book_id)
    out=[]
    for start in range(0,total,chunk):
        stop=min(total,start+chunk)
        left=max(0,stop-context)
        ids=tokens[left:stop].unsqueeze(0)
        _,hidden,_=model(ids)
        pairs=model._r54_geometry().r54_last_pairs.detach().clone()
        logits=model.head(hidden).detach()
        if pairs.shape[:2]!=ids.shape:
            raise AssertionError("Misaligned geometric addresses")
        hist=torch.zeros_like(logits)
        cov=torch.zeros_like(ids,dtype=torch.float32)
        hits=torch.zeros_like(ids,dtype=torch.float32)
        for t in range(start,stop):
            h,c,n=stream.step(book_id,int(tokens[t]),pairs[0,t-left].tolist())
            hist[0,t-left]=h
            cov[0,t-left]=c
            hits[0,t-left]=n
        lo=max(start,warmup)-left
        hi=stop-left
        if lo<hi:
            target=tokens[max(start,warmup)+1:stop+1].clone()
            if len(target)!=hi-lo or torch.any(hist.sum(-1)>1.00001):
                raise AssertionError("Noncausal or unnormalized training data")
            out.append((logits,pairs,ids,hist,cov,hits,lo,hi,target))
    if sum(len(x[-1]) for x in out)!=total-warmup:
        raise AssertionError("TRAIN positions are not unique")
    return out


def train_joint_only(model,optimizer,records,*,updates=256,lr=.01,seed=66):
    if not records or updates<1 or not 0<lr<1 or len(optimizer.param_groups)!=1:
        raise ValueError("Invalid updates, LR or optimizer")
    trainable=model.pair_successor.joint_weight
    for p in model.parameters():
        p.requires_grad_(p is trainable)
    group=optimizer.param_groups[0]
    old_lr,old_decay=group['lr'],group['weight_decay']
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
            logits,pairs,ids,hist,cov,hits,lo,hi,target=records[int(order[cursor])]
            cursor+=1
            optimizer.zero_grad(set_to_none=True)
            corrected,_=model.pair_successor(logits,pairs,ids,
                                  joint_memory_override=(hist,cov,hits))
            loss=content_balanced_ce(corrected[:,lo:hi],target.unsqueeze(0),1.)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite causal loss")
            loss.backward()
            if trainable.grad is None or not torch.isfinite(trainable.grad).all():
                raise FloatingPointError("Nonfinite joint gradient")
            optimizer.step()
            losses.append(float(loss.detach()))
    finally:
        group['lr'],group['weight_decay']=old_lr,old_decay
        optimizer.zero_grad(set_to_none=True)
        for p in model.parameters():
            p.requires_grad_(True)
    return {"updates":updates,"first_ce":losses[0],"last_ce":losses[-1],
            "mean_ce":float(np.mean(losses)),"joint_weights":trainable.tolist(),
            "cached_chunks":len(records),"distinct_training_positions":
              sum(len(x[-1]) for x in records)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint","data-dir","candidate","report"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--expected-parent-sha256",required=True)
    parser.add_argument("--train-books",type=int,default=12)
    parser.add_argument("--warmup",type=int,default=128)
    parser.add_argument("--train-positions",type=int,default=256)
    parser.add_argument("--context",type=int,default=64)
    parser.add_argument("--chunk",type=int,default=16)
    parser.add_argument("--history",type=int,default=32768)
    parser.add_argument("--updates",type=int,default=256)
    parser.add_argument("--lr",type=float,default=.01)
    parser.add_argument("--threads",type=int,default=2)
    parser.add_argument("--seed",type=int,default=66)
    args=parser.parse_args()
    if args.candidate.exists() or args.report.exists():
        parser.error("Refuse to overwrite an existing candidate or report")
    sha=file_hash(args.checkpoint)
    if sha!=args.expected_parent_sha256:
        raise ValueError("Protected parent SHA-256 mismatch")
    torch.set_num_threads(args.threads)
    saved=torch.load(args.checkpoint,map_location="cpu",weights_only=False)
    if saved.get("step")!=3185:
        raise ValueError("R66 must continue the protected R58 parent")
    meta,arrays=checked_corpus(args.data_dir,saved)
    model=MirrorLM(Config(**saved['config'])).eval()
    load_v25_compatible(model,saved['model'])
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
    optimizer.load_state_dict(saved['optimizer'])
    model.enable_joint_pair_successor(optimizer,key_mode="hierarchical")
    before=perf_counter()
    records=[]
    offset=0
    n=min(args.train_books,len(meta['train_doc_lengths']))
    for idx,length in enumerate(meta['train_doc_lengths']):
        if idx==n: break
        length=int(length)
        take=min(length,args.warmup+args.train_positions+1)
        book=torch.tensor(np.asarray(arrays['train'][offset:offset+take],
                                     dtype=np.int64).copy())
        records+=cache_causal_book(model,book,"TRAIN:"+str(idx),
                    context=args.context,chunk=args.chunk,history=args.history,
                    warmup=args.warmup,positions=args.train_positions)
        offset+=length
    prep=perf_counter()-before
    started=perf_counter()
    learning=train_joint_only(model,optimizer,records,updates=args.updates,
                              lr=args.lr,seed=args.seed)
    inheritance=audit_inheritance(saved,model,optimizer,new_gate=True,joint_only=True)
    if not inheritance['pass']:
        raise AssertionError("Inherited parameters/Adam changed: "+str(inheritance['issues']))
    payload=dict(saved)
    payload.update(model=model.state_dict(),optimizer=optimizer.state_dict(),
                   config=asdict(model.cfg),step=int(saved['step'])+args.updates,
                   fa2_r66_training=learning)
    args.candidate.parent.mkdir(parents=True,exist_ok=True)
    args.report.parent.mkdir(parents=True,exist_ok=True)
    torch.save(payload,args.candidate)
    report={"round":"FA2_R66_CAUSAL_TRAIN","parent_step":3185,
            "candidate_step":payload['step'],"parent_sha256":sha,
            "candidate_sha256":file_hash(args.candidate),"train_books":n,
            "train_sha256":meta['train_sha256'],"learning":learning,
            "inherited_optimizer_audit":inheritance,"feature_seconds":prep,
            "optimization_seconds":perf_counter()-started,
            "test_opened":False,"dev_opened":False,"promoted":False}
    args.report.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
