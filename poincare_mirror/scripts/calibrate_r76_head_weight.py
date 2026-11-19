#!/usr/bin/env python3
"""R76 TRAIN-only calibration of confidence-weighted geometric head backoff.

This is parameter fitting, not gradient training. All inherited R69 weights,
198 AdamW states, and the active checkpoint remain untouched. Different
TRAIN books determine the coefficient and test its generalization. DEV/TEST
are never accessed in this screening run.
"""
import argparse
import json
import math
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from book_successor_stream import CausalBookSuccessor,CausalLagBookSuccessor
from fa2_active import load_active_fa2
from poincare_frontier import file_hash
from rank_safe_fusion import protect_parent_space_rank
from train_r55_pair_successor import checked_corpus


WEIGHTS=(.125,.25,.5,1.,2.)
METRICS=('ce_1','ce_2','ce_3','acc_1','acc_2','acc_3',
         'top5_1','top5_2','top5_3')


def _blank():
    return {key:0. for key in METRICS}


@torch.no_grad()
def evaluate_book(model,tokens,identity,*,warmup=256,positions=128,
                  history=32768,context=64,chunk=16,weights=WEIGHTS):
    if tokens.ndim!=1 or len(tokens)<warmup+positions+3:
        raise ValueError('R76 requires complete independent book prefix')
    if model.training or model.pair_successor.future_weight is None:
        raise ValueError('R76 requires eval-mode accepted R69 future memory')
    streams={}
    for weight in weights:
        group=[CausalBookSuccessor(horizon=history,key_mode='hierarchical',
                                   head_backoff_weight=weight)]
        group.extend(CausalLagBookSuccessor(lag=lag,horizon=history,
                     key_mode='hierarchical',head_backoff_weight=weight)
                     for lag in (2,3))
        for stream in group:stream.begin_book(identity)
        streams[weight]=group
    sums={weight:_blank() for weight in weights}
    counts={weight:[0,0,0] for weight in weights}
    changed=0
    first_hist=None
    for start in range(0,warmup+positions,chunk):
        stop=min(warmup+positions,start+chunk)
        left=max(0,stop-context)
        ids=tokens[left:stop].unsqueeze(0)
        _,hidden,_=model(ids)
        pairs=model._r54_geometry().r54_last_pairs.detach()
        raw=model.head(hidden)
        f2,f3=model.future_logits(hidden)
        if stop>warmup:
            null=(torch.zeros_like(raw),torch.zeros_like(ids,dtype=torch.float32),
                  torch.zeros_like(ids,dtype=torch.float32))
            neutral,_=model.pair_successor(raw,pairs,ids,
                                           joint_memory_override=null)
            neutral_future=model.pair_successor.future_forward(
                (f2,f3),null+null)
        for weight,group in streams.items():
            readings=[]
            for stream in group:
                hist=torch.zeros_like(raw)
                cov=torch.zeros_like(ids,dtype=torch.float32)
                hits=torch.zeros_like(cov)
                for t in range(start,stop):
                    h,c,n=stream.step(identity,int(tokens[t]),
                                      pairs[0,t-left].tolist())
                    hist[0,t-left]=h;cov[0,t-left]=c;hits[0,t-left]=n
                readings.append((hist,cov,hits))
            if stop<=warmup:continue
            lo=max(start,warmup)-left
            hi=stop-left
            if weight==weights[0]:
                first_hist=readings[0][0][:,lo:hi].clone()
            if ((weight==1. and weights[0]!=1.) or
                    (weight==weights[-1] and weights[0]==1.)) and first_hist is not None:
                changed+=int((readings[0][0][:,lo:hi]-first_hist).abs()
                              .sum(-1).gt(1e-7).sum())
            pred,_=model.pair_successor(raw,pairs,ids,
                                        joint_memory_override=readings[0])
            pred=protect_parent_space_rank(neutral,pred)
            future=model.pair_successor.future_forward(
                (f2,f3),readings[1]+readings[2])
            future=tuple(protect_parent_space_rank(a,b)
                         for a,b in zip(neutral_future,future))
            for horizon,logits in enumerate((pred,future[0],future[1])):
                horizon+=1
                y=tokens[max(start,warmup)+horizon:stop+horizon]
                logits=logits[0,lo:hi]
                if len(y)!=len(logits):
                    raise AssertionError('Misaligned R76 causal target window')
                loss=torch.nn.functional.cross_entropy(logits,y,reduction='sum')
                sums[weight]['ce_'+str(horizon)]+=float(loss)
                sums[weight]['acc_'+str(horizon)]+=int(logits.argmax(-1).eq(y).sum())
                sums[weight]['top5_'+str(horizon)]+=int(logits.topk(5,-1)
                                     .indices.eq(y[:,None]).any(-1).sum())
                counts[weight][horizon-1]+=len(y)
    output={}
    for weight in weights:
        value=sums[weight]
        result={key:value[key]/counts[weight][int(key[-1])-1]
                for key in METRICS}
        result['bpb_1']=result['ce_1']/math.log(2)
        output[str(weight)]=result
    return {'book_id':identity,'metrics':output,
            'changed_positions_against_low_head_weight':changed,
            'positions':positions,'warmup':warmup}


def mean_metrics(rows,key):
    return {m:float(np.mean([r['metrics'][str(key)][m] for r in rows]))
            for m in (*METRICS,'bpb_1')}


def run(checkpoint,data_dir,report,*,fit_books=8,confirm_books=4,
        warmup=256,positions=128,threads=2):
    if min(fit_books,confirm_books,warmup,positions,threads)<1:
        raise ValueError('Invalid R76 calibration budget')
    torch.set_num_threads(threads)
    model,optimizer,saved=load_active_fa2(checkpoint,include_optimizer=True)
    if saved['step']!=4465 or len(optimizer.state)!=198:
        raise AssertionError('R76 requires complete inherited R69 AdamW')
    meta,arrays=checked_corpus(data_dir,saved)
    rows=[]
    offset=0
    started=perf_counter()
    for i,length in enumerate(meta['train_doc_lengths'][:fit_books+confirm_books]):
        tokens=torch.from_numpy(np.asarray(arrays['train'][
            offset:offset+warmup+positions+3],dtype=np.int64).copy())
        offset+=int(length)
        row=evaluate_book(model,tokens,'TRAIN:'+str(i),
                          warmup=warmup,positions=positions)
        rows.append(row)
        print(f'R76 TRAIN book {i+1}/{fit_books+confirm_books}: '+
              ' '.join(f'{w:g}={row["metrics"][str(w)]["ce_1"]:.5f}'
                       for w in WEIGHTS),flush=True)
    fit=rows[:fit_books]
    confirm=rows[fit_books:]
    # Fit a single global geometric edge-backoff coefficient on TRAIN books.
    # Equal weight 1.0 is the exact accepted-parent baseline.
    objectives={str(w):sum((1.,.35,.35)[i]*mean_metrics(fit,w)[f'ce_{i+1}']
                            for i in range(3)) for w in WEIGHTS}
    selected=min(WEIGHTS,key=lambda w:objectives[str(w)])
    baseline=mean_metrics(confirm,1.)
    candidate=mean_metrics(confirm,selected)
    result={'round':'FA2_R76_CONFIDENCE_BACKOFF','status':'TRAIN-only calibration',
        'parent_checkpoint_sha256':file_hash(checkpoint),'parent_step':4465,
        'train_sha256':meta['train_sha256'],'fit_books':fit_books,
        'confirmation_books':confirm_books,'fit_positions':fit_books*positions,
        'confirmation_positions':confirm_books*positions,'warmup':warmup,
        'candidate_head_weights':list(WEIGHTS),'fitted_objectives':objectives,
        'selected_head_backoff_weight':selected,
        'confirmation_baseline':baseline,'confirmation_candidate':candidate,
        'confirmation_ce1_gain':baseline['ce_1']-candidate['ce_1'],
        'confirmation_ce2_gain':baseline['ce_2']-candidate['ce_2'],
        'confirmation_ce3_gain':baseline['ce_3']-candidate['ce_3'],
        'changed_positions':[r['changed_positions_against_low_head_weight']
                             for r in rows],
        'seconds':perf_counter()-started,
        'preserved_adam_states':198,'dev_opened':False,'test_opened':False,
        'promoted':False,'per_book':rows}
    report.parent.mkdir(parents=True,exist_ok=True)
    report.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('R76 summary',json.dumps({key:value for key,value in result.items()
          if key!='per_book'},indent=2),flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('checkpoint','data-dir','report'):
        p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--fit-books',type=int,default=8)
    p.add_argument('--confirm-books',type=int,default=4)
    p.add_argument('--warmup',type=int,default=256)
    p.add_argument('--positions',type=int,default=128)
    p.add_argument('--threads',type=int,default=2)
    a=p.parse_args()
    run(a.checkpoint,a.data_dir,a.report,fit_books=a.fit_books,
        confirm_books=a.confirm_books,warmup=a.warmup,
        positions=a.positions,threads=a.threads)


if __name__=='__main__':main()
