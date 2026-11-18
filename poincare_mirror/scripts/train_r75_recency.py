#!/usr/bin/env python3
"""R75: fit causal geometric episode recency on NaNROUTER TRAIN only.

Active R69 weights and all 198 original AdamW states remain immutable.
Four confirmation books are excluded from the 8-book fit and TEST stays sealed.
"""
import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fa2_active import load_active_fa2
from poincare_frontier import file_hash
from train_r55_pair_successor import checked_corpus
from train_r74_counterfactual import cache_book,train,metrics


def run(checkpoint,data_dir,report,*,fit_books=8,confirm_books=4,
        positions=128,warmup=128,half_life=48.,updates=96,lr=.003,threads=2):
    if min(fit_books,confirm_books,positions,warmup,updates,threads)<1 or half_life<=0:
        raise ValueError('R75 fit/confirmation/half-life parameters invalid')
    torch.set_num_threads(threads)
    model,optimizer,saved=load_active_fa2(checkpoint,include_optimizer=True)
    if saved['step']!=4465 or len(optimizer.state)!=198:
        raise AssertionError('R75 requires complete accepted R69 optimizer')
    meta,arrays=checked_corpus(data_dir,saved)
    param=model.pair_successor.enable_recency_content()
    optimizer.param_groups[0]['params'].append(param)
    if param.detach().count_nonzero().item():
        raise AssertionError('R75 must initialize to exact R69 function')
    offset=0
    fit=[]
    confirmation=[]
    started=perf_counter()
    for i,length in enumerate(meta['train_doc_lengths'][:fit_books+confirm_books]):
        tokens=torch.from_numpy(np.asarray(arrays['train'][
            offset:offset+warmup+positions+2],dtype=np.int64).copy())
        offset+=int(length)
        record=cache_book(model,tokens,f'TRAIN:{i}',warmup=warmup,
            positions=positions,alternative='recency',recency_half_life=half_life)
        (fit if i<fit_books else confirmation).extend(record)
        print(f'R75 cached TRAIN book {i+1}/{fit_books+confirm_books}',flush=True)
    before=metrics(model,confirmation,rail='recency')
    learning=train(model,optimizer,fit,updates=updates,lr=lr,seed=75,rail='recency')
    after=metrics(model,confirmation,rail='recency')
    for key in ('parent_ce','parent_top1','parent_top5'):
        if abs(before[key]-after[key])>1e-8:
            raise AssertionError('Inherited R69 logits changed during R75')
    for key,value in saved['model'].items():
        if not torch.equal(value,model.state_dict()[key]):
            raise AssertionError('Inherited R69 tensor changed: '+key)
    prior,current=saved['optimizer'],optimizer.state_dict()
    if (len(current['state'])!=199 or
            current['param_groups'][0]['params'][:-1]!=prior['param_groups'][0]['params']):
        raise AssertionError('R75 broke inherited AdamW ordering')
    for pid,slot in prior['state'].items():
        for key,value in slot.items():
            actual=current['state'][pid][key]
            if not (torch.equal(value,actual) if torch.is_tensor(value)
                    else value==actual):
                raise AssertionError('Inherited AdamW slot mutated')
    report.parent.mkdir(parents=True,exist_ok=True)
    result={'round':'FA2_R75_GEOMETRIC_RECENCY','status':'TRAIN-only candidate',
        'parent_checkpoint_sha256':file_hash(checkpoint),'parent_step':4465,
        'train_sha256':meta['train_sha256'],'fit_books':fit_books,
        'confirmation_books':confirm_books,'fit_positions':fit_books*positions,
        'confirmation_positions':confirm_books*positions,'warmup':warmup,
        'recency_half_life':half_life,'seconds':perf_counter()-started,
        'learning':learning,'before':before,'after':after,
        'ce_gain':before['parent_ce']-after['candidate_ce'],
        'top1_gain':after['candidate_top1']-before['parent_top1'],
        'top5_gain':after['candidate_top5']-before['parent_top5'],
        'preserved_adam_states':198,'total_adam_states':199,
        'dev_opened':False,'test_opened':False,'promoted':False}
    report.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2),flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for x in ('checkpoint','data-dir','report'):
        p.add_argument('--'+x,type=Path,required=True)
    p.add_argument('--fit-books',type=int,default=8)
    p.add_argument('--confirm-books',type=int,default=4)
    p.add_argument('--positions',type=int,default=128)
    p.add_argument('--warmup',type=int,default=128)
    p.add_argument('--half-life',type=float,default=48.)
    p.add_argument('--updates',type=int,default=96)
    p.add_argument('--lr',type=float,default=.003)
    p.add_argument('--threads',type=int,default=2)
    a=p.parse_args()
    run(a.checkpoint,a.data_dir,a.report,fit_books=a.fit_books,
        confirm_books=a.confirm_books,positions=a.positions,warmup=a.warmup,
        half_life=a.half_life,updates=a.updates,lr=a.lr,threads=a.threads)


if __name__=='__main__':main()
