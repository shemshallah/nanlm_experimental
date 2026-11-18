#!/usr/bin/env python3
"""Calibrate the opt-in R72 causal temporal bridge using TRAIN books only.

No new checkpoint, no state change, no DEV or TEST during calibration. The
existing R69 future weights and all 198 AdamW histories remain canonical.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fa2_active import load_active_fa2
from poincare_frontier import file_hash
from train_r55_pair_successor import checked_corpus
from train_r69_future_book import cache_future_book


@torch.no_grad()
def score_record(model,record,bridge):
    logits2,logits3,evidence,lo,hi,labels2,labels3=record
    corrected=model.pair_successor.future_forward((logits2,logits3),evidence,
                                                   horizon_bridge=bridge)
    return [float(F.cross_entropy(corrected[k][:,lo:hi].flatten(0,1),labels))
            for k,labels in enumerate((labels2,labels3))]


def run(checkpoint,data_dir,*,books=12,positions=192,context=64,
        strengths=(0.,.1,.2,.35,.5,.75,1.),threads=2):
    if books<4 or positions<4 or threads<1:
        raise ValueError('Need >=4 TRAIN books and positive scoring settings')
    torch.set_num_threads(threads)
    model,optimizer,saved=load_active_fa2(checkpoint,include_optimizer=True)
    if saved['step']!=4465 or len(optimizer.state)!=198:
        raise AssertionError('Calibrate only the original R69 future-memory model')
    before={name:val.detach().clone() for name,val in model.state_dict().items()}
    meta,corpus=checked_corpus(data_dir,saved)
    selected=min(books,len(meta['train_doc_lengths']))
    offset=0
    per_book=[]
    for index,length in enumerate(meta['train_doc_lengths'][:selected]):
        length=int(length)
        take=min(length,positions+3)
        inputs=torch.from_numpy(np.asarray(corpus['train'][offset:offset+take],
                                           dtype=np.int64).copy())
        offset+=length
        records=cache_future_book(model,inputs,'R72_TRAIN_'+str(index),
                                  positions=positions,context=context)
        result={}
        for strength in strengths:
            measurements=[score_record(model,record,strength) for record in records]
            result[str(strength)]={'ce2':float(np.mean([x[0] for x in measurements])),
                                   'ce3':float(np.mean([x[1] for x in measurements]))}
        per_book.append({'book_index':index,'positions':sum(len(r[-1]) for r in records),
                         'scores':result})
        print('R72 TRAIN book',index+1,'/',selected,
              'best CE3',min((result[str(x)]['ce3'],x) for x in strengths),flush=True)
    if any(not torch.equal(val,model.state_dict()[name]) for name,val in before.items()):
        raise AssertionError('R72 TRAIN calibration modified inherited weights')
    calibration_books=max(1,selected*2//3)
    def mean(name,strength,subset):
        return float(np.mean([row['scores'][str(strength)][name] for row in subset]))
    primary=per_book[:calibration_books]
    confirm=per_book[calibration_books:]
    summary={str(x):{'calibration_ce3':mean('ce3',x,primary),
                     'confirm_ce3':mean('ce3',x,confirm),
                     'confirm_ce2':mean('ce2',x,confirm)} for x in strengths}
    chosen=min(strengths,key=lambda x:(summary[str(x)]['calibration_ce3'],x))
    if summary[str(chosen)]['confirm_ce3'] >= summary[str(0.)]['confirm_ce3']:
        chosen=0.
    return {'round':'FA2_R72_CAUSAL_HORIZON_BRIDGE',
            'status':'TRAIN-calibrated only; R69 remains active',
            'checkpoint_sha256':file_hash(checkpoint),
            'train_sha256':meta['train_sha256'],
            'train_books':selected,'positions_per_book':positions,
            'calibration_books':calibration_books,
            'confirm_books':selected-calibration_books,
            'strengths':summary,'selected_strength':chosen,
            'per_book':per_book,'model_weights_unchanged':True,
            'adam_state_count':198,'dev_opened':False,'test_opened':False,
            'promoted':False}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True)
    p.add_argument('--books',type=int,default=12)
    p.add_argument('--positions',type=int,default=192)
    p.add_argument('--threads',type=int,default=2)
    a=p.parse_args()
    if a.report.exists(): p.error('Refuse to overwrite calibration report')
    result=run(a.checkpoint,a.data_dir,books=a.books,
               positions=a.positions,threads=a.threads)
    a.report.parent.mkdir(parents=True,exist_ok=True)
    a.report.write_text(json.dumps(result,indent=2)+'\n')
    print('R72 calibration',json.dumps({k:result[k] for k in
          ('selected_strength','strengths','model_weights_unchanged')}),flush=True)


if __name__=='__main__': main()
