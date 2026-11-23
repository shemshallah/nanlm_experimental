#!/usr/bin/env python3
"""Can the inherited v170 P verifier choose better top-five words?
Calibration is entirely on TRAIN books. VAL books are untouched until a single
best setting is frozen. No reused TEST data is consulted.
"""
import argparse,ctypes,json,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'experiments/nancm/v177'),str(ROOT/'experiments/nancm/v176'),str(ROOT/'experiments/nancm/v175')]
from test_native177 import init
from train_long175 import load_models,load_split
import torch
F=ctypes.c_float;I=ctypes.c_int

def collect(lib,h,names,split,maximum,seed):
    X,Y,B=load_split(split,ROOT/'results/v175_release3')
    rng=np.random.default_rng(seed)
    unique=np.unique(B);per=max(1,maximum//len(unique))
    ids=[]
    for book in unique:
        cand=np.flatnonzero(B==book)
        ids.extend(rng.choice(cand,size=min(per,len(cand)),replace=False).tolist())
    ids=np.array(ids,dtype=int)
    if len(ids)>maximum:ids=ids[:maximum]
    features=[];start=time.perf_counter()
    for k,j in enumerate(ids):
        raw=(' '.join(names[int(x)] if x>=0 else 'zzzzquuxwordunknown' for x in X[j])+' ').encode()
        assert lib.wf177_begin(h,ctypes.c_char_p(raw),len(raw))==0
        scores=(F*5)();rawcalls=I();uniq=I()
        assert lib.wf177_prepare_top5(h,scores,ctypes.byref(rawcalls),ctypes.byref(uniq))==0
        rows=[]
        for rank in range(5):
            ix=I();w=ctypes.create_string_buffer(64);p=F()
            assert lib.wf177_top5(h,rank,ctypes.byref(ix),w,64,ctypes.byref(p))==0
            rows.append((ix.value, float(np.log(max(1e-30,p.value))),float(scores[rank]),len(w.value)+1))
        features.append((int(B[j]),int(Y[j]),rows,int(uniq.value)))
    return features,time.perf_counter()-start

def evaluate(rows,method,alpha):
    n=len(rows);base=top1=cover=0;gold_ce=0.;count_cover=0
    for book,gold,cands,calls in rows:
        base+=cands[0][0]==gold
        found=[k for k,x in enumerate(cands) if x[0]==gold]
        cover+=bool(found)
        scored=[]
        for cid,lp,avg,L in cands:
            byte=avg if method=='per_byte' else L*avg
            scored.append(lp+alpha*byte)
        j=max(range(5),key=lambda k:(scored[k],-k));top1+=cands[j][0]==gold
        if found:
            val=np.array(scored,dtype='float64');mx=max(val);val-=mx
            logz=mx+np.log(np.exp(val).sum())
            gold_ce+=logz-scored[found[0]];count_cover+=1
    return dict(n=n,top1=top1,baseline=base,top5_oracle_ceiling=cover,top1_gain=top1-base,
                conditional_top5_nll=gold_ce/max(1,count_cover))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--train',type=int,default=1200)
    ap.add_argument('--val',type=int,default=1600);a=ap.parse_args()
    torch.set_num_threads(2)
    _,_,names,_,_=load_models()
    old,oldh,lib,h=init()
    try:
        train,ttrain=collect(lib,h,names,'train',a.train,1771)
        methods=['per_byte','whole_word']
        alphas=[-1.,-.5,-.25,-.1,0.,.03,.06,.1,.15,.2,.3,.4,.5,.7,1.]
        # Choose using TRAIN proper conditional top-five NLL. Do not use VAL labels.
        table=[(evaluate(train,m,x),m,x) for m in methods for x in alphas]
        table.sort(key=lambda t:(t[0]['conditional_top5_nll'],-t[0]['top1'],abs(t[2])))
        best,method,alpha=table[0]
        val,tval=collect(lib,h,names,'val',a.val,1772)
        test=evaluate(val,method,alpha)
        baseline=evaluate(val,'per_byte',0.)
        res=dict(status='PASS',training_examples=len(train),validation_examples=len(val),
                 training_books=len(set(x[0] for x in train)),validation_books=len(set(x[0] for x in val)),
                 selected_method=method,selected_alpha=alpha,selection_metric='TRAIN conditional top5 NLL',
                 train_best=best,val_frozen_hybrid=test,val_parent=baseline,
                 val_gain=test['top1']-baseline['top1'],
                 timing_seconds=dict(train=ttrain,val=tval),source='NaNROUTER Release0003, reused 55-book VAL, no TEST labels',
                 note='Only reorders the inherited top5; word distribution not claimed normalized beyond top5. No full-decoder promotion.')
        (ROOT/'results/v177_train_selected_byte_rerank.json').write_text(json.dumps(res,indent=2))
        print('V177 WORD RERANK',json.dumps(res,indent=2),flush=True)
    finally:
        old.wf176_hybrid_close(oldh);lib.wf177_close(h)
if __name__=='__main__':main()