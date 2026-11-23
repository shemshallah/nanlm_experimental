#!/usr/bin/env python3
"""Fully native-compiled feature extraction (Numba) for the opt-in signed-cloud word gate.
Frozen parent v176 and TRAIN-only n-gram memory remain separate and recoverable.
"""
from pathlib import Path
import json,hashlib,time,argparse
import numpy as np,lightgbm as lgb
from numba import njit
ROOT=Path(__file__).resolve().parents[3];R=ROOT/'results/v180';SRC=ROOT/'results/v175_release3'

@njit(cache=True)
def create_features(X,nt,nprob,ng,np_ng,gold,freqlog,lengths):
 n=len(gold);D=38;f=np.zeros((n,72,D),dtype=np.float32);yy=np.zeros((n,72),dtype=np.int8)
 tokens=np.full((n,72),-1,dtype=np.int32);sizes=np.zeros(n,dtype=np.int32)
 for i in range(n):
  top=float(nprob[i,0]);top2=float(nprob[i,1]);mass=0.
  for k in range(64):mass+=nprob[i,k]
  count=0
  for z in range(72):
   if z<64:token=int(nt[i,z]);rank=z;proba=float(nprob[i,z])
   else:
    idx=z-64;order=idx//2;slot=idx%2;token=int(ng[i,order,slot]);rank=64;proba=float(np_ng[i,order,slot])
   if token<0 or token>=12000:continue
   exists=False
   for k in range(count):
    if tokens[i,k]==token:exists=True;break
   if exists:continue
   tokens[i,count]=token;yy[i,count]=int(token==gold[i]);v=f[i,count,:]
   v[0]=np.log(max(1e-30,proba));v[1]=rank/64.;v[2]=np.log(max(1e-30,proba)/max(1e-30,top))
   v[3]=np.log(max(1e-30,top)/max(1e-30,top2));v[4]=mass
   v[5]=freqlog[token];v[6]=lengths[token]/20.
   hits=0;nearest=1000;copy1=0;copy2=0;copy3=0
   for t in range(48):
    if X[i,t]==token:
     hits+=1;nearest=min(nearest,47-t)
   for t in range(47):
    if X[i,t+1]!=token:continue
    if X[i,t]>=0 and X[i,t]==X[i,47]:
     copy1+=1
     if t>=1 and X[i,t-1]>=0 and X[i,t-1]==X[i,46]:
      copy2+=1
      if t>=2 and X[i,t-2]>=0 and X[i,t-2]==X[i,45]:copy3+=1
   v[7]=np.log1p(hits);v[8]=1./(1.+nearest);v[9]=np.log1p(copy1);v[10]=np.log1p(copy2);v[11]=np.log1p(copy3)
   for order in range(4):
    best=int(ng[i,order,0]);second=int(ng[i,order,1]);a=token==best;b=token==second
    total=int(ng[i,order,4]);num=int(ng[i,order,2] if a else ng[i,order,3] if b else 0)
    offset=12+6*order
    v[offset]=int(a);v[offset+1]=int(b);v[offset+2]=np.log1p(num)
    v[offset+3]=np.log1p(total);v[offset+4]=num/(total+1.);v[offset+5]=np.log1p(int(ng[i,order,5]))
   v[36]=int(rank==0);v[37]=int(rank<5)
   count+=1
  sizes[i]=count
 return f,yy,tokens,sizes

def load(split,names,freqlog):
 path=sorted(R.glob(f'{split}_neural_*.npz'),key=lambda x:int(x.stem.split('_')[-1]))[-1]
 d=np.load(path)
 X=np.load(SRC/f'{split}.npz')['X'][d['rows']]
 length=np.array([len(s) for s in names],dtype=np.float32)
 F,Y,tokens,sizes=create_features(X,d['neural_top'],d['neural_probs'],d['ngram'],d['ngram_probs'],d['target'],freqlog,length)
 mask=np.arange(72)[None,:]<sizes[:,None]
 return F[mask],Y[mask],tokens,sizes,d['target'],d['book'],d['neural_top'][:,0]

def accuracy(pred,tokens,sizes,gold):
 scores=np.full(tokens.shape,-1e12,dtype=np.float32)
 mask=np.arange(72)[None,:]<sizes[:,None];scores[mask]=pred
 chosen=tokens[np.arange(len(gold)),np.argmax(scores,axis=1)]
 oracle=(tokens==gold[:,None]).any(axis=1)
 return dict(n=len(gold),top1=int(np.sum(chosen==gold)),top1_percent=100*float(np.mean(chosen==gold)),oracle=int(np.sum(oracle)),oracle_percent=100*float(np.mean(oracle)))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--trees',type=int,default=80);ap.add_argument('--leaves',type=int,default=15);args=ap.parse_args()
 t=time.monotonic()
 import sys
 sys.path.insert(0,str(ROOT/'experiments/nancm/v175'))
 from train_long175 import load_models
 _,_,names,_,_=load_models()
 freqlog=np.log1p(np.bincount(np.load(SRC/'train.npz')['Y'].astype('int64'),minlength=12000).astype('float32'))
 A,Y,tokens,G,gold,books,base=load('train',names,freqlog)
 unique=np.unique(books);holdbooks=unique[::4]
 hold=np.isin(books,holdbooks);fit=~hold
 m=np.repeat(fit,G);V=A[m];vy=Y[m];vg=G[fit];W=A[~m];wy=Y[~m];wg=G[hold]
 print('DATA',A.shape,'FIT',V.shape,'HOLD',W.shape,'seconds',time.monotonic()-t,flush=True)
 model=lgb.LGBMRanker(objective='lambdarank',n_estimators=args.trees,num_leaves=args.leaves,learning_rate=.06,
                     min_child_samples=120,reg_lambda=5.,n_jobs=4,verbosity=-1,max_bin=127,random_state=180)
 model.fit(V,vy,group=vg)
 train_pred=model.predict(V);hold_pred=model.predict(W)
 print('FIT',accuracy(train_pred,tokens[fit],G[fit],gold[fit]),flush=True)
 print('TRAIN_BOOK_HOLDOUT',accuracy(hold_pred,tokens[hold],G[hold],gold[hold]),flush=True)
 print('FROZEN MODEL, NOW EVALUATING VAL',time.monotonic()-t,flush=True)
 B,by,btokens,BG,bgold,bbook,baseline=load('val',names,freqlog)
 val_pred=model.predict(B);res=accuracy(val_pred,btokens,BG,bgold)
 basen=int(np.sum(baseline==bgold));res.update(dict(parent_top1=basen,parent_percent=100*basen/len(bgold),gain=res['top1']-basen,
  benchmark='55 reused book-disjoint NaNROUTER Release3 VAL books, vocab-conditioned 12k targets',
  cutoff=64,ngram_order=4,tree_count=args.trees,leaves=args.leaves,seconds=time.monotonic()-t,
  train_parent_pretrained_on_fit=True,validation_already_explored=True,test_untouched=True,
  full_decoder_promoted=False,byte_probabilities_unchanged=True))
 print('VAL',json.dumps(res,indent=2),flush=True)
 (R/'v180_ranker_val.json').write_text(json.dumps(res,indent=2)+'\n')
 model.booster_.save_model(str(R/'v180_ranker.lgbm.txt'))

if __name__=='__main__':main()
