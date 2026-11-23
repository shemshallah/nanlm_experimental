#!/usr/bin/env python3
"""Train-only word candidate integration: frozen NaNCM signed cloud + 4-order book-disjoint ngrams.
No TEST. The v176 neural parent has previously seen TRAIN books; gate is experimental.
Reference scores on reused VAL must be measured only after model hyperparameters frozen.
"""
from pathlib import Path
import json,hashlib,time,argparse,math
from collections import Counter
import numpy as np,lightgbm as lgb
ROOT=Path(__file__).resolve().parents[3]
RES=ROOT/'results/v180';SRC=ROOT/'results/v175_release3'

def prepare(split,limit=None):
 path=next(RES.glob(f'{split}_neural_*.npz'))
 if split=='train':paths=sorted(RES.glob('train_neural_*.npz'),key=lambda x:int(x.stem.split('_')[-1]));path=paths[-1]
 if split=='val':paths=sorted(RES.glob('val_neural_*.npz'),key=lambda x:int(x.stem.split('_')[-1]));path=paths[-1]
 print('LOADING',path,flush=True)
 d=np.load(path);X=np.load(SRC/(split+'.npz'))['X'][d['rows']]
 return d,X

def observed_copies(x):
 """Only previous observed words, never the unseen target."""
 counts=Counter(int(t) for t in x if t>=0)
 match=[Counter() for _ in range(3)]
 recency={}
 for j in range(47):
  target=int(x[j+1]);
  if target<0:continue
  recency[target]=47-j
  for n in range(1,4):
   if j-n+1<0:continue
   if np.array_equal(x[j-n+1:j+1],x[48-n:48]):match[n-1][target]+=1
 return counts,match,recency

def features(d,X,nrows=None):
 if nrows is None:nrows=len(d['target'])
 wordfreq=np.bincount(np.load(SRC/'train.npz')['Y'].astype('int64'),minlength=12000).astype('float32')
 freq_log=np.log1p(wordfreq)
 # Word lengths are grounded in the same inherited 12k lexical order.
 from sys import path as sys_path
 sys_path.insert(0,str(ROOT/'experiments/nancm/v175'))
 from train_long175 import load_models
 _,_,names,_,_=load_models();length=np.array([len(x) for x in names],dtype=np.int16)
 n=nrows;rows=[];labels=[];sizes=[];meta=[]
 st=time.monotonic()
 for i in range(n):
  gold=int(d['target'][i]);ng=d['ngram'][i];ns=d['ngram_probs'][i];nt=d['neural_top'][i];np0=d['neural_probs'][i]
  dictionary={int(w):[rank,float(p)] for rank,(w,p) in enumerate(zip(nt,np0))}
  for k in range(4):
   for s in (0,1):
    token=int(ng[k,s]);
    if token<12000 and token not in dictionary:dictionary[token]=[64,float(ns[k,s])]
  tokens=list(dictionary);sizes.append(len(tokens));c0,cop,recency=observed_copies(X[i]);top=float(np0[0]);top2=float(np0[1]);mass=float(np0.sum());max_order=0
  for token in tokens:
   rank,p=dictionary[token];fp=[
    math.log(max(1e-30,p)),rank/64.,math.log(max(1e-30,p)/max(1e-30,top)),
    math.log(max(1e-30,top)/max(1e-30,top2)),mass,
    float(freq_log[token]),length[token]/20.,
    math.log1p(c0.get(token,0)), 1./(1.+recency.get(token,1000)),
    math.log1p(cop[0].get(token,0)),math.log1p(cop[1].get(token,0)),math.log1p(cop[2].get(token,0)),
    ]
   for k in range(4):
    best=int(ng[k,0]);second=int(ng[k,1]);a=token==best;b=token==second
    total=int(ng[k,4]);count=int(ng[k,2] if a else ng[k,3] if b else 0)
    fp.extend([int(a),int(b),math.log1p(count),math.log1p(total),count/(total+1),math.log1p(int(ng[k,5]))])
   fp.extend([int(rank==0),int(rank<5)])
   rows.append(fp);labels.append(token==gold)
  meta.append({'gold':gold,'book':int(d['book'][i]),'baseline':int(nt[0]),'oracle':gold in dictionary,'candidates':np.array(tokens,dtype=np.int16)})
  if (i+1)%4000==0:print('FEATURES',i+1,'rows',len(rows),'elapsed',round(time.monotonic()-st,1),flush=True)
 return np.asarray(rows,dtype=np.float32),np.asarray(labels,dtype=np.int8),np.asarray(sizes,dtype=np.int32),meta

def candidate_accuracy(pred,groups,meta):
 offsets=np.r_[0,np.cumsum(groups)];correct=0;oracle=0;books={}
 for i,m in enumerate(meta):
  chosen=int(m['candidates'][np.argmax(pred[offsets[i]:offsets[i+1]])]);correct+=chosen==m['gold'];oracle+=m['oracle']
  books.setdefault(m['book'],[0,0]);books[m['book']][0]+=chosen==m['gold'];books[m['book']][1]+=1
 return {'top1':int(correct),'n':len(meta),'top1_percent':100.*correct/max(1,len(meta)),'oracle_recall':int(oracle),'oracle_percent':100.*oracle/max(1,len(meta)),'books':len(books)}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--trees',type=int,default=90);ap.add_argument('--leaves',type=int,default=15);args=ap.parse_args();st=time.monotonic()
 d,X=prepare('train');A,Y,G,meta=features(d,X)
 books=np.array([m['book'] for m in meta]);trainbooks=np.unique(books)[::4];trainmask=np.isin(books,trainbooks,invert=True)
 # Grouped book separation in gate fitting and selection. Both cohorts are TRAIN books.
 groupmask=np.repeat(trainmask,G);a=A[groupmask];y=Y[groupmask];g=G[trainmask];b=A[~groupmask];yy=Y[~groupmask];gg=G[~trainmask]
 fit_meta=[m for i,m in enumerate(meta) if trainmask[i]];hold_meta=[m for i,m in enumerate(meta) if not trainmask[i]]
 print('CALIB FIT',len(g),'holdout',len(gg),'candidate_oracle',np.mean([m['oracle'] for m in meta]),flush=True)
 model=lgb.LGBMRanker(objective='lambdarank',n_estimators=args.trees,num_leaves=args.leaves,learning_rate=.07,
                     min_child_samples=120,reg_lambda=5.,n_jobs=4,verbosity=-1,max_bin=127,random_state=180)
 model.fit(a,y,group=g)
 fit_pred=model.predict(a);ho_pred=model.predict(b)
 print('GATE FIT',candidate_accuracy(fit_pred,g,fit_meta));print('GATE TRAIN_BOOK_HOLDOUT',candidate_accuracy(ho_pred,gg,hold_meta),flush=True)
 # No VAL labels read until ranker is frozen.
 print('FROZEN trees',args.trees,'leaves',args.leaves,'elapsed',round(time.monotonic()-st,1),flush=True)
 dv,Xv=prepare('val');V,vy,vg,vm=features(dv,Xv)
 pr=model.predict(V);result=candidate_accuracy(pr,vg,vm)
 ref=sum(m['baseline']==m['gold'] for m in vm)
 result.update({'parent_top1':ref,'gain':result['top1']-ref,'parent_top1_percent':100*ref/len(vm),
 'training_gate_books':len(np.unique(books[trainmask])),'selection_gate_books':len(np.unique(books[~trainmask])),
 'validation_status':'reused_55_books_inherited_parent_exposure_uncertain','trees':args.trees,'leaves':args.leaves,
 'training_examples':len(g),'ranker_fit_rows':len(a),'features':A.shape[1],
 'parent_checkpoint_sha256':hashlib.sha256((ROOT/'checkpoints/v175_signed_episodic_48.pt').read_bytes()).hexdigest(),
 'memory_stream_sha256':hashlib.sha256((RES/'train_observed_stream.bin').read_bytes()).hexdigest(),
 'policy':'no TEST; training ngram index only TRAIN; ranker trained TRAIN books and frozen before VAL',
 'seconds':time.monotonic()-st})
 print('FINAL VAL',json.dumps(result,indent=2),flush=True)
 (RES/'v180_fused_val.json').write_text(json.dumps(result,indent=2)+'\n')
 model.booster_.save_model(str(RES/'v180_causal_gate.lgbm.txt'))

if __name__=='__main__':main()
