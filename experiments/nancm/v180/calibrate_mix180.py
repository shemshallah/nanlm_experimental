#!/usr/bin/env python3
"""Fast, train-selected positive additive full-word mixture candidates.
Each ngram mass is a valid Dirichlet posterior count component. Unmodeled
remaining probability is still in the inherited distribution for the full model;
this *ranking-only* screen claims no normalized new distribution or BPB.
"""
import json,time,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3];R=ROOT/'results/v180'

def features(split):
 d=np.load(sorted(R.glob(f'{split}_neural_*.npz'),key=lambda x:int(x.stem.split('_')[-1]))[-1])
 n=len(d['target']);top=d['neural_top'].astype('int32');ng=d['ngram'].astype('int32');p=d['neural_probs'];np0=d['ngram_probs']
 ids=np.concatenate([top,ng[:,:,:2].reshape(n,8)],axis=1)
 probs=np.concatenate([p,np0.reshape(n,8)],axis=1)
 cand=ng[:,:,:2]
 f=np.zeros((n,72,4),dtype=np.float32)
 for order in range(4):
  counts=ng[:,order,2:4].astype('float32');total=ng[:,order,4:5].astype('float32')
  for slot in (0,1):
   matches=(ids==cand[:,order,slot:slot+1]) & (ids<12000)
   f[:,:,order]+=(matches * (counts[:,slot:slot+1]/(total+32.)))
 f=np.maximum(f,0)
 return d,ids,probs,f

def main():
 t=time.monotonic();tdata,tid,tp,tf=features('train');vdata,vid,vp,vf=features('val')
 gold=tdata['target'];goldv=vdata['target'];defpt=np.argmax(tp,axis=1)
 assert np.array_equal(tid[np.arange(len(tid)),defpt],tdata['neural_top'][:,0])
 base=int(np.mean(tid[:,0]==gold)*len(gold));baseval=int(np.mean(vid[:,0]==goldv)*len(goldv))
 candidates=[]
 for weights in ([1,1,1,1],[1,.5,.3,.1],[.5,.6,1,1.5],[.2,.4,.8,2],[2,1,.5,.25],[0,1,1,1]):
  w=np.array(weights,'float32');ff=np.einsum('nko,o->nk',tf,w)
  for alpha in (0,.005,.01,.02,.04,.08,.12,.2,.4,.8,1.6):
   scores=tp+alpha*ff
   pred=tid[np.arange(len(tid)),np.argmax(scores,axis=1)]
   hit=int(np.sum(pred==gold))
   candidates.append((hit,float(alpha),weights))
 candidates.sort(key=lambda z:(z[0],-z[1]),reverse=True)
 selected=candidates[0];_,a,ww=selected;vv=vp+a*np.einsum('nko,o->nk',vf,np.array(ww,'float32'))
 pp=vid[np.arange(len(vid)),np.argmax(vv,axis=1)];hitv=int(np.sum(pp==goldv))
 oracle=int(np.sum((vid==goldv[:,None]).any(axis=1)))
 out=dict(name='v180_train_selected_dirichlet_word_proposal',train_examples=len(gold),validation_examples=len(goldv),
  train_baseline=base,train_best=selected[0],val_baseline=baseval,val_pred=hitv,val_gain=hitv-baseval,
  val_oracle=oracle,selected_alpha=a,selected_order_weights=ww,
  selection='TRAIN book-separated calibration, neural parent itself pretrained on full TRAIN',
  status='research_only; candidate ranker only, not normalized distribution or real byte BPB',
  reused_val=True,elapsed_seconds=time.monotonic()-t)
 (R/'v180_train_selected_mix.json').write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps(out,indent=2),flush=True)

if __name__=='__main__':main()
