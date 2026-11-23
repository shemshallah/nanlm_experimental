#!/usr/bin/env python3
"""Mathematically normalized sparse n-gram residual on inherited 12000-way word P.
P180(w)=(1-sum_k gamma[k]*mass_k)*P176(w)+sum_k gamma[k]*count_k(w)/total_k.
Here count_k(w) retains only top2 per context; mass_k is exactly their sum.
All unrepresented n-gram mass remains in parent P176, so simplex is exact.
Select gamma from TRAIN-only data before scoring reused VAL. No TEST.
"""
from pathlib import Path
import json,time,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[3];R=ROOT/'results/v180'

def get(split):
 d=np.load(sorted(R.glob(f'{split}_neural_*.npz'),key=lambda x:int(x.stem.split('_')[-1]))[-1])
 n=len(d['target']);ng=d['ngram'].astype(np.int32);nt=d['neural_top'].astype(np.int32)
 ids=np.concatenate([nt,ng[:,:,:2].reshape(n,8)],axis=1)
 probs=np.concatenate([d['neural_probs'],d['ngram_probs'].reshape(n,8)],axis=1).astype('float64')
 ratio=np.zeros((n,72,4),dtype='float32');mass=np.zeros((n,4),dtype='float32');targetratio=np.zeros((n,4),dtype='float32')
 for k in range(4):
  total=np.maximum(1,ng[:,k,4:5]);
  for j in range(2):
   token=ng[:,k,j:j+1];counts=ng[:,k,2+j:3+j];r=counts/total
   valid=token<12000
   ratio[:,:,k]+=np.where((ids==token)&valid,r,0).astype('float32')
   mass[:,k]+=np.where(valid[:,0],r[:,0],0)
   targetratio[:,k]+=np.where((d['target']==token[:,0])&valid[:,0],r[:,0],0)
 return d,ids,probs,ratio,mass,targetratio

def calculate(d,ids,probs,ratio,mass,targetratio,gamma):
 g=np.asarray(gamma,dtype='float64')
 scale=1.0-mass@g
 assert np.min(scale)>=-1e-5,(np.min(scale),g)
 scores=probs*scale[:,None]+np.einsum('nko,o->nk',ratio,g)
 # if duplicates exist, each has the SAME normalized word probability
 best=ids[np.arange(len(ids)),np.argmax(scores,axis=1)];target=d['target']
 ptarget=d['target_prob']*scale+targetratio@g
 nll=-np.log(np.clip(ptarget,1e-30,1.)).mean()
 return {'top1':int((best==target).sum()),'nll':float(nll),'n':len(target)}

def main():
 st=time.monotonic();t=get('train');v=get('val');patterns=[[1,1,1,1],[0,1,1,1],[1,1,2,4],[4,2,1,0],[0,0,1,1],[0,0,0,1],[0,1,0,0],[1,0,0,0]]
 strengths=[0,.01,.03,.05,.1,.15,.2,.3,.4,.6,.8,1.]
 cand=[]
 for pat in patterns:
  p=np.array(pat,dtype='float64');p=p/p.sum()
  for s in strengths:
   g=s*p;res=calculate(*t,g);cand.append((res['nll'], -res['top1'],g.tolist(),res))
 cand.sort();_,_,selected,fit=cand[0]
 val=calculate(*v,selected)
 ref=calculate(*v,[0,0,0,0])
 print('NORMALIZED_TRAIN_SELECTED',json.dumps({'selected_gamma':selected,'train':fit,'val':val,'baseline':ref},indent=2),flush=True)
 result={'version':180,'normalized':True,'candidate_residual_exact':True,'selection_metric':'TRAIN-only book subset NLL',
  'train_examples':len(t[0]['target']),'val_examples':len(v[0]['target']),'gamma':selected,
  'train':fit,'val':val,'val_baseline':ref,'val_gain_top1':val['top1']-ref['top1'],
  'val_gain_nll':ref['nll']-val['nll'],'corpus':'NaNROUTER Release3 55-book reused VAL',
  'test_used':False,'parent_v176_frozen':True,'note':'Complete word-level normalized distribution. Does not claim end-to-end byte BPB nor open-vocabulary calibration.',
  'elapsed_seconds':time.monotonic()-st}
 (R/'v180_normalized_mixture.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
