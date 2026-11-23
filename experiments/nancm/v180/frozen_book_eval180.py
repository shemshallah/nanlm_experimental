#!/usr/bin/env python3
"""Frozen gamma=[.1,.1,.1,.1] strict top1/top5, paired whole-book bootstrap.
Reused VAL diagnostic only; no TEST selection or inspection.
"""
import json,time,hashlib
import numpy as np
from normalized_mix180 import get,R

def main():
 d,ids,probs,ratio,mass,targetratio=get('val')
 gamma=np.array([.1,.1,.1,.1],dtype=np.float64)
 scale=1-mass@gamma
 scores=probs*scale[:,None]+np.einsum('nko,o->nk',ratio,gamma)
 n=len(scores);gold=d['target'];books=d['book'];parent=d['neural_top']
 chosen=ids[np.arange(n),np.argmax(scores,axis=1)]
 top5=np.empty((n,5),np.int32);ix=np.argsort(-scores,axis=1,kind='stable')
 for i in range(n):
  found=set();sz=0
  for j in ix[i]:
   word=int(ids[i,j]);
   if word>=12000 or word in found:continue
   top5[i,sz]=word;found.add(word);sz+=1
   if sz==5:break
  assert sz==5
 model_correct=chosen==gold;parent_correct=parent[:,0]==gold
 model_t5=np.any(top5==gold[:,None],axis=1);parent_t5=np.any(parent[:,:5]==gold[:,None],axis=1)
 parent_nll=-np.log(np.maximum(1e-30,d['target_prob'])).astype('float64')
 model_p=d['target_prob']*scale+targetratio@gamma
 model_nll=-np.log(np.maximum(1e-30,model_p))
 assert (model_p>=0).all() and np.isfinite(model_nll).all()
 vals=[]
 for book in np.unique(books):
  mask=books==book;vals.append([int(mask.sum()),int(parent_correct[mask].sum()),int(model_correct[mask].sum()),
   float(parent_nll[mask].sum()),float(model_nll[mask].sum()),int(parent_t5[mask].sum()),int(model_t5[mask].sum())])
 arr=np.array(vals);rng=np.random.default_rng(180);boot=np.zeros((3000,3),dtype=float)
 for i in range(len(boot)):
  v=arr[rng.integers(0,len(arr),size=len(arr))];N=v[:,0].sum()
  boot[i]=[(v[:,2].sum()-v[:,1].sum())/N,(v[:,3].sum()-v[:,4].sum())/N,(v[:,6].sum()-v[:,5].sum())/N]
 result={
  'status':'reused_VAL_diagnostic_unpromoted', 'books':len(vals),'positions':n,
  'parent_top1':int(parent_correct.sum()),'v180_top1':int(model_correct.sum()),
  'parent_top5':int(parent_t5.sum()),'v180_top5':int(model_t5.sum()),
  'parent_word_nll':float(parent_nll.mean()),'v180_word_nll':float(model_nll.mean()),
  'gain_top1':int(model_correct.sum()-parent_correct.sum()),
  'gain_top5':int(model_t5.sum()-parent_t5.sum()),
  'gain_nll':float(parent_nll.mean()-model_nll.mean()),
  'paired_book_bootstrap_reps':len(boot),
  'top1_gain_95':np.quantile(boot[:,0],[.025,.5,.975]).tolist(),
  'nll_gain_95':np.quantile(boot[:,1],[.025,.5,.975]).tolist(),
  'top5_gain_95':np.quantile(boot[:,2],[.025,.5,.975]).tolist(),
  'top5_computation':'unique candidate words from exact normalized top64 neural plus top2 order1-4; all remaining words retain common scale and cannot exceed original top64',
  'train_selected_gamma':gamma.tolist(),
  'training_corpus_transition_count':2650489,
  'test_used':False,
  'inherited_model_weights_and_optimizers_unchanged':True}
 print('FROZEN WORD GATE',json.dumps(result,indent=2),flush=True)
 (R/'v180_frozen_word_gate.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
