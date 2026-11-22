#!/usr/bin/env python3
"""NaNCM v173: exact-normalized softmax training of inherited signed-cloud word head.

NCE -> exact 12k-way cross entropy; retain all 6,000 inherited AdamW moments.
Every checkpoint is a candidate and selected on TRAIN-derived SHA context-key holdout.
Validation is explicitly not book-disjoint. Exposed DEV is evaluated only after selection.
"""
from __future__ import annotations
import os, json, math, time, sys, hashlib, copy
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'v172'))
from train_conditional_cloud172 import ROOT, load_parent,load_examples,CloudNCE,eval_full,dev_cohort

def tensors(snapshot):
 return {k:v.detach().clone() for k,v in snapshot.items()}

def main():
 torch.set_num_threads(int(os.getenv('V173_THREADS','2')))
 torch.manual_seed(173)
 names,C,O,B,spatial,gain,fp=load_parent();vocab={w:i for i,w in enumerate(names)}
 resume_file=ROOT/'checkpoints/v173_exact_ce_word.pt'
 resumed=resume_file.exists()
 ck=torch.load(resume_file if resumed else ROOT/'checkpoints/v172_conditional_signed_cloud.pt',weights_only=False,map_location='cpu')
 assert tuple(fp)==tuple(ck['fingerprints']) and names==ck['word_order']
 X,Y,W,V,freq=load_examples(vocab)
 train=np.where(~V)[0];val=np.where(V)[0]
 model=CloudNCE(C,O,B,spatial,gain,names)
 model.load_state_dict(ck['state_dict'])
 opt=torch.optim.AdamW([{'params':[model.ctx.weight,model.out.weight],'lr':.0004},{'params':[model.bias,model.cloud_projection.weight],'lr':.001},{'params':[model.gain],'lr':.00004}],weight_decay=.002)
 opt.load_state_dict(ck['optimizer_state_dict'])
 if not resumed:
  for group in opt.param_groups: group['lr']*=.2
 rng=np.random.default_rng(173+int(ck['step']))
 iters=int(os.getenv('V173_UPDATES','240'));batch=int(os.getenv('V173_BATCH','96'))
 sample=rng.choice(train,size=iters*batch,p=W[train]/W[train].sum(),replace=True).reshape(iters,batch)
 val_idx=val[:1536]
 init=eval_full(model,X[val_idx],Y[val_idx]);print('INIT_STEP',ck['step'],'VAL',init,flush=True)
 selected={'nll':init['nll_nats'],'step':int(ck['step']),'model':tensors(model.state_dict()),'optimizer':copy.deepcopy(opt.state_dict()),'val':init}
 t=time.perf_counter();history=[]
 for i in range(iters):
  ix=sample[i]
  ids=torch.from_numpy(X[ix].astype('int64'))
  labels=torch.from_numpy(Y[ix].astype('int64'))
  model.train()
  q=model.query(ids)
  logits=model.gain.clamp(0,25)*q@model.out.weight.T+model.bias
  loss=F.cross_entropy(logits,labels)+.0001*model.cloud_projection.weight.square().sum()
  opt.zero_grad(set_to_none=True)
  loss.backward()
  torch.nn.utils.clip_grad_norm_(model.parameters(),2.)
  opt.step()
  if (i+1)%40==0 or i+1==iters:
   met=eval_full(model,X[val_idx],Y[val_idx])
   history.append({'updates':i+1,'validation':met,'loss':float(loss.detach()),'seconds':round(time.perf_counter()-t,2)})
   if met['nll_nats']<selected['nll']:
    selected={'nll':met['nll_nats'],'step':int(ck['step'])+i+1,'model':tensors(model.state_dict()),'optimizer':copy.deepcopy(opt.state_dict()),'val':met}
   print('ITER',i+1,'LOSS',float(loss),'VAL',met,'SEL',selected['step'],'SEC',round(time.perf_counter()-t,2),flush=True)
 model.load_state_dict(selected['model']);Xdev,Ydev,devrows=dev_cohort(vocab);known=np.where(Ydev>=0)[0]
 eval_dev=eval_full(model,Xdev[known],Ydev[known]);print('DEV_REUSED',eval_dev,'STEP',selected['step'],flush=True)
 out={'architecture':'NaNCM v173 exact-normalized continuation of v172 signed conditional word model',
  'step':selected['step'],'previous_step':ck['step'],'optimizer_resumed':True,
  'state_dict':selected['model'],'optimizer_state_dict':selected['optimizer'],
  'word_order':names,'fingerprints':fp,'loss':'exact 12000-word softmax cross-entropy',
  'training':'TRAIN-derived v164 per-context conditional counts; 5% SHA key validation, NOT book-disjoint',
  'selection':'TRAIN context-key holdout NLL only, not DEV'}
 pt=ROOT/'checkpoints/v173_exact_ce_word.pt';torch.save(out,pt)
 report={'name':'NaNCM v173 exact CE refinement','status':'EXPERIMENT_ONLY','steps_run':iters,
  'selected_step':selected['step'],'optimizer_resumed':True,'parent_step':ck['step'],'resumed_from_v173':resumed,
  'validation_parent':init,'validation_selected':selected['val'],
  'dev_exposed_selected':eval_dev,'train_context_rows':len(train),'val_context_rows':len(val),
  'test_not_pristine':True,'val_not_book_disjoint':True,'history':history}
 (ROOT/'results/v173_exact_ce_train.json').write_text(json.dumps(report,indent=2))
 print('SAVED',pt,pt.stat().st_size,flush=True)
if __name__=='__main__':main()