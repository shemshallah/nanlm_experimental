#!/usr/bin/env python3
"""Learn causal native mixture gate from raw TRAIN-book next-word log likelihoods.
Labels enter only training loss; native gate inputs are all pre-target features.
"""
import copy,hashlib,json,math
from pathlib import Path
import numpy as np
import torch
R=Path(__file__).resolve().parents[3]
rows=[json.loads(s) for s in (R/'results/v168_gate_calibration_rows.jsonl').open()]
train=[];val=[]
books=sorted(set(r['book'] for r in rows),key=lambda x:hashlib.sha256(('NaNCM_v168_book_gate:'+x).encode()).digest())
assert len(books)==100
training=set(books[:80])
for r in rows:
 if r['v167_alpha']<=0:continue
 (train if r['book'] in training else val).append(r)
print('COHORTS',json.dumps({'train':len(train),'val':len(val),'train_books':80,'val_books':20}),flush=True)

def arrays(rows):
 f=torch.tensor([r['f'] for r in rows],dtype=torch.float64)
 p=torch.tensor([r['p'] for r in rows],dtype=torch.float64).clamp_min(1e-40)
 q=torch.tensor([r['q'] for r in rows],dtype=torch.float64)
 base=torch.tensor([r['v167_alpha'] for r in rows],dtype=torch.float64)
 return f,p,q,base
T=arrays(train);V=arrays(val)
def scores(x,weight):
 f,p,q,base=x;alpha=.5*torch.sigmoid(f@weight)
 mix=((1-alpha)*p+alpha*q).clamp_min(1e-40)
 return {'n':len(p),'nll':-torch.log(mix).mean().item(),'alpha_mean':alpha.mean().item(),
  'alpha_gt_035':(alpha>.35).double().mean().item(),
  'better_than_v167_pct':(mix>((1-base)*p+base*q)).double().mean().item() *100}
def baselines(x):
 f,p,q,base=x
 def nll(a):return float(-torch.log(((1-a)*p+a*q).clamp_min(1e-40)).mean())
 return {'v167':nll(base),'const_005':nll(.05),'const_025':nll(.25),'const_050':nll(.5)}
print('BASE_TRAIN',baselines(T),flush=True)
print('BASE_VAL',baselines(V),flush=True)
# Default near original v167 agreement heuristic; train a smooth interpolating gate.
w=torch.nn.Parameter(torch.tensor([-2.197224577,0,0,0,0,4.394449,0],dtype=torch.float64))
opt=torch.optim.Adam([w],lr=.018)
best=(float('inf'),None,None)
records=[]
for step in range(1,601):
 opt.zero_grad()
 f,p,q,base=T
 a=.5*torch.sigmoid(f@w)
 loss=-torch.log(((1-a)*p+a*q).clamp_min(1e-40)).mean()
 # Enforce learned smooth moderation; preserve positive parent contribution.
 reg=.004*((a-base)**2).mean()+.00012*(w[1:]**2).sum()
 (loss+reg).backward();opt.step()
 if step%50==0 or step==1:
  with torch.no_grad():
   v=scores(V,w)
   tr=scores(T,w)
   rec={'step':step,'val':v,'train':tr,'weights':[float(k) for k in w]}
   records.append(rec);print('EVAL',json.dumps(rec),flush=True)
   if v['nll']<best[0]:best=(v['nll'],w.detach().clone(),step)
w=best[1]
sidecar=R/'checkpoints/v168_gate_candidate.txt'
sidecar.write_text('N168GATE1\n'+' '.join(format(float(x),'.17g') for x in w)+'\n')
report={'status':'EXPERIMENTAL_NOT_PROMOTED','strategy':'trainable sigmoid gate over pretarget causal native features',
 'TRAIN_only':True,'book_split':'80 TRAIN books train / 20 TRAIN books gate-validation (all seen by pretraining v164 count memory)',
 'train_samples_active':len(train),'val_samples_active':len(val),'base_val_nll':baselines(V),
 'selected_step':best[2],'best_val':scores(V,w),'weights':[float(x) for x in w],
 'history':records,'sidecar_sha256':hashlib.sha256(sidecar.read_bytes()).hexdigest(),
 'caveat':'TRAIN calibration split is book-separated for gating but inherited v164 tables saw these books; separate DEV mandatory'}
(R/'results/v168_gate_train.json').write_text(json.dumps(report,indent=2))
print('FINAL',json.dumps({k:v for k,v in report.items() if k!='history'}),flush=True)