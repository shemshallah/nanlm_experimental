#!/usr/bin/env python3
"""TRAIN-only top-choice expert router; labels never enter native scoring."""
import json,hashlib,math,random
from pathlib import Path
import numpy as np,torch
R=Path(__file__).resolve().parents[3]
rows=[json.loads(s) for s in (R/'results/v168_router_train_rows.jsonl').open()]
books=sorted(set(r['book'] for r in rows),key=lambda x:hashlib.sha256(('NaNCM_v168_book_gate:'+x).encode()).digest())
train_books=set(books[:int(.8*len(books))]);val_books=set(books)-train_books
# Lower-bound genuine whole-book holdout for trainable routing, although v164 saw all 500 TRAIN books.
X=np.array([r['features'] for r in rows],dtype=np.float64)
Y=np.array([r['correct'] for r in rows],dtype=np.float64)
idx_train=np.array([i for i,r in enumerate(rows) if r['book'] in train_books and any(r['correct'])])
idx_val=np.array([i for i,r in enumerate(rows) if r['book'] in val_books])
idx_val_pos=np.array([i for i in idx_val if any(rows[i]['correct'])]);
print('DATA',len(rows),'train_positive',len(idx_train),'val',len(idx_val),'val_positive',len(idx_val_pos),'native_val_top1',int(Y[idx_val,0].sum()),flush=True)
# Normalization fixed only on TRAIN input values; real native feature vector stays identical.
train_all=np.array([i for i,r in enumerate(rows) if r['book'] in train_books]);mu=X[train_all].reshape(-1,10).mean(0);sd=X[train_all].reshape(-1,10).std(0).clip(1e-4)
Z=(X-mu)/sd
X_t=torch.tensor(Z,dtype=torch.float32);Y_t=torch.tensor(Y,dtype=torch.float32)

def evaluate(W):
 with torch.no_grad():
  logits=(X_t@W).detach().numpy();chosen=logits.argmax(1)
  return {'train_top1':int(Y[train_all,chosen[train_all]].sum()),'train_n':len(train_all),
   'val_top1':int(Y[idx_val,chosen[idx_val]].sum()),'val_n':len(idx_val),
   'native_val_top1':int(Y[idx_val,0].sum()),
   'val_by_expert':[int((chosen[idx_val]==i).sum()) for i in range(3)],
   'oracle_val_top1':int(Y[idx_val].max(1).sum())}
# A single shared ranker for 3 expert proposals; model complexity only 10 floats.
trials=[];best=None
for lr,reg in [(.03,.015),(.01,.1),(.025,.002),(.008,.03)]:
 torch.manual_seed(168)
 W=torch.nn.Parameter(torch.zeros(10,dtype=torch.float32));W.data[4]=.02
 opt=torch.optim.Adam([W],lr=lr)
 chosen_state=None
 for step in range(1,901):
  opt.zero_grad()
  scores=X_t[idx_train]@W
  corr=Y_t[idx_train]
  # Full marginal log probability of any correct expert; winner-first training.
  pos=torch.logsumexp(scores.masked_fill(corr==0,-1e6),1)
  loss=-(pos-torch.logsumexp(scores,1)).mean()
  anchor=reg*(W*W).sum()+.005*torch.relu(-W[4])
  (loss+anchor).backward();opt.step()
  if step%50==0:
   stat=evaluate(W);stat.update(lr=lr,reg=reg,step=step,train_margin=float(loss.detach()))
   trials.append(stat)
   score=(stat['val_top1'],stat['train_top1'],-reg)
   if best is None or score>best[0]:best=(score,W.detach().numpy().copy(),stat)
 print('TRIAL',lr,reg,'final',evaluate(W),flush=True)
# Train-val selection is not a strict untouched book test; DEV remains separate.
_,w,stat=best
side=R/'checkpoints/v168_router_candidate.json'
side.write_text(json.dumps({'format':'N168ROUTER1','weights':w.tolist(),'mean':mu.tolist(),'std':sd.tolist(),
 'features':['ln_native_p','ln_neural_q','ln_signed_ann_r','word_length_scaled','is_native','is_neural','is_ann','agreement_count','lexical_alpha','domain_delta']},indent=2))
report={'status':'EXPERIMENTAL_NOT_PROMOTED','baseline_val':int(Y[idx_val,0].sum()),'val_examples':len(idx_val),
 'train_examples':len(train_all),'val_positive_oracle':int(Y[idx_val].max(1).sum()),
 'best':stat,'evaluations':trials,'selected_model':str(side),'strict_dev_pending':True}
(R/'results/v168_router_training.json').write_text(json.dumps(report,indent=2))
print('FINAL',json.dumps({k:v for k,v in report.items() if k!='evaluations'}),flush=True)