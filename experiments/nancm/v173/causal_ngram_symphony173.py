#!/usr/bin/env python3
"""Causal sparse TRAIN ngram memory agreement with signed whole-word candidate pool.
All mixing hyperparams selected on 20 heldout TRAIN books. DEV scored once.
"""
from __future__ import annotations
import re,math,hashlib,collections,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3];PAT=re.compile(rb'[A-Za-z]+')
raw=[json.loads(s) for s in (ROOT/'results/v169_portfolio_train.jsonl').read_text().splitlines()]
met=[json.loads(s) for s in (ROOT/'results/v172_train_context_positions.jsonl').read_text().splitlines()][:len(raw)]
assert all(a['truth']==b['truth'] and a['book']==b['book'] for a,b in zip(raw,met))
dev=[json.loads(s) for s in (ROOT/'results/v169_portfolio_dev.jsonl').read_text().splitlines()]
import sys
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'parent/scripts')]
from train_gate163 import winread
wins=list(winread(ROOT/'external_book_gate/heldout_dev160.bin'))
raw_hist=[m['history'].encode('latin1') for m in met]
dev_hist=[wins[int(r['book'])][0][:r['pos']] for r in dev]
ng={i:{} for i in (1,2,3)}
for i,line in enumerate((ROOT/'checkpoints/v164_native_distribution.tsv').open()):
 if i==0:assert line.startswith('N164DIST1');continue
 order,ctx,word,n,total=line.rstrip('\n').split('\t')
 d=ng[int(order)].setdefault(ctx,[int(total),{}]);d[1][word]=int(n)
print('loaded',len(ng[1]),len(ng[2]),len(ng[3]),flush=True)
freq=collections.Counter()
for line in (ROOT/'external_book_gate/book_unigrams.tsv').read_text().splitlines():
 try:w,c=line.split('\t');freq[w]+=int(c)
 except:pass
norm=sum(freq.values());V=len(freq)

def feats(rows,histories):
 output=[]
 for r,past in zip(rows,histories):
  ws=[m.group().decode('ascii').lower() for m in PAT.finditer(past)];context={o:ng[o].get(' '.join(ws[-o:])) if len(ws)>=o else None for o in (1,2,3)}
  # True probabilities of the *past-only* conditional ngram table.
  cand=[]
  for c in r['candidates']:
   w=c['word'];prior=(freq[w]+.1)/(norm+.1*V)
   a=[math.log(max(c['p'],1e-20)),math.log(max(c['q'],1e-20)),math.log(max(c['r'],1e-20)),math.log(max(prior,1e-20))]
   for o in (1,2,3):
    cc=context[o]
    if cc is None:a.extend([math.log(prior),0.]);continue
    t,v=cc
    # Absolute continuation counts + smoothed probability
    seen=v.get(w,0);alpha={1:100.,2:20.,3:5.}[o]
    a.extend([math.log(max(1e-20,(seen+alpha*prior)/(t+alpha))),math.log1p(t)])
   cand.append(a)
  output.append(np.asarray(cand,dtype=np.float32))
 return output
train=feats(raw,raw_hist);test=feats(dev,dev_hist)
books=sorted({r['book'] for r in raw},key=lambda b:hashlib.sha256(('v173-ngram-'+b).encode()).digest());cal=set(books[-20:]);idx=[i for i,r in enumerate(raw) if r['book'] in cal]
def ev(rows,mx,ids,w):
 hit=0
 for i in ids:
  j=np.argmax(mx[i]@np.asarray(w,dtype=np.float32));hit+=rows[i]['candidates'][j]['word']==rows[i]['truth']
 return hit
opts=[]
for c1 in (0.,.1,.3,.7,1.0):
 for c2 in (0.,.1,.3,.7,1.):
  for c3 in (0.,.1,.3,.7,1.):
   for lp in (0.,.1,.3,.5):
    for lex in (0.,.1,.25):
     w=np.array([1,lex,0,-lp,c1,0.,c2,0.,c3,0.],np.float32)
     opts.append((ev(raw,train,idx,w),-sum(abs(z) for z in (c1,c2,c3,lp,lex)),w))
opts.sort(key=lambda v:(v[0],v[1]),reverse=True)
selected=opts[0];bval=sum(r['truth']==r['baseline'] for i,r in enumerate(raw) if i in set(idx));didx=list(range(len(dev)))
report={'n_train':len(raw),'n_val':len(idx),'n_dev':len(dev),'vocab_freq':V,'contexts':[len(ng[i]) for i in (1,2,3)],'parent_trainval_top1':bval,'selected_trainval_top1':selected[0],'weights':selected[2].tolist(),'dev_parent_top1':sum(r['truth']==r['baseline'] for r in dev),'dev_selected_top1':ev(dev,test,didx,selected[2]),'status':'EXPERIMENT_NOT_PROMOTED_BY_DEFAULT','leakage_note':'DEV historically explored, selected only on TRAIN-book calibration'}
(ROOT/'results/v173_causal_ngram_symphony.json').write_text(json.dumps(report,indent=2));print('RESULT',json.dumps(report),flush=True)