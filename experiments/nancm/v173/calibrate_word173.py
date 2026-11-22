#!/usr/bin/env python3
"""TRAIN-book-only calibration of overconfident common-word lexical head.
Uses only archived v169 candidate lists; strictly independent of DEV label selection.
"""
import json,math,hashlib,collections,statistics
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
rows=[json.loads(s) for s in (ROOT/'results/v169_portfolio_train.jsonl').read_text().splitlines()]
dev=[json.loads(s) for s in (ROOT/'results/v169_portfolio_dev.jsonl').read_text().splitlines()]
count=collections.Counter()
for line in (ROOT/'external_book_gate/book_unigrams.tsv').read_text().splitlines():
 try:
  w,n=line.split('\t');count[w]=int(n)
 except:continue
freq=sum(count.values());print('prior',freq,len(count),'book',len({r['book'] for r in rows}))
books=sorted({r['book'] for r in rows},key=lambda b:hashlib.sha256(('v173-calib-'+b).encode()).digest());cal=set(books[-20:]);tr=[r for r in rows if r['book'] not in cal];va=[r for r in rows if r['book'] in cal];print('train',len(tr),'val',len(va),'dev',len(dev))

def choose(r,params):
 alp,temp,rcoef,qcoef,lengthbias=params
 def score(c):
  prior=(count[c['word']]+1.)/(freq+len(count))
  # Direction: global prior corrects overprediction of function words
  return temp*math.log(max(c['p'],1e-16))-alp*math.log(prior)+rcoef*math.log(max(c['r'],1e-7))+qcoef*math.log(max(c['q'],1e-7))+lengthbias*math.log1p(len(c['word']))
 return max(r['candidates'],key=score)['word']

def ev(rows,p):return sum(choose(r,p)==r['truth'] for r in rows)
opts=[]
for alp in np.arange(0,.91,.1):
 for rcoef in (0,.15,.35,.65):
  for qcoef in (0,.15,.35,.65):
   for lengthbias in (0,.2,.5,.8):
    p=(float(alp),1.,rcoef,qcoef,lengthbias)
    opts.append((ev(va,p),ev(tr,p),p))
opts.sort(key=lambda z:(z[0],z[1],-z[2][0]),reverse=True)
print('baseline',sum(r['baseline']==r['truth'] for r in va),sum(r['baseline']==r['truth'] for r in dev))
print('top sorted heldout',opts[:15]);selected=opts[0][2]
print('SELECTED',selected,'VAL',opts[0][0],'TRAIN',opts[0][1],'DEV',ev(dev,selected),'DEV_BASE',sum(r['baseline']==r['truth'] for r in dev))
report={'ntrain':len(tr),'nval':len(va),'ndev':len(dev),'select':selected,'val_top1':opts[0][0],'val_baseline':sum(r['baseline']==r['truth'] for r in va),'dev_top1':ev(dev,selected),'dev_baseline':sum(r['baseline']==r['truth'] for r in dev),'selected_on':'train-book-heldout'}
(ROOT/'results/v173_calibration.json').write_text(json.dumps(report,indent=2))