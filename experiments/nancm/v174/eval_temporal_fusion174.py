#!/usr/bin/env python3
"""TRAIN-book-selected whole-word native+temporal fusion, then one exposed DEV report.
No DEV labels are used to choose alpha/gate. This ablation never auto-promotes.
"""
from __future__ import annotations
import ctypes,hashlib,json,math,collections,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'parent/scripts')]
from train_gate163 import winread
from train_temporal_signed174 import load_parent
names,*_=load_parent();vocab={w:i for i,w in enumerate(names)}
lib=ctypes.CDLL(str(ROOT/'bin/libwordfield174.so'))
lib.wf174_now_open.argtypes=[ctypes.c_char_p];lib.wf174_now_open.restype=ctypes.c_void_p
lib.wf174_now_close.argtypes=[ctypes.c_void_p]
lib.wf174_now_predict.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_size_t,ctypes.POINTER(ctypes.c_float),ctypes.c_size_t];lib.wf174_now_predict.restype=ctypes.c_int
h=lib.wf174_now_open(str(ROOT/'checkpoints/v174_temporal_signed_now.bin').encode());assert h
train=[json.loads(z) for z in (ROOT/'results/v169_portfolio_train.jsonl').read_text().splitlines()]
trainmeta=[json.loads(z) for z in (ROOT/'results/v172_train_context_positions.jsonl').read_text().splitlines()][:len(train)]
assert all(r['book']==p['book'] and r['truth']==p['truth'] for r,p in zip(train,trainmeta))
dev=[json.loads(z) for z in (ROOT/'results/v169_portfolio_dev.jsonl').read_text().splitlines()]
wins=list(winread(ROOT/'external_book_gate/heldout_dev160.bin'))
prior=collections.Counter()
for line in (ROOT/'external_book_gate/book_unigrams.tsv').read_text().splitlines():
 w,c=line.split('\t');prior[w]=int(c)
Z=sum(prior.values())

def enrich(rows,history):
 arr=(ctypes.c_float*12000)();out=[]
 for i,(r,past) in enumerate(zip(rows,history)):
  assert lib.wf174_now_predict(h,ctypes.c_char_p(past),len(past),arr,12000)==0
  prob=np.ctypeslib.as_array(arr).copy()
  cand=[]
  for c in r['candidates']:
   w=c['word'];q=float(prob[vocab[w]]) if w in vocab else 1e-13
   prior_prob=(prior[w]+1)/(Z+len(prior))
   cand.append((w,math.log(max(float(c['p']),1e-15)),math.log(max(q,1e-15)),math.log(prior_prob)))
  out.append({'book':r['book'],'truth':r['truth'],'baseline':r['baseline'],'cand':cand,'teacher_top1':names[int(np.argmax(prob))], 'teacher_prob':float(np.max(prob))})
  if (i+1)%100==0:print('SCORED',len(rows),i+1,flush=True)
 return out

try:
 T=enrich(train,[p['history'].encode('latin1') for p in trainmeta])
 D=enrich(dev,[wins[int(r['book'])][0][:r['pos']] for r in dev])
finally:lib.wf174_now_close(h)
books=sorted({r['book'] for r in T},key=lambda b:hashlib.sha256(('v174-locked-fusion:'+b).encode()).digest())
cal=set(books[-20:]);hold=[r for r in T if r['book'] in cal]
print('TRAIN-CAL',len(hold),'books',len(cal),'parent',sum(r['baseline']==r['truth'] for r in hold),flush=True)

def score(rows,alpha,mode='raw',gate=0.):
 hit1=hit5=changed=oracle=0
 for row in rows:
  ws=row['cand'];best=max(ws,key=lambda c:c[1]);par=best[0]
  def val(c):
   w,lp,lt,lprior=c
   if mode=='pmi':return lp+alpha*(lt-lprior)
   if mode=='mixed':return (1-alpha)*lp+alpha*lt
   if mode=='difference':return lp+alpha*(lt-lp)
   return lp
  rank=sorted(ws,key=lambda c:(-val(c),-c[1],c[0]))
  scores={c[0]:val(c) for c in ws}
  choice=rank[0][0]
  if scores[choice]-scores[par]<gate:choice=par
  hit1+=choice==row['truth'];hit5+=row['truth'] in [c[0] for c in rank[:5]];changed+=choice!=par;oracle+=row['truth'] in [x[0] for x in ws]
 return {'n':len(rows),'top1':hit1,'top5':hit5,'changed':changed,'oracle':oracle,'parent':sum(r['baseline']==r['truth'] for r in rows)}
options=[]
for mode in ('mixed','pmi'):
 for a in ([0,.025,.05,.1,.15,.2,.3,.4,.5,.6,.8,1] if mode=='mixed' else [0,.002,.005,.01,.02,.03,.05,.075,.1,.15,.2,.3]):
  for gate in (0.,.1,.25,.5,1.,2.):
   r=score(hold,a,mode,gate)
   options.append({'mode':mode,'alpha':a,'gate':gate,'v':r})
selected=max(options,key=lambda r:(r['v']['top1'],r['v']['top5'],-r['v']['changed'],-r['alpha'],-r['gate']))
Dev=score(D,selected['alpha'],selected['mode'],selected['gate'])
print('SELECTED_ON_TRAIN',selected,flush=True)
print('LOCKED_EXPOSED_DEV',Dev,flush=True)
rep={'experiment':'v174 train-book selected whole word native+temporal union fusion','method':selected,
     'train_heldout_books':len(cal),'train_heldout_rows':len(hold),'DEV_reused':Dev,
     'word_vocab':12000,'status':'RESEARCH_ONLY','promotion':False,'not_pristine':True,
     'baseline_vs_teacher_top1':{'train_cal':sum(r['teacher_top1']==r['truth'] for r in hold),'dev_reused':sum(r['teacher_top1']==r['truth'] for r in D)},
     'all_train_cal_options':[r for r in options if r['gate']==0.]}
(ROOT/'results/v174_temporal_fusion.json').write_text(json.dumps(rep,indent=2))