#!/usr/bin/env python3
"""NaNCM v172 TRAIN-book gated causal word ranker; never use DEV for selection.

This is a diagnostic trained ranker on v169/171 proposal universe. It must
outperform a book-disjoint TRAIN calibration split before it is eligible for
independent DEV replay. Large candidate recall is not itself top-1 accuracy.
"""
from __future__ import annotations
import json, math, hashlib, collections, re
from pathlib import Path
import numpy as np
import lightgbm as lgb
ROOT=Path(__file__).resolve().parents[3]
PAT=re.compile(rb'[A-Za-z]+')

def load_memory():
  mem={1:{},2:{},3:{}}
  with (ROOT/'checkpoints/v164_native_distribution.tsv').open() as f:
    head=next(f);assert head.startswith('N164DIST1')
    for line in f:
      order,ctx,word,n,total=line.rstrip('\n').split('\t');d=mem[int(order)].setdefault(ctx,[int(total),{}]);d[1][word]=int(n)
  return mem

def candidate_data(rows,positions,memory,prior):
  xs=[];ys=[];groups=[];source=[];names=[]
  assert len(rows)==len(positions)
  for row,meta in zip(rows,positions):
    assert row['book']==meta['book'] and row['truth']==meta['truth']
    # Only preceding text is available, including current document prefix.
    raw=meta['history'].encode('latin1')
    words=[m.group().decode('ascii').lower() for m in PAT.finditer(raw)]
    local=collections.Counter(words)
    recent=collections.Counter(words[-50:]);recent2=collections.Counter(words[-200:])
    prior_word=words[-1] if words else ''
    dom=row['features']['domain']
    contexts={n:memory[n].get(' '.join(words[-n:])) if len(words)>=n else None for n in (1,2,3)}
    doc_follow={n:collections.Counter() for n in (1,2,3)}
    for n in (1,2,3):
      if len(words)>=n:
        key=words[-n:]
        for j in range(n,len(words)):
          if words[j-n:j]==key:doc_follow[n][words[j]]+=1
    g=[];labels=[];ws=[]
    for c in row['candidates']:
      w=c['word'];p=max(c['p'],1e-25);q=max(c['q'],1e-25);r=max(c['r'],1e-25)
      rank=[min(999,c[k]) for k in ('nr','qr','rr')]
      x=[math.log(p),math.log(q),math.log(r), *[math.log1p(i) for i in rank],
         int(c['q']>0),int(c['r']>0),int(c['nr']==0),int(c['qr']==0),int(c['rr']==0),
         len(w),int(len(w)==1),int(len(w)==2),int(len(w)>8),
         math.log1p(prior[w]),math.log1p(local[w]),math.log1p(recent[w]),math.log1p(recent2[w]),
         min(200,words[::-1].index(w)+1) if w in local else 250,
         float(w==prior_word),float(w.startswith(prior_word[-2:]) if prior_word else False),
         math.log1p(dom),
         int(w.endswith('ed')),int(w.endswith('ing')),int(w.endswith('ly')),
         float(w in ('e','ll','s','d','ve','re','t','m','n','j'))]
      for n in (1,2,3):
        d=contexts[n];cw=d[1].get(w,0) if d else 0
        x.extend([math.log1p(cw), cw/d[0] if d else 0.,float(bool(d))])
      for n in (1,2,3):
        d=doc_follow[n];z=sum(d.values());x.extend([math.log1p(d[w]),d[w]/z if z else 0.])
      g.append(x);labels.append(int(w==row['truth']));ws.append(w)
    xs.extend(g);ys.extend(labels);groups.append(len(g));source.append(row['baseline']);names.append(ws)
  return np.asarray(xs,'float32'),np.asarray(ys,'int8'),np.asarray(groups,'int32'),source,names

def rank_eval(rows,dat,preds,threshold=0):
  _,_,groups,parents,names=dat
  ans=collections.Counter();book=collections.defaultdict(collections.Counter);off=0
  for row,n,base,ws in zip(rows,groups,parents,names):
    cand=preds[off:off+n];off+=n
    if n<1:continue
    idx=sorted(range(n),key=lambda i:(-cand[i],-row['candidates'][i]['p'],ws[i]))
    top=ws[idx[0]]
    parent_score=max([float(cand[j]) for j,w in enumerate(ws) if w==base] or [-1e3]);margin=float(cand[idx[0]])-parent_score
    if margin<threshold:top=base
    for c in (ans,book[row['book']]):
      c['n']+=1;c['top1']+=top==row['truth'];c['baseline']+=base==row['truth'];c['top5']+=row['truth'] in [ws[j] for j in idx[:5]]
      c['oracle']+=row['truth'] in ws;c['changed']+=top!=base
  ans['books']=len(book)
  return dict(ans),{b:dict(c) for b,c in book.items()}

def main():
  original=[json.loads(s) for s in (ROOT/'results/v169_portfolio_train.jsonl').read_text().splitlines()]
  pos=[json.loads(s) for s in (ROOT/'results/v172_train_context_positions.jsonl').read_text().splitlines()][:len(original)]
  dev=[json.loads(s) for s in (ROOT/'results/v169_portfolio_dev.jsonl').read_text().splitlines()]
  from sys import path
  path[:0]=[str(ROOT/'scripts'),str(ROOT/'parent/scripts')]
  from train_gate163 import winread
  wins=list(winread(ROOT/'external_book_gate/heldout_dev160.bin'))
  devmeta=[{'book':r['book'],'truth':r['truth'],'history':wins[int(r['book'])][0][:r['pos']].decode('latin1')} for r in dev]
  mem=load_memory();prior=collections.Counter()
  with (ROOT/'external_book_gate/book_unigrams.tsv').open() as f:
    for line in f:
      w,n=line.rstrip('\n').split('\t');prior[w]=int(n)
  books=sorted({r['book'] for r in original},key=lambda b:hashlib.sha256(('v172-calib-'+b).encode()).digest());cal=set(books[-20:])
  tr=[(r,p) for r,p in zip(original,pos) if r['book'] not in cal];va=[(r,p) for r,p in zip(original,pos) if r['book'] in cal]
  trr,trp=map(list,zip(*tr));var,vap=map(list,zip(*va))
  td=candidate_data(trr,trp,mem,prior);vd=candidate_data(var,vap,mem,prior);dd=candidate_data(dev,devmeta,mem,prior)
  print('SPLIT',len(trr),len(var),len(dev),'books',len(books)-len(cal),len(cal),'features',td[0].shape[-1],flush=True)
  options=[]
  for leaf in (3,5,9):
    for estim in (40,100,200):
      p={'leaves':leaf,'iters':estim,'lr':.035,'minleaf':35,'reg':12.}
      model=lgb.LGBMRanker(objective='lambdarank',n_estimators=estim,learning_rate=p['lr'],num_leaves=leaf,min_child_samples=p['minleaf'],reg_lambda=p['reg'],random_state=172,n_jobs=2,verbosity=-1)
      model.fit(td[0],td[1],group=td[2])
      pred=model.predict(vd[0]);res,_=rank_eval(var,vd,pred)
      for thr in (0.,.1,.25,.5,1.,2.,3.):
        res,_=rank_eval(var,vd,pred,thr)
        options.append((res['top1'],res['top5'],-res['changed'],p,thr))
      print('VAL',p['leaves'],p['iters'],'best',res,flush=True)
  best=sorted(options,key=lambda z:(z[0],z[1],z[2],-z[3]['leaves'],-z[3]['iters']),reverse=True)[0]
  print('VAL_SELECTED',best,flush=True)
  p=best[3];thr=best[4];model=lgb.LGBMRanker(objective='lambdarank',n_estimators=p['iters'],learning_rate=p['lr'],num_leaves=p['leaves'],min_child_samples=p['minleaf'],reg_lambda=p['reg'],random_state=172,n_jobs=2,verbosity=-1)
  # Final model retrained on all TRAIN books. The heldout split above selected hyperparameters.
  alld=candidate_data(original,pos,mem,prior)
  model.fit(alld[0],alld[1],group=alld[2]);pd=model.predict(dd[0]);res,bybook=rank_eval(dev,dd,pd,thr)
  print('DEV_LOCKED',res,flush=True)
  report={'name':'v172 causal-document candidate reranker','source':'850 archived TRAIN contexts with reconstructed exact history; 91 books','split':{'fit_books':len(books)-len(cal),'validation_books':len(cal),'fit':len(trr),'validation':len(var)},'selected_by':'TRAIN-book heldout only','selected':p,'gate_margin':thr,'val_best':{'top1':best[0],'top5':best[1]},'dev':res,'bybook':bybook,'status':'RESEARCH_ONLY','promote':res['top1']>res['baseline'] and best[0]>sum(r['baseline']==r['truth'] for r in var)}
  (ROOT/'results/v172_causal_ranker_report.json').write_text(json.dumps(report,indent=2));model.booster_.save_model(str(ROOT/'checkpoints/v172_causal_ranker_lgbm.txt'))
  print('REPORT',json.dumps({k:v for k,v in report.items() if k!='bybook'}),flush=True)
if __name__=='__main__':main()