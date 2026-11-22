#!/usr/bin/env python3
"""Frozen grouped-by-book TRAIN calibration and DEV evaluation of complete-word rank.
Scores only words retrieved causally before the target, and never promotes on TRAIN fit.
"""
import json,math,numpy as np,collections,argparse
from pathlib import Path
R=Path(__file__).resolve().parents[3]

def load(split):return [json.loads(x) for x in (R/f'results/v169_portfolio_{split}.jsonl').read_text().splitlines()]

def scores(row,pow_q,pow_r,bonus_ann,bonus_native):
 out=[]
 for c in row['candidates']:
  p=max(1e-30,c['p']);q=max(1e-12,c['q']);r=max(1e-12,c['r']);
  score=math.log(p)+pow_q*math.log(q)+pow_r*(math.log(r) if c['r']>0 else -12)+bonus_ann*int(c['rr']<999)+bonus_native*int(c['nr']<999)
  out.append((score,c['word']))
 return [w for sc,w in sorted(out,key=lambda x:(-x[0],x[1]))]

def evaluate(rows,values):
 hits=[0,0,0];b=0
 for r in rows:
  p=scores(r,*values);hits[0]+=p[0]==r['truth'];hits[1]+=r['truth'] in p[:5];hits[2]+=r['truth'] in p[:32];b+=r['baseline']==r['truth']
 return {'n':len(rows),'baseline':b,'top1':hits[0],'top5':hits[1],'top32':hits[2]}

def main():
 a=argparse.ArgumentParser();a.add_argument('--split',default='train',choices=['train','dev']);a.add_argument('--train-results',default='v169_selected.json');args=a.parse_args()
 rows=load(args.split)
 if args.split=='train':
  books=sorted(set(r['book'] for r in rows));rng=np.random.default_rng(169);rng.shuffle(books);cut=int(.75*len(books));tr=set(books[:cut]);train=[r for r in rows if r['book'] in tr];val=[r for r in rows if r['book'] not in tr]
  print('SPLIT',len(train),len(val),len(tr),len(books)-len(tr))
  variants=[]
  for q in (-.1,0,.05,.1,.2,.4):
   for r in (-.1,0.,.025,.05,.1,.2):
    for a in (-.2,0.,.2):
     for n in (0.,.2):
      config=(q,r,a,n);res=evaluate(val,config)
      variants.append((res['top1'],res['top5'],res['top32'],config))
  variants.sort(reverse=True);winner=variants[0];conservative=sorted([x for x in variants if x[0]>=evaluate(val,(0,0,0,0))['top1']],key=lambda x:(-x[1],-x[2],sum(abs(z) for z in x[3])))[0]
  report={'selected':winner[3],'validation':winner[:3],'conservative':conservative[3],'conservative_val':conservative[:3],'baseline_val':evaluate(val,(0,0,0,0)),'baseline_train':evaluate(train,(0,0,0,0)),'train_books':sorted(tr),'calibration_books':sorted(set(books)-tr),'selection':'TRAIN calibration book split; no DEV labels used'}
  (R/f'results/{args.train_results}').write_text(json.dumps(report,indent=2))
  print('SELECTED',json.dumps({k:v for k,v in report.items() if not k.endswith('books')}))
 else:
  selection=json.loads((R/f'results/{args.train_results}').read_text())
  print('DEV',{t:evaluate(rows,tuple(w)) for t,w in [('base',(0,0,0,0)),('selected',selection['selected']),('conservative',selection['conservative'])]})
if __name__=='__main__':main()