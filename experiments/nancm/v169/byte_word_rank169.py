#!/usr/bin/env python3
"""Probe 256-byte symbiotic likelihood as a complete-word ranking signal over
frozen, causally generated v169 candidate portfolios. This is not an exhaustive
96-beam word decoder: report candidate-set ranking separately.
"""
import argparse,json,math,os,sys,time,collections
from pathlib import Path
R=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(Path(__file__).parent),str(R/'scripts'),str(R/'parent/scripts')]
from train_gate163 import winread
from byte_system169 import NaNCM169BytePredictor

def main():
 a=argparse.ArgumentParser();a.add_argument('--max',type=int,default=244);args=a.parse_args()
 rows=[json.loads(s) for s in (R/'results/v169_portfolio_dev.jsonl').read_text().splitlines()][:args.max]
 books=[raw for raw,skip in winread(R/'external_book_gate/heldout_dev160.bin')][:24]
 valid=0;hits=collections.Counter();changes=[];t=time.monotonic()
 with NaNCM169BytePredictor() as model:
  for ix,row in enumerate(rows):
   past=books[int(row['book'])][:row['pos']];cand=sorted({v['word'] for v in row['candidates']})
   # One native call per distinct candidate prefix, shared by every completion.
   prefixes={''}
   for w in cand:
    for k in range(len(w)+1):prefixes.add(w[:k])
   dist={pre:model.next256(past+pre.encode()) for pre in sorted(prefixes,key=lambda x:(len(x),x))}
   ratings=[]
   for w in cand:
    lp=0.
    for j,c in enumerate(w):
     b=dist[w[:j]]
     lp+=math.log(max(1e-30,b[ord(c)]+b[ord(c.upper())]))
    end=max(1e-30,1.-sum(dist[w][65:91])-sum(dist[w][97:123]))
    lp+=math.log(end)
    ratings.append((lp,w))
   ranked=[w for p,w in sorted(ratings,key=lambda a:(-a[0],a[1]))]
   valid+=1
   for k in (1,5,32):hits[k]+=row['truth'] in ranked[:k]
   changes.append({'book':row['book'],'truth':row['truth'],'native':row['baseline'],'byte_rerank':ranked[0],'candidate_set_size':len(cand)})
   if (ix+1)%50==0:print('PROGRESS',ix+1,'top1',hits[1],'sec',round(time.monotonic()-t,1),flush=True)
 report={'n':valid,'top1_candidate_portfolio':hits[1],'top5_candidate_portfolio':hits[5], 'top32_candidate_portfolio':hits[32], 'baseline_top1':sum(r['baseline']==r['truth'] for r in rows), 'seconds':round(time.monotonic()-t,1), 'method':'all causal v169 portfolio proposals scored by true normalized 256-byte mixture; casefolded prefix approximation; not exhaustive native beam'}
 (R/'results/v169_byte_word_rerank_DEV.json').write_text(json.dumps(report,indent=2))
 (R/'results/v169_byte_word_rerank_DEV_rows.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in changes))
 print('FINAL',json.dumps(report),flush=True)
if __name__=='__main__':main()