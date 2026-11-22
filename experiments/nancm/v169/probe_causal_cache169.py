#!/usr/bin/env python3
"""Causal document-local suffix-automaton probe; only prior words used.
Not a full model benchmark. Use as a support/precision audit, no truth in predictor.
"""
import re,json,sys,collections,os,hashlib
from pathlib import Path
R=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(R/'scripts'),str(R/'parent/scripts')]
from train_gate163 import winread
from test_native_speller156 import getwords

def cache(past):
 toks=[m.group().decode('ascii').lower() for m in re.finditer(rb'[A-Za-z]+',past)]
 if not toks:return []
 # For each order, count historical transitions where the next word
 # is fully inside *past*. Include exponential recency only.
 proposals=[]
 for k in range(min(5,len(toks)),0,-1):
  suf=tuple(toks[-k:]);cnt=collections.Counter();seen=[]
  for j in range(k,len(toks)):
   if tuple(toks[j-k:j])==suf:
    w=toks[j];cnt[w]+=1;seen.append(j)
  if cnt:
   top=cnt.most_common()
   total=sum(cnt.values());purity=top[0][1]/total
   proposals.append({'order':k,'word':top[0][0],'support':total,'purity':purity,'dist':len(toks)-seen[-1],'top':top[:5]})
 return proposals

def main():
 rows=[]
 for bi,(raw,skip) in enumerate(winread(R/'external_book_gate/heldout_dev160.bin')):
  if bi>=24:break
  for pos,truth in getwords(raw,skip):
   p=cache(raw[:pos]);rows.append({'book':bi,'word':truth,'match':p})
 print('TOTAL',len(rows))
 for k in [1,2,3,4,5]:
  sub=[r for r in rows if r['match'] and r['match'][0]['order']>=k]
  hits=sum(r['match'][0]['word']==r['word'] for r in sub)
  print('ORD>=',k,'coverage',len(sub),'top1_hits',hits,'precision',round(hits/max(1,len(sub)),3))
 for support in [1,2,3]:
  sub=[r for r in rows if r['match'] and r['match'][0]['support']>=support]
  hits=sum(r['match'][0]['word']==r['word'] for r in sub)
  print('SUP>=',support,'coverage',len(sub),'top1_hits',hits,'precision',round(hits/max(1,len(sub)),3))
 sample=[]
 for b in range(24):
  chunk=[r for r in rows if r['book']==b]
  chunk_idxs=[min(len(chunk)-1,int((j+.5)*len(chunk)/min(20,len(chunk)))) for j in range(min(20,len(chunk)))]
  sample.extend(chunk[i] for i in chunk_idxs)
 print('SAMPLE',len(sample),'max_order_1',sum(bool(r['match']) for r in sample),'order_2+',sum(bool(r['match']) and r['match'][0]['order']>=2 for r in sample))
 path=R/'results/v169_cache_dev_probe.jsonl';path.write_text(''.join(json.dumps(r)+'\n' for r in sample))
if __name__=='__main__':main()