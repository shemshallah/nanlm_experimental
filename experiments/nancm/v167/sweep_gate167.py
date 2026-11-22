#!/usr/bin/env python3
"""DEV-only trust/routing ablations; frozen TEST only after selection."""
import argparse,ctypes,json,math,os,sys,time
from pathlib import Path
R=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(R/'scripts'))
from train_gate163 import setup,winread
sys.path.insert(0,str(R/'parent/scripts'))
from test_native_speller156 import load,MODEL,F,getwords
from beam_words156 import decode

def samples(split):
 for bi,(raw,skip) in enumerate(winread(R/f'external_book_gate/heldout_{split}160.bin')):
  if bi>=24:break
  words=getwords(raw,skip)
  if not words:continue
  chosen=[words[min(len(words)-1,int((j+.5)*len(words)/min(20,len(words))))] for j in range(min(20,len(words)))]
  for pos,truth in chosen:yield bi,raw[:pos],truth

def run(arms,split='dev'):
 setup(core=False,domain=True,gate=None)
 os.environ.update(V164_DISTRIBUTION_BUDGET='.6',V164_DISTRIBUTION_INDEX=str(R/'checkpoints/v164_native_distribution.tsv'),V164_DOMAIN_GATE='1',V164_DOMAIN_THRESHOLD='1.5',V166_BIRTH_MODEL=str(R/'checkpoints/v166_lexical_birth.bin'))
 lib=load(R/'bin/libwordfield167.so')
 lib.wf167_effective_alpha.argtypes=[ctypes.c_void_p];lib.wf167_effective_alpha.restype=ctypes.c_double
 lib.wf167_parent_ratio.argtypes=[ctypes.c_void_p];lib.wf167_parent_ratio.restype=ctypes.c_double
 lib.wf167_neural_top_q.argtypes=[ctypes.c_void_p];lib.wf167_neural_top_q.restype=ctypes.c_double
 rows=list(samples(split));assert len(rows)==480
 for name,mode,alpha,high,thresh,domain in arms:
  os.environ.update(V166_LEXICAL_BIRTH=str(alpha),V167_GATE_MODE=str(mode),V167_HIGH_ALPHA=str(high),V167_TRUST_THRESHOLD=str(thresh),V167_DOMAIN_MIN=str(domain))
  h=lib.wf148_open(*MODEL);assert h,name;t0=time.monotonic();loss=0;symbols=0;hits={1:0,5:0,32:0};active=0;boosts=0
  diagnostic=[]
  for bi,past,truth in rows:
   assert lib.wf148_begin(h,ctypes.c_char_p(past),len(past),F(.25))>=0
   ae=lib.wf167_effective_alpha(h);active+=int(ae>0);boosts+=int(ae>.05+1e-8)
   for i in range(len(truth)+1):
    pre=truth[:i].encode();o=(F*27)();e=F();assert lib.wf148_next(h,pre,len(pre),o,ctypes.byref(e))==0
    p=o[26] if i==len(truth) else o[ord(truth[i])-97]
    loss-=math.log2(max(1e-25,float(p)+e.value/27));symbols+=1
   ranked,_=decode(lib,h,beam=96,steps=20,topk=32)
   for k in hits:hits[k]+=int(truth in ranked[:k])
   diagnostic.append({'book':bi,'word':truth,'pred':ranked[:5],'alpha':ae,'parent_ratio':lib.wf167_parent_ratio(h),'top_q':lib.wf167_neural_top_q(h)})
  lib.wf148_close(h)
  report={'name':name,'mode':mode,'min_domain':domain,'alpha':alpha,'high_alpha':high,'trust_threshold':thresh,'active':active,'boosted':boosts,'n':480,'top1':hits[1],'top5':hits[5],'top32':hits[32],'bits_per_spelling_symbol':loss/symbols,'seconds':round(time.monotonic()-t0,2)}
  (R/f'results/v167_{split}_gate_{name}.json').write_text(json.dumps(report,indent=2))
  (R/f'results/v167_{split}_gate_{name}.jsonl').write_text('\n'.join(json.dumps(x) for x in diagnostic)+'\n')
  print('ARM',json.dumps(report),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--split',default='dev',choices=['dev','test']);p.add_argument('--arms',required=True,help='Comma-separated name:mode:alpha:high:threshold:domain')
 a=p.parse_args()
 arms=[]
 for item in a.arms.split(','):
  name,mode,alpha,high,th,domain=item.split(':');arms.append((name,int(mode),float(alpha),float(high),float(th),float(domain)))
 run(arms,a.split)