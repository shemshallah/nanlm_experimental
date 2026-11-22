#!/usr/bin/env python3
"""Study Q166-informed reranking of signed ANN proposals, without target leakage.
Source data are TRAIN-book calibration windows; DEV is evaluation only.
"""
import ctypes,os,sys,json,math,re,time,collections,argparse
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(R/'scripts'),str(R/'parent/scripts')]
from train_gate163 import setup,winread
from test_native_speller156 import load,MODEL,F,getwords
setup(core=False,domain=True,gate=None)
os.environ.update(V164_DISTRIBUTION_BUDGET='.6',V164_DISTRIBUTION_INDEX=str(R/'checkpoints/v164_native_distribution.tsv'),V164_DOMAIN_GATE='1',V164_DOMAIN_THRESHOLD='1.5',V166_BIRTH_MODEL=str(R/'checkpoints/v166_lexical_birth.bin'),V166_LEXICAL_BIRTH='.05',V167_GATE_MODE='1',V167_DOMAIN_MIN='1.4',V167_HIGH_ALPHA='.5',V167_TRUST_THRESHOLD='.5',V168_ANN_INDEX=str(R/'checkpoints/v168_signed_ann.bin'),V168_ANN_SIGNED_SHARE='.66',V168_ANN_DOMAIN_MIN='1.4',V168_ANN_STRENGTH='.05')
l=load(R/'bin/libwordfield168_ann.so')
l.wf168_ann_alpha.argtypes=[ctypes.c_void_p];l.wf168_ann_alpha.restype=ctypes.c_double
l.wf168_ann_candidate_count.argtypes=[ctypes.c_void_p];l.wf168_ann_candidate_count.restype=ctypes.c_int
l.wf168_ann_candidate.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_char_p,ctypes.c_size_t,ctypes.POINTER(ctypes.c_double)];l.wf168_ann_candidate.restype=ctypes.c_int
l.wf168_neural_word_prob.argtypes=[ctypes.c_void_p,ctypes.c_char_p];l.wf168_neural_word_prob.restype=ctypes.c_double

def train_windows():
 cohort=json.loads((R/'external_v163/COHORTS.json').read_text())
 for part in range(3):
  for (raw,skip),info in zip(winread(R/f'external_v163/v163_calibration_{part}.bin'),cohort['calibration_windows'][part]):
   toks=[(m.start(),m.group().lower().decode('ascii')) for m in re.finditer(rb'[A-Za-z]+',raw)]
   words=[(p,w) for j,(p,w) in enumerate(toks) if j>=3 and p>=skip and len(w)<=20]
   if len(words)>8:words=[words[min(len(words)-1,int((i+.5)*len(words)/8))] for i in range(8)]
   yield info['name'],raw,words

def dev_windows():
 for b,(raw,skip) in enumerate(winread(R/'external_book_gate/heldout_dev160.bin')):
  if b>=24:break
  words=getwords(raw,skip)
  chosen=[words[min(len(words)-1,int((j+.5)*len(words)/min(20,len(words))))] for j in range(min(20,len(words)))] if words else []
  yield str(b),raw,chosen

def extract(split):
 h=l.wf148_open(*MODEL);assert h
 out=[];t=time.monotonic(); active=0
 for book,raw,words in (train_windows() if split=='train' else dev_windows()):
  for pos,truth in words:
   assert l.wf148_begin(h,ctypes.c_char_p(raw[:pos]),pos,F(.25))>=0
   if l.wf168_ann_alpha(h)<=0:continue
   active+=1
   n=l.wf168_ann_candidate_count(h)
   candidates=[]
   for j in range(n):
    wb=ctypes.create_string_buffer(40);p=ctypes.c_double()
    assert l.wf168_ann_candidate(h,j,wb,40,ctypes.byref(p))==0
    w=wb.value.decode('ascii');q=float(l.wf168_neural_word_prob(h,wb.value));candidates.append((w,float(p.value),q))
   out.append({'book':book,'truth':truth,'candidates':candidates})
  if split=='train' and len(out)%100<10: print('BOOK',book,'examples',len(out),'time',round(time.monotonic()-t,1),flush=True)
 l.wf148_close(h)
 output=R/f'results/v169_ann_{split}_candidates.jsonl';output.write_text(''.join(json.dumps(x,separators=(',',':'))+'\n' for x in out))
 print('EXTRACT',split,len(out),'active',active,'truth_in_top64',sum(r['truth'] in {w for w,_,_ in r['candidates']} for r in out),'time',round(time.monotonic()-t,1),flush=True)
 return out

def compare(rows, b):
 hits=collections.Counter();n=len(rows)
 for x in rows:
  cand=x['candidates']
  scores=[math.log(max(1e-20,p))+b*math.log(max(1e-12,q if q>0 else 1e-12)) for w,p,q in cand]
  idx=sorted(range(len(cand)),key=lambda j:(-scores[j],cand[j][0]))
  rank=[cand[i][0] for i in idx]
  for k in (1,5,16,32,64):hits[k]+=x['truth'] in rank[:k]
 return {'n':n,'top1':hits[1],'top5':hits[5],'top16':hits[16],'top32':hits[32],'top64':hits[64]}

def main():
 a=argparse.ArgumentParser();a.add_argument('--split',choices=['train','dev'],default='train');a.add_argument('--extract',action='store_true');a.add_argument('--beta',type=float,default=0.25);p=a.parse_args()
 path=R/f'results/v169_ann_{p.split}_candidates.jsonl'
 rows=extract(p.split) if p.extract or not path.exists() else [json.loads(s) for s in path.read_text().splitlines()]
 if p.split=='train':
  books=sorted(set(r['book'] for r in rows));rng=np.random.default_rng(169);rng.shuffle(books);cut=int(len(books)*.8);tr=set(books[:cut]);subsets={'train': [r for r in rows if r['book'] in tr],'validation':[r for r in rows if r['book'] not in tr]}
 else:subsets={'dev':rows}
 for name,rs in subsets.items():
  for b in (-.4,-.2,0.,.05,.1,.2,.4,.8,1.6):
   print('ARM',name,'weight',b,compare(rs,b),flush=True)
if __name__=='__main__':main()