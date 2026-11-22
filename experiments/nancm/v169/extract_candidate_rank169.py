#!/usr/bin/env python3
"""Causal candidate portfolio for v169: native, neural, analogical.
No next-word labels enter feature extraction. Books are kept separate for
train, train-calibration, and held-out DEV diagnostics.
"""
import ctypes,os,sys,json,math,re,time,collections,argparse
from pathlib import Path
R=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(R/'scripts'),str(R/'parent/scripts')]
from train_gate163 import setup,winread
from test_native_speller156 import load,MODEL,F,getwords
from beam_words156 import decode
setup(core=False,domain=True,gate=None)
os.environ.update(V164_DISTRIBUTION_BUDGET='.6',V164_DISTRIBUTION_INDEX=str(R/'checkpoints/v164_native_distribution.tsv'),V164_DOMAIN_GATE='1',V164_DOMAIN_THRESHOLD='1.5',V166_BIRTH_MODEL=str(R/'checkpoints/v166_lexical_birth.bin'),V166_LEXICAL_BIRTH='.05',V167_GATE_MODE='1',V167_DOMAIN_MIN='1.4',V167_HIGH_ALPHA='.5',V167_TRUST_THRESHOLD='.5',V168_ANN_INDEX=str(R/'checkpoints/v168_signed_ann.bin'),V168_ANN_SIGNED_SHARE='.66',V168_ANN_DOMAIN_MIN='1.4')
l=load(R/'bin/libwordfield168_ann.so')
for name in ('wf168_ann_alpha','wf167_effective_alpha','wf167_route_score'):
 f=getattr(l,name);f.argtypes=[ctypes.c_void_p];f.restype=ctypes.c_double
for name in ('wf166_candidate','wf168_ann_candidate'):
 f=getattr(l,name);f.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_char_p,ctypes.c_size_t,ctypes.POINTER(ctypes.c_double)];f.restype=ctypes.c_int
l.wf168_neural_word_prob.argtypes=[ctypes.c_void_p,ctypes.c_char_p];l.wf168_neural_word_prob.restype=ctypes.c_double

def windows(split):
 if split=='train':
  cohorts=json.loads((R/'external_v163/COHORTS.json').read_text())
  for part in range(3):
   for (raw,skip),info in zip(winread(R/f'external_v163/v163_calibration_{part}.bin'),cohorts['calibration_windows'][part]):
    toks=[(m.start(),m.group().lower().decode('ascii')) for m in re.finditer(rb'[A-Za-z]+',raw)]
    words=[(p,w) for j,(p,w) in enumerate(toks) if j>=3 and p>=skip and len(w)<=20]
    if len(words)>8: words=[words[min(len(words)-1,int((i+.5)*len(words)/8))] for i in range(8)]
    yield info['name'],raw,words
 else:
  for bi,(raw,skip) in enumerate(winread(R/'external_book_gate/heldout_dev160.bin')):
   if bi>=24:break
   words=getwords(raw,skip)
   selected=[words[min(len(words)-1,int((j+.5)*len(words)/min(20,len(words))))] for j in range(min(20,len(words)))] if words else []
   yield str(bi),raw,selected

def wf_wordp(handle,word):
 p=1.;b=word.encode()
 for j in range(len(word)+1):
  o=(F*27)();e=F();assert l.wf148_next(handle,b,j,o,ctypes.byref(e))==0
  p*=max(1e-30,(o[26] if j==len(word) else o[ord(word[j])-97])+e.value/27)
 return p

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--split',choices=['train','dev'],default='train');ap.add_argument('--max',type=int,default=999999);a=ap.parse_args()
 os.environ['V168_ANN_STRENGTH']='0';h0=l.wf148_open(*MODEL);assert h0
 os.environ['V168_ANN_STRENGTH']='.05';h1=l.wf148_open(*MODEL);assert h1
 t=time.monotonic();rows=[];seen=0
 for book,raw,items in windows(a.split):
  for pos,truth in items:
   past=raw[:pos];assert l.wf148_begin(h0,ctypes.c_char_p(past),len(past),F(.25))>=0
   assert l.wf148_begin(h1,ctypes.c_char_p(past),len(past),F(.25))>=0
   if l.wf168_ann_alpha(h1)<=0:continue
   base,_=decode(l,h0,beam=96,steps=20,topk=16)
   assert base
   props=collections.defaultdict(lambda:{'ann':0.,'neural_rank':999,'native_rank':999,'ann_rank':999})
   for i,w in enumerate(base):props[w]['native_rank']=i
   for i in range(16):
    word=ctypes.create_string_buffer(64);p=ctypes.c_double()
    if l.wf166_candidate(h1,i,word,64,ctypes.byref(p))!=0:break
    w=word.value.decode();props[w]['neural_rank']=i
   for i in range(48):
    word=ctypes.create_string_buffer(64);p=ctypes.c_double()
    if l.wf168_ann_candidate(h1,i,word,64,ctypes.byref(p))!=0:break
    w=word.value.decode();props[w]['ann_rank']=i;props[w]['ann']=p.value
   candidates=[]
   for w,d in props.items():
    q=l.wf168_neural_word_prob(h1,w.encode());p=wf_wordp(h0,w)
    candidates.append({'word':w,'p':p,'q':max(0.,q),'r':d['ann'],'nr':d['native_rank'],'qr':d['neural_rank'],'rr':d['ann_rank']})
   rows.append({'book':book,'pos':pos,'truth':truth,'baseline':base[0],'features':{'domain':l.wf167_route_score(h1),'lex_alpha':l.wf167_effective_alpha(h1)},'candidates':candidates})
   seen+=1
   if seen>=a.max:break
  if seen>=a.max:break
  if a.split=='train' and seen%200<=8:print('PROGRESS',a.split,seen,'time',round(time.monotonic()-t,1),flush=True)
 for h in (h0,h1):l.wf148_close(h)
 out=R/f'results/v169_portfolio_{a.split}.jsonl';out.write_text(''.join(json.dumps(r,separators=(',',':'))+'\n' for r in rows))
 print('FINAL',json.dumps({'split':a.split,'n':seen,'books':len(set(r['book'] for r in rows)),'base_hits':sum(r['truth']==r['baseline'] for r in rows),'oracle_hits':sum(any(c['word']==r['truth'] for c in r['candidates']) for r in rows),'file':str(out),'secs':round(time.monotonic()-t,1)}),flush=True)
if __name__=='__main__':main()