#!/usr/bin/env python3
"""TRAIN-book causal top-choice expert selector dataset; labels are used only for training."""
import ctypes,json,os,re,sys,time
from pathlib import Path
R=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(R/'scripts'),str(R/'parent/scripts')]
from train_gate163 import setup,winread
from test_native_speller156 import load,MODEL,F
from beam_words156 import decode
setup(core=False,domain=True,gate=None)
os.environ.update(V164_DISTRIBUTION_BUDGET='.6',V164_DISTRIBUTION_INDEX=str(R/'checkpoints/v164_native_distribution.tsv'),V164_DOMAIN_GATE='1',V164_DOMAIN_THRESHOLD='1.5',V166_BIRTH_MODEL=str(R/'checkpoints/v166_lexical_birth.bin'),V166_LEXICAL_BIRTH='.05',V167_GATE_MODE='1',V167_DOMAIN_MIN='1.4',V167_HIGH_ALPHA='.5',V167_TRUST_THRESHOLD='.5',V168_ANN_INDEX=str(R/'checkpoints/v168_signed_ann.bin'),V168_ANN_SIGNED_SHARE='.66',V168_ANN_DOMAIN_MIN='1.4',V168_ANN_STRENGTH='.05')
cohort=json.loads((R/'external_v163/COHORTS.json').read_text());cal=set(cohort['calibration_books'])
l=load(R/'bin/libwordfield168_ann.so')
for n in ('wf167_effective_alpha','wf167_route_score','wf168_ann_alpha'):
 f=getattr(l,n);f.argtypes=[ctypes.c_void_p];f.restype=ctypes.c_double
for n in ('wf166_candidate','wf168_ann_candidate'):
 f=getattr(l,n);f.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_char_p,ctypes.c_size_t,ctypes.POINTER(ctypes.c_double)];f.restype=ctypes.c_int
l.wf168_neural_word_prob.argtypes=[ctypes.c_void_p,ctypes.c_char_p];l.wf168_neural_word_prob.restype=ctypes.c_double
h=l.wf148_open(*MODEL);assert h
def candidate(n):
 word=ctypes.create_string_buffer(64);p=ctypes.c_double()
 assert getattr(l,n)(h,0,word,64,ctypes.byref(p))==0
 return word.value.decode(),p.value
def wordp(w):
 p=1.;b=w.encode()
 for i in range(len(w)+1):
  out=(F*27)();e=F();assert l.wf148_next(h,b,i,out,ctypes.byref(e))==0
  p*=max(1e-30,(out[26] if i==len(w) else out[ord(w[i])-97])+e.value/27)
 return p
rows=[];t0=time.monotonic()
for part in range(3):
 wins=list(winread(R/f'external_v163/v163_calibration_{part}.bin'));names=cohort['calibration_windows'][part]
 for (raw,skip),info in zip(wins,names):
  book=info['name'];assert book in cal
  toks=[(m.start(),m.group().lower().decode('ascii')) for m in re.finditer(rb'[A-Za-z]+',raw)]
  targets=[(p,w) for j,(p,w) in enumerate(toks) if j>=3 and p>=skip and len(w)<=20]
  if len(targets)>8:targets=[targets[min(len(targets)-1,int((i+.5)*len(targets)/8))] for i in range(8)]
  for p,truth in targets:
   past=raw[:p]
   assert l.wf148_begin(h,ctypes.c_char_p(past),len(past),F(.25))>=0
   if l.wf168_ann_alpha(h)<=0 or l.wf167_effective_alpha(h)<=0:continue
   rank,_=decode(l,h,beam=96,steps=20,topk=1)
   if not rank:continue
   native=rank[0];q,_=candidate('wf166_candidate');a,ap=candidate('wf168_ann_candidate')
   cands=[native,q,a]
   weights=[]
   for j,w in enumerate(cands):
    pn=wordp(w);qn=l.wf168_neural_word_prob(h,w.encode());an=ap if w==a else 0.
    # Can check ANN distribution for matching native/Q in first 64 candidates
    if w!=a:
     for k in range(1,64):
      bw=ctypes.create_string_buffer(64);pp=ctypes.c_double()
      if l.wf168_ann_candidate(h,k,bw,64,ctypes.byref(pp))!=0:break
      if bw.value.decode()==w:an=pp.value;break
    import math
    weights.append([math.log(max(1e-30,pn)),math.log(max(1e-30,qn)),math.log(max(1e-30,an)),len(w)/20.,1. if j==0 else 0.,1. if j==1 else 0.,1. if j==2 else 0.,float(cands.count(w)-1),l.wf167_effective_alpha(h),max(-5.,min(5.,l.wf167_route_score(h)-1.5))])
   rows.append({'book':book,'truth':truth,'words':cands,'features':weights,'correct':[int(x==truth) for x in cands]})
 print('PART',part,'rows',len(rows),'sec',round(time.monotonic()-t0,1),flush=True)
l.wf148_close(h)
assert len(rows)>800
out=R/'results/v168_router_train_rows.jsonl';out.write_text(''.join(json.dumps(r,separators=(',',':'))+'\n' for r in rows))
print('FINAL',json.dumps({'n':len(rows),'books':len(set(r['book'] for r in rows)), 'native_top1':sum(r['correct'][0] for r in rows),'lex_top1':sum(r['correct'][1] for r in rows),'ann_top1':sum(r['correct'][2] for r in rows),'oracle_any':sum(any(r['correct']) for r in rows),'seconds':round(time.monotonic()-t0,2)}),flush=True)