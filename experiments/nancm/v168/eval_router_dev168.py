#!/usr/bin/env python3
"""Frozen TRAIN-book selected expert gate on independent 24-book DEV positions."""
import ctypes,json,math,os,sys,time,numpy as np
from pathlib import Path
R=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(R/'scripts'),str(R/'parent/scripts')]
from train_gate163 import setup,winread
from test_native_speller156 import load,MODEL,F,getwords
from beam_words156 import decode
pack=json.loads((R/'checkpoints/v168_router_candidate.json').read_text());W=np.array(pack['weights']);mu=np.array(pack['mean']);sd=np.array(pack['std'])
setup(core=False,domain=True,gate=None)
os.environ.update(V164_DISTRIBUTION_BUDGET='.6',V164_DISTRIBUTION_INDEX=str(R/'checkpoints/v164_native_distribution.tsv'),V164_DOMAIN_GATE='1',V164_DOMAIN_THRESHOLD='1.5',V166_BIRTH_MODEL=str(R/'checkpoints/v166_lexical_birth.bin'),V166_LEXICAL_BIRTH='.05',V167_GATE_MODE='1',V167_DOMAIN_MIN='1.4',V167_HIGH_ALPHA='.5',V167_TRUST_THRESHOLD='.5',V168_ANN_INDEX=str(R/'checkpoints/v168_signed_ann.bin'),V168_ANN_SIGNED_SHARE='.66',V168_ANN_DOMAIN_MIN='1.4',V168_ANN_STRENGTH='.05')
l=load(R/'bin/libwordfield168_ann.so')
for n in ('wf167_effective_alpha','wf167_route_score','wf168_ann_alpha'):
 f=getattr(l,n);f.argtypes=[ctypes.c_void_p];f.restype=ctypes.c_double
for n in ('wf166_candidate','wf168_ann_candidate'):
 f=getattr(l,n);f.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_char_p,ctypes.c_size_t,ctypes.POINTER(ctypes.c_double)];f.restype=ctypes.c_int
l.wf168_neural_word_prob.argtypes=[ctypes.c_void_p,ctypes.c_char_p];l.wf168_neural_word_prob.restype=ctypes.c_double
h=l.wf148_open(*MODEL);assert h
def candidate(n):
 word=ctypes.create_string_buffer(64);p=ctypes.c_double();assert getattr(l,n)(h,0,word,64,ctypes.byref(p))==0
 return word.value.decode(),p.value
def wordp(w):
 p=1.;b=w.encode()
 for i in range(len(w)+1):
  o=(F*27)();e=F();assert l.wf148_next(h,b,i,o,ctypes.byref(e))==0
  p*=max(1e-30,(o[26] if i==len(w) else o[ord(w[i])-97])+e.value/27)
 return p
rows=[];active=0;t0=time.monotonic()
for bi,(raw,skip) in enumerate(winread(R/'external_book_gate/heldout_dev160.bin')):
 if bi>=24:break
 words=getwords(raw,skip);chosen=[words[min(len(words)-1,int((j+.5)*len(words)/min(20,len(words))))] for j in range(min(20,len(words)))] if words else []
 for pos,truth in chosen:
  assert l.wf148_begin(h,ctypes.c_char_p(raw[:pos]),pos,F(.25))>=0
  ranked,_=decode(l,h,beam=96,steps=20,topk=5)
  native=ranked[0]
  route=native;three=[native,native,native];pred_idx=0;features=None
  if l.wf168_ann_alpha(h)>0 and l.wf167_effective_alpha(h)>0:
   active+=1
   q,_=candidate('wf166_candidate');a,ap=candidate('wf168_ann_candidate');three=[native,q,a]
   feats=[]
   for j,w in enumerate(three):
    pn=wordp(w);qn=l.wf168_neural_word_prob(h,w.encode());an=ap if w==a else 0.
    if w!=a:
     for k in range(1,64):
      bw=ctypes.create_string_buffer(64);pp=ctypes.c_double()
      if l.wf168_ann_candidate(h,k,bw,64,ctypes.byref(pp))!=0:break
      if bw.value.decode()==w:an=pp.value;break
    feats.append([math.log(max(1e-30,pn)),math.log(max(1e-30,qn)),math.log(max(1e-30,an)),len(w)/20.,float(j==0),float(j==1),float(j==2),float(three.count(w)-1),l.wf167_effective_alpha(h),max(-5.,min(5.,l.wf167_route_score(h)-1.5))])
   features=np.array(feats);scores=(features-mu)/sd@W
   pred_idx=int(np.argmax(scores));route=three[pred_idx]
  rows.append({'book':bi,'truth':truth,'native':native,'route':route,'experts':three,'expert_selected':pred_idx})
 print('BOOK',bi,'n',len(rows),'route_hits',sum(r['route']==r['truth'] for r in rows),'base_hits',sum(r['native']==r['truth'] for r in rows),flush=True)
l.wf148_close(h)
report={'n':len(rows),'books':24,'active':active,'baseline_top1':sum(r['native']==r['truth'] for r in rows),'router_top1':sum(r['route']==r['truth'] for r in rows),'oracle_any':sum(r['truth'] in r['experts'] for r in rows),'new_hits':sum(r['native']!=r['truth'] and r['route']==r['truth'] for r in rows),'lost_hits':sum(r['native']==r['truth'] and r['route']!=r['truth'] for r in rows),'seconds':time.monotonic()-t0,'selection':'model trained/selected on 80+20 TRAIN books only'}
(R/'results/v168_router_DEV_frozen.json').write_text(json.dumps(report,indent=2));(R/'results/v168_router_DEV_rows.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
print('FINAL',json.dumps(report),flush=True)