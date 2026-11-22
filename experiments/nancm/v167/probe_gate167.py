#!/usr/bin/env python3
import ctypes,json,os,sys,statistics
from pathlib import Path
R=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(R/'scripts'))
from train_gate163 import setup,winread
sys.path.insert(0,str(R/'parent/scripts'))
from test_native_speller156 import load,MODEL,F,getwords
setup(core=False,domain=True,gate=None)
os.environ.update(V164_DISTRIBUTION_BUDGET='.6',V164_DISTRIBUTION_INDEX=str(R/'checkpoints/v164_native_distribution.tsv'),V164_DOMAIN_GATE='1',V164_DOMAIN_THRESHOLD='1.5',V166_LEXICAL_BIRTH='.05',V166_BIRTH_MODEL=str(R/'checkpoints/v166_lexical_birth.bin'),V167_GATE_MODE='0',V167_DOMAIN_MIN='1.4')
l=load(R/'bin/libwordfield167.so')
for n in ('wf167_effective_alpha','wf167_route_score','wf167_parent_ratio','wf167_neural_top_q'):
 f=getattr(l,n);f.argtypes=[ctypes.c_void_p];f.restype=ctypes.c_double
h=l.wf148_open(*MODEL);assert h
r=[]
for bi,(raw,skip) in enumerate(winread(R/'external_book_gate/heldout_dev160.bin')):
 if bi>=24:break
 words=getwords(raw,skip)
 chosen=[words[min(len(words)-1,int((j+.5)*len(words)/min(20,len(words))))] for j in range(min(20,len(words)))] if words else []
 for pos,truth in chosen:
  past=raw[:pos];assert l.wf148_begin(h,ctypes.c_char_p(past),len(past),F(.25))>=0
  r.append({'book':bi,'truth':truth,'alpha':l.wf167_effective_alpha(h),'score':l.wf167_route_score(h),'relative_q':l.wf167_parent_ratio(h),'top_q':l.wf167_neural_top_q(h)})
for k in ('relative_q','top_q','score'):
 vals=sorted([x[k] for x in r if x['alpha']>.0]);n=len(vals)
 print(k,'active',n,'min,max',vals[0],vals[-1],'q10 q25 q50 q75 q90',[round(vals[int((n-1)*q)],6) for q in [.1,.25,.5,.75,.9]])
print('active',sum(x['alpha']>0 for x in r),'in dev 480')
Path('/mnt/data/v167_gate_features_dev.json').write_text(json.dumps(r,indent=2))
l.wf148_close(h)