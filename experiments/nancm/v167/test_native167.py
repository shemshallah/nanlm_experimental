#!/usr/bin/env python3
"""Exact v166 feature-off rollback, documentation isolation, valid normalized output."""
import ctypes,json,os,sys,math,random
from pathlib import Path
R=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(R/'scripts'))
from train_gate163 import setup,winread
sys.path.insert(0,str(R/'parent/scripts'))
from test_native_speller156 import load,MODEL,F,getwords,wins
setup(core=False,domain=True,gate=None)
os.environ.update(V164_DISTRIBUTION_BUDGET='.6',V164_DISTRIBUTION_INDEX=str(R/'checkpoints/v164_native_distribution.tsv'),V164_DOMAIN_GATE='1',V164_DOMAIN_THRESHOLD='1.5',V166_LEXICAL_BIRTH='.05',V166_BIRTH_MODEL=str(R/'checkpoints/v166_lexical_birth.bin'),V167_GATE_MODE='0',V167_DOMAIN_MIN='1.5')
old=load(R/'bin/libwordfield166.so');new=load(R/'bin/libwordfield167.so')
for l in (old,new):l.wf166_candidate_count.argtypes=[ctypes.c_void_p];l.wf166_candidate_count.restype=ctypes.c_int
new.wf167_effective_alpha.argtypes=[ctypes.c_void_p];new.wf167_effective_alpha.restype=ctypes.c_double
A=old.wf148_open(*MODEL);B=new.wf148_open(*MODEL);assert A and B
vectors=0
for bi,(raw,skip) in enumerate(winread(R/'external_book_gate/heldout_dev160.bin')):
 if bi>=24:break
 words=getwords(raw,skip)
 chosen=[words[min(len(words)-1,int((j+.5)*len(words)/min(20,len(words))))] for j in range(min(20,len(words)))] if words else []
 for pos,truth in chosen:
  if vectors>=950:break
  past=raw[:pos];assert old.wf148_begin(A,ctypes.c_char_p(past),len(past),F(.25))>=0
  assert new.wf148_begin(B,ctypes.c_char_p(past),len(past),F(.25))>=0
  assert old.wf166_candidate_count(A)==new.wf166_candidate_count(B)
  for k in ('',truth[:2],truth[:4]):
   outs=[]
   for lib,h in ((old,A),(new,B)):
    f=(F*27)();e=F();assert lib.wf148_next(h,k.encode(),len(k),f,ctypes.byref(e))==0
    outs.append((bytes(f),bytes(e)))
   assert outs[0]==outs[1],(bi,k)
   vectors+=1
 if vectors>=950:break
old.wf148_close(A);new.wf148_close(B)
# Decoupled gate must NOT allow generator activity on historical documentation.
os.environ['V167_DOMAIN_MIN']='1.4';os.environ['V167_GATE_MODE']='0';os.environ['V166_LEXICAL_BIRTH']='.5'
X=old.wf148_open(*MODEL);Y=new.wf148_open(*MODEL);assert X and Y
exact=0;allowed=0
for fname in ('win.bin','win_test.bin'):
 for raw,skip in wins(fname):
  for pos,truth in getwords(raw,skip):
   past=raw[:pos]
   assert old.wf148_begin(X,ctypes.c_char_p(past),len(past),F(.25))>=0
   assert new.wf148_begin(Y,ctypes.c_char_p(past),len(past),F(.25))>=0
   allowed+=int(new.wf166_candidate_count(Y)>0)
   for pre in ('',truth[:1],truth[:4]):
    outs=[]
    for lib,h in ((old,X),(new,Y)):
     f=(F*27)();e=F();assert lib.wf148_next(h,pre.encode(),len(pre),f,ctypes.byref(e))==0
     outs.append((bytes(f),bytes(e)))
    assert outs[0]==outs[1],(fname,pos,pre)
    exact+=1
assert allowed==0
old.wf148_close(X);new.wf148_close(Y)
# Malformed / out-of-range control settings fail closed.
for e,v in [('V167_DOMAIN_MIN','nan'),('V167_GATE_MODE','7'),('V167_HIGH_ALPHA','0.9'),('V167_TRUST_THRESHOLD','-1')]:
 os.environ[e]=v;assert not new.wf148_open(*MODEL),(e,v)
 os.environ.pop(e)
report={'status':'PASS','v166_optout_exact_vectors':vectors,'historical_docs_exact_vectors_at_domain_1_4':exact,'docs_generator_admissions':allowed,'fail_closed':True,'no_production_promotion':True}
(R/'results/v167_native_regression.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))