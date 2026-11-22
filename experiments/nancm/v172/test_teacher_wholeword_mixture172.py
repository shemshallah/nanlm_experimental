#!/usr/bin/env python3
"""Check normalized native word mixture is equal to exact mixture in word space."""
import os,sys,ctypes,json,math,random
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'parent/scripts'),str(ROOT/'experiments/nancm/v172')]
from train_gate163 import setup,winread
from test_native_speller156 import load,MODEL,F
from train_conditional_cloud172 import load_parent
setup(core=False,domain=True,gate=None)
os.environ.update(V164_DISTRIBUTION_BUDGET='.6',V164_DISTRIBUTION_INDEX=str(ROOT/'checkpoints/v164_native_distribution.tsv'),V164_DOMAIN_GATE='1',V164_DOMAIN_THRESHOLD='1.5',V166_BIRTH_MODEL=str(ROOT/'checkpoints/v166_lexical_birth.bin'),V166_LEXICAL_BIRTH='.05',V167_GATE_MODE='1',V167_DOMAIN_MIN='1.4',V167_HIGH_ALPHA='.5',V167_TRUST_THRESHOLD='.5',V168_ANN_INDEX=str(ROOT/'checkpoints/v168_signed_ann.bin'),V168_ANN_SIGNED_SHARE='.66',V168_ANN_DOMAIN_MIN='1.4',V168_ANN_STRENGTH='.05',V169_NEURAL_RERANK='.05',V171_ANN_NEIGHBORS='32',V171_ANN_CANDIDATES='64',V172_ANN_RAILS='1',V172_TEACHER_SHARE='.4',V172_TEACHER_MODEL=str(ROOT/'checkpoints/v172_native_signed_teacher.bin'))
native=load(ROOT/'bin/libwordfield172.so');parent=load(ROOT/'bin/libwordfield171.so')
native.wf172_teacher_distribution.argtypes=[ctypes.c_void_p,ctypes.POINTER(F),ctypes.c_size_t];native.wf172_teacher_distribution.restype=ctypes.c_int
h=native.wf148_open(*MODEL);bh=parent.wf148_open(*MODEL);assert h and bh
names,*_=load_parent();lookup={w:i for i,w in enumerate(names)}
wins=list(winread(ROOT/'external_book_gate/heldout_dev160.bin'))
rows=[json.loads(s) for s in (ROOT/'results/v169_portfolio_dev.jsonl').read_text().splitlines()]
random.seed(172);err=0.;testwords=0;contexts=0
for r in rows[::40]:
 raw=wins[int(r['book'])][0];past=raw[:r['pos']]
 assert native.wf148_begin(h,ctypes.c_char_p(past),len(past),F(.25))>=0
 assert parent.wf148_begin(bh,ctypes.c_char_p(past),len(past),F(.25))>=0
 output=(F*12000)();assert native.wf172_teacher_distribution(h,output,12000)==0
 candidates=[r['truth']]+random.sample(names,25)+['xyzxyzxyzxy','zoological']
 for w in candidates:
  if not (0<len(w)<=20 and w.isascii() and w.islower() and w.isalpha()):continue
  def prob(lib,handle):
   p=1.;data=w.encode()
   for j in range(len(w)+1):
    pre=data[:j];out=(F*27)();escape=F()
    rc=lib.wf148_next(handle,ctypes.c_char_p(pre),j,out,ctypes.byref(escape));assert rc==0,(w,j,rc)
    p*=out[26 if j==len(w) else ord(w[j])-97]+escape.value/27
   return p
  pmix=prob(native,h);pbase=prob(parent,bh)
  pteacher=float(output[lookup[w]]) if w in lookup else 0.
  expected=.6*pbase+.4*pteacher
  e=abs(pmix-expected);err=max(err,e)
  if e>2e-5:raise AssertionError((w,pmix,expected,e))
  testwords+=1
 contexts+=1
native.wf148_close(h);parent.wf148_close(bh)
rep={'status':'PASS','contexts':contexts,'word_checks':testwords,'max_absolute_mixture_mass_error':err,'teacher_share':.4,'normalized_whole_word_identity':True}
(ROOT/'results/v172_teacher_word_probability_parity.json').write_text(json.dumps(rep,indent=2));print('FINAL',rep,flush=True)