#!/usr/bin/env python3
"""Causal threshold/normalization assertions for frozen v167 active candidate."""
import ctypes,hashlib,json,math,os,sys
from pathlib import Path
R=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(R/'scripts'))
from train_gate163 import setup,winread
sys.path.insert(0,str(R/'parent/scripts'))
from test_native_speller156 import load,MODEL,F,getwords
setup(core=False,domain=True,gate=None)
os.environ.update(V164_DISTRIBUTION_BUDGET='.6',V164_DISTRIBUTION_INDEX=str(R/'checkpoints/v164_native_distribution.tsv'),V164_DOMAIN_GATE='1',V164_DOMAIN_THRESHOLD='1.5',V166_LEXICAL_BIRTH='.05',V166_BIRTH_MODEL=str(R/'checkpoints/v166_lexical_birth.bin'),V167_GATE_MODE='1',V167_DOMAIN_MIN='1.4',V167_HIGH_ALPHA='.5',V167_TRUST_THRESHOLD='.5')
l=load(R/'bin/libwordfield167.so')
for name in ('wf167_effective_alpha','wf167_route_score','wf167_parent_ratio','wf167_neural_top_q'):
 f=getattr(l,name);f.argtypes=[ctypes.c_void_p];f.restype=ctypes.c_double
h=l.wf148_open(*MODEL);assert h
vec=0;total=0;boost=0;active=0;causal=0
for bi,(raw,skip) in enumerate(winread(R/'external_book_gate/heldout_dev160.bin')):
 if bi>=24:break
 words=getwords(raw,skip)
 if not words:continue
 chosen=[words[min(len(words)-1,int((j+.5)*len(words)/min(20,len(words))))] for j in range(min(20,len(words)))]
 for pos,truth in chosen:
  past=raw[:pos];assert l.wf148_begin(h,ctypes.c_char_p(past),len(past),F(.25))>=0
  a=l.wf167_effective_alpha(h);score=l.wf167_route_score(h);ratio=l.wf167_parent_ratio(h)
  total+=1;active+=a>0;boost+=a>.05+1e-8
  if score<1.4:assert a==0
  else:assert a==(.5 if ratio>=.5 else .05),(score,ratio,a)
  before=[]
  for pref in ('',truth[:1],truth[:3]):
   prefix=pref.encode();o=(F*27)();e=F();assert l.wf148_next(h,prefix,len(prefix),o,ctypes.byref(e))==0
   vals=[float(z)+e.value/27 for z in o]
   assert abs(sum(vals)-1)<1e-4 and all(math.isfinite(v) and v>=0 for v in vals)
   before.append(bytes(o)+bytes(e));vec+=1
  # Present arbitrary extra suffix in the allocated buffer, but pass the
  # original prefix length to begin(). The future bytes MUST be ignored.
  extended=past+b' DO NOT READ THIS FUTURE TARGET.'
  assert l.wf148_begin(h,ctypes.c_char_p(extended),len(past),F(.25))>=0
  assert a==l.wf167_effective_alpha(h)
  for pref,ref in zip(('',truth[:1],truth[:3]),before):
   prefix=pref.encode();o=(F*27)();e=F();assert l.wf148_next(h,prefix,len(prefix),o,ctypes.byref(e))==0
   assert bytes(o)+bytes(e)==ref
   causal+=1
l.wf148_close(h)
assert total==480 and active==244 and boost==81
report={'status':'PASS','score_positions':total,'generator_active':active,'agreement_boosted':boost,'normalized_27_symbol_vectors':vec,'future_suffix_ignored_bit_identically':causal,
 'signed_v166_checkpoint_sha':hashlib.sha256((R/'checkpoints/v166_lexical_birth.bin').read_bytes()).hexdigest(),
 'full_byte_BPB_tested':False}
(R/'results/v167_trust_regression.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))