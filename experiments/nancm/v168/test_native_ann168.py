#!/usr/bin/env python3
"""Native v168 analogical birth tests: rollback, causal input boundary,
normalized whole-word conditional API, historical documentation guard and
fail-closed ANN sidecar. No hard-coded assumed accuracy counts.
"""
import ctypes,hashlib,json,math,os,sys,time
from pathlib import Path
R=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(R/'scripts'),str(R/'parent/scripts')]
from train_gate163 import setup,winread
from test_native_speller156 import load,MODEL,F,getwords,wins
setup(core=False,domain=True,gate=None)
os.environ.update(V164_DISTRIBUTION_BUDGET='.6',V164_DISTRIBUTION_INDEX=str(R/'checkpoints/v164_native_distribution.tsv'),V164_DOMAIN_GATE='1',V164_DOMAIN_THRESHOLD='1.5',V166_BIRTH_MODEL=str(R/'checkpoints/v166_lexical_birth.bin'),V166_LEXICAL_BIRTH='.05',V167_GATE_MODE='1',V167_DOMAIN_MIN='1.4',V167_HIGH_ALPHA='.5',V167_TRUST_THRESHOLD='.5',V168_ANN_INDEX=str(R/'checkpoints/v168_signed_ann.bin'),V168_ANN_SIGNED_SHARE='.66',V168_ANN_DOMAIN_MIN='1.4')
os.environ.pop('V168_GATE_FILE',None)
old=load(R/'bin/libwordfield167.so');new=load(R/'bin/libwordfield168_ann.so')
new.wf168_ann_alpha.argtypes=[ctypes.c_void_p];new.wf168_ann_alpha.restype=ctypes.c_double
for l in (old,new):
 for name in ('wf167_effective_alpha','wf167_route_score'):
  f=getattr(l,name);f.argtypes=[ctypes.c_void_p];f.restype=ctypes.c_double

def probs(l,h,prefix):
 a=(F*27)();e=F();b=prefix.encode()
 assert l.wf148_next(h,b,len(b),a,ctypes.byref(e))==0
 return bytes(a)+bytes(e),[float(x)+e.value/27 for x in a]
rows=[]
for bi,(raw,skip) in enumerate(winread(R/'external_book_gate/heldout_dev160.bin')):
 if bi>=24:break
 words=getwords(raw,skip)
 selected=[words[min(len(words)-1,int((j+.5)*len(words)/min(20,len(words))))] for j in range(min(20,len(words)))] if words else []
 for pos,truth in selected:rows.append((raw[:pos],truth))
assert len(rows)==480
os.environ['V168_ANN_STRENGTH']='0'
A=old.wf148_open(*MODEL);B=new.wf148_open(*MODEL);assert A and B
rollback=0
for past,truth in rows[:150]:
 for l,h in ((old,A),(new,B)):assert l.wf148_begin(h,ctypes.c_char_p(past),len(past),F(.25))>=0
 for pre in ('',truth[:1],truth[:3]):
  aa,_=probs(old,A,pre);bb,_=probs(new,B,pre)
  assert aa==bb,(pre,truth)
  rollback+=1
old.wf148_close(A);new.wf148_close(B)
# Check active ANN and future-target exclusion (only passed n is visible).
os.environ['V168_ANN_STRENGTH']='.05';B=new.wf148_open(*MODEL);assert B
normalized=0;causal=0;enabled=0
for past,truth in rows:
 assert new.wf148_begin(B,ctypes.c_char_p(past),len(past),F(.25))>=0
 enabled+=int(new.wf168_ann_alpha(B)>0)
 orig=[]
 for pre in ('',truth[:1],truth[:3]):
  bb,p=probs(new,B,pre)
  assert all(math.isfinite(v) and v>=0. for v in p)
  assert abs(sum(p)-1)<2e-4
  normalized+=1;orig.append((pre,bb))
 poison=past+b' future suffix deliberately hidden from input API'
 assert new.wf148_begin(B,ctypes.c_char_p(poison),len(past),F(.25))>=0
 for pre,ref in orig:
  bb,_=probs(new,B,pre);assert bb==ref
  causal+=1
new.wf148_close(B)
# Historical documentation corpus. The new branch MUST be gated off exactly.
A=old.wf148_open(*MODEL);B=new.wf148_open(*MODEL);assert A and B
doc_exact=0;doc_active=0
for fname in ('win.bin','win_test.bin'):
 for raw,skip in wins(fname):
  for pos,word in getwords(raw,skip):
   past=raw[:pos]
   assert old.wf148_begin(A,ctypes.c_char_p(past),len(past),F(.25))>=0
   assert new.wf148_begin(B,ctypes.c_char_p(past),len(past),F(.25))>=0
   doc_active+=int(new.wf168_ann_alpha(B)>0)
   for pre in ('',word[:1],word[:4]):
    aa,_=probs(old,A,pre);bb,_=probs(new,B,pre)
    assert aa==bb,(fname,pos,pre)
    doc_exact+=1
old.wf148_close(A);new.wf148_close(B)
assert doc_active==0
# Invalid settings and wrong model index fail closed, not silently ignored.
fail=0
for key,value in [('V168_ANN_STRENGTH','inf'),('V168_ANN_STRENGTH','.501'),('V168_ANN_SIGNED_SHARE','-1'),('V168_ANN_DOMAIN_MIN','nan'),('V168_ANN_INDEX','/tmp/n168-missing-index-does-not-exist.bin')]:
 prev=os.environ.get(key);os.environ[key]=value
 assert not new.wf148_open(*MODEL),(key,value)
 fail+=1
 if prev is None:os.environ.pop(key,None)
 else:os.environ[key]=prev
report={'status':'PASS','v167_optout_exact_27way_vectors':rollback,'active_normalized_27way_vectors':normalized,
 'future_suffix_ignored_bit_identically':causal,'ann_active_dev_positions':enabled,'historical_doc_exact_vectors':doc_exact,
 'historical_doc_ann_admissions':doc_active,'invalid_config_fail_closed':fail,
 'source_v166_sha256':hashlib.sha256((R/'checkpoints/v166_lexical_birth.bin').read_bytes()).hexdigest(),
 'signed_ann_sha256':hashlib.sha256((R/'checkpoints/v168_signed_ann.bin').read_bytes()).hexdigest(),
 'full_256_byte_BPB_measured':False,'promotion':'NO'}
(R/'results/v168_ann_native_regression.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2),flush=True)