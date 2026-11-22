#!/usr/bin/env python3
"""Native symbiotic S+/S- ANN neural rescoring trust suite.
Includes exact rollback, future-suffix exclusion, domain isolation, failure
closure and checkpoint-preservation commitments.
"""
import ctypes,os,sys,json,math,hashlib
from pathlib import Path
R=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(R/'scripts'),str(R/'parent/scripts')]
from train_gate163 import setup,winread
from test_native_speller156 import load,MODEL,F,getwords,wins
setup(core=False,domain=True,gate=None)
os.environ.update(V164_DISTRIBUTION_BUDGET='.6',V164_DISTRIBUTION_INDEX=str(R/'checkpoints/v164_native_distribution.tsv'),V164_DOMAIN_GATE='1',V164_DOMAIN_THRESHOLD='1.5',V166_BIRTH_MODEL=str(R/'checkpoints/v166_lexical_birth.bin'),V166_LEXICAL_BIRTH='.05',V167_GATE_MODE='1',V167_DOMAIN_MIN='1.4',V167_HIGH_ALPHA='.5',V167_TRUST_THRESHOLD='.5',V168_ANN_INDEX=str(R/'checkpoints/v168_signed_ann.bin'),V168_ANN_SIGNED_SHARE='.66',V168_ANN_DOMAIN_MIN='1.4',V168_ANN_STRENGTH='.05')
parent=load(R/'bin/libwordfield168_ann.so')
new=load(R/'bin/libwordfield169.so')
old=load(R/'bin/libwordfield167.so')
new.wf168_ann_alpha.argtypes=[ctypes.c_void_p];new.wf168_ann_alpha.restype=ctypes.c_double
new.wf169_neural_rerank.argtypes=[ctypes.c_void_p];new.wf169_neural_rerank.restype=ctypes.c_double

def begin(lib,h,past):
 assert lib.wf148_begin(h,ctypes.c_char_p(past),len(past),F(.25))>=0

def vectors(lib,h,prefix):
 a=(F*27)();e=F();b=prefix.encode()
 assert lib.wf148_next(h,b,len(b),a,ctypes.byref(e))==0
 return bytes(a)+bytes(e),[float(x)+e.value/27. for x in a]

rows=[]
for bi,(raw,skip) in enumerate(winread(R/'external_book_gate/heldout_dev160.bin')):
 if bi>=24:break
 ws=getwords(raw,skip)
 for pos,truth in ([ws[min(len(ws)-1,int((j+.5)*len(ws)/min(20,len(ws))))] for j in range(min(20,len(ws)))] if ws else []):
  rows.append((raw[:pos],truth))
assert len(rows)==480

# Stage 1: rescoring disabled => exact v168 native 27-way parity.
os.environ['V169_NEURAL_RERANK']='0'
a=parent.wf148_open(*MODEL);b=new.wf148_open(*MODEL);assert a and b
rollback=0
for past,truth in rows[:150]:
 begin(parent,a,past);begin(new,b,past)
 for pre in ('',truth[:1],truth[:3]):
  assert vectors(parent,a,pre)[0]==vectors(new,b,pre)[0],('exact v168 rollback',truth,pre)
  rollback+=1
parent.wf148_close(a);new.wf148_close(b)

# Stage 2: ANN entirely off => exact v167 regardless of rescoring strength.
os.environ['V169_NEURAL_RERANK']='.05';os.environ['V168_ANN_STRENGTH']='0'
a=old.wf148_open(*MODEL);b=new.wf148_open(*MODEL);assert a and b
rollback167=0
for past,truth in rows[:100]:
 begin(old,a,past);begin(new,b,past)
 for pre in ('',truth[:1],truth[:3]):
  assert vectors(old,a,pre)[0]==vectors(new,b,pre)[0],('exact v167 rollback',truth,pre)
  rollback167+=1
old.wf148_close(a);new.wf148_close(b)

os.environ['V168_ANN_STRENGTH']='.05';h=new.wf148_open(*MODEL);assert h
enabled=0;norm=0;causal=0;changed=0
for past,truth in rows:
 begin(new,h,past); enabled+=int(new.wf168_ann_alpha(h)>0)
 before=[]
 for pre in ('',truth[:1],truth[:3]):
  blob,probs=vectors(new,h,pre)
  assert all(math.isfinite(q) and q>=0 for q in probs) and abs(sum(probs)-1)<2e-4
  before.append((pre,blob));norm+=1
 # Passing poisoned buffer with shorter declared length must be bit-identical.
 poison=past+b' unseen-target-future-word'
 assert new.wf148_begin(h,ctypes.c_char_p(poison),len(past),F(.25))>=0
 for pre,blob in before:
  assert vectors(new,h,pre)[0]==blob,('future suffix leaked',truth,pre)
  causal+=1
new.wf148_close(h)
assert enabled==244,enabled
# Stage 3: no change on protected docs domain; v167 exact bytes.
a=old.wf148_open(*MODEL);b=new.wf148_open(*MODEL);assert a and b
docs=0;docs_enabled=0
for fixture in ('win.bin','win_test.bin'):
 for raw,skip in wins(fixture):
  for pos,word in getwords(raw,skip):
   past=raw[:pos];begin(old,a,past);begin(new,b,past)
   docs_enabled+=int(new.wf168_ann_alpha(b)>0)
   for pre in ('',word[:1],word[:4]):
    assert vectors(old,a,pre)[0]==vectors(new,b,pre)[0],(fixture,pos,pre)
    docs+=1
old.wf148_close(a);new.wf148_close(b)
assert docs==4545 and docs_enabled==0,(docs,docs_enabled)
# Stage 4: invalid config must fail at open, not silently disable.
fail=0
for val in ('nan','inf','-0.001','2.01','garbage'):
 os.environ['V169_NEURAL_RERANK']=val
 assert not new.wf148_open(*MODEL),val
 fail+=1
os.environ['V169_NEURAL_RERANK']='.05'
# Immutable weight/checkpoint digests are recorded for recovery lineage.
protected=['checkpoints/v164_native_distribution.tsv','checkpoints/v165_signed_cloud.bin',
 'checkpoints/v166_lexical_birth.bin','checkpoints/v166_lexical_birth.pt',
 'checkpoints/v168_signed_ann.bin','parent/checkpoints/native_cloud155/v155_joint_trained.cloud']
shas={p:hashlib.sha256((R/p).read_bytes()).hexdigest() for p in protected}
result={'status':'PASS','exact_v168_rollback_vectors':rollback,'exact_v167_rollback_vectors':rollback167,
 'active_normalized_vectors':norm,'future_suffix_exclusion_exact_vectors':causal,
 'docs_domain_exact_vectors':docs,'docs_domain_ann_admissions':docs_enabled,
 'active_heldout_dev_positions':enabled,'invalid_settings_fail_closed':fail,'protected_checkpoint_digests':shas,
 'native_full_256_byte_bpb':'provided by wf169_byte_* and verified in test_native_byte169.py',
 'promotion':'NOT_PROMOTED'}
(R/'results/v169_native_regression.json').write_text(json.dumps(result,indent=2))
print('FINAL',json.dumps({k:v for k,v in result.items() if k!='protected_checkpoint_digests'}),flush=True)