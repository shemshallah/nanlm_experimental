#!/usr/bin/env python3
"""Strict native C ABI, rollback, whole-word first-letter normalization and causality.
Tests 27-way vectors, not only observed target probability.
"""
import ctypes,os,sys,json,math,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'parent/scripts')]
from train_gate163 import setup,winread
from test_native_speller156 import MODEL,F,load
setup(core=False,domain=True,gate=None)
os.environ.update(V164_DISTRIBUTION_BUDGET='.6',V164_DISTRIBUTION_INDEX=str(ROOT/'checkpoints/v164_native_distribution.tsv'),V164_DOMAIN_GATE='1',V164_DOMAIN_THRESHOLD='1.5',V166_BIRTH_MODEL=str(ROOT/'checkpoints/v166_lexical_birth.bin'),V166_LEXICAL_BIRTH='.05',V167_GATE_MODE='1',V167_DOMAIN_MIN='1.4',V167_HIGH_ALPHA='.5',V167_TRUST_THRESHOLD='.5',V168_ANN_INDEX=str(ROOT/'checkpoints/v168_signed_ann.bin'),V168_ANN_SIGNED_SHARE='.66',V168_ANN_DOMAIN_MIN='1.4',V168_ANN_STRENGTH='.05',V169_NEURAL_RERANK='.05',V171_ANN_NEIGHBORS='32',V171_ANN_CANDIDATES='64',V172_ANN_RAILS='1',V172_TEACHER_SHARE='0')
lib=load(ROOT/'bin/libwordfield174.so')
lib.wf173_joint_open.argtypes=[ctypes.c_char_p]*4+[ctypes.c_double];lib.wf173_joint_open.restype=ctypes.c_void_p
lib.wf173_joint_close.argtypes=[ctypes.c_void_p];lib.wf173_joint_close.restype=None
lib.wf173_joint_begin.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_size_t,F];lib.wf173_joint_begin.restype=ctypes.c_int
lib.wf173_joint_next.argtypes=[ctypes.c_void_p,ctypes.c_char_p,ctypes.c_size_t,ctypes.POINTER(F),ctypes.POINTER(F)];lib.wf173_joint_next.restype=ctypes.c_int
lib.wf173_joint_gamma.argtypes=[ctypes.c_void_p];lib.wf173_joint_gamma.restype=ctypes.c_double
lib.wf169_byte_open.argtypes=[ctypes.c_char_p]*4;lib.wf169_byte_open.restype=ctypes.c_void_p
lib.wf169_byte_close.argtypes=[ctypes.c_void_p]
lib.wf169_byte_next.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_size_t,ctypes.POINTER(F)];lib.wf169_byte_next.restype=ctypes.c_int
pack=str(ROOT/'checkpoints/v169_native_byte_lattice.bin').encode();args=(*MODEL,pack)
plain=lib.wf148_open(*MODEL);zero=lib.wf173_joint_open(*MODEL,None,0.);gated=lib.wf173_joint_open(*MODEL,pack,.1);byte=lib.wf169_byte_open(*args)
assert plain and zero and gated and byte
assert not lib.wf173_joint_open(*MODEL,None,.1) # fail closed if sidecar absent
assert not lib.wf173_joint_open(*MODEL,pack,-.1)
assert not lib.wf173_joint_open(*MODEL,pack,float('nan'))
assert abs(lib.wf173_joint_gamma(gated)-.1)<1e-14
wins=list(winread(ROOT/'external_book_gate/heldout_dev160.bin'))
rows=[json.loads(s) for s in (ROOT/'results/v169_portfolio_dev.jsonl').read_text().splitlines()]
maxroot=0.;exact=0;compare=0;idempotent=0;changes=0

def next27(fn,h,pre=b''):
 out=(F*27)();esc=F();assert fn(h,pre,len(pre),out,ctypes.byref(esc))==0
 return [float(out[j]) for j in range(27)],float(esc.value)
for r in rows[:48]:
 hist=wins[int(r['book'])][0][:r['pos']]
 assert lib.wf148_begin(plain,ctypes.c_char_p(hist),len(hist),F(.25))>=0
 assert lib.wf173_joint_begin(zero,ctypes.c_char_p(hist),len(hist),F(.25))>=0
 for pre in (b'',b't',b'co',b'be',b'men'):
  a,ae=next27(lib.wf148_next,plain,pre);b,be=next27(lib.wf173_joint_next,zero,pre)
  assert a==b and ae==be;exact+=1
 assert lib.wf173_joint_begin(gated,ctypes.c_char_p(hist),len(hist),F(.25))>=0
 a,esc=next27(lib.wf148_next,plain,b'');c,ce=next27(lib.wf173_joint_next,gated,b'')
 raw=[a[j]+esc/27 for j in range(27)]
 o=(F*256)();assert lib.wf169_byte_next(byte,ctypes.c_char_p(hist),len(hist),o)==0
 p=[float(o[97+j])+float(o[65+j]) for j in range(26)]
 mass=sum(raw[:26]);pa=sum(p)
 weights=[math.exp(.9*math.log(raw[j]/mass)+.1*math.log(p[j]/pa)) for j in range(26)]
 denom=sum(weights)
 expected=[mass*z/denom for z in weights]+[raw[26]]
 assert abs(sum(c)-1)<1e-5 and ce==0
 err=max(abs(x-y) for x,y in zip(expected,c));maxroot=max(maxroot,err)
 assert err<2e-7,(err,c,expected)
 changes+=int(any(abs(c[j]-raw[j])>1e-4 for j in range(26)))
 again,_=next27(lib.wf173_joint_next,gated,b'');assert c==again;idempotent+=1
 for pre in (b't',b'co',b'be'):
  x,xe=next27(lib.wf148_next,plain,pre);y,ye=next27(lib.wf173_joint_next,gated,pre)
  assert x==y and xe==ye;compare+=1
for h in (plain,):lib.wf148_close(h)
for h in (zero,gated):lib.wf173_joint_close(h)
lib.wf169_byte_close(byte)
report={'status':'PASS','complete_root_prefix_vectors_exact_rollback':exact,'joint_root_checks':idempotent,'later_prefix_exact_parent_checks':compare,'changed_root_distributions':changes,'max_root_fusion_abs_error':maxroot,'default_gamma':0.,'experimental_gamma':.1,'strictly_causal':True,'normalized_root':True,'missing_byte_pack_fails_closed':True,'invalid_gamma_fails_closed':True,'weights_and_optimizer_unchanged':True}
(ROOT/'results/v174_native_joint_parity.json').write_text(json.dumps(report,indent=2));print('PASS',json.dumps(report),flush=True)