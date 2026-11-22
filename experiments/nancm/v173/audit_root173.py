#!/usr/bin/env python3
"""Full 27-way word-root head-to-head at every historically exposed DEV word start.
Causal; no target suffix in calls, no model fitting on DEV.
"""
import os,sys,ctypes,math,json,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'parent/scripts')]
from train_gate163 import setup,winread
from test_native_speller156 import MODEL,F
setup(core=False,domain=True,gate=None)
os.environ.update(V164_DISTRIBUTION_BUDGET='.6',V164_DISTRIBUTION_INDEX=str(ROOT/'checkpoints/v164_native_distribution.tsv'),V164_DOMAIN_GATE='1',V164_DOMAIN_THRESHOLD='1.5',V166_BIRTH_MODEL=str(ROOT/'checkpoints/v166_lexical_birth.bin'),V166_LEXICAL_BIRTH='.05',V167_GATE_MODE='1',V167_DOMAIN_MIN='1.4',V167_HIGH_ALPHA='.5',V167_TRUST_THRESHOLD='.5',V168_ANN_INDEX=str(ROOT/'checkpoints/v168_signed_ann.bin'),V168_ANN_SIGNED_SHARE='.66',V168_ANN_DOMAIN_MIN='1.4',V168_ANN_STRENGTH='.05',V169_NEURAL_RERANK='.05',V171_ANN_NEIGHBORS='32',V171_ANN_CANDIDATES='64',V172_ANN_RAILS='1',V172_TEACHER_SHARE='0')
lib=ctypes.CDLL(str(ROOT/'bin/libwordfield173.so'))
lib.wf173_joint_open.argtypes=[ctypes.c_char_p]*4+[ctypes.c_double];lib.wf173_joint_open.restype=ctypes.c_void_p
lib.wf173_joint_begin.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_size_t,F];lib.wf173_joint_begin.restype=ctypes.c_int
lib.wf173_joint_next.argtypes=[ctypes.c_void_p,ctypes.c_char_p,ctypes.c_size_t,ctypes.POINTER(F),ctypes.POINTER(F)];lib.wf173_joint_next.restype=ctypes.c_int
lib.wf173_joint_close.argtypes=[ctypes.c_void_p]
pack=str(ROOT/'checkpoints/v169_native_byte_lattice.bin').encode()
ref=lib.wf173_joint_open(*MODEL,None,0.);exp=lib.wf173_joint_open(*MODEL,pack,.1);assert ref and exp
wins=list(winread(ROOT/'external_book_gate/heldout_dev160.bin'))
rows=[json.loads(s) for s in (ROOT/'results/v169_portfolio_dev.jsonl').read_text().splitlines()]
metric=collections.defaultdict(lambda:{'n':0,'top1':0,'top5':0,'nll':0.})
for r in rows:
 past=wins[int(r['book'])][0][:r['pos']];word=r['truth'];target=ord(word[0])-97
 for name,h in (('parent',ref),('joint',exp)):
  assert lib.wf173_joint_begin(h,ctypes.c_char_p(past),len(past),F(.25))>=0
  o=(F*27)();e=F()
  assert lib.wf173_joint_next(h,b'',0,o,ctypes.byref(e))==0
  p=[float(o[j])+e.value/27 for j in range(26)];z=sum(p)
  p=[x/z for x in p];rank=sorted(range(26),key=lambda j:p[j],reverse=True)
  d=metric[name];d['n']+=1;d['top1']+=int(rank[0]==target);d['top5']+=int(target in rank[:5]);d['nll']-=math.log(max(1e-18,p[target]))
lib.wf173_joint_close(ref);lib.wf173_joint_close(exp)
out={k:{**v,'nll_nats_per_first_letter':v['nll']/v['n']} for k,v in metric.items()}
(ROOT/'results/v173_first_letter_root.json').write_text(json.dumps(out,indent=2))
print('FIRST_LETTER',json.dumps(out),flush=True)