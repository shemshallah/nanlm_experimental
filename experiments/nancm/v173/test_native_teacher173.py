#!/usr/bin/env python3
"""Independent PyTorch/native v173 exact CE whole-word teacher and 27-symbol mix audit."""
import os,sys,ctypes,math,json
from pathlib import Path
import torch,numpy as np
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'parent/scripts'),str(ROOT/'experiments/nancm/v172')]
from train_gate163 import setup,winread
from test_native_speller156 import load,MODEL,F
from train_conditional_cloud172 import load_parent,CloudNCE,word_ids
setup(core=False,domain=True,gate=None)
os.environ.update(V164_DISTRIBUTION_BUDGET='.6',V164_DISTRIBUTION_INDEX=str(ROOT/'checkpoints/v164_native_distribution.tsv'),V164_DOMAIN_GATE='1',V164_DOMAIN_THRESHOLD='1.5',V166_BIRTH_MODEL=str(ROOT/'checkpoints/v166_lexical_birth.bin'),V166_LEXICAL_BIRTH='.05',V167_GATE_MODE='1',V167_DOMAIN_MIN='1.4',V167_HIGH_ALPHA='.5',V167_TRUST_THRESHOLD='.5',V168_ANN_INDEX=str(ROOT/'checkpoints/v168_signed_ann.bin'),V168_ANN_SIGNED_SHARE='.66',V168_ANN_DOMAIN_MIN='1.4',V168_ANN_STRENGTH='.05',V169_NEURAL_RERANK='.05',V171_ANN_NEIGHBORS='32',V171_ANN_CANDIDATES='64',V172_ANN_RAILS='1',V172_TEACHER_SHARE='.2',V172_TEACHER_MODEL=str(ROOT/'checkpoints/v173_native_signed_teacher.bin'))
new=load(ROOT/'bin/libwordfield173.so');old=load(ROOT/'bin/libwordfield171.so')
new.wf172_teacher_strength.argtypes=[ctypes.c_void_p];new.wf172_teacher_strength.restype=ctypes.c_double
new.wf172_teacher_distribution.argtypes=[ctypes.c_void_p,ctypes.POINTER(F),ctypes.c_size_t];new.wf172_teacher_distribution.restype=ctypes.c_int
h=new.wf148_open(*MODEL);assert h,'trained teacher sidecar refused native open'
assert abs(new.wf172_teacher_strength(h)-.2)<1e-12
names,C,O,B,spatial,gain,fp=load_parent();mapping={w:i for i,w in enumerate(names)}
ck=torch.load(ROOT/'checkpoints/v173_exact_ce_word.pt',weights_only=False,map_location='cpu')
m=CloudNCE(C,O,B,spatial,gain,names);m.load_state_dict(ck['state_dict']);m.eval()
oldh=old.wf148_open(*MODEL);assert oldh
wins=list(winread(ROOT/'external_book_gate/heldout_dev160.bin'))
rows=[json.loads(s) for s in (ROOT/'results/v169_portfolio_dev.jsonl').read_text().splitlines()]
maxt=0.;maxroot=0.;prefix_vectors=0
with torch.no_grad():
 for r in rows[::12]:
  b=int(r['book']);pos=int(r['pos']);past=wins[b][0][:pos]
  assert new.wf148_begin(h,ctypes.c_char_p(past),pos,F(.25))>=0
  assert old.wf148_begin(oldh,ctypes.c_char_p(past),pos,F(.25))>=0
  arr=(F*12000)();assert new.wf172_teacher_distribution(h,arr,12000)==0
  p=np.array(arr,dtype='float64');assert (p>0).all() and abs(p.sum()-1)<1e-5
  import re
  context=[v.group().lower().decode('ascii') for v in re.finditer(rb'[A-Za-z]+',past)]
  ids=torch.tensor([word_ids(' '.join(context[-3:]),mapping)])
  q=m.query(ids);sc=m.gain.clamp(0,25)*q@m.out.weight.T+m.bias;ref=torch.softmax(sc,dim=1).numpy()[0]
  err=np.max(np.abs(p-ref));maxt=max(maxt,err);assert err<2e-4,(b,pos,err)
  o=(F*27)();e=F();z=(F*27)();f=F()
  assert old.wf148_next(oldh,b'',0,o,ctypes.byref(e))==0
  assert new.wf148_next(h,b'',0,z,ctypes.byref(f))==0
  root=np.zeros(27)
  for word,pi in zip(names,p):root[ord(word[0])-97]+=pi
  expected=.8*(np.array(o,dtype='float64')+e.value/27)+.2*root
  me=np.max(np.abs(np.array(z,dtype='float64')-expected));maxroot=max(maxroot,me)
  assert me<2e-6,(b,pos,me)
  assert abs(sum(z)-1)<1e-5 and f.value==0
  prefix_vectors+=1
new.wf148_close(h);old.wf148_close(oldh)
# Invalid sidecar path must fail closed if opt-in requested.
os.environ['V172_TEACHER_MODEL']=str(ROOT/'checkpoints/does_not_exist.bin')
assert not new.wf148_open(*MODEL)
os.environ['V172_TEACHER_SHARE']='0'
back=new.wf148_open(*MODEL);assert back;new.wf148_close(back)
report={'status':'PASS','contexts':prefix_vectors,'max_teacher_probability_difference':maxt,'max_root_mixture_difference':maxroot,'native_12000_way_normalized':True,'missing_optin_sidecar_fails_closed':True,'disabled_parent_path_opens_without_sidecar':True,'gating':'expert shares only if explicit V172_TEACHER_SHARE >0','provenance':'DEV exposed diagnostic only'}
(ROOT/'results/v173_native_teacher_parity.json').write_text(json.dumps(report,indent=2));print('FINAL',json.dumps(report),flush=True)