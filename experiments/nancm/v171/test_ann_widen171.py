#!/usr/bin/env python3
"""Native v171 widened proposal harness. Requires full cumulative recovery ZIP."""
import ctypes,json,os,sys
from pathlib import Path
R=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(R/'scripts'),str(R/'parent/scripts')]
from train_gate163 import setup,winread
from test_native_speller156 import load,MODEL,F
setup(core=False,domain=True,gate=None)
os.environ.update(V164_DISTRIBUTION_BUDGET='.6',V164_DISTRIBUTION_INDEX=str(R/'checkpoints/v164_native_distribution.tsv'),V164_DOMAIN_GATE='1',V164_DOMAIN_THRESHOLD='1.5',V166_BIRTH_MODEL=str(R/'checkpoints/v166_lexical_birth.bin'),V166_LEXICAL_BIRTH='.05',V167_GATE_MODE='1',V167_DOMAIN_MIN='1.4',V167_HIGH_ALPHA='.5',V167_TRUST_THRESHOLD='.5',V168_ANN_INDEX=str(R/'checkpoints/v168_signed_ann.bin'),V168_ANN_SIGNED_SHARE='.66',V168_ANN_DOMAIN_MIN='1.4',V168_ANN_STRENGTH='.05',V169_NEURAL_RERANK='.05')
old=load(R/'bin/libwordfield169.so')
new=load(R/'bin/libwordfield171.so')
for l in (old,new):
 l.wf168_ann_candidate_count.argtypes=[ctypes.c_void_p];l.wf168_ann_candidate_count.restype=ctypes.c_int
 l.wf168_ann_candidate.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_char_p,ctypes.c_size_t,ctypes.POINTER(ctypes.c_double)];l.wf168_ann_candidate.restype=ctypes.c_int
rows=[json.loads(s) for s in (R/'results/v169_portfolio_dev.jsonl').read_text().splitlines()]
wins=list(winread(R/'external_book_gate/heldout_dev160.bin'))
def get(l,h,past):
 assert l.wf148_begin(h,ctypes.c_char_p(past),len(past),F(.25))>=0
 a=[]
 for i in range(l.wf168_ann_candidate_count(h)):
  w=ctypes.create_string_buffer(32);p=ctypes.c_double()
  assert l.wf168_ann_candidate(h,i,w,32,ctypes.byref(p))==0
  a.append((w.value.decode(),p.value))
 assert not a or abs(sum(p for w,p in a)-1.)<1e-6
 return a
os.environ['V171_ANN_NEIGHBORS']='32';os.environ['V171_ANN_CANDIDATES']='64'
h0=old.wf148_open(*MODEL);h1=new.wf148_open(*MODEL);assert h0 and h1
for r in rows[:40]:
 past=wins[int(r['book'])][0][:r['pos']]
 assert get(old,h0,past)==get(new,h1,past)
 a=(F*27)();b=(F*27)();e=F();f=F()
 assert old.wf148_next(h0,b'',0,a,ctypes.byref(e))==0
 assert new.wf148_next(h1,b'',0,b,ctypes.byref(f))==0
 assert list(a)==list(b) and e.value==f.value
old.wf148_close(h0);new.wf148_close(h1)
print('PASS exact v169 rollback: 40 signed-proposal and 27-way probability vectors',flush=True)
for n,k in ((32,64),(64,128),(128,256),(256,512),(512,1024)):
 os.environ['V171_ANN_NEIGHBORS']=str(n);os.environ['V171_ANN_CANDIDATES']=str(k)
 h=new.wf148_open(*MODEL);assert h
 hits=0;top1=0
 for r in rows:
  past=wins[int(r['book'])][0][:r['pos']];ws=[w for w,p in get(new,h,past)]
  hits+=r['truth'] in ws
  top1+=bool(ws and ws[0]==r['truth'])
 new.wf148_close(h)
 print('ANN',n,k,'recall',hits,'of',len(rows),'top1',top1,flush=True)
