#!/usr/bin/env python3
"""Active v166 must preserve every historical documentation prefix exactly."""
import ctypes,os,sys
from pathlib import Path
R=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(R/'scripts'))
from train_gate163 import setup
sys.path.insert(0,str(R/'parent/scripts'))
from test_native_speller156 import load,wins,getwords,MODEL,F
setup(core=False,domain=True)
os.environ.update(V160_ASSOC_BUDGET='.4',V160_ASSOC_INDEX=str(R/'checkpoints/v161_unified.tsv'),V164_DISTRIBUTION_BUDGET='.6',V164_DISTRIBUTION_INDEX=str(R/'checkpoints/v164_native_distribution.tsv'),V164_DOMAIN_GATE='1',V164_DOMAIN_THRESHOLD='1.5',V166_LEXICAL_BIRTH='.5',V166_BIRTH_MODEL=str(R/'checkpoints/v166_lexical_birth.bin'))
a=load(R/'bin/libwordfield164.so');b=load(R/'bin/libwordfield166.so')
b.wf166_candidate_count.argtypes=[ctypes.c_void_p];b.wf166_candidate_count.restype=ctypes.c_int
x=a.wf148_open(*MODEL);y=b.wf148_open(*MODEL);assert x and y
check=0;active=0
for fname in ('win.bin','win_test.bin'):
 for raw,skip in wins(fname):
  for pos,word in getwords(raw,skip):
   past=raw[:pos];assert a.wf148_begin(x,ctypes.c_char_p(past),len(past),F(.25))>=0
   assert b.wf148_begin(y,ctypes.c_char_p(past),len(past),F(.25))>=0
   active+=b.wf166_candidate_count(y)>0
   for prefix in ('',word[:1],word[:4]):
    outs=[]
    for l,h in ((a,x),(b,y)):
     p=(F*27)();e=F();assert l.wf148_next(h,prefix.encode(),len(prefix),p,ctypes.byref(e))==0
     outs.append((bytes(p),bytes(e)))
    assert outs[0]==outs[1],(fname,pos,prefix)
    check+=1
assert active==0,(active,check)
a.wf148_close(x);b.wf148_close(y)
print(f'PASS v166 active documentary-domain exclusion: {check} exact prefix distributions; {active} proposals')