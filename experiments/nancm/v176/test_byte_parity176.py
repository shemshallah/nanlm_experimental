#!/usr/bin/env python3
"""Exact 256-way byte law (v176 inherits v175 and v170) on old DEV/TEST."""
import argparse,ctypes,json,sys,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'parent/scripts')]
from train_gate163 import setup,winread
from test_native_speller156 import MODEL,F

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--n',type=int,default=200);args=ap.parse_args()
 setup(core=False,domain=True,gate=None)
 import os
 os.environ.update(V164_DISTRIBUTION_BUDGET='.6',V164_DISTRIBUTION_INDEX=str(ROOT/'checkpoints/v164_native_distribution.tsv'),V164_DOMAIN_GATE='1',V164_DOMAIN_THRESHOLD='1.5',V166_BIRTH_MODEL=str(ROOT/'checkpoints/v166_lexical_birth.bin'),V166_LEXICAL_BIRTH='.05',V167_GATE_MODE='1',V167_DOMAIN_MIN='1.4',V167_HIGH_ALPHA='.5',V167_TRUST_THRESHOLD='.5',V168_ANN_INDEX=str(ROOT/'checkpoints/v168_signed_ann.bin'),V168_ANN_SIGNED_SHARE='.66',V168_ANN_DOMAIN_MIN='1.4',V168_ANN_STRENGTH='.05',V169_NEURAL_RERANK='.05')
 old=ctypes.CDLL(str(ROOT/'bin/libwordfield175.so'));new=ctypes.CDLL(str(ROOT/'bin/libwordfield176.so'))
 for l in (old,new):
  l.wf170_byte_open.argtypes=[ctypes.c_char_p]*4;l.wf170_byte_open.restype=ctypes.c_void_p
  l.wf170_byte_close.argtypes=[ctypes.c_void_p]
  l.wf170_byte_next.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_size_t,ctypes.POINTER(F)];l.wf170_byte_next.restype=ctypes.c_int
  l.wf170_byte_set_adaptive.argtypes=[ctypes.c_void_p,ctypes.c_int];l.wf170_byte_set_adaptive.restype=ctypes.c_int
 args2=(*MODEL,str(ROOT/'checkpoints/v169_native_byte_lattice.bin').encode())
 a=old.wf170_byte_open(*args2);b=new.wf170_byte_open(*args2);assert a and b
 assert old.wf170_byte_set_adaptive(a,0)==0 and new.wf170_byte_set_adaptive(b,0)==0
 checks=0;maxd=0.
 try:
  for cohort in ('dev','test'):
   wins=list(winread(ROOT/f'external_book_gate/heldout_{cohort}160.bin'))[:24]
   for book,text in enumerate(wins):
    raw=text[0]
    for pos in range(0,min(len(raw),800),max(1,800//max(1,args.n//48))):
     past=raw[:pos];q=(F*256)();p=(F*256)();c=ctypes.c_char_p(past)
     assert old.wf170_byte_next(a,c,len(past),p)==0
     assert new.wf170_byte_next(b,c,len(past),q)==0
     assert bytes(p)==bytes(q)
     assert abs(sum(p)-1)<1e-5 and all(math.isfinite(x) and x>0 for x in p)
     checks+=1
 finally:old.wf170_byte_close(a);new.wf170_byte_close(b)
 result={'status':'PASS','exact_256way_byte_rollback_vectors':checks,'max_abs_error':maxd,'adaptive_disabled_on_both':True,'old_archive_dev_test_exposed':True,'no_model_weight_changes':True}
 (ROOT/'results/v176_byte_rollback.json').write_text(json.dumps(result,indent=2));print(result,flush=True)
if __name__=='__main__':main()