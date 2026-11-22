#!/usr/bin/env python3
"""History utility audit for the exact native S+/S- neighbor branch.
Donor contexts are from different books. Neither query can see target suffix.
"""
import ctypes,os,sys,json,math,collections,hashlib,time
from pathlib import Path
R=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(R/'scripts'),str(R/'parent/scripts')]
from train_gate163 import setup,winread
from test_native_speller156 import load,MODEL,F,getwords
setup(core=False,domain=True,gate=None)
os.environ.update(V164_DISTRIBUTION_BUDGET='.6',V164_DISTRIBUTION_INDEX=str(R/'checkpoints/v164_native_distribution.tsv'),V164_DOMAIN_GATE='1',V164_DOMAIN_THRESHOLD='1.5',V166_BIRTH_MODEL=str(R/'checkpoints/v166_lexical_birth.bin'),V166_LEXICAL_BIRTH='.05',V167_GATE_MODE='1',V167_DOMAIN_MIN='1.4',V167_HIGH_ALPHA='.5',V167_TRUST_THRESHOLD='.5',V168_ANN_INDEX=str(R/'checkpoints/v168_signed_ann.bin'),V168_ANN_SIGNED_SHARE='.66',V168_ANN_DOMAIN_MIN='1.4',V168_ANN_STRENGTH='.05')
l=load(R/'bin/libwordfield168_ann.so')
l.wf168_ann_alpha.argtypes=[ctypes.c_void_p];l.wf168_ann_alpha.restype=ctypes.c_double
l.wf168_ann_candidate.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_char_p,ctypes.c_size_t,ctypes.POINTER(ctypes.c_double)];l.wf168_ann_candidate.restype=ctypes.c_int

def get(h,past):
 assert l.wf148_begin(h,ctypes.c_char_p(past),len(past),F(.25))>=0
 if l.wf168_ann_alpha(h)<=0:return {}
 out={}
 for i in range(64):
  w=ctypes.create_string_buffer(32);p=ctypes.c_double()
  if l.wf168_ann_candidate(h,i,w,32,ctypes.byref(p))!=0:break
  out[w.value.decode()]=p.value
 return out

def main():
 rows=[]
 for bi,(raw,skip) in enumerate(winread(R/'external_book_gate/heldout_dev160.bin')):
  if bi>=24:break
  ws=getwords(raw,skip)
  sample=[ws[min(len(ws)-1,int((j+.5)*len(ws)/min(20,len(ws))))] for j in range(min(20,len(ws)))] if ws else []
  for pos,truth in sample: rows.append({'book':bi,'truth':truth,'past':raw[:pos]})
 assert len(rows)==480
 h=l.wf148_open(*MODEL);assert h;t=time.monotonic();pairs=[]
 # No label or target influences donor selection: fixed cyclic permutation over
 # only gate-active contexts; reject donor from same book.
 active=[]
 for r in rows:
  original=get(h,r['past'])
  if original:active.append((r,original))
 for idx,(r,original) in enumerate(active):
  j=(idx+len(active)//2)%len(active)
  while active[j][0]['book']==r['book']:j=(j+1)%len(active)
  donor=active[j][0]
  unrelated=active[j][1]
  if original and unrelated:
   def rank(v,w):return next((j+1 for j,(x,p) in enumerate(v.items()) if x==w),65)
   pairs.append({'book':r['book'],'target':r['truth'],'actual_rank':rank(original,r['truth']),'wrong_rank':rank(unrelated,r['truth']),'actual_mass':original.get(r['truth'],0.),'wrong_mass':unrelated.get(r['truth'],0.),'actual_top':next(iter(original)),'wrong_top':next(iter(unrelated))})
 l.wf148_close(h)
 res={'n':len(pairs),'books':len(set(r['book'] for r in pairs)),'actual':{str(k):sum(r['actual_rank']<=k for r in pairs) for k in [1,5,16,32,64]},'unrelated':{str(k):sum(r['wrong_rank']<=k for r in pairs) for k in [1,5,16,32,64]},'mean_log2_true_mass_actual':sum(math.log2(max(1e-12,r['actual_mass'])) for r in pairs)/len(pairs),'mean_log2_true_mass_unrelated':sum(math.log2(max(1e-12,r['wrong_mass'])) for r in pairs)/len(pairs),'paired_better':sum(r['actual_rank']<r['wrong_rank'] for r in pairs),'paired_worse':sum(r['actual_rank']>r['wrong_rank'] for r in pairs),'time':round(time.monotonic()-t,2),'claim':'ANN branch only, not whole model; forced wrong donor from another book; target never in begin'}
 (R/'results/v169_history_counterfactual.json').write_text(json.dumps(res,indent=2));print('FINAL',json.dumps(res),flush=True)
 (R/'results/v169_history_counterfactual_rows.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in pairs))
if __name__=='__main__':main()