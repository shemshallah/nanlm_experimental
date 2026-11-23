#!/usr/bin/env python3
"""Native speculative prefix sampler with exact P/Q residual acceptance."""
import argparse,ctypes,json,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'experiments/nancm/v176'),str(ROOT/'experiments/nancm/v175')]
from bench_native176 import init_lib,top5,prefix
from train_long175 import load_models,load_split

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--n',type=int,default=640);args=ap.parse_args()
 import torch;torch.set_num_threads(2)
 _,_,names,_,_=load_models();X,Y,B=load_split('val',ROOT/'results/v175_release3');rng=np.random.default_rng(176)
 indices=rng.choice(len(Y),size=min(args.n,len(Y)),replace=False)
 lib,h=init_lib(full=True)
 stats={'attempts':0,'accepted':0,'emitted':0,'rejected':0,'exact_word_top5_hits':0}
 try:
  for j in indices:
   raw=(' '.join(names[int(i)] if i>=0 else 'zzzzquuxwordunknown' for i in X[j])+' ').encode()
   assert lib.wf176_hybrid_begin(h,ctypes.c_char_p(raw),len(raw))==0
   stats['exact_word_top5_hits']+=int(names[int(Y[j])] in [w for _,w,_ in top5(lib,h)])
   out=ctypes.create_string_buffer(48);written=ctypes.c_int();accepted=ctypes.c_int();rejected=ctypes.c_int()
   assert lib.wf176_hybrid_sample(h,176+int(j),16,out,48,ctypes.byref(written),ctypes.byref(accepted),ctypes.byref(rejected))==0
   stats['attempts']+=1;stats['accepted']+=accepted.value;stats['emitted']+=written.value;stats['rejected']+=rejected.value
 finally:lib.wf176_hybrid_close(h)
 report=dict(status='PASS',**stats,mean_accepted_bytes=stats['accepted']/stats['attempts'],mean_emitted_bytes=stats['emitted']/stats['attempts'],no_parallel_verifier_speedup_claim=True)
 (ROOT/'results/v176_speculative_native_pilot.json').write_text(json.dumps(report,indent=2))
 print(report,flush=True)
if __name__=='__main__':main()