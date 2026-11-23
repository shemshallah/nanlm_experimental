#!/usr/bin/env python3
"""Cold single speculative draft per context: tests actual stream-use case,
without the favorable repeated-query caching assumed in warm lattice benchmark.
"""
import argparse,ctypes,json,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'experiments/nancm/v177'),str(ROOT/'experiments/nancm/v176'),str(ROOT/'experiments/nancm/v175')]
from test_native177 import init,sample
from train_long175 import load_models,load_split
import torch

def main():
    a=argparse.ArgumentParser();a.add_argument('--n',type=int,default=96);args=a.parse_args()
    torch.set_num_threads(2)
    _,_,names,_,_=load_models();X,Y,B=load_split('val',ROOT/'results/v175_release3')
    indices=np.random.default_rng(17771).choice(len(Y),min(args.n,len(Y)),replace=False)
    old,oldh,lib,h=init()
    old_t=0.;new_t=0.;w_old=0.;w_new=0.;matches=0;bytes_emitted=0
    try:
        for j in indices:
            raw=(' '.join(names[int(x)] if x>=0 else 'zzzzquuxwordunknown' for x in X[j])+' ').encode()
            # Alternate AB/BA order over contexts to limit cache-temperature bias.
            if matches%2:
                t=time.perf_counter();assert lib.wf177_begin(h,ctypes.c_char_p(raw),len(raw))==0
                s1=sample(lib,h,lib.wf177_sample,1770000+int(j))
                new_t+=time.perf_counter()-t
                t=time.perf_counter();assert old.wf176_hybrid_begin(oldh,ctypes.c_char_p(raw),len(raw))==0
                s0=sample(old,oldh,old.wf176_hybrid_sample,1770000+int(j))
                old_t+=time.perf_counter()-t
            else:
                t=time.perf_counter();assert old.wf176_hybrid_begin(oldh,ctypes.c_char_p(raw),len(raw))==0
                s0=sample(old,oldh,old.wf176_hybrid_sample,1770000+int(j))
                old_t+=time.perf_counter()-t
                t=time.perf_counter();assert lib.wf177_begin(h,ctypes.c_char_p(raw),len(raw))==0
                s1=sample(lib,h,lib.wf177_sample,1770000+int(j))
                new_t+=time.perf_counter()-t
            assert s0==s1
            matches+=1;bytes_emitted+=len(s0[0])
    finally:
        old.wf176_hybrid_close(oldh);lib.wf177_close(h)
    d=dict(status='PASS',contexts=matches,byte_output=bytes_emitted,exact_matches=matches,
           v176_seconds=old_t,v177_seconds=new_t,
           v177_relative_throughput=old_t/new_t,
           includes_begin=True,cache_first_use_only=True,
           notes='one draft/context, no warm-cache advantage; causal next-word begin counted; 55 known VAL books')
    (ROOT/'results/v177_cold_stream_benchmark.json').write_text(json.dumps(d,indent=2))
    print('V177 COLD STREAM',d,flush=True)
if __name__=='__main__':main()