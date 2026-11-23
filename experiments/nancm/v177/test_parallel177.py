#!/usr/bin/env python3
"""Native v177 parallel path: one v170 state per worker, no mutable sharing.
Verify every P/Q probability against serial v176, then deterministic sampler.
"""
import argparse,ctypes,json,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'experiments/nancm/v177'),str(ROOT/'experiments/nancm/v176'),str(ROOT/'experiments/nancm/v175')]
from test_native177 import init,sample
from bench_native176 import top5
from train_long175 import load_models,load_split
import torch
F=ctypes.c_float;I=ctypes.c_int

def init_parallel(lib,workers):
 lib.wf177_parallel_open_full.argtypes=[ctypes.c_char_p]*5+[F,ctypes.c_char_p,ctypes.c_char_p,I]
 lib.wf177_parallel_open_full.restype=ctypes.c_void_p
 lib.wf177_parallel_close.argtypes=[ctypes.c_void_p]
 lib.wf177_parallel_begin.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_size_t];lib.wf177_parallel_begin.restype=I
 lib.wf177_parallel_prefill_top5.argtypes=[ctypes.c_void_p,ctypes.POINTER(F),ctypes.POINTER(I),ctypes.POINTER(I)];lib.wf177_parallel_prefill_top5.restype=I
 lib.wf177_parallel_next.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_size_t,ctypes.POINTER(F),ctypes.POINTER(F)];lib.wf177_parallel_next.restype=I
 lib.wf177_parallel_sample.argtypes=lib.wf177_sample.argtypes;lib.wf177_parallel_sample.restype=I
 lib.wf177_parallel_stats.argtypes=[ctypes.c_void_p,ctypes.POINTER(I),ctypes.POINTER(I),ctypes.POINTER(I)];lib.wf177_parallel_stats.restype=I
 from test_native_speller156 import MODEL
 par=lib.wf177_parallel_open_full(str(ROOT/'checkpoints/v175_native_episodic_48.bin').encode(),*MODEL,
         str(ROOT/'checkpoints/v169_native_byte_lattice.bin').encode(),F(.35),
         str(ROOT/'checkpoints/v176_native_crosslevel.bin').encode(),
         str(ROOT/'checkpoints/v176_native_bpefirst.bin').encode(),I(workers))
 assert par,'parallel init failed'
 return par

def main():
 a=argparse.ArgumentParser();a.add_argument('--n',type=int,default=20);a.add_argument('--workers',type=int,default=2);args=a.parse_args()
 torch.set_num_threads(2)
 _,_,names,_,_=load_models();X,Y,B=load_split('val',ROOT/'results/v175_release3')
 rng=np.random.default_rng(1772);idx=rng.choice(len(Y),min(len(Y),args.n),replace=False)
 old,oldh,lib,h=init()
 start=time.perf_counter();parallel=init_parallel(lib,args.workers)
 init_elapsed=time.perf_counter()-start
 counts=dict(status='PASS',contexts=0,workers=args.workers,parallel_init_seconds=init_elapsed,
             full_vector_exact=0,topfive_likelihood_exact=0,identical_samples=0,
             parallel_prefix_calls=0,workers_used_sum=0,byte_model_state_clones=args.workers)
 try:
  for j in idx:
   raw=(' '.join(names[int(w)] if w>=0 else 'zzzzquuxwordunknown' for w in X[j])+' ').encode()
   assert old.wf176_hybrid_begin(oldh,ctypes.c_char_p(raw),len(raw))==0
   assert lib.wf177_parallel_begin(parallel,ctypes.c_char_p(raw),len(raw))==0
   scores=(F*5)();new_calls=I();used=I()
   rc=lib.wf177_parallel_prefill_top5(parallel,scores,ctypes.byref(new_calls),ctypes.byref(used))
   assert rc==0,('parallel prefill',j,rc)
   reference=(F*5)();calls=I()
   assert old.wf176_hybrid_verify_top5(oldh,reference,ctypes.byref(calls))==0
   assert max(abs(x-y) for x,y in zip(scores,reference))<1e-7
   counts['topfive_likelihood_exact']+=1
   assert new_calls.value==calls.value
   counts['parallel_prefix_calls']+=new_calls.value;counts['workers_used_sum']+=used.value
   words=top5(old,oldh)
   for _,word,_ in words:
    for k in range(len(word)+1):
     prefix=word.encode()[:k]
     q=(F*256)();p=(F*256)();p2=(F*256)();q2=(F*256)()
     assert lib.wf177_parallel_next(parallel,ctypes.c_char_p(prefix),len(prefix),q,p)==0
     assert old.wf176_hybrid_next(oldh,ctypes.c_char_p(prefix),len(prefix),q2,p2)==0
     if bytes(q)!=bytes(q2) or bytes(p)!=bytes(p2):
      raise AssertionError((j,prefix,max(abs(x-y) for x,y in zip(p,p2)),max(abs(x-y) for x,y in zip(q,q2))))
     counts['full_vector_exact']+=1
   for seed in range(8):
    s=177200+int(j)*32+seed
    original=sample(old,oldh,old.wf176_hybrid_sample,s)
    parallel_sample=sample(lib,parallel,lib.wf177_parallel_sample,s)
    assert original==parallel_sample,(j,seed,original,parallel_sample)
    counts['identical_samples']+=1
   counts['contexts']+=1
 finally:
  old.wf176_hybrid_close(oldh);lib.wf177_close(h);lib.wf177_parallel_close(parallel)
 (ROOT/'results/v177_parallel_exact_parity.json').write_text(json.dumps(counts,indent=2))
 print('V177 REAL THREAD PARITY',json.dumps(counts,indent=2),flush=True)
if __name__=='__main__':main()