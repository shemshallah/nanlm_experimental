#!/usr/bin/env python3
"""Fair serial vs independent-thread byte-verifier prefill timing.
No claim of application speed unless amortization after clone loading is included.
"""
import argparse,ctypes,json,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'experiments/nancm/v177'),str(ROOT/'experiments/nancm/v176'),str(ROOT/'experiments/nancm/v175')]
from test_native177 import init
from test_parallel177 import init_parallel
from train_long175 import load_models,load_split
import torch
F=ctypes.c_float;I=ctypes.c_int

def rss():
 for x in Path('/proc/self/status').read_text().splitlines():
  if x.startswith('VmRSS:'):return int(x.split()[1])*1024
 return None

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--n',type=int,default=32);ap.add_argument('--workers',type=int,default=2);a=ap.parse_args()
 torch.set_num_threads(2)
 _,_,names,_,_=load_models();X,Y,B=load_split('val',ROOT/'results/v175_release3')
 rng=np.random.default_rng(1770+a.workers);idx=rng.choice(len(Y),min(a.n,len(Y)),replace=False)
 old,oldh,lib,h=init()
 before=rss();start=time.perf_counter();parallel=init_parallel(lib,a.workers)
 setup_seconds=time.perf_counter()-start;additional_memory=(rss() or 0)-(before or 0)
 serial_seconds=parallel_seconds=0.;calls_s=calls_p=0
 try:
  for z,j in enumerate(idx):
   raw=(' '.join(names[int(w)] if w>=0 else 'zzzzquuxwordunknown' for w in X[j])+' ').encode()
   def serial():
    assert lib.wf177_begin(h,ctypes.c_char_p(raw),len(raw))==0
    scores=(F*5)();a=I();b=I()
    assert lib.wf177_prepare_top5(h,scores,ctypes.byref(a),ctypes.byref(b))==0
    return list(scores),b.value
   def para():
    assert lib.wf177_parallel_begin(parallel,ctypes.c_char_p(raw),len(raw))==0
    scores=(F*5)();b=I();used=I()
    assert lib.wf177_parallel_prefill_top5(parallel,scores,ctypes.byref(b),ctypes.byref(used))==0
    return list(scores),b.value
   if z%2:
    t=time.perf_counter();p=para();parallel_seconds+=time.perf_counter()-t
    t=time.perf_counter();s=serial();serial_seconds+=time.perf_counter()-t
   else:
    t=time.perf_counter();s=serial();serial_seconds+=time.perf_counter()-t
    t=time.perf_counter();p=para();parallel_seconds+=time.perf_counter()-t
   assert s[1]==p[1]
   assert max(abs(a-b) for a,b in zip(s[0],p[0]))<1e-7
   calls_s+=s[1];calls_p+=p[1]
 finally:
  old.wf176_hybrid_close(oldh);lib.wf177_close(h);lib.wf177_parallel_close(parallel)
 n=len(idx);relative=serial_seconds/parallel_seconds
 saved=(serial_seconds-parallel_seconds)/n
 result=dict(status='PASS',contexts=n,workers=a.workers,parallel_open_seconds=setup_seconds,
  parallel_clone_memory_bytes=additional_memory,
  serial_seconds=serial_seconds,parallel_seconds=parallel_seconds,
  serial_ms_context=1000*serial_seconds/n,parallel_ms_context=1000*parallel_seconds/n,
  parallel_relative_speedup=relative,serial_calls=calls_s,parallel_calls=calls_p,
  break_even_contexts=(setup_seconds/saved if saved>0 else None),
  caveat='Includes begin+prefill, excludes cold open in per-context values; break-even includes clone setup. Source-only 55-book reused VAL, 2/4 independent byte states; no default promotion.')
 path=ROOT/f'results/v177_parallel_cpu_{a.workers}workers.json';path.write_text(json.dumps(result,indent=2))
 print('V177 PARALLEL BENCH',json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()