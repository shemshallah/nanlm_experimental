#!/usr/bin/env python3
"""Matched first-pass and warm-prefix timing, using real inherited sidecars.
Cold: each history fresh, each distinct prefix evaluated once; no cache prefill.
Warm: repeat identical prefixes, preserving context. Source-only research.
"""
import argparse,ctypes,json,sys,time,statistics
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'experiments/nancm/v178'),str(ROOT/'experiments/nancm/v176'),str(ROOT/'experiments/nancm/v175'),str(ROOT/'parent/scripts')]
from test_trained178 import rss,raw_text
from bench_native176 import init_lib,top5
from test_native_speller156 import MODEL
F=ctypes.c_float;I=ctypes.c_int;P=ctypes.c_void_p;S=ctypes.c_size_t

def main():
 p=argparse.ArgumentParser();p.add_argument('--n',type=int,default=24);a=p.parse_args()
 from train_long175 import load_models,load_split
 import torch;torch.set_num_threads(2)
 _,_,names,_,_=load_models();X,Y,B=load_split('val',ROOT/'results/v175_release3')
 rng=np.random.default_rng(1781);idx=[];seen=set()
 for j in rng.permutation(len(Y)):
  if B[j] not in seen:seen.add(int(B[j]));idx.append(int(j))
  if len(idx)>=a.n:break
 old,h=init_lib(full=True);lib=ctypes.CDLL(str(ROOT/'bin/libwordfield178.so'))
 lib.wf178_hub_open_full.argtypes=[ctypes.c_char_p]*5+[F,ctypes.c_char_p,ctypes.c_char_p];lib.wf178_hub_open_full.restype=P
 lib.wf178_hub_close.argtypes=[P];lib.wf178_session_open.argtypes=[P,P,S];lib.wf178_session_open.restype=P
 lib.wf178_session_close.argtypes=[P]
 lib.wf178_session_next.argtypes=[P,P,S,ctypes.POINTER(F),ctypes.POINTER(F)];lib.wf178_session_next.restype=I
 args=(str(ROOT/'checkpoints/v175_native_episodic_48.bin').encode(),*MODEL,str(ROOT/'checkpoints/v169_native_byte_lattice.bin').encode(),F(.35),str(ROOT/'checkpoints/v176_native_crosslevel.bin').encode(),str(ROOT/'checkpoints/v176_native_bpefirst.bin').encode())
 hub=lib.wf178_hub_open_full(*args);assert hub
 total={'old_cold':0.,'new_cold':0.,'old_warm':0.,'new_warm':0.,'old_begin':0.,'new_begin':0.};ctx={'old_cold':[],'new_cold':[],'old_warm':[],'new_warm':[]};count=0
 try:
  for j in idx:
   raw=raw_text(X[j],names)
   t=time.perf_counter();assert old.wf176_hybrid_begin(h,ctypes.c_char_p(raw),len(raw))==0;total['old_begin']+=time.perf_counter()-t
   t=time.perf_counter();s=lib.wf178_session_open(hub,ctypes.c_char_p(raw),len(raw));total['new_begin']+=time.perf_counter()-t
   assert s
   try:
    words=top5(old,h);pr={b'',b'xyz'}
    for _,word,_ in words:
     z=(word+' ').encode()
     for k in range(len(z)):pr.add(z[:k])
    prefixes=sorted(pr)
    for stage in ('cold','warm'):
     old_t=new_t=0.
     for pre in prefixes:
      q=(F*256)();p=(F*256)();rq=(F*256)();rp=(F*256)()
      start=time.perf_counter();assert old.wf176_hybrid_next(h,ctypes.c_char_p(pre),len(pre),rq,rp)==0;old_t+=time.perf_counter()-start
      start=time.perf_counter();assert lib.wf178_session_next(s,ctypes.c_char_p(pre),len(pre),q,p)==0;new_t+=time.perf_counter()-start
      assert bytes(q)==bytes(rq) and bytes(p)==bytes(rp)
      count+=1
     total['old_'+stage]+=old_t;total['new_'+stage]+=new_t
     ctx['old_'+stage].append(old_t);ctx['new_'+stage].append(new_t)
   finally:lib.wf178_session_close(s)
 finally:lib.wf178_hub_close(hub);old.wf176_hybrid_close(h)
 res={'status':'PASS','contexts':len(idx),'exact_256way_vector_pairs':count,'timing_s':total,
      'first_pass_speed_ratio_without_begin':total['old_cold']/total['new_cold'],
      'warm_speed_ratio_without_begin':total['old_warm']/total['new_warm'],
      'first_pass_speed_ratio_with_begin':(total['old_cold']+total['old_begin'])/(total['new_cold']+total['new_begin']),
      'cold_ms_context':{'old':1000*(total['old_cold']+total['old_begin'])/len(idx),'v178':1000*(total['new_cold']+total['new_begin'])/len(idx)},
      'warm_ms_context':{'old':1000*total['old_warm']/len(idx),'v178':1000*total['new_warm']/len(idx)},
      'p95_cold_ms_context':{k:1000*np.percentile(v,95) for k,v in ctx.items() if k.endswith('cold')},
      'caveat':'single-process paired order old then new; independent books from the repeatedly examined validation split; temperature and warm hardware caches not controlled; not end-to-end text throughput or unseen test.'}
 (ROOT/'results/v178_cold_warm_trained.json').write_text(json.dumps(res,indent=2)+'\n')
 print(json.dumps(res,indent=2))
if __name__=='__main__':main()
