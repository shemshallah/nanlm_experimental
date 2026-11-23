#!/usr/bin/env python3
"""Matched v176 versus v177 CPU cost with identical trained weights/P/Q.
Crucial distinction: warm same-context reuse versus first-pass latency.
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
F=ctypes.c_float;I=ctypes.c_int;U=ctypes.c_uint32

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--n',type=int,default=16);ap.add_argument('--samples',type=int,default=12);args=ap.parse_args()
    torch.set_num_threads(2)
    _,_,names,_,_=load_models();X,Y,B=load_split('val',ROOT/'results/v175_release3')
    rng=np.random.default_rng(1771);ids=rng.choice(len(Y),size=min(len(Y),args.n),replace=False)
    old,oldh,lib,h=init()
    times={x:0. for x in ['old_begin','new_begin','naive_five','old_shared_five','new_shared_five_cold','new_shared_five_warm','new_batch_cached','old_sample','new_sample']}
    counts=dict(status='PASS',contexts=0,draft_samples=0,verified_seed_identical=0,raw_uncached_oracle_calls=0,
                v176_shared_oracle_calls=0,v177_cold_oracle_calls=0,v177_warm_oracle_calls=0,
                v177_batch_cached_oracle_calls=0,v177_total_oracle_calls=0,cached_nodes=0)
    try:
        for j in ids:
            raw=(' '.join(names[int(x)] if x>=0 else 'zzzzquuxwordunknown' for x in X[j])+' ').encode()
            t=time.perf_counter();assert old.wf176_hybrid_begin(oldh,ctypes.c_char_p(raw),len(raw))==0
            times['old_begin']+=time.perf_counter()-t
            t=time.perf_counter();assert lib.wf177_begin(h,ctypes.c_char_p(raw),len(raw))==0
            times['new_begin']+=time.perf_counter()-t
            words=top5(old,oldh)
            # Cost of independent 5 path verification: no prefix sharing.
            paths=[w.encode()+b' ' for _,w,_ in words]
            assert all(len(w)<41 for w in paths)
            t=time.perf_counter();P=(F*256)();Q=(F*256)()
            for word in paths:
                for k in range(len(word)):
                    prefix=word[:k]
                    assert old.wf176_hybrid_next(oldh,ctypes.c_char_p(prefix),len(prefix),Q,P)==0
                    counts['raw_uncached_oracle_calls']+=1
            times['naive_five']+=time.perf_counter()-t
            t=time.perf_counter();scores=(F*5)();shared=I()
            assert old.wf176_hybrid_verify_top5(oldh,scores,ctypes.byref(shared))==0
            times['old_shared_five']+=time.perf_counter()-t
            counts['v176_shared_oracle_calls']+=shared.value
            t=time.perf_counter();newscore=(F*5)();rawcnt=I();newcnt=I()
            assert lib.wf177_prepare_top5(h,newscore,ctypes.byref(rawcnt),ctypes.byref(newcnt))==0
            times['new_shared_five_cold']+=time.perf_counter()-t
            counts['v177_cold_oracle_calls']+=newcnt.value
            assert max(abs(a-b) for a,b in zip(scores,newscore))<1e-7
            t=time.perf_counter();assert lib.wf177_prepare_top5(h,newscore,ctypes.byref(rawcnt),ctypes.byref(newcnt))==0
            times['new_shared_five_warm']+=time.perf_counter()-t
            counts['v177_warm_oracle_calls']+=newcnt.value
            # One-call grouped batch of all proposal-node prefixes, duplicated on purpose.
            prefixes=[]
            for word in paths:
                for k in range(len(word)):
                    prefixes.extend([word[:k],word[:k]])
            buf=b''.join(prefixes);at=0;offsets=[]
            for pref in prefixes:offsets.append(at);at+=len(pref)
            n=len(prefixes);off=(U*n)(*offsets);lens=(U*n)(*[len(x) for x in prefixes]);Bq=(F*(n*256))();Bp=(F*(n*256))()
            prev=I();hits=I();nodes=I();lib.wf177_stats(h,ctypes.byref(prev),ctypes.byref(hits),ctypes.byref(nodes))
            t=time.perf_counter();assert lib.wf177_verify_batch(h,ctypes.c_char_p(buf),len(buf),off,lens,n,Bq,Bp)==0
            times['new_batch_cached']+=time.perf_counter()-t
            after=I();lib.wf177_stats(h,ctypes.byref(after),ctypes.byref(hits),ctypes.byref(nodes))
            counts['v177_batch_cached_oracle_calls']+=after.value-prev.value
            for k in range(len(paths)):
                assert abs(float(newscore[k])-float(scores[k]))<1e-7
            # Samples use the exact *same* random choices and distributions.
            for seed in range(args.samples):
                sd=17701+int(j)*101+seed
                # AB/BA alternation reduces systematic second-in-order cache effects.
                if (counts['contexts']+seed)%2:
                    t=time.perf_counter();alt=sample(lib,h,lib.wf177_sample,sd)
                    times['new_sample']+=time.perf_counter()-t
                    t=time.perf_counter();ref=sample(old,oldh,old.wf176_hybrid_sample,sd)
                    times['old_sample']+=time.perf_counter()-t
                else:
                    t=time.perf_counter();ref=sample(old,oldh,old.wf176_hybrid_sample,sd)
                    times['old_sample']+=time.perf_counter()-t
                    t=time.perf_counter();alt=sample(lib,h,lib.wf177_sample,sd)
                    times['new_sample']+=time.perf_counter()-t
                assert ref==alt,(j,seed,ref,alt)
                counts['verified_seed_identical']+=1
            misses=I();lib.wf177_stats(h,ctypes.byref(misses),ctypes.byref(hits),ctypes.byref(nodes))
            counts['cached_nodes']+=nodes.value;counts['v177_total_oracle_calls']+=misses.value
            counts['draft_samples']+=args.samples;counts['contexts']+=1
    finally:
        old.wf176_hybrid_close(oldh);lib.wf177_close(h)
    means={k:1000*v/max(1,counts['contexts']) for k,v in times.items()}
    sample_mean={x:1000*times[x]/max(1,counts['draft_samples']) for x in ('old_sample','new_sample')}
    out=dict(counts,timings_seconds=times,mean_ms_per_context=means,
             mean_ms_per_draft_sample=sample_mean,
             warm_sample_speedup=times['old_sample']/times['new_sample'] if times['new_sample'] else None,
             top5_unique_prefix_reduction=1-counts['v176_shared_oracle_calls']/max(1,counts['raw_uncached_oracle_calls']),
             notes='Exact P/Q law; measurements include repeated same-context drafts. No real-world token/s guarantee. Verifier is serial; prefix-DAG removes duplicated work and Python boundary overhead.',
             dataset='55 known VAL books, reconstructed 48-word ASCII histories',
             parallel_verification=False)
    path=ROOT/'results/v177_cpu_batch_benchmark.json';path.write_text(json.dumps(out,indent=2))
    print('V177 CPU PREFIX DAG',json.dumps(out,indent=2),flush=True)
if __name__=='__main__':main()