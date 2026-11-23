#!/usr/bin/env python3
"""Test v177 exact verifier memoization vs its inherited v176 P and Q.
All checkpoints and native source are immutable; evaluation is on 55 known val books.
"""
import argparse,ctypes,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'experiments/nancm/v176'),str(ROOT/'experiments/nancm/v175')]
from bench_native176 import init_lib,top5
from train_long175 import load_models,load_split
import torch
F=ctypes.c_float;I=ctypes.c_int;U=ctypes.c_uint32

def init():
    inherited,old=init_lib(full=True)
    lib=ctypes.CDLL(str(ROOT/'bin/libwordfield177.so'))
    lib.wf177_open_full.argtypes=[ctypes.c_char_p]*5+[F,ctypes.c_char_p,ctypes.c_char_p]
    lib.wf177_open_full.restype=ctypes.c_void_p
    lib.wf177_begin.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_size_t];lib.wf177_begin.restype=I
    lib.wf177_next.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_size_t,ctypes.POINTER(F),ctypes.POINTER(F)];lib.wf177_next.restype=I
    lib.wf177_verify_batch.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_size_t,ctypes.POINTER(U),ctypes.POINTER(U),ctypes.c_size_t,ctypes.POINTER(F),ctypes.POINTER(F)];lib.wf177_verify_batch.restype=I
    lib.wf177_prepare_top5.argtypes=[ctypes.c_void_p,ctypes.POINTER(F),ctypes.POINTER(I),ctypes.POINTER(I)];lib.wf177_prepare_top5.restype=I
    lib.wf177_stats.argtypes=[ctypes.c_void_p,ctypes.POINTER(I),ctypes.POINTER(I),ctypes.POINTER(I)];lib.wf177_stats.restype=I
    lib.wf177_top5.argtypes=inherited.wf176_hybrid_top5.argtypes;lib.wf177_top5.restype=I
    lib.wf177_sample.argtypes=inherited.wf176_hybrid_sample.argtypes;lib.wf177_sample.restype=I
    lib.wf177_close.argtypes=[ctypes.c_void_p]
    args=(str(ROOT/'checkpoints/v175_native_episodic_48.bin').encode(),
          *(__import__('test_native_speller156').MODEL),
          str(ROOT/'checkpoints/v169_native_byte_lattice.bin').encode(),F(.35),
          str(ROOT/'checkpoints/v176_native_crosslevel.bin').encode(),
          str(ROOT/'checkpoints/v176_native_bpefirst.bin').encode())
    h=lib.wf177_open_full(*args)
    assert h,'v177 full handle failed to load current trained sidecars'
    return inherited,old,lib,h

def sample(lib,handle,fn,seed):
    out=ctypes.create_string_buffer(40);n=I();a=I();r=I()
    assert fn(handle,seed,12,out,40,ctypes.byref(n),ctypes.byref(a),ctypes.byref(r))==0
    return out.raw[:n.value],a.value,r.value

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--n',type=int,default=30);args=ap.parse_args()
    torch.set_num_threads(2)
    _,_,names,_,_=load_models();X,Y,B=load_split('val',ROOT/'results/v175_release3')
    rng=np.random.default_rng(177);ids=rng.choice(len(Y),size=min(len(Y),args.n),replace=False)
    old,oldh,lib,h=init()
    counts=dict(status='PASS',contexts=0,full_vector_exact=0,batch_prefixes=0,sampler_seed_matches=0,
                sample_accepted_bytes=0,oracle_misses=0,raw_top5_calls=0,top5_cached_savings=0,
                second_prepare_calls=0,cache_reset_checks=0,full_word_top5_identical=0)
    try:
        for j in ids:
            raw=(' '.join(names[int(x)] if x>=0 else 'zzzzquuxwordunknown' for x in X[j])+' ').encode()
            assert old.wf176_hybrid_begin(oldh,ctypes.c_char_p(raw),len(raw))==0
            assert lib.wf177_begin(h,ctypes.c_char_p(raw),len(raw))==0
            cands=top5(old,oldh)
            for k,(_,word,prob) in enumerate(cands):
                ix=I();w=ctypes.create_string_buffer(64);p=F()
                assert lib.wf177_top5(h,k,ctypes.byref(ix),w,64,ctypes.byref(p))==0
                assert (ix.value,w.value.decode(),p.value)==cands[k]
                counts['full_word_top5_identical']+=1
            # Include trie-shared prefixes, off-trie bytes, and deliberate duplicates.
            prefixes=[b'',b't',b'th',b'the',b'xyz',b'']
            for _,word,_ in cands:
                for length in range(len(word)+1):prefixes.append(word.encode()[:length])
            data=b''.join(prefixes)
            offsets=[];at=0
            for pref in prefixes:offsets.append(at);at+=len(pref)
            n=len(prefixes);pos=(U*n)(*offsets);sizes=(U*n)(*[len(s) for s in prefixes]);Q=(F*(n*256))();P=(F*(n*256))()
            assert lib.wf177_verify_batch(h,ctypes.c_char_p(data),len(data),pos,sizes,n,Q,P)==0
            for z,prefix in enumerate(prefixes):
                q=(F*256)();p=(F*256)()
                assert old.wf176_hybrid_next(oldh,ctypes.c_char_p(prefix),len(prefix),q,p)==0
                assert list(Q[z*256:(z+1)*256])==list(q)
                assert list(P[z*256:(z+1)*256])==list(p)
                counts['full_vector_exact']+=1
            scores=(F*5)();calls=I();unique=I()
            assert lib.wf177_prepare_top5(h,scores,ctypes.byref(calls),ctypes.byref(unique))==0
            native=(F*5)();old_calls=I()
            assert old.wf176_hybrid_verify_top5(oldh,native,ctypes.byref(old_calls))==0
            assert calls.value>=unique.value and calls.value==sum(len(w)+1 for _,w,_ in cands)
            assert calls.value>=old_calls.value
            assert max(abs(a-b) for a,b in zip(scores,native))<1e-7
            counts['raw_top5_calls']+=calls.value;counts['top5_cached_savings']+=calls.value-unique.value
            before_miss=I();hits=I();nodes=I();lib.wf177_stats(h,ctypes.byref(before_miss),ctypes.byref(hits),ctypes.byref(nodes))
            assert lib.wf177_prepare_top5(h,scores,ctypes.byref(calls),ctypes.byref(unique))==0
            assert unique.value==0
            counts['second_prepare_calls']+=unique.value
            for seed in range(10):
                sd=177000+int(j)*10+seed
                a=sample(lib,h,lib.wf177_sample,sd)
                b=sample(old,oldh,old.wf176_hybrid_sample,sd)
                assert a==b,(j,seed,a,b)
                counts['sampler_seed_matches']+=1;counts['sample_accepted_bytes']+=a[1]
            misses=I();hits=I();nodes=I();lib.wf177_stats(h,ctypes.byref(misses),ctypes.byref(hits),ctypes.byref(nodes))
            assert nodes.value<=256 and misses.value>=before_miss.value
            counts['oracle_misses']+=misses.value
            assert lib.wf177_begin(h,ctypes.c_char_p(raw),len(raw))==0
            lib.wf177_stats(h,ctypes.byref(misses),ctypes.byref(hits),ctypes.byref(nodes))
            assert (misses.value,hits.value,nodes.value)==(0,0,0)
            counts['cache_reset_checks']+=1
            counts['contexts']+=1;counts['batch_prefixes']+=len(prefixes)
    finally:
        old.wf176_hybrid_close(oldh);lib.wf177_close(h)
    out=ROOT/'results/v177_exact_dag_parity.json';out.write_text(json.dumps(counts,indent=2))
    print('V177 EXACT DAG NATIVE',counts,flush=True)
if __name__=='__main__':main()