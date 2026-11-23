#!/usr/bin/env python3
"""Native v176 top-five beam / speculative exact-verifier probe. The source byte
oracle remains v170; the proposal is trained signed/letter/BPE word top-five."""
import ctypes, json, os, sys, time, argparse
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'parent/scripts')]
from train_gate163 import setup
from test_native_speller156 import MODEL,F

def init_lib(full=True,escape=.35):
    setup(core=False,domain=True,gate=None)
    os.environ.update(V164_DISTRIBUTION_BUDGET='.6',V164_DISTRIBUTION_INDEX=str(ROOT/'checkpoints/v164_native_distribution.tsv'),V164_DOMAIN_GATE='1',V164_DOMAIN_THRESHOLD='1.5',V166_BIRTH_MODEL=str(ROOT/'checkpoints/v166_lexical_birth.bin'),V166_LEXICAL_BIRTH='.05',V167_GATE_MODE='1',V167_DOMAIN_MIN='1.4',V167_HIGH_ALPHA='.5',V167_TRUST_THRESHOLD='.5',V168_ANN_INDEX=str(ROOT/'checkpoints/v168_signed_ann.bin'),V168_ANN_SIGNED_SHARE='.66',V168_ANN_DOMAIN_MIN='1.4',V168_ANN_STRENGTH='.05',V169_NEURAL_RERANK='.05')
    lib=ctypes.CDLL(str(ROOT/'bin/libwordfield176.so'))
    lib.wf176_hybrid_open_full.argtypes=[ctypes.c_char_p]*5+[ctypes.c_float,ctypes.c_char_p,ctypes.c_char_p]
    lib.wf176_hybrid_open_full.restype=ctypes.c_void_p
    lib.wf176_hybrid_open.argtypes=[ctypes.c_char_p]*5+[ctypes.c_float]
    lib.wf176_hybrid_open.restype=ctypes.c_void_p
    lib.wf176_hybrid_close.argtypes=[ctypes.c_void_p]
    lib.wf176_hybrid_begin.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_size_t];lib.wf176_hybrid_begin.restype=ctypes.c_int
    lib.wf176_hybrid_next.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_size_t,ctypes.POINTER(F),ctypes.POINTER(F)];lib.wf176_hybrid_next.restype=ctypes.c_int
    lib.wf176_hybrid_top5.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.POINTER(ctypes.c_int),ctypes.c_char_p,ctypes.c_size_t,ctypes.POINTER(F)];lib.wf176_hybrid_top5.restype=ctypes.c_int
    lib.wf176_hybrid_verify_top5.argtypes=[ctypes.c_void_p,ctypes.POINTER(F),ctypes.POINTER(ctypes.c_int)];lib.wf176_hybrid_verify_top5.restype=ctypes.c_int
    lib.wf176_hybrid_sample.argtypes=[ctypes.c_void_p,ctypes.c_uint64,ctypes.c_int,ctypes.c_void_p,ctypes.c_size_t,ctypes.POINTER(ctypes.c_int),ctypes.POINTER(ctypes.c_int),ctypes.POINTER(ctypes.c_int)];lib.wf176_hybrid_sample.restype=ctypes.c_int
    args=(str(ROOT/'checkpoints/v175_native_episodic_48.bin').encode(),*MODEL,str(ROOT/'checkpoints/v169_native_byte_lattice.bin').encode())
    if full:
        h=lib.wf176_hybrid_open_full(*args,F(escape),str(ROOT/'checkpoints/v176_native_crosslevel.bin').encode(),str(ROOT/'checkpoints/v176_native_bpefirst.bin').encode())
    else:h=lib.wf176_hybrid_open(*args,F(escape))
    assert h,'native v176 failed to open inherited sidecars'
    return lib,h

def prefix(lib,h,data):
    q=(F*256)();p=(F*256)()
    rc=lib.wf176_hybrid_next(h,ctypes.c_char_p(data),len(data),q,p)
    assert rc==0,('native proposal',rc,len(data))
    a=np.asarray(list(q),dtype='float64');b=np.asarray(list(p),dtype='float64')
    assert np.isfinite(a).all() and np.isfinite(b).all() and np.all(a>=0) and np.all(b>=0)
    assert abs(a.sum()-1)<1e-5 and abs(b.sum()-1)<1e-5
    return a,b

def top5(lib,h):
    words=[]
    for k in range(5):
        index=ctypes.c_int();word=ctypes.create_string_buffer(64);prob=F()
        assert lib.wf176_hybrid_top5(h,k,ctypes.byref(index),word,64,ctypes.byref(prob))==0
        words.append((index.value,word.value.decode('ascii'),float(prob.value)))
    return words

def main():
    p=argparse.ArgumentParser();p.add_argument('--n',type=int,default=40);a=p.parse_args()
    sys.path[:0]=[str(ROOT/'experiments/nancm/v175')]
    from train_long175 import load_models,load_split
    import torch;torch.set_num_threads(2)
    _,_,names,_,_=load_models();X,Y,B=load_split('val',ROOT/'results/v175_release3')
    lib,h=init_lib(full=True)
    rng=np.random.default_rng(176);idx=rng.choice(len(Y),size=min(a.n,len(Y)),replace=False)
    cachecalls=0;accepted=0;emitted=0;matched=0;top5_hits=0;start=time.monotonic()
    try:
        for j in idx:
            raw=(' '.join(names[int(w)] if w>=0 else 'zzzzquuxwordunknown' for w in X[j])+' ').encode('ascii')
            assert lib.wf176_hybrid_begin(h,ctypes.c_char_p(raw),len(raw))==0
            ws=top5(lib,h);top5_hits+=any(w==names[int(Y[j])] for _,w,_ in ws)
            q,p=prefix(lib,h,b'');scores=(F*5)();calls=ctypes.c_int()
            assert lib.wf176_hybrid_verify_top5(h,scores,ctypes.byref(calls))==0
            cachecalls+=calls.value
            out=ctypes.create_string_buffer(40);n=ctypes.c_int();ac=ctypes.c_int();rej=ctypes.c_int()
            assert lib.wf176_hybrid_sample(h,176+int(j),12,out,40,ctypes.byref(n),ctypes.byref(ac),ctypes.byref(rej))==0
            emitted+=n.value;accepted+=ac.value
    finally:lib.wf176_hybrid_close(h)
    res=dict(status='PASS',contexts=len(idx),word_top5_hits=top5_hits,accepted_draft_bytes=accepted,
       mean_accepted_bytes=accepted/len(idx),mean_emitted_bytes=emitted/len(idx),byte_oracle_calls_for_rank=cachecalls,
       verifier_runs_sequentially=True,speculation_speedup_claim=False,source='reconstructed word histories, 55 validation books',seconds=time.monotonic()-start)
    (ROOT/'results/v176_native_speculative.json').write_text(json.dumps(res,indent=2))
    print('V176 NATIVE SPEC',res,flush=True)
if __name__=='__main__':main()