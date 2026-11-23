#!/usr/bin/env python3
"""Model-grounded macro coding: trained v176 top-five, exact native v170 P.
For each 48-word history the top-five six-event code is a *factorization* of P.
"""
import argparse,ctypes,json,math,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'experiments/nancm/v177'),str(ROOT/'experiments/nancm/v176'),str(ROOT/'experiments/nancm/v175')]
from test_native177 import init
from macro_code177 import MacroByteModel
from train_long175 import load_models,load_split
from range_code176 import encode as plain_encode,decode as plain_decode
import torch
F=ctypes.c_float;I=ctypes.c_int

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--n',type=int,default=60);a=ap.parse_args()
    torch.set_num_threads(2)
    _,_,names,_,_=load_models();X,Y,B=load_split('val',ROOT/'results/v175_release3')
    rng=np.random.default_rng(1778);idx=rng.choice(len(Y),size=min(a.n,len(Y)),replace=False)
    old,oldh,lib,h=init()
    d=dict(status='PASS',contexts=0,macro_candidate_hits=0,exact_probability_law=0,
           lossless_macro_roundtrip=0,lossless_baseline_roundtrip=0,
           oracle_prefix_calls=0,plain_coder_bits=0,macro_coder_bits=0,
           base_byte_symbols=0,third_code_macro_symbols=0)
    try:
        for j in idx:
            raw=(' '.join(names[int(x)] if x>=0 else 'zzzzquuxwordunknown' for x in X[j])+' ').encode()
            assert lib.wf177_begin(h,ctypes.c_char_p(raw),len(raw))==0
            wordlist=[]
            for k in range(5):
                ix=I();w=ctypes.create_string_buffer(64);p=F()
                assert lib.wf177_top5(h,k,ctypes.byref(ix),w,64,ctypes.byref(p))==0
                wordlist.append(w.value+b' ')
            assert len(wordlist)==len(set(wordlist))==5
            scores=(F*5)();raw_calls=I();uniq=I()
            assert lib.wf177_prepare_top5(h,scores,ctypes.byref(raw_calls),ctypes.byref(uniq))==0
            def oracle(prefix):
                q=(F*256)();p=(F*256)()
                assert lib.wf177_next(h,ctypes.c_char_p(prefix),len(prefix),q,p)==0
                return p
            mdl=MacroByteModel(wordlist,oracle)
            target=names[int(Y[j])].encode()+b' '
            assert len(target)<=40
            macro,mbits=mdl.encode(target)
            assert mdl.decode(macro,mbits)==target
            plain,bits=plain_encode(target,oracle)
            assert plain_decode(plain,bits,len(target),oracle)==target
            true_p=mdl.oracle_probability(target);macro_p=mdl.macro_probability(target)
            assert math.isclose(true_p,macro_p,rel_tol=2e-8,abs_tol=1e-17),(j,target,true_p,macro_p)
            d['contexts']+=1;d['macro_candidate_hits']+=target in wordlist
            d['exact_probability_law']+=1;d['lossless_macro_roundtrip']+=1
            d['lossless_baseline_roundtrip']+=1;d['plain_coder_bits']+=bits;d['macro_coder_bits']+=mbits
            d['base_byte_symbols']+=len(target)
            d['third_code_macro_symbols']+=1 if target in wordlist else 1+len(target)
            misses=I();hits=I();nodes=I();lib.wf177_stats(h,ctypes.byref(misses),ctypes.byref(hits),ctypes.byref(nodes))
            d['oracle_prefix_calls']+=misses.value
    finally:
        old.wf176_hybrid_close(oldh);lib.wf177_close(h)
    d['top5_candidate_fraction']=d['macro_candidate_hits']/max(1,d['contexts'])
    d['bit_delta_macro_minus_plain']=d['macro_coder_bits']-d['plain_coder_bits']
    d['interpretation']='Exact law and round-trip, not new full-text BPB; precomputation of all 5 P(word) costs oracle calls.'
    (ROOT/'results/v177_real_macro_coding.json').write_text(json.dumps(d,indent=2))
    print('V177 REAL MACRO',json.dumps(d,indent=2),flush=True)
if __name__=='__main__':main()