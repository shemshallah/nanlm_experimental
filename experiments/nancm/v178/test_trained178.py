#!/usr/bin/env python3
"""Real-checkpoint cloned-v176 versus shared-v178 causal parity and CPU timing.
No model training or weight edits; book-disjoint *within selected validation set*.
Report exact float-vector agreement and repeated-context cost separately.
"""
import argparse, ctypes, json, sys, time, os
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'experiments/nancm/v176'),str(ROOT/'experiments/nancm/v175'),str(ROOT/'scripts'),str(ROOT/'parent/scripts')]
from bench_native176 import init_lib,top5
from test_native_speller156 import MODEL
F=ctypes.c_float;I=ctypes.c_int;P=ctypes.c_void_p;S=ctypes.c_size_t

def rss():
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith('VmRSS:'):return int(line.split()[1])*1024
    return -1

def raw_text(row,names):
    return (' '.join(names[int(x)] if x>=0 else 'zzzzquuxwordunknown' for x in row)+' ').encode()

def sample(fn,h,seed):
    out=ctypes.create_string_buffer(40);n=I();a=I();r=I()
    rc=fn(h,ctypes.c_uint64(seed),12,out,40,ctypes.byref(n),ctypes.byref(a),ctypes.byref(r))
    assert rc==0,(fn,rc)
    return (out.raw[:n.value],a.value,r.value)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--n',type=int,default=12);a=parser.parse_args()
    from train_long175 import load_models,load_split
    import torch;torch.set_num_threads(2)
    _,_,names,_,_=load_models();X,Y,B=load_split('val',ROOT/'results/v175_release3')
    rng=np.random.default_rng(178)
    # One distinct context for each book until we run out of books.
    idx=[];used=set()
    for j in rng.permutation(len(Y)):
        book=int(B[j]);
        if book not in used:used.add(book);idx.append(int(j))
        if len(idx)>=a.n:break
    if len(idx)<2:raise RuntimeError('not enough independent books')
    baseline,old=init_lib(full=True)
    lib=ctypes.CDLL(str(ROOT/'bin/libwordfield178.so'))
    lib.wf178_hub_open_full.argtypes=[ctypes.c_char_p]*5+[F,ctypes.c_char_p,ctypes.c_char_p];lib.wf178_hub_open_full.restype=P
    lib.wf178_hub_close.argtypes=[P]
    lib.wf178_hub_stats.argtypes=[P,ctypes.POINTER(I),ctypes.POINTER(I)];lib.wf178_hub_stats.restype=I
    lib.wf178_session_open.argtypes=[P,P,S];lib.wf178_session_open.restype=P
    lib.wf178_session_close.argtypes=[P]
    lib.wf178_session_begin.argtypes=[P,P,S];lib.wf178_session_begin.restype=I
    lib.wf178_session_next.argtypes=[P,P,S,ctypes.POINTER(F),ctypes.POINTER(F)];lib.wf178_session_next.restype=I
    lib.wf178_session_top5.argtypes=[P,I,P,ctypes.c_char_p,S,ctypes.POINTER(F)];lib.wf178_session_top5.restype=I
    lib.wf178_session_prepare_top5.argtypes=[P,ctypes.POINTER(F),ctypes.POINTER(I),ctypes.POINTER(I)];lib.wf178_session_prepare_top5.restype=I
    lib.wf178_session_sample.argtypes=baseline.wf176_hybrid_sample.argtypes;lib.wf178_session_sample.restype=I
    lib.wf178_session_stats.argtypes=[P,ctypes.POINTER(I),ctypes.POINTER(I),ctypes.POINTER(I)];lib.wf178_session_stats.restype=I
    args=(str(ROOT/'checkpoints/v175_native_episodic_48.bin').encode(),*MODEL,str(ROOT/'checkpoints/v169_native_byte_lattice.bin').encode(),F(.35),str(ROOT/'checkpoints/v176_native_crosslevel.bin').encode(),str(ROOT/'checkpoints/v176_native_bpefirst.bin').encode())
    before=rss();t=time.perf_counter();hub=lib.wf178_hub_open_full(*args);open_s=time.perf_counter()-t
    assert hub,'failed to load v178 trained hub'
    after=rss();handles=[];count=dict(status='PASS',real_books=len(idx),vectors=0,wholeword_top5=0,seeded_samples=0,exact_byte_vector_differences=0,interleaved_sessions=0,oracle_model_instances=0,model_open_seconds=open_s,model_open_incremental_rss_bytes=after-before,source='NaNCM v177 cumulative recovery ZIP - original pretrained sidecars')
    timings={'begin_v176_seconds':0.,'open_v178_session_seconds':0.,'next_v176_seconds':0.,'next_v178_seconds':0.,'samples_v176_seconds':0.,'samples_v178_seconds':0.,'top5_v176_seconds':0.,'top5_v178_seconds':0.}
    prefixes=[b'',b't',b'th',b'the',b'xyz',b'pro',b'hello']
    try:
        for j in idx:
            raw=raw_text(X[j],names)
            t=time.perf_counter();s=lib.wf178_session_open(hub,ctypes.c_char_p(raw),len(raw));timings['open_v178_session_seconds']+=time.perf_counter()-t
            assert s
            handles.append((s,raw,j))
        models=I();live=I()
        assert lib.wf178_hub_stats(hub,ctypes.byref(models),ctypes.byref(live))==0
        assert models.value==1 and live.value==len(idx)
        count['oracle_model_instances']=models.value;count['sessions_created']=live.value
        count['rss_with_all_sessions_bytes']=rss();count['session_incremental_rss_bytes']=rss()-after
        # Pass twice in deliberately permuted order, retaining all independent sessions.
        for step,(s,raw,j) in enumerate(handles+list(reversed(handles))):
            t=time.perf_counter();assert baseline.wf176_hybrid_begin(old,ctypes.c_char_p(raw),len(raw))==0;timings['begin_v176_seconds']+=time.perf_counter()-t
            parent=top5(baseline,old)
            for k,(ix,word,prob) in enumerate(parent):
                ri=I();rw=ctypes.create_string_buffer(48);rp=F()
                assert lib.wf178_session_top5(s,k,ctypes.byref(ri),rw,48,ctypes.byref(rp))==0
                assert (ri.value,rw.value.decode('ascii'),float(rp.value))==(ix,word,prob)
                count['wholeword_top5']+=1
            chosen=prefixes+[(parent[k][1]+' ').encode()[:n] for k in range(5) for n in range(len(parent[k][1])+2)]
            for pre in chosen:
                q=(F*256)();p=(F*256)();rq=(F*256)();rp=(F*256)()
                t=time.perf_counter();assert baseline.wf176_hybrid_next(old,ctypes.c_char_p(pre),len(pre),rq,rp)==0;timings['next_v176_seconds']+=time.perf_counter()-t
                t=time.perf_counter();rc=lib.wf178_session_next(s,ctypes.c_char_p(pre),len(pre),q,p);timings['next_v178_seconds']+=time.perf_counter()-t
                assert rc==0,(j,pre,rc)
                assert bytes(q)==bytes(rq) and bytes(p)==bytes(rp),(j,pre)
                count['vectors']+=1
            for seed in (100000+int(j),100001+int(j),100002+int(j),100003+int(j)):
                t=time.perf_counter();ref=sample(baseline.wf176_hybrid_sample,old,seed);timings['samples_v176_seconds']+=time.perf_counter()-t
                t=time.perf_counter();got=sample(lib.wf178_session_sample,s,seed);timings['samples_v178_seconds']+=time.perf_counter()-t
                assert got==ref,(j,seed,got,ref)
                count['seeded_samples']+=1
            t=time.perf_counter();rscore=(F*5)();rcalls=I();assert baseline.wf176_hybrid_verify_top5(old,rscore,ctypes.byref(rcalls))==0;timings['top5_v176_seconds']+=time.perf_counter()-t
            t=time.perf_counter();sc=(F*5)();calls=I();uniq=I();assert lib.wf178_session_prepare_top5(s,sc,ctypes.byref(calls),ctypes.byref(uniq))==0;timings['top5_v178_seconds']+=time.perf_counter()-t
            assert max(abs(x-y) for x,y in zip(sc,rscore))<1e-7
            count['interleaved_sessions']+=1
    finally:
        for s,_,_ in handles:lib.wf178_session_close(s)
        lib.wf178_hub_close(hub)
        baseline.wf176_hybrid_close(old)
    count['timings_seconds']=timings
    count['mean_next_ms_v176']=1000*timings['next_v176_seconds']/max(1,count['vectors'])
    count['mean_next_ms_v178']=1000*timings['next_v178_seconds']/max(1,count['vectors'])
    count['note']='Interleaved request histories force hub rebegin on uncached misses; v178 model opens once. Timing not a controlled end-to-end workload speedup; no promotion automatically.'
    out=ROOT/'results/v178_shared_trained_parity.json';out.write_text(json.dumps(count,indent=2)+'\n')
    print(json.dumps(count,indent=2),flush=True)
if __name__=='__main__':main()
