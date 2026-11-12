#!/usr/bin/env python3
"""R7.20 packed 128K index: causality, expiry, doc boundaries, score parity."""
from __future__ import annotations
import json,math,time
import numpy as np
from fa1_r720_packed_index import PackedFarIndex
from fa1_r719_streaming_index import HORIZON, EXCLUSION, far_reference


def compare(idx,seq,p,q,tar,doc=0):
    idx.advance(p)
    old=far_reference(seq,p,q,tar,document_start=doc)
    new=idx.query(q,tar)
    for order,(a,b) in enumerate(zip(old,new),1):
        assert a['n']==b['n'], ('n',p,order,a['n'],b['n'])
        assert a['modal']==b['modal'],('modal',p,order,a['modal'],b['modal'])
        for key in ('weight_total','peak','entropy','recent','target_probability'):
            assert math.isclose(a[key],b[key],abs_tol=8e-5,rel_tol=8e-5),(key,p,order,a[key],b[key])
    return new


def run():
    rng=np.random.default_rng(720)
    tests=[];metrics={}
    for kind in ('constant','periodic','zipf'):
        N=HORIZON+250
        if kind=='constant':seq=np.full(N,6,dtype='int32')
        elif kind=='periodic':seq=np.resize(np.array([9,8,7,6,5,4,3,2],dtype='int32'),N)
        else:seq=np.minimum(32000,np.maximum(2,rng.zipf(1.45,size=N))).astype('int32')
        p=HORIZON
        t=time.perf_counter();idx=PackedFarIndex(seq,p,merge_stride=32)
        build=time.perf_counter()-t
        for delta in (0,1,2,4,6,13,31,32,33,65,127,233):
            now=p+delta
            compare(idx,seq,now,seq[now-4:now],int(seq[now]))
        assert idx.rebuilds>=7,idx.stats()
        tests.append(kind+'_collision_free_expiry_and_merge_rebuild')
        q=seq[now-4:now]
        a=idx.query(q,target=3);b=idx.query(q,target=32000)
        for left,right in zip(a,b):
            for key in ('n','weight_total','modal','peak','entropy','recent'):
                assert left[key]==right[key],(kind,key)
        tests.append(kind+'_target_not_a_route_feature')
        metrics[kind]={'build_s':build,'index':idx.stats()}
    seq=np.full(HORIZON+128,9,dtype='int32')
    seq[:90000]=6;doc=90000;pos=doc+EXCLUSION+9
    idx=PackedFarIndex(seq,pos,document_start=doc,merge_stride=17)
    for delta in (0,1,2,3,16,17,19,34,48,60):
        p=pos+delta;compare(idx,seq,p,[9,9,9,9],9,doc=doc)
    tests.append('document_boundary_no_cross_book_history')
    seq=np.full(HORIZON+145,2,dtype='int32')
    seq[3:7]=[9,8,7,6];seq[7]=11
    seq[HORIZON-EXCLUSION-12:HORIZON-EXCLUSION-7]=[9,8,7,6,12]
    idx=PackedFarIndex(seq,HORIZON,merge_stride=21)
    for now in range(HORIZON,HORIZON+110):
        if now%11==0:compare(idx,seq,now,[9,8,7,6],11)
        else:idx.advance(now)
    tests.append('age_128k_and_32k_exclusion_exact')
    bad=np.ones(HORIZON+4,dtype=np.int32)*2;bad[100]=40000
    try:PackedFarIndex(bad,HORIZON)
    except ValueError:pass
    else:raise AssertionError('out-of-range key must fail closed')
    tests.append('packed_15bit_capacity_fail_closed')
    N=HORIZON+1000
    seq=np.minimum(32000,np.maximum(2,rng.zipf(1.6,size=N))).astype('int32')
    p=HORIZON;t=time.perf_counter();idx=PackedFarIndex(seq,p)
    init_s=time.perf_counter()-t
    t=time.perf_counter()
    for k in range(700):
        now=p+k;idx.advance(now);idx.query(seq[now-4:now],int(seq[now]))
    packed_s=time.perf_counter()-t
    t=time.perf_counter()
    for k in range(700):
        now=p+k;far_reference(seq,now,seq[now-4:now],int(seq[now]))
    ref_s=time.perf_counter()-t
    metrics['benchmark']={'positions':700,'build_s':init_s,'packed_update_query_s':packed_s,
        'reference_rescan_s':ref_s,'speedup_excluding_build':ref_s/packed_s,
        'speedup_including_build':ref_s/(init_s+packed_s),'memory':idx.stats(),
        'note':'CPU retrieval microbenchmark; not full-model inference and not independent-book loss'}
    tests.append('CPU_microbenchmark_observed')
    out={'round':'FA1 R7.20 packed index','passed':len(tests),'total':len(tests),
        'tests':tests,'details':metrics,'accepted_step':14416,'weights_changed':False,
        'new_book_validation':False}
    print(json.dumps(out,indent=2));return out
if __name__=='__main__':run()