#!/usr/bin/env python3
"""FA1 R7.19 parity, causality, document boundaries and CPU benchmark."""
from __future__ import annotations
import json, math, time
import numpy as np
import torch
from fa1_r719_streaming_index import StreamingFarIndex, far_reference, HORIZON, EXCLUSION
from fa1_r718_far_residual import FarResidual, make_feature, loss_far


def compare(idx, sequence, context, target, document_start=0, atol=0.00006):
    old = far_reference(sequence, idx.position, context, target, document_start)
    new = idx.query(context, target)
    for order, (a,b) in enumerate(zip(old,new),1):
        for key in ('n', 'modal'):
            assert a[key] == b[key], f'{key} order={order}, time={idx.position}: {a[key]} != {b[key]}'
        for key in ('weight_total','peak','entropy','recent','target_probability'):
            # R7.18 sums exponentials rounded to float32, R7.19 uses
            # float64 running sums; differences at ~few ppm are expected.
            assert math.isclose(a[key], b[key], abs_tol=atol, rel_tol=atol), (key, order, idx.position, a[key], b[key])
    return old,new


def run():
    torch.set_num_threads(2)
    tests=[];summary={};rng=np.random.default_rng(719)
    q = np.array([9,8,7,6],dtype=np.int64)
    # Three adversarial regimes: high-frequency all-one key, periodic
    # collision with several target values, and Zipfian long-tail words.
    for kind in ('constant', 'periodic', 'zipf'):
        length=HORIZON+300
        if kind=='constant':
            seq=np.full(length,6,dtype=np.int64)
        elif kind=='periodic':
            seq=np.tile(np.array([5,6,7,8,9,7,6,9,5,8,7,6,4,5,7,6]),math.ceil(length/16))[:length]
        else:
            seq=np.minimum(1000,np.maximum(2,rng.zipf(1.45,size=length))).astype(np.int64)
        t=HORIZON+2
        init=time.perf_counter();idx=StreamingFarIndex(seq,t);build=time.perf_counter()-init
        maxdiff=0.0
        for delta in (0,1,2,3,4,5,19,31,63):
            idx.advance(t+delta)
            ctx=seq[t+delta-4:t+delta].copy()
            tar=int(seq[t+delta]);old,new=compare(idx,seq,ctx,tar)
            for a,b in zip(old,new):
                maxdiff=max(maxdiff,abs(a['weight_total']-b['weight_total']),abs(a['target_probability']-b['target_probability']))
        tests.append(f'{kind}_exact_ngram_modes_and_float_parity')
        # No target was observed by the index: changing the test target
        # changes its probability only, never route features.
        r1=idx.query(seq[idx.position-4:idx.position],target=2)
        r2=idx.query(seq[idx.position-4:idx.position],target=999)
        assert all(all(a[k]==b[k] for k in ('n','weight_total','peak','entropy','recent','modal')) for a,b in zip(r1,r2))
        tests.append(f'{kind}_target_independent_features')
        summary[kind]={'build_s':round(build,3),'max_absolute_diff':maxdiff,'size':idx.stats()}
    # Advance vs rebuilding at a boundary where expired n-grams differ
    # from 4-gram transitions, and backfill prefixes are invalid.
    seq=np.ones(HORIZON+400,dtype=np.int64)
    seq[10:15]=[9,8,7,6,55]
    seq[HORIZON-EXCLUSION-20:HORIZON-EXCLUSION-15]=[9,8,7,6,56]
    t=HORIZON+10
    idx=StreamingFarIndex(seq,t)
    for step in range(50):
        idx.advance(t+step)
        compare(idx,seq,np.array([9,8,7,6]),55)
    tests.append('time_expiry_per_order_and_exact_32k_boundaries')
    # Source document boundary: prevent cross-book matches and long
    # history label contamination in the first 128K positions of a book.
    seq=np.full(180000,9,dtype=np.int64)
    seq[:90000]=6;doc=90000;t=doc+32774
    idx=StreamingFarIndex(seq,t,document_start=doc)
    compare(idx,seq,[9,9,9,9],9,document_start=doc)
    idx.advance(t+10);compare(idx,seq,[9,9,9,9],9,document_start=doc)
    tests.append('document_boundary_and_BOS_padding_parity')
    # Rebuild = same output as incremental at each step.
    for delta in (0,1,7,15):
        a=StreamingFarIndex(seq,t+delta,document_start=doc)
        b=StreamingFarIndex(seq,t,document_start=doc).advance(t+delta)
        for old,new in zip(a.query([9]*4,9),b.query([9]*4,9)):
            assert old['n']==new['n'] and old['modal']==new['modal']
            assert math.isclose(old['target_probability'],new['target_probability'],abs_tol=1e-9)
    tests.append('build_equals_incremental')
    # Model-facing features, unmodified frozen R7.18 gate, and resulting
    # NLL must remain numerically close, with no target access in gates.
    rng=np.random.default_rng(421)
    seq=np.minimum(60,np.maximum(2,rng.zipf(1.2,size=HORIZON+100))).astype(np.int64)
    idx=StreamingFarIndex(seq,HORIZON+3)
    near=torch.tensor([5.7],dtype=torch.float32)
    net=FarResidual();net.eval()
    for p in range(HORIZON+3,HORIZON+10):
        idx.advance(p);ctx=seq[p-4:p];target=int(seq[p]);old,new=compare(idx,seq,ctx,target)
        pr=np.array([.06,.03,.07,.03])
        fea=torch.tensor([make_feature(old,pr,ctx,far=True)],dtype=torch.float32)
        feb=torch.tensor([make_feature(new,pr,ctx,far=True)],dtype=torch.float32)
        da={'near':near,'far_features':fea,'far_count':torch.tensor([[x['n'] for x in old]],dtype=torch.float32),'far_prob':torch.tensor([[x['target_probability'] for x in old]],dtype=torch.float32)}
        db={'near':near,'far_features':feb,'far_count':torch.tensor([[x['n'] for x in new]],dtype=torch.float32),'far_prob':torch.tensor([[x['target_probability'] for x in new]],dtype=torch.float32)}
        la,ga=loss_far(net,da);lb,gb=loss_far(net,db)
        assert torch.allclose(la,lb,atol=1e-6,rtol=1e-6), (la,lb)
        assert torch.allclose(ga,gb,atol=1e-6,rtol=1e-6)
    tests.append('r718_gate_NLL_and_probability_parity')
    # CPU throughput: compare index update+query vs unmodified R7.18
    # on the SAME sequential positions; report both timings, exclude build.
    seq=np.minimum(250,np.maximum(2,rng.zipf(1.24,size=HORIZON+1000))).astype(np.int64)
    t=HORIZON+3
    idx=StreamingFarIndex(seq,t)
    start=time.perf_counter()
    for p in range(t,t+100):
        idx.advance(p);idx.query(seq[p-4:p],int(seq[p]))
    indexed=(time.perf_counter()-start)/100
    start=time.perf_counter()
    for p in range(t,t+16):
        far_reference(seq,p,seq[p-4:p],int(seq[p]))
    reference=(time.perf_counter()-start)/16
    assert reference>0 and indexed>0
    summary['benchmark']={'index_update_plus_query_ms':indexed*1000,'reference_rescan_query_ms':reference*1000,
                          'speedup_excluding_initial_build':reference/indexed,'reference_queries':16,'indexed_queries':100,
                          'index_size':idx.stats(),'note':'Synthetic Zipf distribution, not full LM throughput'}
    tests.append('CPU_benchmark_recorded_on_identical_causal_stream')
    out={'round':'FA1 R7.19','tests_passed':len(tests),'tests_total':len(tests),'tests':tests,'details':summary,
         'promoted':False,'base_parent_step':14416,'independent_books_evaluated':False}
    print(json.dumps(out,indent=2))
    return out

if __name__ == '__main__':
    run()