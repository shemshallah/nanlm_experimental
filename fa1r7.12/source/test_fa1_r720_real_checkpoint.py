#!/usr/bin/env python3
"""R7.20 full frozen checkpoint parity: original, rescan, packed index.

Historical test positions are not a fresh book-level generalization split.
No training, optimizer updates, or checkpoint promotion.
"""
from __future__ import annotations
import argparse,json,pathlib,time
from unittest.mock import patch
import numpy as np
import torch
import fa1_r718_far_residual as ref
from fa1_r720_runtime import restore_frozen,FA1R720Runtime
from r710_checkpoint_integrity import sha256
from fa1_r716_router import PARENT


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',default='/mnt/data/fa1_full')
    ap.add_argument('--near',default='/mnt/data/fa1r719/adaptation/fa1_r716_225.pt')
    ap.add_argument('--far',default='/mnt/data/fa1r719/adaptation/fa1_r718_full_candidate.pt')
    ap.add_argument('--split',default='test')
    ap.add_argument('--start',type=int,default=134000)
    ap.add_argument('--positions',type=int,default=1024)
    ap.add_argument('--batch',type=int,default=16)
    ap.add_argument('--output',default=None)
    ap.add_argument('--reverse',action='store_true',help='run packed first to control warm-cache order')
    a=ap.parse_args();t0=time.monotonic();root=pathlib.Path(a.root)
    model,opts,base,comp,near,far=restore_frozen(root,a.near,a.far)
    if len(opts)!=5:raise AssertionError('optimizer histories missing')
    positions=np.arange(a.start,a.start+a.positions,dtype=np.int64)
    rtref=FA1R720Runtime(root,model,base,comp,near,far,split=a.split)
    rtfast=FA1R720Runtime(root,model,base,comp,near,far,split=a.split)
    def timed(runtime,mode):
        t=time.monotonic()
        result=runtime.score_positions(positions,mode=mode,batch_size=a.batch)
        return result,time.monotonic()-t
    if a.reverse:
        p,packed_s=timed(rtfast,'packed_index')
        r,original_s=timed(rtref,'exact_reference')
    else:
        r,original_s=timed(rtref,'exact_reference')
        p,packed_s=timed(rtfast,'packed_index')
    err={k:max(abs(x-y) for x,y in zip(r[k],p[k])) for k in ('baseline_nll','near_nll','far_nll')}
    assert all(v<2e-6 for v in err.values()),err
    sample_pos=positions[:min(4,len(positions))]
    with patch.object(ref,'pick_positions',lambda corpus,n,seed,min_start=ref.HORIZON:sample_pos):
        old=ref.gather(root,model,base,comp,near,a.split,len(sample_pos),720,batch=len(sample_pos))
    near=np.asarray(r['near_nll'][:len(sample_pos)])
    farold,_=ref.loss_far(far,old)
    external={'baseline':float(np.max(np.abs(old['base'].numpy()-r['baseline_nll'][:len(sample_pos)]))),
              'near':float(np.max(np.abs(old['near'].numpy()-near))),
              'far':float(np.max(np.abs(farold.numpy()-np.asarray(r['far_nll'][:len(sample_pos)]))))}
    assert all(v<2e-6 for v in external.values()),external
    assert sha256(root/'checkpoint/pal_active.pt')==PARENT
    report={'round':'FA1 R7.20 real checkpoint and real LM runtime','accepted_step':14416,
        'five_inherited_optimizer_histories':len(opts), 'parent_sha256':PARENT,
        'test_type':'historical contiguous corpus (NOT new books)',
        'execution_order':'packed_first' if a.reverse else 'reference_first',
        'split':a.split,'start_position':a.start,'positions':len(positions),'batch_size':a.batch,
        'mean_original_baseline_nll':float(np.mean(r['baseline_nll'])),
        'mean_frozen_near_nll':float(np.mean(r['near_nll'])),
        'mean_r718_far_nll':float(np.mean(r['far_nll'])),
        'mean_r720_packed_far_nll':float(np.mean(p['far_nll'])),
        'maximum_abs_parity_error':err,'old_gather_crosscheck_4_positions':external,
        'whole_runtime_seconds':{'r718_reference':original_s,'r720_packed':packed_s},
        'model_and_retrieval_profile_seconds':{'r718_reference':r['profile_s'],'r720_packed':p['profile_s']},
        'whole_runtime_speedup_including_index_build':original_s/packed_s,
        'far_phase_speedup_including_index_build':r['profile_s']['far']/p['profile_s']['far'],
        'packed_index_resets':p['resets'],'packed_incremental_steps':p['steps'],
        'total_elapsed_seconds':time.monotonic()-t0,
        'checkpoint_modified':False,'new_book_validation':False,'promoted':False}
    if a.output:pathlib.Path(a.output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2));return report
if __name__=='__main__':main()