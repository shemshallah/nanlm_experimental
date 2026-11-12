#!/usr/bin/env python3
"""FA1 R7.21 single-architecture checkpoint, optimizer and causal-parity audit."""
from __future__ import annotations
import argparse, copy, json, pathlib, tempfile
import numpy as np
import torch
from r710_checkpoint_integrity import sha256, deep_equal
from fa1_r721_unified import (PARENT, FORMAT, UnifiedR7, bootstrap, blank,
                               load_candidate,trainable_memory_only,write_candidate)
from fa1_r720_runtime import FA1R720Runtime


def _cmp_states(a,b):
    assert a.keys()==b.keys(),(a.keys(),b.keys())
    return all(torch.equal(a[k],b[k]) for k in a)


def run(args):
    torch.manual_seed(721);np.random.seed(721)
    parent=pathlib.Path(args.root)/'checkpoint/pal_active.pt'
    before=sha256(parent)
    assert before==PARENT
    original,origopts,base,comp,migration=bootstrap(args.root,args.near,args.far)
    unified,opts,base2,comp2,blob=load_candidate(args.candidate,args.root)
    assert blob['format']==FORMAT and blob['accepted_joint_steps']==14416
    assert blob['candidate_joint_adapter_updates']==0 and not blob['promotion']
    assert len(list(unified.parameters()))==len(list(original.parameters()))
    assert _cmp_states(original.state_dict(),unified.state_dict())
    for i,(a,b) in enumerate(zip(origopts,opts)):
        assert deep_equal(a.state_dict(),b.state_dict()),f'optimizer {i} mismatch'
    positions=np.arange(args.start,args.start+args.positions,dtype=np.int64)
    prev=original.runtime(args.root,base,comp,split='test').score_positions(positions,mode='packed_index',batch_size=args.batch)
    new=unified.runtime(args.root,base2,comp2,split='test').score_positions(positions,mode='packed_index',batch_size=args.batch)
    errs={k:max(abs(x-y) for x,y in zip(prev[k],new[k])) for k in ('baseline_nll','near_nll','far_nll')}
    assert all(e<2e-6 for e in errs.values()),errs
    # 1-step differentiable joint near/far update on causal-shaped synthetic
    # features. Verifies the same fifth optimizer and checkpoint can resume.
    t=copy.deepcopy(unified.core.state_dict())
    opt0=copy.deepcopy(opts[4].state_dict())
    trainable_memory_only(unified)
    n=16
    g=torch.Generator().manual_seed(721)
    nf=torch.randn(n,60,generator=g)*.3
    ff=torch.randn(n,32,generator=g)*.3
    npb=torch.rand(n,8,generator=g)*.2
    fpb=torch.rand(n,4,generator=g)*.2
    nc=torch.ones(n,8)
    fc=torch.ones(n,4)
    data={'base':torch.full((n,),4.6),'near_features':nf,'near_prob':npb,
          'near_count':nc,'far_features':ff,'far_prob':fpb,'far_count':fc}
    opts[4].zero_grad(set_to_none=True)
    before_loss=unified.joint_nll(data)[0].mean()
    before_loss.backward()
    neargrad=sum(float(p.grad.abs().sum()) for p in unified.near.parameters() if p.grad is not None)
    fargrad=sum(float(p.grad.abs().sum()) for p in unified.far.parameters() if p.grad is not None)
    assert neargrad>0 and fargrad>0
    opts[4].step()
    assert _cmp_states(t,unified.core.state_dict()),'core modified despite frozen mode'
    newopt=opts[4].state_dict()
    assert deep_equal(opt0['param_groups'],newopt['param_groups'])
    assert all(deep_equal(opt0['state'][key],newopt['state'][key]) for key in opt0['state'] if key<25)
    near_changed=any(not torch.equal(blob['model']['near.'+k],v) for k,v in unified.near.state_dict().items())
    far_changed=any(not torch.equal(blob['model']['far.'+k],v) for k,v in unified.far.state_dict().items())
    assert near_changed and far_changed
    with tempfile.TemporaryDirectory() as td:
        dest=pathlib.Path(td)/'resumed_unified.pt'
        write_candidate(dest,unified,opts,args.root,args.near,args.far,candidate_updates=1)
        resumed,ropts,_,_,meta=load_candidate(dest,args.root)
        assert _cmp_states(unified.state_dict(),resumed.state_dict())
        assert all(deep_equal(o.state_dict(),p.state_dict()) for o,p in zip(opts,ropts))
        assert meta['candidate_joint_adapter_updates']==1
    assert sha256(parent)==before
    return {'round':'FA1 R7.21','result':'PASS','accepted_step':14416,
            'model_parameter_tensors':len(unified.state_dict()),
            'optimizer_histories':len(opts),
            'near_and_far_moments_merged_into_fifth':True,
            'core_bitwise_frozen_one_joint_step':True,
            'original_optimizer_groups_and_moments_preserved':True,
            'single_optimizer_updates_both_near_and_far':True,
            'save_and_resume_candidate_parity':True,
            'real_historical_positions':len(positions),
            'real_checkpoint_parity_max_abs_nll':errs,
            'gradient_l1':{'near':neargrad,'far':fargrad},
            'accepted_sha256_before_after':before,
            'new_book_validation':False,'promoted':False}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='/mnt/data/fa1_full');p.add_argument('--near',default='/mnt/data/fa1r720/adaptation/fa1_r716_225.pt');p.add_argument('--far',default='/mnt/data/fa1r720/adaptation/fa1_r718_full_candidate.pt');p.add_argument('--candidate',default='/mnt/data/fa1r721/checkpoint/r7_unified_candidate.pt');p.add_argument('--start',type=int,default=134000);p.add_argument('--positions',type=int,default=64);p.add_argument('--batch',type=int,default=16);p.add_argument('--output');a=p.parse_args()
    r=run(a)
    if a.output:pathlib.Path(a.output).write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps(r,indent=2))