#!/usr/bin/env python3
"""R68: extend the SAME R67 FA2 model and optimizer through 1,024 updates.

Preserve the historical Adam group ordering, inherited 196 model moment sets,
and R66 joint moment set. TRAIN only; the independent TEST remains sealed.
"""
import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from poincare_frontier import Config,MirrorLM,file_hash,load_v25_compatible
from train_r55_pair_successor import checked_corpus
from train_r66_causal_book import cache_causal_book,train_joint_only


def restore_joint_optimizer(model,saved):
    """Restore R54 -> R58 -> R66 appended migration ordering exactly.

    Natural model registration places successor parameters near the output
    head, but the historical Adam group appended these AFTER every inherited
    backbone parameter at their separate migration steps.
    """
    joint=model.pair_successor.joint_weight
    module=model.pair_successor
    if joint is None or module.weight is None or module.content_weight is None:
        raise ValueError('Missing inherited R54/R58/R66 successor parameters')
    ordered=[p for name,p in model.named_parameters()
             if not name.startswith('pair_successor.')]
    ordered.extend((module.weight,module.content_weight,joint))
    optimizer=torch.optim.AdamW(ordered,lr=.001)
    optimizer.load_state_dict(saved['optimizer'])
    if len(optimizer.param_groups)!=1:
        raise AssertionError('Optimizer group was split')
    for p in ordered:
        state=optimizer.state.get(p,{})
        for name in ('exp_avg','exp_avg_sq','max_exp_avg_sq'):
            if name in state and state[name].shape!=p.shape:
                raise AssertionError('AdamW state shape mismatch: '+name)
    return optimizer


def audit_inherited(saved,model,optimizer):
    for name,old in saved['model'].items():
        if name!='pair_successor.joint_weight':
            if not torch.equal(old,model.state_dict()[name]):
                raise AssertionError('Inherited model tensor changed: '+name)
    old=saved['optimizer']
    new=optimizer.state_dict()
    if (len(old['param_groups'])!=1 or len(new['param_groups'])!=1 or
            old['param_groups'][0]['params']!=new['param_groups'][0]['params'] or
            set(old['state'])!=set(new['state'])):
        raise AssertionError('Optimizer lineage changed')
    final=old['param_groups'][0]['params'][-1]
    for pid,items in old['state'].items():
        if pid==final:
            continue
        for name,prior in items.items():
            now=new['state'][pid][name]
            equal=(torch.equal(prior,now) if torch.is_tensor(prior)
                   else prior==now)
            if not equal:
                raise AssertionError('Inherited Adam moment changed')
    return {'preserved_model_parameters':len(saved['model'])-1,
            'retained_adam_states':len(old['state']),
            'unchanged_inherited_adam_states':len(old['state'])-1,
            'trained_joint_parameter_only':True}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('checkpoint','data-dir','candidate','report'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--expected-parent-sha256',required=True)
    p.add_argument('--train-books',type=int,default=12)
    p.add_argument('--positions',type=int,default=768)
    p.add_argument('--context',type=int,default=64)
    p.add_argument('--chunk',type=int,default=16)
    p.add_argument('--history',type=int,default=32768)
    p.add_argument('--updates',type=int,default=768)
    p.add_argument('--lr',type=float,default=.003)
    p.add_argument('--seed',type=int,default=68)
    p.add_argument('--threads',type=int,default=2)
    args=p.parse_args()
    if args.candidate.exists() or args.report.exists():
        p.error('Refuse to overwrite candidate/report')
    sha=file_hash(args.checkpoint)
    if sha!=args.expected_parent_sha256:
        raise ValueError('Protected R67 parent checksum mismatch')
    torch.set_num_threads(args.threads)
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    if saved.get('step')!=3441 or not saved['config'].get(
            'pair_successor_space_rank_guard'):
        raise ValueError('Only unified R67 step3441 can resume in R68')
    meta,arrays=checked_corpus(args.data_dir,saved)
    model=MirrorLM(Config(**saved['config'])).eval()
    load_v25_compatible(model,saved['model'])
    optimizer=restore_joint_optimizer(model,saved)
    start=perf_counter()
    records=[]
    offset=0
    n=min(args.train_books,len(meta['train_doc_lengths']))
    for idx,length in enumerate(meta['train_doc_lengths']):
        if idx>=n: break
        length=int(length)
        take=min(length,args.positions+1)
        book=torch.from_numpy(np.asarray(arrays['train'][offset:offset+take],
                                         dtype=np.int64).copy())
        records.extend(cache_causal_book(model,book,'TRAIN:'+str(idx),
                       context=args.context,chunk=args.chunk,
                       history=args.history,warmup=0,positions=args.positions))
        offset+=length
    prep=perf_counter()-start
    begin=perf_counter()
    learning=train_joint_only(model,optimizer,records,updates=args.updates,
                              lr=args.lr,seed=args.seed)
    optim_seconds=perf_counter()-begin
    audit=audit_inherited(saved,model,optimizer)
    result=dict(saved)
    result.update(model=model.state_dict(),optimizer=optimizer.state_dict(),
                  step=int(saved['step'])+args.updates,
                  fa2_r68_training=learning)
    args.candidate.parent.mkdir(parents=True,exist_ok=True)
    args.report.parent.mkdir(parents=True,exist_ok=True)
    torch.save(result,args.candidate)
    report={'round':'FA2_R68_MULTIDEPTH_1K','parent_step':saved['step'],
            'candidate_step':result['step'],'parent_sha256':sha,
            'candidate_sha256':file_hash(args.candidate),
            'train_books':n,'positions_per_book':args.positions,
            'train_sha256':meta['train_sha256'],'learning':learning,
            'inheritance':audit,'feature_seconds':prep,
            'optimization_seconds':optim_seconds,'dev_opened':False,
            'test_opened':False,'promoted':False}
    args.report.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
