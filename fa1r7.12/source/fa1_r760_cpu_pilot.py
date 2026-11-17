#!/usr/bin/env python3
"""FA1 R7.60: checkpoint-backed, book-disjoint support calibration pilot."""
from __future__ import annotations
import argparse,copy,json,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from fa1_r723_train import prepare,forward,take
from fa1_r727_checkpoint import load_unified,save_unpromoted,digest
from fa1_r727_optimizer import _deep_equal
from fa1_r744_cpu_pilot import bootstrap_by_book
from fa1_r748_cpu_pilot import verify_train_dev_only
from fa1_r751_cpu_pilot import balanced_batch_indices
from fa1_r753_corpus import balanced_causal_positions,with_successor_history
from fa1_r754_cpu_pilot import frozen_fingerprint
from fa1_r759_cpu_pilot import evaluate as evaluate_r759
from fa1_r760_geometric_calibration import (
    ROUND, attach_geometric_calibration,calibration_control)

PARENT='FA1_R7_59_WORD_CONDITIONED_GEOMETRIC_PRIOR_V1'

@torch.no_grad()
def evaluate(model,data,seed):
    measured=evaluate_r759(model,data,seed)
    with calibration_control(model,False):
        nll,_=forward(model,data)
    with calibration_control(model,True):
        adjusted,_=forward(model,data)
    advantage=nll-adjusted
    book=bootstrap_by_book(advantage,data['book_ids'],seed+99)
    measured['uncalibrated_r759_nll']=float(nll.mean())
    measured['calibration_advantage']=float(advantage.mean())
    measured['calibration_bootstrap']=book
    measured['learned_shrinkage']=float(model.word_backoff.r759_prior.r760_shrinkage)
    measured['strict_geometry_gate_passed']=bool(
        measured['strict_geometry_gate_passed'] and advantage.mean()>0
        and book['book_bootstrap_ci95'][0]>0)
    return measured

def run(a):
    start=time.monotonic()
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed);np.random.seed(a.seed)
    if any(Path(p).exists() for p in (a.output_checkpoint,a.report)):
        raise FileExistsError('new R7.60 output paths required')
    inventory=verify_train_dev_only(a.corpus_dir,a.split_metadata)
    parent_hash=digest(a.checkpoint)
    model,opt,base,comp,meta=load_unified(a.checkpoint,a.root,a.threads)
    if (meta.get('architecture_round')!=PARENT or meta.get('promoted') is not False
            or len(opt.param_groups)!=33):
        raise ValueError('R7.60 requires complete unpromoted R7.59/33-group parent')
    args=SimpleNamespace(root=a.root,corpus_dir=a.corpus_dir,batch=8,
        r753_position_selector=balanced_causal_positions)
    train=with_successor_history(prepare(args,model,base,comp,'train',a.train,a.seed+101),
        a.root,'train',a.corpus_dir,a.context)
    dev=with_successor_history(prepare(args,model,base,comp,'dev',a.dev,a.seed+202),
        a.root,'dev',a.corpus_dir,a.context)
    if int(train['book_ids'].unique().numel())<200 or int(dev['book_ids'].unique().numel())!=15:
        raise AssertionError('TRAIN/DEV book separation broken')
    inherited=dict(model.named_parameters())
    initial=frozen_fingerprint(inherited,opt)
    with torch.no_grad():baseline,_=forward(model,dev)
    for p in inherited.values():p.requires_grad_(False)
    attach_geometric_calibration(model,opt,learning_rate=a.lr)
    with torch.no_grad():restored,_=forward(model,dev)
    if not torch.equal(baseline,restored):
        raise AssertionError('R7.60 attachment must exactly preserve R7.59')
    if frozen_fingerprint(inherited,opt)!=initial:
        raise AssertionError('inherited weights/moments changed on attachment')
    new={n:p for n,p in model.named_parameters() if n not in inherited}
    if len(new)!=3 or len(opt.param_groups)!=34:
        raise AssertionError('R7.60 must add 3 parameters, 1 optimizer group')
    train=dict(train,r757_leave_one_out=True,r758_leave_one_out=True,
               r759_leave_one_out=True)
    rng=np.random.default_rng(a.seed+303)
    history=[];best=None;selected=None;weights=None;moments=None
    for step in range(1,a.steps+1):
        batch=take(train,balanced_batch_indices(train['book_ids'],a.minibatch,rng))
        model.eval()
        loss=forward(model,batch)[0].mean()
        opt._adamw.zero_grad(set_to_none=True)
        loss.backward()
        if any(p.grad is not None for p in inherited.values()):
            raise AssertionError('R7.59 inherited parameter acquired gradient')
        torch.nn.utils.clip_grad_norm_(list(new.values()),1.)
        opt._adamw.step()
        if step==1 or step==a.steps or step%a.eval_every==0:
            m=evaluate(model,dev,a.seed+step);m['step']=step;history.append(m)
            rank=(int(m['strict_geometry_gate_passed']),-m['candidate_nll'])
            if best is None or rank>best:
                best,selected=rank,m
                weights={n:p.detach().clone() for n,p in new.items()}
                moments={n:copy.deepcopy(opt._adamw.state.get(p,{})) for n,p in new.items()}
            print('R760_DEV',json.dumps({'step':step,'candidate':m['candidate_nll'],
                'geo_over_lex':m['geometry_over_lexical_matched'],
                'calibration':m['calibration_advantage'],
                'gate':m['strict_geometry_gate_passed']}),flush=True)
    with torch.no_grad():
        for n,p in new.items():p.copy_(weights[n])
    for n,p in new.items():opt._adamw.state[p]=moments[n]
    final=evaluate(model,dev,a.seed+9000)
    if frozen_fingerprint(inherited,opt)!=initial or digest(a.checkpoint)!=parent_hash:
        raise AssertionError('R7.59 parent or optimizer was mutated')
    saved=save_unpromoted(a.output_checkpoint,model,opt,a.root,a.checkpoint,
        source_metadata=meta,candidate_trained=True,training_round=ROUND,
        training_updates=selected['step'],development_gate_passed=False,
        architecture_round=ROUND)
    reload_model,reload_opt,_,_,reload_meta=load_unified(
        a.output_checkpoint,a.root,a.threads)
    loaded=dict(reload_model.named_parameters())
    if (reload_meta['architecture_round']!=ROUND or len(reload_opt.param_groups)!=34
        or frozen_fingerprint({n:p for n,p in loaded.items() if n in inherited},
            reload_opt)!=initial):
        raise AssertionError('R7.60 inherited checkpoint reload mismatch')
    for n,p in new.items():
        if not torch.equal(p,loaded[n]) or not _deep_equal(
                opt._adamw.state.get(p,{}),reload_opt._adamw.state.get(loaded[n],{})):
            raise AssertionError('R7.60 new weights or optimizer moments lost')
    out=dict(round=ROUND,promoted=False,independent_test_opened=False,
        parent_sha256=parent_hash,candidate_sha256=saved['sha256'],
        one_optimizer_groups=34,inherited_groups_preserved=33,
        full_checkpoint_reload_exact=True,train_positions=len(train['target']),
        dev_positions=len(dev['target']),selected_step=selected['step'],
        steps_executed=a.steps,matched_dev=final,dev_history=history,
        corpus_inventory=inventory,strict_geometry_gate_passed=final['strict_geometry_gate_passed'],
        eligible_for_9000=bool(final['strict_geometry_gate_passed']),
        raw_byte_bpb_available=False,full_word_top5_available=False,
        elapsed_seconds=time.monotonic()-start)
    Path(a.report).write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    print('R760_SAVED',saved['sha256'],final['strict_geometry_gate_passed'],flush=True)
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for field in ('root','corpus_dir','split_metadata','checkpoint',
                  'output_checkpoint','report'):
        p.add_argument('--'+field.replace('_','-'),required=True)
    p.add_argument('--threads',type=int,default=4,choices=(1,2,4))
    p.add_argument('--train',type=int,default=4096)
    p.add_argument('--dev',type=int,default=2048)
    p.add_argument('--context',type=int,default=512)
    p.add_argument('--steps',type=int,default=1000)
    p.add_argument('--minibatch',type=int,default=16)
    p.add_argument('--eval-every',type=int,default=250)
    p.add_argument('--lr',type=float,default=.002)
    p.add_argument('--seed',type=int,default=760)
    run(p.parse_args())
