#!/usr/bin/env python3
"""R7.58 TRAIN-only 1k CPU continuation of the real R7.57 31-group checkpoint."""
from __future__ import annotations
import argparse, copy, json, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from fa1_r723_train import prepare, forward, take
from fa1_r727_checkpoint import load_unified, save_unpromoted, digest
from fa1_r727_optimizer import _deep_equal
from fa1_r744_cpu_pilot import bootstrap_by_book
from fa1_r748_cpu_pilot import verify_train_dev_only
from fa1_r751_cpu_pilot import balanced_batch_indices
from fa1_r753_corpus import balanced_causal_positions, with_successor_history
from fa1_r754_cpu_pilot import frozen_fingerprint
from fa1_r758_contextual_geometry import (
    ROUND, build_context_banks, attach_contextual_geometry, contextual_control)

PARENT='FA1_R7_57_TRAIN_ONLY_SIGNED_PAIR_ROUTER_V1'

@torch.no_grad()
def evaluate(model, data, seed):
    model.eval()
    outputs={}
    for mode in ('disabled','full','exact_matched','geometry_matched',
                 'wrong_older','mirror_swapped','wrong_geo_values','no_apex'):
        with contextual_control(model,mode):
            outputs[mode],info=forward(model,data)
            if mode=='full': full_info=info
    f=outputs['full']
    measures={
        'gain_over_r757':outputs['disabled']-f,
        'geometry_over_exact_bigram_matched':outputs['exact_matched']-f,
        'correct_vs_wrong_older':outputs['wrong_older']-f,
        'correct_vs_mirror':outputs['mirror_swapped']-f,
        'correct_vs_wrong_geo_values':outputs['wrong_geo_values']-f,
        'apex_reliance':outputs['no_apex']-f,
        'full_over_geometry_matched':outputs['geometry_matched']-f}
    boots={n:bootstrap_by_book(v,data['book_ids'],seed+i)
           for i,(n,v) in enumerate(measures.items())}
    required=('gain_over_r757','geometry_over_exact_bigram_matched',
              'correct_vs_wrong_older','correct_vs_wrong_geo_values')
    passed=all(float(measures[n].mean())>0 and
               boots[n]['book_bootstrap_ci95'][0]>0 for n in required)
    return dict(books=int(data['book_ids'].unique().numel()),positions=len(f),
                parent_nll=float(outputs['disabled'].mean()),
                candidate_nll=float(f.mean()),
                exact_bigram_matched_nll=float(outputs['exact_matched'].mean()),
                mean_exact_gain=float(full_info['r758_exact_gain'].mean()),
                mean_geometric_gain=float(full_info['r758_geometry_gain'].mean()),
                strict_geometry_gate_passed=bool(passed),book_bootstrap=boots,
                **{k:float(v.mean()) for k,v in measures.items()})

def run(a):
    started=time.monotonic()
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    if any(Path(x).exists() for x in (a.output_checkpoint,a.report)):
        raise FileExistsError('outputs must be new and immutable')
    if Path(a.checkpoint).resolve()==Path(a.output_checkpoint).resolve():
        raise ValueError('cannot overwrite R7.57 parent')
    inventory=verify_train_dev_only(a.corpus_dir,a.split_metadata)
    parent_sha=digest(a.checkpoint)
    model,opt,base,comp,meta=load_unified(a.checkpoint,a.root,a.threads)
    if (meta.get('architecture_round')!=PARENT or meta.get('promoted') is not False
            or len(opt.param_groups)!=31):
        raise ValueError('requires complete unpromoted R7.57 model and 31 optimizers')
    args=SimpleNamespace(root=a.root,corpus_dir=a.corpus_dir,batch=8,
                         r753_position_selector=balanced_causal_positions)
    train=with_successor_history(
        prepare(args,model,base,comp,'train',a.train,a.seed+101),
        a.root,'train',a.corpus_dir,a.context)
    dev=with_successor_history(
        prepare(args,model,base,comp,'dev',a.dev,a.seed+202),
        a.root,'dev',a.corpus_dir,a.context)
    if (int(train['book_ids'].unique().numel())<200 or
            int(dev['book_ids'].unique().numel())!=15):
        raise AssertionError('book-disjoint train/dev coverage absent')
    inherited=dict(model.named_parameters())
    original_fingerprint=frozen_fingerprint(inherited,opt)
    with torch.no_grad(): original,_=forward(model,dev)
    for p in inherited.values():p.requires_grad_(False)
    banks=build_context_banks(a.root,a.corpus_dir,model.word_backoff.word_head_lookup)
    attach_contextual_geometry(model,opt,banks,learning_rate=a.lr)
    with torch.no_grad(): attached,_=forward(model,dev)
    if not torch.equal(original,attached):
        raise AssertionError('zero-gain R7.58 attach broke exact parent parity')
    if frozen_fingerprint(inherited,opt)!=original_fingerprint:
        raise AssertionError('R7.57 weights/moments changed during attachment')
    new={n:p for n,p in model.named_parameters() if n not in inherited}
    if len(new)!=3 or len(opt.param_groups)!=32:
        raise AssertionError('expected exactly one new 32nd optimizer group')
    with torch.no_grad(): model.word_backoff.r758_context.strength.fill_(a.open_strength)
    # Remove supervised TRAIN events from BOTH inherited and new banks.
    train=dict(train,r757_leave_one_out=True,r758_leave_one_out=True)
    rng=np.random.default_rng(a.seed+303)
    records=[];best=None;selected=None;weights=None;moments=None
    for step in range(1,a.steps+1):
        ids=balanced_batch_indices(train['book_ids'],a.minibatch,rng)
        batch=take(train,ids)
        model.eval()
        nll,_=forward(model,batch)
        opt._adamw.zero_grad(set_to_none=True)
        loss=nll.mean()
        loss.backward()
        if any(p.grad is not None for p in inherited.values()):
            raise AssertionError('inherited weights received gradient')
        torch.nn.utils.clip_grad_norm_(list(new.values()),1.)
        opt._adamw.step()
        if step==1 or step==a.steps or step%a.eval_every==0:
            metric=evaluate(model,dev,a.seed+step)
            metric['step']=step
            records.append(metric)
            ranking=(int(metric['strict_geometry_gate_passed']),-metric['candidate_nll'])
            if best is None or ranking>best:
                best,selected=ranking,metric
                weights={n:p.detach().clone() for n,p in new.items()}
                moments={n:copy.deepcopy(opt._adamw.state.get(p,{})) for n,p in new.items()}
            print('R758_DEV',json.dumps({'step':step,'parent':metric['parent_nll'],
                'candidate':metric['candidate_nll'],
                'geometry_over_bigram':metric['geometry_over_exact_bigram_matched'],
                'gate':metric['strict_geometry_gate_passed']}),flush=True)
    with torch.no_grad():
        for n,p in new.items():p.copy_(weights[n])
    for n,p in new.items():opt._adamw.state[p]=moments[n]
    final=evaluate(model,dev,a.seed+9000)
    if (frozen_fingerprint(inherited,opt)!=original_fingerprint or
            digest(a.checkpoint)!=parent_sha):
        raise AssertionError('R7.57 parent mutated')
    saved=save_unpromoted(a.output_checkpoint,model,opt,a.root,a.checkpoint,
        source_metadata=meta,candidate_trained=True,training_round=ROUND,
        training_updates=selected['step'],development_gate_passed=False,
        architecture_round=ROUND)
    recovered,reopt,_,_,rmeta=load_unified(a.output_checkpoint,a.root,a.threads)
    names=dict(recovered.named_parameters())
    if (rmeta['architecture_round']!=ROUND or len(reopt.param_groups)!=32 or
        frozen_fingerprint({n:p for n,p in names.items() if n in inherited},reopt)
        !=original_fingerprint):
        raise AssertionError('inherited checkpoint history failed strict reload')
    for n,p in new.items():
        if not torch.equal(p,names[n]) or not _deep_equal(
            opt._adamw.state.get(p,{}),reopt._adamw.state.get(names[n],{})):
            raise AssertionError('new AdamW moments failed reload')
    report=dict(round=ROUND,promoted=False,independent_test_opened=False,
                candidate_sha256=saved['sha256'],parent_sha256=parent_sha,
                inherited_groups_preserved=31,unified_optimizer_groups=32,
                checkpoint_reload_exact=True,trained_steps=a.steps,
                selected_step=selected['step'],matched_dev=final,
                dev_history=records,corpus_inventory=inventory,
                train_positions=len(train['target']),dev_positions=len(dev['target']),
                strict_geometry_gate_passed=final['strict_geometry_gate_passed'],
                eligible_for_more_training=bool(final['strict_geometry_gate_passed']),
                true_byte_bpb_available=False,full_word_top5_available=False,
                elapsed_seconds=time.monotonic()-started)
    Path(a.report).write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print('R758_SAVED',saved['sha256'],final['strict_geometry_gate_passed'],flush=True)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for field in ('root','corpus_dir','split_metadata','checkpoint','output_checkpoint','report'):
        p.add_argument('--'+field.replace('_','-'),required=True)
    p.add_argument('--threads',type=int,default=4,choices=(1,2,4))
    p.add_argument('--train',type=int,default=4096)
    p.add_argument('--dev',type=int,default=2048)
    p.add_argument('--context',type=int,default=512)
    p.add_argument('--steps',type=int,default=1000)
    p.add_argument('--minibatch',type=int,default=16)
    p.add_argument('--eval-every',type=int,default=250)
    p.add_argument('--lr',type=float,default=.003)
    p.add_argument('--open-strength',type=float,default=.5)
    p.add_argument('--seed',type=int,default=758)
    run(p.parse_args())
