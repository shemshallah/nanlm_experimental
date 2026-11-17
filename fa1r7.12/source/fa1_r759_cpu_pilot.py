#!/usr/bin/env python3
"""R7.59 CPU continuation: conditioned geometric Bayes prior vs lexical match."""
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
from fa1_r759_geometric_prior import (
    ROUND,build_conditioned_bank,attach_geometric_prior,prior_control)

PARENT='FA1_R7_58_CAUSAL_COMPOSITIONAL_MIRROR_V1'

@torch.no_grad()
def evaluate(model,data,seed):
    model.eval()
    scores={}
    for mode in ('disabled','full','lexical_matched','wrong_older',
                 'wrong_geometry_values','no_apex'):
        with prior_control(model,mode):
            scores[mode],info=forward(model,data)
            if mode=='full':true_info=info
    full=scores['full']
    tracks={
        'gain_over_r758':scores['disabled']-full,
        'geometry_over_lexical_matched':scores['lexical_matched']-full,
        'correct_vs_wrong_older':scores['wrong_older']-full,
        'correct_vs_wrong_geometry_values':scores['wrong_geometry_values']-full,
        'apex_reliance':scores['no_apex']-full}
    boots={k:bootstrap_by_book(v,data['book_ids'],seed+i)
           for i,(k,v) in enumerate(tracks.items())}
    mandatory=('gain_over_r758','geometry_over_lexical_matched',
               'correct_vs_wrong_older','correct_vs_wrong_geometry_values')
    gate=all(float(tracks[k].mean())>0 and
             boots[k]['book_bootstrap_ci95'][0]>0 for k in mandatory)
    return dict(books=int(data['book_ids'].unique().numel()),positions=len(full),
        parent_nll=float(scores['disabled'].mean()),candidate_nll=float(full.mean()),
        matched_lexical_nll=float(scores['lexical_matched'].mean()),
        mean_geometric_gain=float(true_info['r759_geometry_gain'].mean()),
        geometric_pseudocount=float(true_info['r759_pseudocount']),
        strict_geometry_gate_passed=bool(gate),book_bootstrap=boots,
        **{k:float(v.mean()) for k,v in tracks.items()})

def run(a):
    start=time.monotonic()
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    if any(Path(p).exists() for p in (a.output_checkpoint,a.report)):
        raise FileExistsError('candidate output must be new')
    if Path(a.checkpoint).resolve()==Path(a.output_checkpoint).resolve():
        raise ValueError('cannot overwrite R7.58 parent')
    inventory=verify_train_dev_only(a.corpus_dir,a.split_metadata)
    parent_hash=digest(a.checkpoint)
    model,opt,base,comp,meta=load_unified(a.checkpoint,a.root,a.threads)
    if (meta.get('architecture_round')!=PARENT or meta.get('promoted') is not False
            or len(opt.param_groups)!=32):
        raise ValueError('requires intact unpromoted R7.58 with 32 groups')
    x=SimpleNamespace(root=a.root,corpus_dir=a.corpus_dir,batch=8,
                      r753_position_selector=balanced_causal_positions)
    train=with_successor_history(
        prepare(x,model,base,comp,'train',a.train,a.seed+101),
        a.root,'train',a.corpus_dir,a.context)
    dev=with_successor_history(
        prepare(x,model,base,comp,'dev',a.dev,a.seed+202),
        a.root,'dev',a.corpus_dir,a.context)
    if (int(train['book_ids'].unique().numel())<200 or
            int(dev['book_ids'].unique().numel())!=15):
        raise AssertionError('invalid TRAIN/DEV book coverage')
    inherited=dict(model.named_parameters())
    inherited_hash=frozen_fingerprint(inherited,opt)
    with torch.no_grad():baseline,_=forward(model,dev)
    for p in inherited.values():p.requires_grad_(False)
    banks=build_conditioned_bank(a.root,a.corpus_dir,
                                 model.word_backoff.word_head_lookup)
    attach_geometric_prior(model,opt,banks,learning_rate=a.lr)
    with torch.no_grad():recovered,_=forward(model,dev)
    if not torch.equal(baseline,recovered):
        raise AssertionError('R7.59 must attach with exact zero-gain parity')
    if frozen_fingerprint(inherited,opt)!=inherited_hash:
        raise AssertionError('R7.58 inherited weights/moments changed')
    new={n:p for n,p in model.named_parameters() if n not in inherited}
    if len(new)!=4 or len(opt.param_groups)!=33:
        raise AssertionError('R7.59 needs 4 new parameters, 33 groups total')
    with torch.no_grad():model.word_backoff.r759_prior.gain.fill_(a.open_gain)
    train=dict(train,r757_leave_one_out=True,r758_leave_one_out=True,
               r759_leave_one_out=True)
    rng=np.random.default_rng(a.seed+303)
    records=[];best=None;chosen=None;weights=None;moments=None
    for step in range(1,a.steps+1):
        batch=take(train,balanced_batch_indices(train['book_ids'],a.minibatch,rng))
        model.eval()
        loss=forward(model,batch)[0].mean()
        opt._adamw.zero_grad(set_to_none=True)
        loss.backward()
        if any(p.grad is not None for p in inherited.values()):
            raise AssertionError('inherited FA1 parameter got gradient')
        torch.nn.utils.clip_grad_norm_(list(new.values()),1.)
        opt._adamw.step()
        if step==1 or step==a.steps or step%a.eval_every==0:
            m=evaluate(model,dev,a.seed+step)
            m['step']=step
            records.append(m)
            rank=(int(m['strict_geometry_gate_passed']),-m['candidate_nll'])
            if best is None or rank>best:
                best,chosen=rank,m
                weights={n:p.detach().clone() for n,p in new.items()}
                moments={n:copy.deepcopy(opt._adamw.state.get(p,{}))
                         for n,p in new.items()}
            print('R759_DEV',json.dumps({'step':step,'parent':m['parent_nll'],
                'candidate':m['candidate_nll'],
                'geo_over_lexical':m['geometry_over_lexical_matched'],
                'gate':m['strict_geometry_gate_passed']}),flush=True)
    with torch.no_grad():
        for n,p in new.items():p.copy_(weights[n])
    for n,p in new.items():opt._adamw.state[p]=moments[n]
    selected=evaluate(model,dev,a.seed+9000)
    if (frozen_fingerprint(inherited,opt)!=inherited_hash or
            digest(a.checkpoint)!=parent_hash):
        raise AssertionError('protected R7.58 lineage mutated')
    saved=save_unpromoted(a.output_checkpoint,model,opt,a.root,a.checkpoint,
        source_metadata=meta,candidate_trained=True,training_round=ROUND,
        training_updates=chosen['step'],development_gate_passed=False,
        architecture_round=ROUND)
    recovered,reopt,_,_,rmeta=load_unified(a.output_checkpoint,a.root,a.threads)
    parameters=dict(recovered.named_parameters())
    if (rmeta['architecture_round']!=ROUND or len(reopt.param_groups)!=33 or
        frozen_fingerprint({n:p for n,p in parameters.items() if n in inherited},
                           reopt)!=inherited_hash):
        raise AssertionError('R7.59 inherited checkpoint reload mismatch')
    for n,p in new.items():
        if not torch.equal(p,parameters[n]) or not _deep_equal(
            opt._adamw.state.get(p,{}),reopt._adamw.state.get(parameters[n],{})):
            raise AssertionError('R7.59 new AdamW state lost on reload')
    out=dict(round=ROUND,promoted=False,independent_test_opened=False,
        parent_sha256=parent_hash,candidate_sha256=saved['sha256'],
        one_optimizer_groups=33,inherited_groups_preserved=32,
        full_checkpoint_reload_exact=True,train_positions=len(train['target']),
        dev_positions=len(dev['target']),selected_step=chosen['step'],
        steps_executed=a.steps,matched_dev=selected,dev_history=records,
        corpus_inventory=inventory,strict_geometry_gate_passed=selected['strict_geometry_gate_passed'],
        eligible_for_9000=bool(selected['strict_geometry_gate_passed']),
        raw_byte_bpb_available=False,full_word_top5_available=False,
        elapsed_seconds=time.monotonic()-start)
    Path(a.report).write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    print('R759_SAVED',saved['sha256'],selected['strict_geometry_gate_passed'],flush=True)
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
    p.add_argument('--open-gain',type=float,default=.5)
    p.add_argument('--seed',type=int,default=759)
    run(p.parse_args())
