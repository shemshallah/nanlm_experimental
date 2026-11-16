#!/usr/bin/env python3
"""Checkpoint-backed, DEV-only FA1 R7.44 causal word-backoff pilot.

No accepted checkpoint or sealed TEST prediction is opened. All inherited
parameters and optimizer moments remain in the same model/optimizer; only
the small new backoff group trains. The selected candidate is *unpromoted*.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch

from fa1_r723_train import prepare, take, forward
from fa1_r725_controls import different_book_donors
from fa1_r727_checkpoint import load_unified, save_unpromoted, digest
from fa1_r727_eval import validate_book_corpus, normalized_trunk_word_topk
from fa1_r728_geometric_memory import layered_byte_metrics
from fa1_r744_causal_word_backoff import (ROUND, attach_causal_word_backoff,
                                          causal_word_backoff_disabled)


def bootstrap_by_book(values, book_ids, seed, resamples=1000):
    unique = torch.unique(book_ids)
    means = np.array([float(values[book_ids == book].mean()) for book in unique])
    gen = np.random.default_rng(seed)
    picks = gen.integers(0,len(means),(resamples,len(means)))
    estimates = means[picks].mean(axis=1)
    return {'books':len(means), 'book_average':float(means.mean()),
            'book_bootstrap_ci95':np.quantile(estimates,[.025,.975]).tolist(),
            'resampling_unit':'independent_book'}


def foreign_word_history(data, seed):
    donors = different_book_donors(data['book_ids'], seed)
    wrong = dict(data)
    wrong['diamond_backoff_recent_ids'] = data['word_ids'][donors]
    if 'diamond_backoff_recent_mask' in data:
        wrong['diamond_backoff_recent_mask'] = data['diamond_backoff_recent_mask'][donors]
    wrong['diamond_backoff_modal_ids'] = data['memory_modal_ids'][donors]
    wrong['diamond_backoff_modal_mask'] = data['memory_modal_mask'][donors]
    # Donor coverage statistics are causal and target-free, and should move
    # with the donor history while the inherited R7 features remain fixed.
    for key in ('near_count','far_count'):
        if key in data:
            wrong['diamond_backoff_'+key] = data[key][donors]
    return wrong, donors


@torch.no_grad()
def evaluate(model, data, foreign, seed):
    model.eval()
    candidate, ex = forward(model,data)
    parent = ex['backoff_parent_nll']
    wrong,wrong_ex = forward(model,foreign)
    with causal_word_backoff_disabled(model):
        disabled, _ = forward(model,data)
    if not torch.equal(parent,disabled):
        raise AssertionError('backoff-off must exactly reproduce inherited R7.40')
    no_apex, ap_ex = forward(model,data,'no_apex')
    with causal_word_backoff_disabled(model):
        no_apex_parent, _ = forward(model,data,'no_apex')
    improvement=parent-candidate
    advantage=wrong-candidate
    geometric_specificity=(parent-candidate)-(no_apex_parent-no_apex)
    books=data['book_ids']
    gain_boot=bootstrap_by_book(improvement,books,seed)
    history_boot=bootstrap_by_book(advantage,books,seed+17)
    geo_boot=bootstrap_by_book(geometric_specificity,books,seed+31)
    result={
        'parent_word_nll':float(parent.mean()),
        'candidate_word_nll':float(candidate.mean()),
        'r744_word_backoff_gain':float(improvement.mean()),
        'r744_correct_vs_foreign_word_advantage':float(advantage.mean()),
        'r744_geometry_specificity':float(geometric_specificity.mean()),
        'r744_word_gain_book_bootstrap':gain_boot,
        'r744_history_book_bootstrap':history_boot,
        'r744_geometry_book_bootstrap':geo_boot,
        'r744_mean_mixture_weight':float(ex['backoff_alpha'].mean()),
        'r744_backoff_target_nonzero_fraction':float((ex['backoff_target_probability']>0).float().mean()),
        'r744_target_probability_when_present':float(ex['backoff_target_probability'][
             ex['backoff_target_probability']>0].mean()) if bool((ex['backoff_target_probability']>0).any()) else 0.,
        'r744_route_does_not_observe_target':True,
        'r744_parent_fallback_exact':True,
        'independent_dev_books':int(torch.unique(books).numel()),
        'dev_word_targets':len(parent),
        'raw_contiguous_text_bpb':None,
        'full_mixture_word_top5':None,
        'book_disjoint_from_accepted_R7_ancestor_unknown':True,
    }
    result['r744_candidate_only_dev_gate_passed'] = bool(
        result['r744_word_backoff_gain'] > 1.e-4 and
        result['r744_correct_vs_foreign_word_advantage'] > 0 and
        result['r744_geometry_specificity'] > 0 and
        gain_boot['book_bootstrap_ci95'][0] > 0 and
        history_boot['book_bootstrap_ci95'][0] > 0 and
        geo_boot['book_bootstrap_ci95'][0] > 0)
    return result


def saved_new_optimizer_state(model, optimizer):
    return {name:copy.deepcopy(optimizer._adamw.state.get(param,{}))
            for name,param in model.word_backoff.named_parameters()}


def restore_new_optimizer_state(model, optimizer, saved):
    for name,param in model.word_backoff.named_parameters():
        optimizer._adamw.state[param] = copy.deepcopy(saved[name])


def run(args):
    if args.checkpoint == args.output_checkpoint:
        raise ValueError('source checkpoint must never be overwritten')
    if not (0 <= args.open_gain < .5):
        raise ValueError('candidate opening gate must be in [0, 0.5)')
    for p in (args.report,args.output_checkpoint):
        if Path(p).exists():
            raise FileExistsError('pilot output already exists: '+str(p))
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    start=time.monotonic()
    inventory=validate_book_corpus(args.corpus_dir,'dev')
    source_hash=digest(args.checkpoint)
    model,optimizer,base,comp,meta=load_unified(args.checkpoint,args.root,args.threads)
    if (meta.get('architecture_round')!='FA1_R7_40_NORMALIZED_SIGNED_RAILS_V1'
            or len(optimizer.param_groups)!=24):
        raise ValueError('pilot requires the real intact 24-group R7.40 checkpoint')
    print('R744 RESTORED 24 parent groups',source_hash,flush=True)
    train=prepare(args,model,base,comp,'train',args.train,args.seed+101)
    dev=prepare(args,model,base,comp,'dev',args.dev,args.seed+202)
    if int(torch.unique(dev['book_ids']).numel())<12:
        raise ValueError('development sample must span at least 12 independent books')
    wrong,donors=foreign_word_history(dev,args.seed+303)
    model.eval()
    with torch.no_grad():
        previous,_=forward(model,dev)
    inherited_params = list(model.parameters())
    for p in inherited_params:
        p.requires_grad_(False)
    attach_causal_word_backoff(model,optimizer,learning_rate=args.lr)
    with torch.no_grad():
        identical,_=forward(model,dev)
    if not torch.equal(previous,identical):
        raise AssertionError('R7.44 closed-gate migration must preserve parent loss exactly')
    print('R744 PARITY exact; 25 groups; DEV books',len(torch.unique(dev['book_ids'])),flush=True)
    for name,p in model.word_backoff.named_parameters():
        if not p.requires_grad:
            raise AssertionError('new backoff parameter was not enabled: '+name)
    optimizer._adamw.zero_grad(set_to_none=True)
    # Candidate-only warm-up. The complete R7.40 source remains immutable.
    with torch.no_grad():model.word_backoff.gain.fill_(args.open_gain)
    rng=np.random.default_rng(args.seed+404)
    snapshots=[]
    best=None
    best_weights=None
    best_moments=None
    evaluations=set([1,args.steps,*args.eval_steps])
    for step in range(1,args.steps+1):
        idx=torch.as_tensor(rng.choice(args.train,min(args.minibatch,args.train),replace=False),
                            dtype=torch.long)
        batch=take(train,idx)
        model.eval()  # trainable new parameters still receive gradients
        nll,ex=forward(model,batch)
        # The only objective is the normalized word mixture. Every earlier
        # word/byte/sentence/geometry weight is frozen, with moments preserved.
        loss=nll.mean()
        optimizer._adamw.zero_grad(set_to_none=True)
        loss.backward()
        if any(p.grad is not None for p in inherited_params):
            raise AssertionError('R7.44 accidentally updated inherited gradients')
        torch.nn.utils.clip_grad_norm_(model.word_backoff.parameters(),1.)
        optimizer._adamw.step()
        if step in evaluations:
            metrics=evaluate(model,dev,wrong,args.seed+step)
            metrics['step']=step
            metrics['train_word_nll']=float(loss.detach())
            snapshots.append(metrics)
            print('R744 STEP',step,'DEV nll',round(metrics['candidate_word_nll'],6),
                  'gain',round(metrics['r744_word_backoff_gain'],8),
                  'foreign',round(metrics['r744_correct_vs_foreign_word_advantage'],8),
                  'geometry',round(metrics['r744_geometry_specificity'],8),
                  'gate',metrics['r744_candidate_only_dev_gate_passed'],flush=True)
            # Select best valid candidate if one exists; otherwise preserve
            # the best exploratory loss but never call it a promotion.
            rank=(int(metrics['r744_candidate_only_dev_gate_passed']),
                  -metrics['candidate_word_nll'])
            if best is None or rank>best[0]:
                best=(rank,metrics)
                best_weights=copy.deepcopy(model.word_backoff.state_dict())
                best_moments=saved_new_optimizer_state(model,optimizer)
    model.word_backoff.load_state_dict(best_weights)
    restore_new_optimizer_state(model,optimizer,best_moments)
    selected=best[1]
    with torch.no_grad():
        model.eval()
        _,aux=forward(model,dict(dev,layered_byte_supervision=True))
        spelling=layered_byte_metrics(aux,dev)
        trunk=normalized_trunk_word_topk(model,base,dev['hidden'],dev['target'])
    for name,old in zip((n for n,p in model.named_parameters()
                         if not n.startswith('word_backoff.')),inherited_params):
        if old.grad is not None:
            raise AssertionError('unexpected inherited gradient after training: '+name)
    if digest(args.checkpoint)!=source_hash:
        raise AssertionError('source checkpoint changed during training')
    saved=save_unpromoted(args.output_checkpoint,model,optimizer,args.root,
                           args.checkpoint,source_metadata=meta,
                           candidate_trained=True,training_round=ROUND,
                           training_updates=selected['step'],
                           development_gate_passed=False,architecture_round=ROUND)
    # Verify the selected candidate, optimizer groups, and exactly preserved
    # source weights using the *normal* loader, not an ad hoc state restore.
    verified,checked_opt,_,_,checked_meta=load_unified(args.output_checkpoint,
                                                        args.root,args.threads)
    if (len(checked_opt.param_groups)!=25 or checked_meta['promoted'] is not False
            or checked_meta['architecture_round']!=ROUND):
        raise AssertionError('recovered R7.44 optimizer/metadata mismatch')
    for key,value in model.state_dict().items():
        if not torch.equal(value,verified.state_dict()[key]):
            raise AssertionError('round-trip model changed: '+key)
    result={'round':ROUND, 'branch':'master', 'accepted_step':14416,
            'source_r740_sha256':source_hash, 'one_optimizer_groups':25,
            'source_checkpoint_never_overwritten':True,
            'candidate_checkpoint':saved, 'candidate_checkpoint_promoted':False,
            'independent_test_opened':False, 'corpus_inventory':inventory,
            'train_positions':args.train,'dev_positions':args.dev,
            'train_updates':args.steps,'selected_update':selected['step'],
            'selected_gate_passed':selected['r744_candidate_only_dev_gate_passed'],
            'promotion_status':'REJECTED: exploratory R7.44; inherited full strict R7 gate not established',
            'conditional_spelling':spelling, 'inherited_trunk_topk':trunk,
            'development_history':snapshots, 'elapsed_seconds':time.monotonic()-start}
    out=Path(args.report)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print('R744 COMPLETED',json.dumps({'candidate_sha256':saved['sha256'],
         'step':selected['step'],'dev_word_nll':selected['candidate_word_nll'],
         'elapsed':round(result['elapsed_seconds'],1)}),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',required=True)
    p.add_argument('--corpus-dir',required=True)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--output-checkpoint',required=True)
    p.add_argument('--report',required=True)
    p.add_argument('--train',type=int,default=240)
    p.add_argument('--dev',type=int,default=192)
    p.add_argument('--batch',type=int,default=8)
    p.add_argument('--minibatch',type=int,default=8)
    p.add_argument('--steps',type=int,default=48)
    p.add_argument('--eval-steps',type=int,nargs='*',default=[8,16,24,48])
    p.add_argument('--seed',type=int,default=744)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--open-gain',type=float,default=.2)
    p.add_argument('--lr',type=float,default=.0008)
    run(p.parse_args())
