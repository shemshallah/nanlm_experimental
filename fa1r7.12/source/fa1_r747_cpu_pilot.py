#!/usr/bin/env python3
"""R7.47 one-checkpoint CPU pilot: make exact head addresses predictive.

All inherited R7.45 weights and optimizer moments survive unchanged. Only
new address coefficients and two bounded probability gates are trained.
Checkpoint promotion is deliberately disabled until strict DEV + sealed TEST
and long-context gates have been independently satisfied.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from fa1_r723_train import prepare, take, forward
from fa1_r727_checkpoint import load_unified, save_unpromoted, digest
from fa1_r727_eval import validate_book_corpus
from fa1_r744_cpu_pilot import bootstrap_by_book, foreign_word_history
from fa1_r746_factorial_dev import perturbed_context
from fa1_r747_head_address_copy import (ROUND,attach_exact_head_copy,
                                         exact_head_copy_disabled)


@torch.no_grad()
def evaluate(model,dev,foreign_hist,wrong_addr,seed):
    model.eval()
    good,ex=forward(model,dev)
    bad_address,_=forward(model,wrong_addr)
    bad_history,_=forward(model,foreign_hist)
    no_apex,_=forward(model,dev,'no_apex')
    no_pair,_=forward(model,dev,'no_pair')
    with exact_head_copy_disabled(model):
        inherited,_=forward(model,dev)
        inherited_wrong,_=forward(model,wrong_addr)
        inherited_no_apex,_=forward(model,dev,'no_apex')
        inherited_no_pair,_=forward(model,dev,'no_pair')
    gain=inherited-good
    address=gain-(inherited_wrong-bad_address)
    apex=gain-(inherited_no_apex-no_apex)
    pair=gain-(inherited_no_pair-no_pair)
    history=bad_history-good
    books=dev['book_ids']
    summaries={
        'r745_parent_word_nll':float(inherited.mean()),
        'r747_candidate_word_nll':float(good.mean()),
        'r747_incremental_word_gain':float(gain.mean()),
        'r747_address_specificity':float(address.mean()),
        'r747_pair_specificity':float(pair.mean()),
        'r747_apex_specificity':float(apex.mean()),
        'r747_correct_vs_foreign_history':float(history.mean()),
        'mean_r747_head_mixture':float(ex['r747_head_mixture'].mean()),
        'mean_r747_extra_copy_mass':float(ex['r747_extra_copy_mass'].mean()),
        'mean_r747_head_membership':float(ex['r747_head_membership'].mean()),
        'mean_total_copy_alpha':float(ex['backoff_alpha'].mean()),
        'dev_books':int(torch.unique(books).numel()),
        'dev_positions':len(good),
    }
    for i,(name,array) in enumerate((('word',gain),('address',address),
                                     ('pair',pair),('apex',apex),('history',history))):
        summaries[f'r747_{name}_book_bootstrap']=bootstrap_by_book(array,books,seed+i)
    summaries['r747_strict_dev_gate_passed']=bool(
        gain.mean().item()>1.e-4 and address.mean().item()>0 and
        pair.mean().item()>0 and apex.mean().item()>0 and history.mean().item()>0 and
        all(summaries[f'r747_{name}_book_bootstrap']['book_bootstrap_ci95'][0]>0
            for name in ('word','address','pair','apex','history')))
    return summaries


def run(a):
    if Path(a.output_checkpoint).exists() or Path(a.report).exists():
        raise FileExistsError('R7.47 outputs must be new and immutable')
    started=time.monotonic()
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)
    inventory=validate_book_corpus(a.corpus_dir,'dev')
    parent_sha=digest(a.checkpoint)
    model,opt,base,comp,meta=load_unified(a.checkpoint,a.root,a.threads)
    if (meta['architecture_round']!='FA1_R7_45_GEOMETRIC_VALUE_READ_V1' or
            len(opt.param_groups)!=26 or meta['promoted'] is not False):
        raise ValueError('R7.47 must inherit the full 26-group R7.45 model')
    print('R747 RESTORED',parent_sha,flush=True)
    train=prepare(a,model,base,comp,'train',a.train,a.seed+101)
    dev=prepare(a,model,base,comp,'dev',a.dev,a.seed+202)
    if int(torch.unique(dev['book_ids']).numel())<12:
        raise ValueError('evaluation must include at least 12 separate DEV books')
    wrong_hist,_=foreign_word_history(dev,a.seed+303)
    wrong_addr=perturbed_context(dev,'addresses_only',a.seed+404)
    model.eval()
    with torch.no_grad():before,_=forward(model,dev)
    old_ids={id(p) for p in model.parameters()}
    for p in model.parameters():p.requires_grad_(False)
    attach_exact_head_copy(model,opt,base,learning_rate=a.lr)
    with torch.no_grad():after,_=forward(model,dev)
    if not torch.equal(before,after):
        raise AssertionError('R7.47 exact-zero migration failed')
    new=[(n,p) for n,p in model.named_parameters() if id(p) not in old_ids]
    if len(new)!=5 or len(opt.param_groups)!=27:
        raise AssertionError('expected exactly 5 new parameters in optimizer group 27')
    for _,p in new:p.requires_grad_(True)
    with torch.no_grad():
        model.word_backoff.address_gain.fill_(a.open_attention)
        model.word_backoff.address_copy_gain.fill_(a.open_copy)
        model.word_backoff.address_coefficients.copy_(
            torch.tensor([0.,0.,.10,.10,-.10]))
    print('R747 ZERO PARITY AND 27-GROUP CONTINUITY PASSED',flush=True)
    minibatch_rng=np.random.default_rng(a.seed+505)
    history=[]
    best=None
    selected_weights=None
    selected_moments=None
    evaluation_steps=set(a.eval_steps)|{1,a.steps}
    for step in range(1,a.steps+1):
        ids=torch.as_tensor(minibatch_rng.choice(a.train,a.minibatch,
                                                replace=False),dtype=torch.long)
        batch=take(train,ids)
        wrong=perturbed_context(batch,'addresses_only',a.seed+step+606)
        model.eval()
        # Calculate frozen inherited scores before the differentiable
        # candidate forwards: toggling a Parameter after graph construction
        # would invalidate PyTorch's saved tensor version checks.
        with exact_head_copy_disabled(model),torch.no_grad():
            inherited,_=forward(model,batch)
            inherited_wrong,_=forward(model,wrong)
        good,_=forward(model,batch)
        unrelated,_=forward(model,wrong)
        specificity=((inherited-good)-(inherited_wrong-unrelated)).mean()
        penalty=F.softplus((a.address_margin-specificity)*10.)/10.
        loss=good.mean()+a.address_contrast*penalty
        opt._adamw.zero_grad(set_to_none=True)
        loss.backward()
        if any(p.grad is not None for p in model.parameters() if id(p) in old_ids):
            raise AssertionError('inherited R7.45 gradient appeared')
        torch.nn.utils.clip_grad_norm_([p for _,p in new],1.)
        opt._adamw.step()
        if step in evaluation_steps:
            metric=evaluate(model,dev,wrong_hist,wrong_addr,a.seed+step)
            metric.update({'step':step,'train_objective':float(loss.detach()),
                           'train_address_specificity':float(specificity.detach())})
            history.append(metric)
            print('R747 STEP',step,'word',round(metric['r747_incremental_word_gain'],7),
                  'address',round(metric['r747_address_specificity'],7),
                  'pair',round(metric['r747_pair_specificity'],7),
                  'gate',metric['r747_strict_dev_gate_passed'],flush=True)
            ranking=(int(metric['r747_strict_dev_gate_passed']),
                     -metric['r747_candidate_word_nll'])
            if best is None or ranking>best[0]:
                best=ranking,metric
                selected_weights={n:p.detach().clone() for n,p in new}
                selected_moments={n:copy.deepcopy(opt._adamw.state.get(p,{})) for n,p in new}
    for n,p in new:
        with torch.no_grad():p.copy_(selected_weights[n])
        opt._adamw.state[p]=copy.deepcopy(selected_moments[n])
    if digest(a.checkpoint)!=parent_sha:
        raise AssertionError('protected R7.45 source checkpoint was modified')
    saved=save_unpromoted(a.output_checkpoint,model,opt,a.root,a.checkpoint,
                          source_metadata=meta,candidate_trained=True,
                          training_round=ROUND,training_updates=best[1]['step'],
                          development_gate_passed=False,architecture_round=ROUND)
    recovered,rec_opt,_,_,rec_meta=load_unified(a.output_checkpoint,a.root,a.threads)
    if rec_meta['architecture_round']!=ROUND or len(rec_opt.param_groups)!=27:
        raise AssertionError('R7.47 failed unified checkpoint reload')
    for n,w in model.state_dict().items():
        if not torch.equal(w,recovered.state_dict()[n]):
            raise AssertionError('R7.47 reload tensor mismatch: '+n)
    out={
        'round':ROUND,'accepted_parent_step':14416,'branch':'master',
        'source_checkpoint_sha256':parent_sha,'one_optimizer_groups':27,
        'candidate_checkpoint_promoted':False,'independent_test_opened':False,
        'train_positions':a.train,'dev_positions':a.dev,'train_updates':a.steps,
        'selected_step':best[1]['step'],
        'selected_strict_dev_gate_passed':best[1]['r747_strict_dev_gate_passed'],
        'promotion':'rejected: no full independent TEST or long-context gate',
        'dataset_inventory':inventory,'candidate_checkpoint':saved,
        'development_history':history,'elapsed_seconds':time.monotonic()-started,
    }
    dest=Path(a.report)
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(out,sort_keys=True,indent=2)+'\n')
    print('R747 DONE',json.dumps({'sha256':saved['sha256'],
          'selected_step':best[1]['step'],
          'word_gain':best[1]['r747_incremental_word_gain'],
          'seconds':round(out['elapsed_seconds'],1)}),flush=True)
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',required=True)
    p.add_argument('--corpus-dir',required=True)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--output-checkpoint',required=True)
    p.add_argument('--report',required=True)
    p.add_argument('--train',type=int,default=1536)
    p.add_argument('--dev',type=int,default=384)
    p.add_argument('--batch',type=int,default=8)
    p.add_argument('--minibatch',type=int,default=12)
    p.add_argument('--steps',type=int,default=96)
    p.add_argument('--eval-steps',type=int,nargs='*',default=[16,32,64,96])
    p.add_argument('--seed',type=int,default=747)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--lr',type=float,default=.0007)
    p.add_argument('--open-attention',type=float,default=.40)
    p.add_argument('--open-copy',type=float,default=.25)
    p.add_argument('--address-contrast',type=float,default=.4)
    p.add_argument('--address-margin',type=float,default=.001)
    run(p.parse_args())
