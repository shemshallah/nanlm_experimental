#!/usr/bin/env python3
"""Train-only FA1 R7.50 continuation, group-30 isolated geometric attribution."""
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
from fa1_r744_cpu_pilot import bootstrap_by_book, foreign_word_history
from fa1_r746_factorial_dev import perturbed_context
from fa1_r748_cpu_pilot import verify_train_dev_only
from fa1_r749_address_reliability import ROUND as R749
from fa1_r750_calibrated_inhibition import (ROUND, attach_calibrated_inhibition,
                                             inhibition_control)


@torch.no_grad()
def evaluate(model, dev, seed):
    model.eval()
    wrong = perturbed_context(dev, 'addresses_only', seed + 1)
    foreign, _ = foreign_word_history(dev, seed + 2)
    good, telemetry = forward(model, dev)
    wrong_good, _ = forward(model, wrong)
    foreign_good, _ = forward(model, foreign)
    with inhibition_control(model):
        parent, _ = forward(model, dev)
        wrong_parent, _ = forward(model, wrong)
    with inhibition_control(model, 'permute_inhibitory_values'):
        permuted, _ = forward(model, dev)
    with inhibition_control(model, 'constant_reliability'):
        constant, _ = forward(model, dev)
    metrics = {
        'absolute_word_nll':good,
        'incremental_inhibition_gain':parent-good,
        'correct_minus_wrong_address_specificity':(
            parent-good)-(wrong_parent-wrong_good),
        'learned_inhibition_row_permutation':permuted-good,
        'context_reliability_difference':constant-good,
        'correct_history_advantage':foreign_good-good,
    }
    books = dev['book_ids']
    result = {'dev_books':int(torch.unique(books).numel()),
              'dev_positions':len(good),
              'r749_control_nll':float(parent.mean()),
              'exact_pair_candidate_fraction':float(telemetry['r750_exact_hit_fraction'].mean()),
              'mean_applied_penalty':float(telemetry['r750_inhibition_mean'].mean()),
              'mean_context_reliability':float(telemetry['r750_inhibition_trust'].mean()),
              'mean_gain':float(telemetry['r750_inhibition_gain'])}
    for i, (key, values) in enumerate(metrics.items()):
        result[key] = float(values.mean())
        result[key+'_book_bootstrap'] = bootstrap_by_book(values, books, seed + 20 + i)
    gate = ('incremental_inhibition_gain',
            'correct_minus_wrong_address_specificity',
            'learned_inhibition_row_permutation',
            'correct_history_advantage')
    result['strict_dev_gate_passed'] = all(
        result[key] > 0 and result[key+'_book_bootstrap']['book_bootstrap_ci95'][0] > 0
        for key in gate)
    return result


def run(a):
    if Path(a.output_checkpoint).exists() or Path(a.report).exists():
        raise FileExistsError('R7.50 experiment outputs are immutable')
    t0 = time.monotonic()
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    np.random.seed(a.seed)
    inventory = verify_train_dev_only(a.corpus_dir, a.split_metadata)
    sha = digest(a.checkpoint)
    model, optimizer, base, comp, meta = load_unified(a.checkpoint, a.root, a.threads)
    if (meta['architecture_round'] != R749 or meta['promoted'] is not False or
            len(optimizer.param_groups) != 29):
        raise ValueError('R7.50 requires the current authentic R7.49 checkpoint')
    train = prepare(a, model, base, comp, 'train', a.train, a.seed + 101)
    dev = prepare(a, model, base, comp, 'dev', a.dev, a.seed + 202)
    if int(torch.unique(dev['book_ids']).numel()) != 15:
        raise AssertionError('all 15 DEV books are required')
    model.eval()
    with torch.no_grad():
        before, _ = forward(model, dev)
    old_params = {name:p for name,p in model.named_parameters()}
    for p in model.parameters():
        p.requires_grad_(False)
    attach_calibrated_inhibition(model, optimizer, a.lr)
    with torch.no_grad():
        after, _ = forward(model, dev)
    if not torch.equal(before, after):
        raise AssertionError('R7.49 -> R7.50 zero-migration prediction parity failed')
    new = [(name, p) for name,p in model.named_parameters() if name not in old_params]
    if len(new) != 5 or len(optimizer.param_groups) != 30:
        raise AssertionError('R7.50 must append five tensors and group 30')
    for _,p in new:
        p.requires_grad_(True)
    with torch.no_grad():
        model.word_backoff.mirror_inhibition_gain.fill_(a.open_gain)
    rng = np.random.default_rng(a.seed + 303)
    selected = None
    weights = None
    moments = None
    history = []
    print('R750 BITWISE PARENT PARITY; 29 GROUPS FROZEN; GROUP 30 APPENDED',sha,flush=True)
    check_steps = set(a.eval_steps)|{1,a.steps}
    for step in range(1,a.steps+1):
        idx = torch.tensor(rng.choice(a.train,a.minibatch,replace=False),dtype=torch.long)
        batch = take(train,idx)
        wrong = perturbed_context(batch,'addresses_only',a.seed + step + 401)
        good, extras = forward(model,batch)
        wrong_good, _ = forward(model,wrong)
        with inhibition_control(model), torch.no_grad():
            old_good,_ = forward(model,batch)
            old_wrong,_ = forward(model,wrong)
        with inhibition_control(model,'permute_inhibitory_values'):
            permute_good,_ = forward(model,batch)
        delta = ((old_good-good)-(old_wrong-wrong_good)).mean()
        row_delta = (permute_good-good).mean()
        copied_prob = extras['r748_pair_target_probability']
        copied = copied_prob > 0.
        # Gold labels appear only in this TRAIN loss; the actual pair read
        # remains target-free and is evaluated with immutable DEV targets.
        copy_loss = (-copied_prob[copied].clamp_min(1.e-12).log().mean()
                     if bool(copied.any()) else good.sum()*0.)
        loss = (good.mean() + a.copy_aux_weight*copy_loss +
                a.address_contrast*F.softplus(10*(a.margin-delta))/10 +
                a.row_contrast*F.softplus(10*(a.margin-row_delta))/10)
        optimizer._adamw.zero_grad(set_to_none=True)
        loss.backward()
        if any(p.grad is not None for p in old_params.values()):
            raise AssertionError('R7.49 inherited model received a gradient')
        torch.nn.utils.clip_grad_norm_([p for _,p in new],1.)
        optimizer._adamw.step()
        if step in check_steps:
            measures = evaluate(model,dev,a.seed+step)
            measures.update({'step':step,'train_loss':float(loss.detach()),
                             'train_copy_aux_loss':float(copy_loss.detach()),
                             'train_address_delta':float(delta.detach()),
                             'train_row_delta':float(row_delta.detach())})
            history.append(measures)
            print('R750 DEV',step,json.dumps({k:measures[k] for k in (
                'absolute_word_nll','incremental_inhibition_gain',
                'correct_minus_wrong_address_specificity',
                'learned_inhibition_row_permutation','strict_dev_gate_passed')}),flush=True)
            score = (int(measures['strict_dev_gate_passed']), -measures['absolute_word_nll'])
            if selected is None or score>selected[0]:
                selected = (score, measures)
                weights = {name:p.detach().clone() for name,p in new}
                moments = {name:copy.deepcopy(optimizer._adamw.state.get(p,{}))
                           for name,p in new}
    for name,p in new:
        with torch.no_grad():
            p.copy_(weights[name])
        optimizer._adamw.state[p] = copy.deepcopy(moments[name])
    if digest(a.checkpoint) != sha:
        raise AssertionError('protected R7.49 checkpoint changed')
    saved = save_unpromoted(a.output_checkpoint,model,optimizer,a.root,a.checkpoint,
                            source_metadata=meta,candidate_trained=True,
                            training_round=ROUND,training_updates=selected[1]['step'],
                            development_gate_passed=False,architecture_round=ROUND)
    report = {'round':ROUND,'branch':'master','promoted':False,
              'independent_test_opened':False,'source_sha256':sha,
              'optimizer_count':1,'optimizer_groups':30,
              'frozen_inherited_groups':29,'trainable_new_group':30,
              'corpus_inventory':inventory,
              'train_positions':a.train,'dev_positions':a.dev,
              'updates_executed':a.steps,'selected_step':selected[1]['step'],
              'selected_strict_dev_gate_passed':selected[1]['strict_dev_gate_passed'],
              'candidate_checkpoint':saved,
              'full_word_top5_available':False,'raw_byte_bpb_available':False,
              'promotion':'rejected pending causal attribution, long-context, sealed TEST',
              'development_history':history,'elapsed_seconds':time.monotonic()-t0}
    Path(a.report).parent.mkdir(parents=True,exist_ok=True)
    Path(a.report).write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    print('R750 SAVED',saved['sha256'],'SELECTED',selected[1]['step'],flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',required=True)
    p.add_argument('--split-metadata',required=True)
    p.add_argument('--corpus-dir',required=True)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--output-checkpoint',required=True)
    p.add_argument('--report',required=True)
    p.add_argument('--train',type=int,default=1024)
    p.add_argument('--dev',type=int,default=256)
    p.add_argument('--batch',type=int,default=8)
    p.add_argument('--minibatch',type=int,default=12)
    p.add_argument('--steps',type=int,default=96)
    p.add_argument('--eval-steps',type=int,nargs='*',default=[24,48,72,96])
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--seed',type=int,default=750)
    p.add_argument('--lr',type=float,default=8.e-4)
    p.add_argument('--open-gain',type=float,default=1.5)
    p.add_argument('--copy-aux-weight',type=float,default=.025)
    p.add_argument('--address-contrast',type=float,default=.5)
    p.add_argument('--row-contrast',type=float,default=.2)
    p.add_argument('--margin',type=float,default=.0005)
    run(p.parse_args())
