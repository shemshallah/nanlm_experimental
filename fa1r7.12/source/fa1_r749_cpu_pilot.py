#!/usr/bin/env python3
"""Train R7.49 in the ONE R7.48 model, preserving 28 AdamW groups.

TRAIN-only updates; book-disjoint DEV for selection; sealed TEST never read.
Save and stop without reloading the huge checkpoint in the same process.
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
from fa1_r744_cpu_pilot import bootstrap_by_book, foreign_word_history
from fa1_r746_factorial_dev import perturbed_context
from fa1_r748_mirror_pair_values import ROUND as R748, pair_value_control
from fa1_r748_cpu_pilot import verify_train_dev_only
from fa1_r749_address_reliability import (ROUND, attach_address_reliability,
                                          address_reliability_control)


@torch.no_grad()
def evaluate(model, dev, seed):
    model.eval()
    wrong = perturbed_context(dev, 'addresses_only', seed + 1)
    foreign, _ = foreign_word_history(dev, seed + 2)
    full, telemetry = forward(model, dev)
    wrong_full, _ = forward(model, wrong)
    foreign_full, _ = forward(model, foreign)
    with address_reliability_control(model):
        r748, _ = forward(model, dev)
        wrong_r748, _ = forward(model, wrong)
    with address_reliability_control(model, 'without_new_address'):
        no_address, _ = forward(model, dev)
    with address_reliability_control(model, 'without_new_signed_gate'):
        no_gate, _ = forward(model, dev)
    with pair_value_control(model, 'without_signed_orientation'):
        no_signed_rail, _ = forward(model, dev)
    with pair_value_control(model, 'scramble_pair_values'):
        shuffled, _ = forward(model, dev)
    books = dev['book_ids']
    measures = {
        'absolute_word_nll': full,
        'incremental_over_current_r748': r748 - full,
        'canonical_address_delta': (r748 - full) - (wrong_r748 - wrong_full),
        'new_address_contribution': no_address - full,
        'new_signed_reliability_contribution': no_gate - full,
        'total_signed_rail_contribution': no_signed_rail - full,
        'whole_pair_bank_permutation_contribution': shuffled - full,
        'correct_history_advantage': foreign_full - full,
    }
    out = {'dev_books': int(torch.unique(books).numel()),
           'dev_positions': int(len(full)),
           'r748_control_nll': float(r748.mean()),
           'exact_active_pair_fraction': float(telemetry['r749_exact_active_pair_fraction'].mean()),
           'any_endpoint_fraction': float(telemetry['r749_any_endpoint_fraction'].mean()),
           'signed_reliability': float(telemetry['r749_signed_reliability'].mean()),
           'signed_suppression': float(telemetry['r749_signed_suppression']),
           'mixing_beta': float(telemetry['r748_pair_mixture'].mean())}
    for i, (key, values) in enumerate(measures.items()):
        out[key] = float(values.mean())
        out[key + '_whole_book_bootstrap'] = bootstrap_by_book(values, books, seed + 11 + i)
    # Evidence gate: improvements on all the critical *causal* controls,
    # not merely a favorable NLL. Independent TEST is a separate gate.
    positive = ('incremental_over_current_r748', 'canonical_address_delta',
                'new_address_contribution', 'total_signed_rail_contribution',
                'whole_pair_bank_permutation_contribution', 'correct_history_advantage')
    out['strict_dev_gate_passed'] = all(
        out[key] > 0 and
        out[key + '_whole_book_bootstrap']['book_bootstrap_ci95'][0] > 0
        for key in positive)
    return out


def run(a):
    if Path(a.output_checkpoint).exists() or Path(a.report).exists():
        raise FileExistsError('new R7.49 checkpoint/report must be immutable')
    t0 = time.monotonic()
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    np.random.seed(a.seed)
    identity = verify_train_dev_only(a.corpus_dir, a.split_metadata)
    source_sha = digest(a.checkpoint)
    model, optimizer, base, compositor, meta = load_unified(a.checkpoint, a.root, a.threads)
    continuing = meta['architecture_round'] == ROUND
    if (meta['architecture_round'] not in (R748, ROUND) or meta['promoted'] is not False or
            len(optimizer.param_groups) != (29 if continuing else 28)):
        raise ValueError('R7.49 must continue the genuine R7.48/R7.49 lineage')
    train = prepare(a, model, base, compositor, 'train', a.train, a.seed + 101)
    dev = prepare(a, model, base, compositor, 'dev', a.dev, a.seed + 202)
    if int(torch.unique(dev['book_ids']).numel()) != 15:
        raise ValueError('DEV needs every one of the 15 disjoint books')
    model.eval()
    with torch.no_grad():
        original, _ = forward(model, dev)
    inherited = {name: p for name, p in model.named_parameters()
                 if not name.startswith('word_backoff.mirror_')}
    for p in model.parameters():
        p.requires_grad_(False)
    if not continuing:
        attach_address_reliability(model, optimizer, a.lr)
    with torch.no_grad():
        migrated, _ = forward(model, dev)
    if not torch.equal(original, migrated):
        raise AssertionError('R7.49 migration or continuation changed original scores')
    tunable = [(name, p) for name, p in model.named_parameters()
               if name.startswith('word_backoff.mirror_')]
    for _, p in tunable:
        p.requires_grad_(True)
    optimizer._adamw.param_groups[-2]['lr'] = a.old_pair_lr
    optimizer._adamw.param_groups[-1]['lr'] = a.lr
    # Enabling suppression follows exact migration, and is a candidate
    # choice that training/DEV may reject.
    with torch.no_grad():
        module = model.word_backoff
        module.mirror_signed_suppression.fill_(max(float(module.mirror_signed_suppression),
                                                   a.open_suppression))
        module.mirror_reliability_bias.fill_(max(float(module.mirror_reliability_bias),
                                                a.open_reliability_bias))
    print('R749 BITWISE PARITY 29 GROUPS', source_sha,
          'RESUMED MOMENTS' if continuing else 'APPENDED', flush=True)
    rng = np.random.default_rng(a.seed + 303)
    history = []
    chosen, selected = None, None
    for step in range(1, a.steps + 1):
        ix = torch.tensor(rng.choice(a.train, a.minibatch, replace=False), dtype=torch.long)
        batch = take(train, ix)
        address_wrong = perturbed_context(batch, 'addresses_only', a.seed + step + 401)
        good, _ = forward(model, batch)
        wrong, _ = forward(model, address_wrong)
        with address_reliability_control(model):
            old_good, _ = forward(model, batch)
            old_wrong, _ = forward(model, address_wrong)
        with address_reliability_control(model, 'without_new_address'):
            no_addr, _ = forward(model, batch)
        with pair_value_control(model, 'without_signed_orientation'):
            no_sign, _ = forward(model, batch)
        address_delta = ((old_good - good) - (old_wrong - wrong)).mean()
        address_necessity = (no_addr - good).mean()
        signed_delta = (no_sign - good).mean()
        loss = (good.mean() +
                a.address_contrast * F.softplus(10. * (a.margin - address_delta)) / 10. +
                a.address_necessity_contrast * F.softplus(10. * (a.margin - address_necessity)) / 10. +
                a.sign_contrast * F.softplus(10. * (a.margin - signed_delta)) / 10.)
        optimizer._adamw.zero_grad(set_to_none=True)
        loss.backward()
        if any(p.grad is not None for p in inherited.values()):
            raise AssertionError('old FA1 backbone gradient detected')
        torch.nn.utils.clip_grad_norm_([p for _, p in tunable], 1.)
        optimizer._adamw.step()
        if step in set(a.eval_steps) | {1, a.steps}:
            result = evaluate(model, dev, a.seed + step)
            result.update({'step': step, 'train_loss': float(loss.detach()),
                           'train_address_delta': float(address_delta.detach()),
                           'train_signed_delta': float(signed_delta.detach())})
            history.append(result)
            print('R749 DEV', step, json.dumps({k: result[k] for k in
                  ('absolute_word_nll', 'incremental_over_current_r748',
                   'canonical_address_delta', 'total_signed_rail_contribution',
                   'strict_dev_gate_passed')}), flush=True)
            # Prefer strict-causal passage, then matched same-book DEV NLL.
            score = (int(result['strict_dev_gate_passed']), -result['absolute_word_nll'])
            if selected is None or score > selected[0]:
                selected = (score, result)
                chosen = ({name: p.detach().clone() for name, p in tunable},
                          {name: copy.deepcopy(optimizer._adamw.state.get(p, {}))
                           for name, p in tunable})
    if chosen is None:
        raise AssertionError('no selected DEV candidate')
    for name, p in tunable:
        with torch.no_grad():
            p.copy_(chosen[0][name])
        optimizer._adamw.state[p] = copy.deepcopy(chosen[1][name])
    if digest(a.checkpoint) != source_sha:
        raise AssertionError('original R7.48 checkpoint was altered')
    cumulative = (int(meta.get('candidate_training_updates', 0)) if continuing else 0) + selected[1]['step']
    saved = save_unpromoted(a.output_checkpoint, model, optimizer, a.root, a.checkpoint,
                            source_metadata=meta, candidate_trained=True,
                            training_round=ROUND, training_updates=cumulative,
                            development_gate_passed=False, architecture_round=ROUND)
    report = {'round': ROUND, 'branch': 'master', 'promoted': False,
              'independent_test_opened': False, 'source_sha256': source_sha,
              'optimizer_count': 1, 'optimizer_groups': 29,
              'inherited_groups_preserved': 28,
              'frozen_older_groups': 27, 'source_optimizer_state_preserved': True,
              'resumed_r749_with_adam_moments': continuing,
              'cumulative_r749_training_updates': cumulative,
              'corpus_inventory': identity, 'train_positions': a.train,
              'dev_positions': a.dev, 'updates_executed': a.steps,
              'selected_update': selected[1]['step'],
              'selected_strict_dev_gate_passed': selected[1]['strict_dev_gate_passed'],
              'candidate_checkpoint': saved,
              'full_byte_bpb_available': False, 'full_word_top5_available': False,
              'promotion': 'rejected until independent-book and full attribution gates',
              'development_history': history, 'elapsed_seconds': time.monotonic()-t0}
    Path(a.report).parent.mkdir(parents=True, exist_ok=True)
    Path(a.report).write_text(json.dumps(report, sort_keys=True, indent=2)+'\n')
    print('R749 SAVED', saved['sha256'], 'SELECTED', selected[1]['step'], flush=True)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--split-metadata', required=True)
    p.add_argument('--corpus-dir', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output-checkpoint', required=True)
    p.add_argument('--report', required=True)
    p.add_argument('--train', type=int, default=1024)
    p.add_argument('--dev', type=int, default=256)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--minibatch', type=int, default=12)
    p.add_argument('--steps', type=int, default=64)
    p.add_argument('--eval-steps', type=int, nargs='*', default=[16, 32, 48, 64])
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--seed', type=int, default=749)
    p.add_argument('--lr', type=float, default=4.e-4)
    p.add_argument('--old-pair-lr', type=float, default=2.e-4)
    p.add_argument('--open-suppression', type=float, default=.2)
    p.add_argument('--open-reliability-bias', type=float, default=0.)
    p.add_argument('--address-contrast', type=float, default=.65)
    p.add_argument('--address-necessity-contrast', type=float, default=.3)
    p.add_argument('--sign-contrast', type=float, default=.5)
    p.add_argument('--margin', type=float, default=.0005)
    run(p.parse_args())
