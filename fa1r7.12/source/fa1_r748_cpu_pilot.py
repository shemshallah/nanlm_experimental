#!/usr/bin/env python3
"""R7.48 one-checkpoint TRAIN-only pair-value continuation and DEV attribution.

The R7.47 checkpoint remains immutable. Inherited tensors and all 27
optimizer groups/moments are frozen; only appended group 28 is updated.
No TEST data is loaded or evaluated; no checkpoint promotion is permitted.
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
from fa1_r746_factorial_dev import perturbed_context, same_points_new_order
from fa1_r747_head_address_copy import ROUND as R747
from fa1_r748_mirror_pair_values import (ROUND, attach_mirror_pair_values,
                                         pair_value_control)


def verify_train_dev_only(corpus_dir, archived_meta):
    """Inspect exactly TRAIN/DEV document inventories, never sealed TEST."""
    state = {}
    seen = set()
    for split, expected_count in (('train', 270), ('dev', 15)):
        original = json.loads((Path(archived_meta) / f'{split}_documents.json').read_text())
        prepared = json.loads((Path(corpus_dir) / f'{split}_documents.json').read_text())
        if prepared != original or len(original['documents']) != expected_count:
            raise AssertionError('prepared TRAIN/DEV corpus identity mismatch')
        digests = {row['sha256'] for row in original['documents']}
        if len(digests) != expected_count or digests & seen:
            raise AssertionError('train/dev book overlap')
        seen.update(digests)
        positions = np.memmap(Path(corpus_dir) / f'{split}_doc_start.u32', dtype='<u4')
        if len(positions) != original['total_word_targets']:
            raise AssertionError('train/dev position mismatch')
        for row in original['documents']:
            idx = int(row['first_record'])
            if int(positions[idx]) != idx:
                raise AssertionError('unaligned book boundary')
        state[split] = {'books': expected_count, 'words': len(positions)}
    return state


@torch.no_grad()
def evaluate(model, dev, seed):
    model.eval()
    wrong_address = perturbed_context(dev, 'addresses_only', seed + 1)
    reordered = same_points_new_order(dev)
    wrong_history, _ = foreign_word_history(dev, seed + 2)
    good, telemetry = forward(model, dev)
    wrong, _ = forward(model, wrong_address)
    ordered, _ = forward(model, reordered)
    foreign, _ = forward(model, wrong_history)
    with pair_value_control(model, 'scramble_pair_values'):
        scramble, _ = forward(model, dev)
    with pair_value_control(model, 'without_pair_values'):
        no_bank, _ = forward(model, dev)
    with pair_value_control(model, 'without_signed_orientation'):
        no_sign, _ = forward(model, dev)
    with pair_value_control(model):
        parent, _ = forward(model, dev)
        wrong_parent, _ = forward(model, wrong_address)
        ordered_parent, _ = forward(model, reordered)
        foreign_parent, _ = forward(model, wrong_history)
    gain = parent - good
    metrics = {
        'r747_parent_nll': float(parent.mean()),
        'r748_candidate_nll': float(good.mean()),
        'gain': gain,
        'address': gain - (wrong_parent - wrong),
        'same_points_reordered': gain - (ordered_parent - ordered),
        'pair_value_bank': no_bank - good,
        'pair_value_permutation': scramble - good,
        'signed_orientation': no_sign - good,
        'correct_history': foreign - good,
        'incremental_history': (foreign - good) - (foreign_parent - parent),
    }
    books = dev['book_ids']
    out = {'dev_books': int(torch.unique(books).numel()),
           'dev_positions': int(len(good)),
           'mean_new_pair_copy_mass': float(telemetry['r748_pair_copy_mass'].mean()),
           'mean_new_pair_mixture': float(telemetry['r748_pair_mixture'].mean()),
           'mean_valid_word_pair_fraction': float(telemetry['r748_valid_pair_fraction'].mean())}
    for i, (key, val) in enumerate(metrics.items()):
        if not torch.is_tensor(val):
            out[key] = val
        else:
            out[key] = float(val.mean())
            out[f'{key}_book_bootstrap'] = bootstrap_by_book(val, books, seed + 10 + i)
    needed = ('gain', 'address', 'pair_value_bank', 'pair_value_permutation',
              'signed_orientation', 'correct_history')
    out['strict_dev_gate_passed'] = bool(all(
        out[key] > 0 and out[f'{key}_book_bootstrap']['book_bootstrap_ci95'][0] > 0
        for key in needed))
    return out


def run(a):
    if Path(a.output_checkpoint).exists() or Path(a.report).exists():
        raise FileExistsError('R7.48 outputs must be new and immutable')
    start = time.monotonic()
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)
    inventory = verify_train_dev_only(a.corpus_dir, a.split_metadata)
    source_sha = digest(a.checkpoint)
    model, optimizer, base, compositor, metadata = load_unified(a.checkpoint, a.root, a.threads)
    continuing = metadata['architecture_round'] == ROUND
    if (metadata['architecture_round'] not in (R747, ROUND) or
            metadata['promoted'] is not False or
            len(optimizer.param_groups) != (28 if continuing else 27)):
        raise ValueError('R7.48 requires the exact R7.47 or R7.48 unified lineage')
    print('R748 SOURCE VERIFIED', source_sha, flush=True)
    train = prepare(a, model, base, compositor, 'train', a.train, a.seed + 101)
    dev = prepare(a, model, base, compositor, 'dev', a.dev, a.seed + 202)
    if torch.unique(dev['book_ids']).numel() != 15:
        raise ValueError('DEV selection must include all 15 books')
    model.eval()
    with torch.no_grad():
        before, _ = forward(model, dev)
    inherited_ids = {id(p) for name, p in model.named_parameters()
                     if not name.startswith('word_backoff.mirror_')}
    for p in model.parameters():
        p.requires_grad_(False)
    if not continuing:
        attach_mirror_pair_values(model, optimizer, a.lr)
    else:
        optimizer._adamw.param_groups[-1]['lr'] = a.lr
    new = [(name, p) for name, p in model.named_parameters() if id(p) not in inherited_ids]
    with torch.no_grad():
        after, _ = forward(model, dev)
    if not torch.equal(before, after) or len(new) != 6 or len(optimizer.param_groups) != 28:
        raise AssertionError('R7.48 zero-gate migration broke R7.47')
    for _, p in new:
        p.requires_grad_(True)
    if not continuing:
        with torch.no_grad():
            model.word_backoff.mirror_pair_gain.fill_(a.open_mixture)
            model.word_backoff.mirror_pair_copy_gain.fill_(a.open_copy)
    print('R748 EXACT PARITY; 28-GROUP STATE CONTINUITY',
          'RESUMED WITH SAVED ADAM MOMENTS' if continuing else 'APPENDED', flush=True)
    rng = np.random.default_rng(a.seed + 303)
    best, history, chosen_weights, chosen_moments = None, [], None, None
    eval_steps = set(a.eval_steps) | {1, a.steps}
    for step in range(1, a.steps + 1):
        ids = torch.as_tensor(rng.choice(a.train, a.minibatch, replace=False), dtype=torch.long)
        batch = take(train, ids)
        wrong = perturbed_context(batch, 'addresses_only', a.seed + step + 401)
        with pair_value_control(model), torch.no_grad():
            inherited, _ = forward(model, batch)
            inherited_wrong, _ = forward(model, wrong)
        good, _ = forward(model, batch)
        bad, _ = forward(model, wrong)
        with pair_value_control(model, 'scramble_pair_values'):
            scramble, _ = forward(model, batch)
        with pair_value_control(model, 'without_pair_values'):
            no_bank, _ = forward(model, batch)
        address_delta = ((inherited - good) - (inherited_wrong - bad)).mean()
        bank_delta = (scramble - good).mean()
        bank_necessity = (no_bank - good).mean()
        loss = (good.mean() + a.address_contrast *
                F.softplus(10 * (a.margin - address_delta)) / 10 +
                a.bank_contrast * F.softplus(10 * (a.margin - bank_delta)) / 10 +
                a.bank_necessity_contrast * F.softplus(10 * (a.margin - bank_necessity)) / 10)
        optimizer._adamw.zero_grad(set_to_none=True)
        loss.backward()
        if any(p.grad is not None for p in model.parameters() if id(p) in inherited_ids):
            raise AssertionError('an inherited R7.47 tensor received a gradient')
        torch.nn.utils.clip_grad_norm_([p for _, p in new], 1.)
        optimizer._adamw.step()
        if step in eval_steps:
            measures = evaluate(model, dev, a.seed + step)
            measures.update({'step': step, 'train_loss': float(loss.detach()),
                             'train_address_specificity': float(address_delta.detach()),
                             'train_bank_specificity': float(bank_delta.detach()),
                             'train_bank_necessity': float(bank_necessity.detach())})
            history.append(measures)
            print('R748 DEV', step, 'gain', round(measures['gain'], 7),
                  'address', round(measures['address'], 7),
                  'pairbank', round(measures['pair_value_bank'], 7),
                  'sign', round(measures['signed_orientation'], 7),
                  'strict', measures['strict_dev_gate_passed'], flush=True)
            score = (int(measures['strict_dev_gate_passed']), -measures['r748_candidate_nll'])
            if best is None or score > best[0]:
                best = (score, measures)
                chosen_weights = {name: p.detach().clone() for name, p in new}
                chosen_moments = {name: copy.deepcopy(optimizer._adamw.state.get(p, {}))
                                  for name, p in new}
    for name, param in new:
        with torch.no_grad():
            param.copy_(chosen_weights[name])
        optimizer._adamw.state[param] = copy.deepcopy(chosen_moments[name])
    if digest(a.checkpoint) != source_sha:
        raise AssertionError('protected R7.47 checkpoint changed')
    cumulative_updates = (metadata.get('candidate_training_updates', 0)
                          if continuing else 0) + best[1]['step']
    saved = save_unpromoted(a.output_checkpoint, model, optimizer, a.root, a.checkpoint,
                            source_metadata=metadata, candidate_trained=True,
                            training_round=ROUND, training_updates=cumulative_updates,
                            development_gate_passed=False, architecture_round=ROUND)
    restored, ropt, _, _, restored_meta = load_unified(a.output_checkpoint, a.root, a.threads)
    if restored_meta['architecture_round'] != ROUND or len(ropt.param_groups) != 28:
        raise AssertionError('failed to reload unified R7.48 candidate')
    for name, value in model.state_dict().items():
        if not torch.equal(value, restored.state_dict()[name]):
            raise AssertionError('checkpoint roundtrip tensor mismatch: ' + name)
    report = {'round': ROUND, 'branch': 'master', 'promoted': False,
              'independent_test_opened': False, 'full_byte_bpb_available': False,
              'full_word_top5_available': False, 'source_sha256': source_sha,
              'optimizer_count': 1, 'optimizer_groups': 28,
              'inherited_groups_preserved': 27,
              'resumed_r748_optimizer_moments': continuing,
              'corpus_inventory': inventory,
              'train_positions': a.train, 'dev_positions': a.dev,
              'updates_executed': a.steps, 'selected_update': best[1]['step'],
              'cumulative_r748_training_updates': cumulative_updates,
              'selected_strict_dev_gate_passed': best[1]['strict_dev_gate_passed'],
              'candidate_checkpoint': saved,
              'promotion': 'rejected: exploratory DEV and sealed independent TEST untouched',
              'development_history': history, 'elapsed_seconds': time.monotonic() - start}
    output = Path(a.report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, sort_keys=True, indent=2) + '\n')
    print('R748 FINISHED', json.dumps({'checkpoint_sha256': saved['sha256'],
          'selected_step': best[1]['step'], 'word_gain': best[1]['gain'],
          'seconds': round(report['elapsed_seconds'], 2)}), flush=True)
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
    p.add_argument('--steps', type=int, default=48)
    p.add_argument('--eval-steps', type=int, nargs='*', default=[12, 24, 48])
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--seed', type=int, default=748)
    p.add_argument('--lr', type=float, default=7.e-4)
    p.add_argument('--open-mixture', type=float, default=.4)
    p.add_argument('--open-copy', type=float, default=.25)
    p.add_argument('--address-contrast', type=float, default=.4)
    p.add_argument('--bank-contrast', type=float, default=.4)
    p.add_argument('--bank-necessity-contrast', type=float, default=.4)
    p.add_argument('--margin', type=float, default=.001)
    run(p.parse_args())
