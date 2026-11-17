#!/usr/bin/env python3
"""FA1 R7.57: trained lexical-vs-signed-pair router, one FA1 checkpoint.

Inherits the immutable R7.54 30-group checkpoint, appends one optimizer
group, freezes all inherited weights/moments, and trains using leave-one-out
TRAIN transition counts. DEV is book-disjoint; TEST remains sealed.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import time

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
from fa1_r757_geometric_transition_router import (
    ROUND, attach_transition_router, build_transition_banks,
    transition_control)

PARENT = 'FA1_R7_53_CAUSAL_SUCCESSOR_MIXTURE_V1'


@torch.no_grad()
def evaluate(model, data, seed):
    model.eval()
    with transition_control(model, 'disabled'):
        baseline, _ = forward(model, data)
    full, info = forward(model, data)
    with transition_control(model, 'lexical_matched'):
        lexical, _ = forward(model, data)
    with transition_control(model, 'wrong_pair_addresses'):
        wrong_addr, _ = forward(model, data)
    with transition_control(model, 'wrong_pair_values'):
        wrong_value, _ = forward(model, data)
    with transition_control(model, 'no_apex'):
        no_apex, _ = forward(model, data)
    tracks = {
        'gain_over_r754': baseline - full,
        'geometry_over_matched_lexical': lexical - full,
        'correct_vs_wrong_pair_addresses': wrong_addr - full,
        'correct_vs_wrong_pair_values': wrong_value - full,
        'successor_apex_reliance': no_apex - full,
    }
    books = data['book_ids']
    boots = {key: bootstrap_by_book(val, books, seed + k)
             for k, (key, val) in enumerate(tracks.items())}
    gain = tracks['gain_over_r754']
    geogain = tracks['geometry_over_matched_lexical']
    passed = all(
        float(tracks[key].mean()) > 0 and
        boots[key]['book_bootstrap_ci95'][0] > 0
        for key in ('gain_over_r754', 'geometry_over_matched_lexical',
                    'correct_vs_wrong_pair_values'))
    return {
        'books': int(books.unique().numel()), 'positions': len(books),
        'r754_next_word_nll': float(baseline.mean()),
        'r757_full_next_word_nll': float(full.mean()),
        'lexical_only_matched_nll': float(lexical.mean()),
        'mean_word_gate': float(info['r757_word_gain'].mean()),
        'mean_pair_gate': float(info['r757_signed_pair_gain'].mean()),
        'geometry_gate_passed': bool(passed),
        'book_bootstrap': boots,
        **{k: float(v.mean()) for k, v in tracks.items()},
    }


def run(a):
    started = time.monotonic()
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    for target in (a.report, a.output_checkpoint):
        if Path(target).exists():
            raise FileExistsError(target)
    if Path(a.checkpoint).resolve() == Path(a.output_checkpoint).resolve():
        raise ValueError('cannot overwrite R7.54 parent')
    inventory = verify_train_dev_only(a.corpus_dir, a.split_metadata)
    parent_sha = digest(a.checkpoint)
    model, optimizer, base, compositor, metadata = load_unified(
        a.checkpoint, a.root, a.threads)
    if (metadata.get('architecture_round') != PARENT or
            len(optimizer.param_groups) != 30 or
            metadata.get('promoted') is not False):
        raise ValueError('R7.57 requires intact unpromoted R7.54 candidate')
    args = SimpleNamespace(root=a.root, corpus_dir=a.corpus_dir, batch=8,
                           r753_position_selector=balanced_causal_positions)
    train = with_successor_history(
        prepare(args, model, base, compositor, 'train', a.train, a.seed + 101),
        a.root, 'train', a.corpus_dir, a.context)
    dev = with_successor_history(
        prepare(args, model, base, compositor, 'dev', a.dev, a.seed + 202),
        a.root, 'dev', a.corpus_dir, a.context)
    if (int(train['book_ids'].unique().numel()) < 200 or
            int(dev['book_ids'].unique().numel()) != 15):
        raise AssertionError('missing TRAIN/DEV independent book coverage')
    existing = dict(model.named_parameters())
    original_hash = frozen_fingerprint(existing, optimizer)
    model.eval()
    with torch.no_grad():
        baseline, _ = forward(model, dev)
    for p in existing.values():
        p.requires_grad_(False)
    banks = build_transition_banks(
        a.root, a.corpus_dir, model.word_backoff.word_head_lookup)
    attach_transition_router(model, optimizer, banks, learning_rate=a.lr)
    with torch.no_grad():
        migrated, _ = forward(model, dev)
    if not torch.equal(baseline, migrated):
        raise AssertionError('R7.57 zero-gain migration lacks bitwise parity')
    if frozen_fingerprint(existing, optimizer) != original_hash:
        raise AssertionError('prior model/optimizer changed during migration')
    new = {n: p for n, p in model.named_parameters() if n not in existing}
    if len(new) != 3 or len(optimizer.param_groups) != 31:
        raise AssertionError('not one added optimizer group')
    train = dict(train, r757_leave_one_out=True)
    with torch.no_grad():
        model.word_backoff.r757_router.strength.fill_(a.open_strength)
    rng = np.random.default_rng(a.seed + 303)
    best = None
    winner = None
    weights = None
    moments = None
    snapshots = []
    for step in range(1, a.steps + 1):
        indices = balanced_batch_indices(train['book_ids'], a.minibatch, rng)
        batch = take(train, indices)
        model.eval()
        nll, _ = forward(model, batch)
        loss = nll.mean()
        optimizer._adamw.zero_grad(set_to_none=True)
        loss.backward()
        if any(p.grad is not None for p in existing.values()):
            raise AssertionError('frozen inherited weights received gradients')
        torch.nn.utils.clip_grad_norm_(list(new.values()), 1.)
        optimizer._adamw.step()
        if step == 1 or step == a.steps or step % a.eval_every == 0:
            m = evaluate(model, dev, a.seed + step)
            m.update({'step': step, 'train_word_nll': float(loss.detach())})
            snapshots.append(m)
            score = (int(m['geometry_gate_passed']), -m['r757_full_next_word_nll'])
            if best is None or score > best:
                best = score
                winner = m
                weights = {n: p.detach().clone() for n, p in new.items()}
                moments = {n: copy.deepcopy(optimizer._adamw.state.get(p, {}))
                           for n, p in new.items()}
            print('FA1_R757_DEV', json.dumps({
                'step': step, 'r754': m['r754_next_word_nll'],
                'candidate': m['r757_full_next_word_nll'],
                'geometry_over_lexical': m['geometry_over_matched_lexical'],
                'strict_geometry_gate': m['geometry_gate_passed']}), flush=True)
    with torch.no_grad():
        for n, p in new.items():
            p.copy_(weights[n])
    for n, p in new.items():
        optimizer._adamw.state[p] = moments[n]
    selected = evaluate(model, dev, a.seed + 9000)
    if frozen_fingerprint(existing, optimizer) != original_hash:
        raise AssertionError('one or more inherited Adam moments were overwritten')
    if digest(a.checkpoint) != parent_sha:
        raise AssertionError('protected R7.54 original checkpoint mutated')
    output = save_unpromoted(
        a.output_checkpoint, model, optimizer, a.root, a.checkpoint,
        source_metadata=metadata, candidate_trained=True,
        training_round=ROUND, training_updates=winner['step'],
        development_gate_passed=False, architecture_round=ROUND)
    reloaded, reoptim, _, _, reloadmeta = load_unified(
        a.output_checkpoint, a.root, a.threads)
    if (reloadmeta.get('architecture_round') != ROUND or
            len(reoptim.param_groups) != 31 or
            frozen_fingerprint({n: p for n, p in reloaded.named_parameters()
                                if n in existing}, reoptim) != original_hash):
        raise AssertionError('full 31-group reload did not preserve FA1 parent')
    restored = dict(reloaded.named_parameters())
    for n, p in new.items():
        if not torch.equal(p, restored[n]) or not _deep_equal(
            optimizer._adamw.state.get(p, {}),
            reoptim._adamw.state.get(restored[n], {})):
            raise AssertionError('router parameter/moment failed strict reload')
    report = {
        'round': ROUND, 'promoted': False, 'independent_test_opened': False,
        'steps_executed': a.steps, 'selected_step': winner['step'],
        'source_sha256': parent_sha, 'candidate_sha256': output['sha256'],
        'one_optimizer_groups': 31, 'inherited_groups_preserved': 30,
        'reloaded_exact': True, 'frozen_parent_verified': True,
        'train_books': int(train['book_ids'].unique().numel()),
        'dev_books': int(dev['book_ids'].unique().numel()),
        'train_positions': len(train['target']),
        'dev_positions': len(dev['target']),
        'transition_training_leave_one_out': True,
        'train_transition_total': int(banks['word_counts'].sum()),
        'corpus_inventory': inventory,
        'matched_dev': selected,
        'all_dev_checkpoints': snapshots,
        'strict_geometry_gate_passed': selected['geometry_gate_passed'],
        'eligible_for_more_training': bool(selected['geometry_gate_passed']),
        'true_byte_bpb_available': False,
        'full_word_top5_available': False,
        'elapsed_seconds': time.monotonic() - started,
    }
    Path(a.report).write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print('FA1_R757_SAVED', output['sha256'], 'STRICT',
          report['strict_geometry_gate_passed'], flush=True)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for field in ('root', 'corpus_dir', 'split_metadata', 'checkpoint',
                  'output_checkpoint', 'report'):
        p.add_argument('--' + field.replace('_', '-'), required=True)
    p.add_argument('--threads', type=int, default=4, choices=(1, 2, 4))
    p.add_argument('--train', type=int, default=4096)
    p.add_argument('--dev', type=int, default=1024)
    p.add_argument('--context', type=int, default=512)
    p.add_argument('--steps', type=int, default=1000)
    p.add_argument('--minibatch', type=int, default=16)
    p.add_argument('--eval-every', type=int, default=250)
    p.add_argument('--lr', type=float, default=.005)
    p.add_argument('--open-strength', type=float, default=.5)
    p.add_argument('--seed', type=int, default=757)
    run(p.parse_args())
