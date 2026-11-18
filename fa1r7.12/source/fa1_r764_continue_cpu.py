#!/usr/bin/env python3
"""Resume *the same* R7.64 weights/AdamW, with all R7.63 paths frozen.

This is a candidate continuation, not promotion. Selection uses only the
original development sampling protocol. A later independent evaluation must
be run on previously unseen DEV positions with the weights frozen.
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from fa1_r723_train import forward, prepare, take
from fa1_r727_checkpoint import digest, load_unified, save_unpromoted
from fa1_r727_optimizer import _deep_equal
from fa1_r748_cpu_pilot import verify_train_dev_only
from fa1_r751_cpu_pilot import balanced_batch_indices
from fa1_r753_corpus import balanced_causal_positions, with_successor_history
from fa1_r754_cpu_pilot import frozen_fingerprint
from fa1_r764_cpu_pilot import evaluate
from fa1_r764_multiresolution import ROUND


def run(a):
    started = time.monotonic()
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    if Path(a.output_checkpoint).exists() or Path(a.report).exists():
        raise FileExistsError('R7.64 continuation cannot overwrite any result')
    inventory = verify_train_dev_only(a.corpus_dir, a.split_metadata)
    ancestor_hash = digest(a.checkpoint)
    model, optimizer, base, compositor, meta = load_unified(
        a.checkpoint, a.root, a.threads)
    if (meta.get('architecture_round') != ROUND or
            meta.get('promoted') is not False or
            not meta.get('candidate_trained') or
            len(optimizer.param_groups) != 37):
        raise ValueError('requires exact 37-group unpromoted R7.64 continuation')
    names = dict(model.named_parameters())
    new = {n: p for n, p in names.items() if '.r764_' in n}
    inherited = {n: p for n, p in names.items() if n not in new}
    if len(new) != 6:
        raise AssertionError('R7.64 added parameters must be exactly 6')
    frozen_hash = frozen_fingerprint(inherited, optimizer)
    for p in inherited.values():
        p.requires_grad_(False)
    x = SimpleNamespace(root=a.root, corpus_dir=a.corpus_dir, batch=8,
                        r753_position_selector=balanced_causal_positions)
    train = with_successor_history(
        prepare(x, model, base, compositor, 'train', a.train, a.seed + 101),
        a.root, 'train', a.corpus_dir, a.context)
    # Reuse only the original DEV sample for candidate selection.
    dev = with_successor_history(
        prepare(x, model, base, compositor, 'dev', a.dev, 764 + 202),
        a.root, 'dev', a.corpus_dir, a.context)
    if (int(train['book_ids'].unique().numel()) < 200 or
            int(dev['book_ids'].unique().numel()) != 15):
        raise AssertionError('TRAIN or DEV split coverage failed')
    train = dict(train, r757_leave_one_out=True, r758_leave_one_out=True,
                 r759_leave_one_out=True)
    rng = np.random.default_rng(a.seed + 303)
    historical_step = int(meta.get('candidate_training_updates', 0))
    best = None
    chosen = None
    retained = None
    retained_adam = None
    history = []
    for step in range(1, a.steps + 1):
        batch = take(train, balanced_batch_indices(
            train['book_ids'], a.minibatch, rng))
        model.eval()
        loss = forward(model, batch)[0].mean()
        optimizer._adamw.zero_grad(set_to_none=True)
        loss.backward()
        if any(p.grad is not None for p in inherited.values()):
            raise AssertionError('inherited R7.63 pathway acquired gradients')
        torch.nn.utils.clip_grad_norm_(list(new.values()), 1.)
        optimizer._adamw.step()
        if step in (1, a.steps) or step % a.eval_every == 0:
            result = evaluate(model, dev, 764 + historical_step + step)
            result['step'] = historical_step + step
            history.append(result)
            rank = (int(result['strict_geometry_gate_passed']),
                    -result['candidate_nll'])
            if best is None or rank > best:
                best, chosen = rank, result
                retained = {n: p.detach().clone() for n, p in new.items()}
                retained_adam = {n: copy.deepcopy(optimizer._adamw.state[p])
                                 for n, p in new.items()}
            print('R764_CONT_DEV', json.dumps({
                'step': result['step'],
                'nll': result['candidate_nll'],
                'hierarchy_gain': result['hierarchy_gain'],
                'geometry_over_lexical': result['geometry_over_lexical_matched'],
                'gate': result['strict_geometry_gate_passed']}), flush=True)
    with torch.no_grad():
        for n, p in new.items():
            p.copy_(retained[n])
    for n, p in new.items():
        optimizer._adamw.state[p] = retained_adam[n]
    final = evaluate(model, dev, 764 + 9000)
    if (frozen_fingerprint(inherited, optimizer) != frozen_hash or
            digest(a.checkpoint) != ancestor_hash):
        raise AssertionError('protected FA1 inherited tensors/moments mutated')
    saved = save_unpromoted(
        a.output_checkpoint, model, optimizer, a.root, a.checkpoint,
        source_metadata=meta, candidate_trained=True, training_round=ROUND,
        training_updates=chosen['step'], development_gate_passed=False,
        architecture_round=ROUND)
    model2, optimizer2, _, _, meta2 = load_unified(
        a.output_checkpoint, a.root, a.threads)
    all2 = dict(model2.named_parameters())
    if (meta2['architecture_round'] != ROUND or
            len(optimizer2.param_groups) != 37 or
            frozen_fingerprint({n: p for n, p in all2.items()
                                if n in inherited}, optimizer2) != frozen_hash):
        raise AssertionError('inherited optimizer not recoverable')
    for n, p in new.items():
        if (not torch.equal(p, all2[n]) or
                not _deep_equal(optimizer._adamw.state[p],
                                optimizer2._adamw.state[all2[n]])):
            raise AssertionError('R7.64 AdamW continuation not exactly recoverable')
    report = dict(
        round=ROUND, candidate_sha256=saved['sha256'],
        parent_sha256=ancestor_hash, prior_updates=historical_step,
        steps_executed=a.steps, selected_step=chosen['step'],
        optimizer_groups=37, inherited_groups_preserved=36,
        full_checkpoint_reload_exact=True,
        train_positions=len(train['target']),
        dev_positions=len(dev['target']),
        independent_test_opened=False, promoted=False,
        matched_dev=final, dev_history=history,
        corpus_inventory=inventory, elapsed_seconds=time.monotonic() - started)
    Path(a.report).write_text(json.dumps(report, indent=2, sort_keys=True)+'\n')
    print('R764_CONT_SAVED', json.dumps({
        'sha256': saved['sha256'],
        'selected_step': chosen['step'],
        'strict_geometry_gate_passed': final['strict_geometry_gate_passed']}),
        flush=True)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for field in ('root', 'corpus_dir', 'split_metadata', 'checkpoint',
                  'output_checkpoint', 'report'):
        p.add_argument('--' + field.replace('_', '-'), required=True)
    p.add_argument('--threads', type=int, default=4, choices=(1, 2, 4))
    p.add_argument('--train', type=int, default=16384)
    p.add_argument('--dev', type=int, default=2048)
    p.add_argument('--context', type=int, default=512)
    p.add_argument('--steps', type=int, default=2000)
    p.add_argument('--minibatch', type=int, default=16)
    p.add_argument('--eval-every', type=int, default=500)
    p.add_argument('--seed', type=int, default=2764)
    run(p.parse_args())
