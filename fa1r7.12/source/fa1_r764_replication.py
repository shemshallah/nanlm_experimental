#!/usr/bin/env python3
"""Independent-position DEV replication for frozen FA1 R7.64 weights.

This scorer does NO training, selects a new deterministic disjoint set of
target positions across the same 15 independent DEV books, and reports
independent-book bootstrap controls. TEST is never opened. It must not
reselect a checkpoint or change promotion parameters based on the result.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from fa1_r723_train import prepare
from fa1_r727_checkpoint import digest, load_unified
from fa1_r748_cpu_pilot import verify_train_dev_only
from fa1_r753_corpus import balanced_causal_positions, with_successor_history
from fa1_r764_cpu_pilot import ROUND as R764_ROUND, evaluate


def run(a):
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    if Path(a.report).exists():
        raise FileExistsError('replication report cannot be replaced')
    inventory = verify_train_dev_only(a.corpus_dir, a.split_metadata)
    checkpoint_hash = digest(a.checkpoint)
    model, optimizer, base, compositor, meta = load_unified(
        a.checkpoint, a.root, a.threads)
    if (meta.get('architecture_round') != R764_ROUND
            or len(optimizer.param_groups) != 37
            or meta.get('candidate_trained') is not True):
        raise AssertionError('replication requires trained R7.64 / 37 groups')
    args = SimpleNamespace(
        root=a.root, corpus_dir=a.corpus_dir, batch=8,
        r753_position_selector=balanced_causal_positions)
    data = with_successor_history(
        prepare(args, model, base, compositor, 'dev', a.dev, a.seed + 202),
        a.root, 'dev', a.corpus_dir, a.context)
    if int(data['book_ids'].unique().numel()) != 15:
        raise AssertionError('all 15 DEV books are required')
    model.eval()
    metrics = evaluate(model, data, a.seed + 1001)
    assert digest(a.checkpoint) == checkpoint_hash
    report = dict(round='FA1_R7_64_INDEPENDENT_DEV_POSITIONS_V1',
        model_round=R764_ROUND, checkpoint_sha256=checkpoint_hash,
        train_books=inventory['train']['books'], dev_books=15,
        dev_positions=len(data['target']),
        independent_test_accessed=False, optimizer_groups=37,
        new_evaluation_seed=a.seed, metrics=metrics,
        strictly_passed=bool(metrics['strict_geometry_gate_passed']))
    Path(a.report).parent.mkdir(parents=True, exist_ok=True)
    Path(a.report).write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print('R764_REPLICATION', json.dumps({
        'candidate_nll': metrics['candidate_nll'],
        'geometry_over_lexical': metrics['geometry_over_lexical_matched'],
        'correct_history': metrics['correct_vs_wrong_older'],
        'strict_gate': report['strictly_passed'],
        'positions': len(data['target'])}), flush=True)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for field in ('root', 'corpus_dir', 'split_metadata',
                  'checkpoint', 'report'):
        p.add_argument('--' + field.replace('_', '-'), required=True)
    p.add_argument('--threads', type=int, default=4, choices=(1, 2, 4))
    p.add_argument('--dev', type=int, default=4096)
    p.add_argument('--context', type=int, default=512)
    p.add_argument('--seed', type=int, default=1764)
    run(p.parse_args())
