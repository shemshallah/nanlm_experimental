#!/usr/bin/env python3
"""Read-only, explicitly exploratory exact-pair coefficient stress test.

This sweep is performed on development books and cannot establish promotion.
It probes whether the canonical addresses contain predictive information at
all before allocating further training to their coefficient.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from fa1_r723_train import prepare, forward
from fa1_r727_checkpoint import load_unified, digest
from fa1_r746_factorial_dev import perturbed_context
from fa1_r748_cpu_pilot import verify_train_dev_only
from fa1_r749_address_reliability import ROUND, address_reliability_control


def run(a):
    if Path(a.output).exists():
        raise FileExistsError('coefficient sweep already exists')
    torch.set_num_threads(a.threads)
    verify_train_dev_only(a.corpus_dir, a.split_metadata)
    model, opt, base, comp, meta = load_unified(a.checkpoint, a.root, a.threads)
    if meta['architecture_round'] != ROUND or len(opt.param_groups) != 29:
        raise ValueError('genuine R7.49 checkpoint required')
    dev = prepare(a, model, base, comp, 'dev', a.targets, a.seed)
    if int(torch.unique(dev['book_ids']).numel()) != 15:
        raise AssertionError('15 DEV books required')
    wrong = perturbed_context(dev, 'addresses_only', a.seed + 29)
    m = model.word_backoff
    old = m.mirror_exact_address_gain.detach().clone()
    row = []
    try:
        model.eval()
        with torch.no_grad():
            with address_reliability_control(model, 'without_new_address'):
                parent, _ = forward(model, dev)
                wrong_parent, _ = forward(model, wrong)
            for x in a.exact_gains:
                m.mirror_exact_address_gain.fill_(x)
                good, _ = forward(model, dev)
                other, _ = forward(model, wrong)
                delta = (parent - good) - (wrong_parent - other)
                row.append({'exact_address_gain': x, 'nll': float(good.mean()),
                            'new_address_delta': float((parent - good).mean()),
                            'canonical_address_specificity': float(delta.mean())})
    finally:
        with torch.no_grad():
            m.mirror_exact_address_gain.copy_(old)
    doc = {'round': ROUND + '_EXPLORATORY_DEV_SWEEP',
           'checkpoint_sha256': digest(a.checkpoint),
           'book_count': 15, 'positions': a.targets, 'independent_test_opened': False,
           'chosen_checkpoint_unchanged': True, 'model_promoted': False,
           'original_exact_address_gain': float(old), 'sweep': row,
           'interpretation': 'A DEV hyperparameter sweep; never a confirmatory significance test'}
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(json.dumps(doc, indent=2, sort_keys=True)+'\n')
    for x in row:
        print('SWEEP', x, flush=True)
    return doc


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--split-metadata', required=True)
    p.add_argument('--corpus-dir', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--targets', type=int, default=512)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--seed', type=int, default=913)
    p.add_argument('--exact-gains', type=float, nargs='+',
                   default=[-.75, -.3, -.1, -.05, 0., .05, .1, .3, .75])
    run(p.parse_args())
