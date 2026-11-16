#!/usr/bin/env python3
"""Checkpoint-reloaded 15-book DEV comparison and geometric attribution.

No TEST data or TEST book content is accessed. The full word NLL is measured;
first-byte accuracy is auxiliary, not full byte top-k or raw-text BPB.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from fa1_r723_train import prepare, forward
from fa1_r727_checkpoint import load_unified, digest
from fa1_r744_cpu_pilot import bootstrap_by_book
from fa1_r747_head_address_copy import ROUND as R747
from fa1_r748_mirror_pair_values import ROUND, pair_value_control
from fa1_r748_cpu_pilot import evaluate, verify_train_dev_only


def run(a):
    if Path(a.output).exists():
        raise FileExistsError('DEV audit output already exists')
    start = time.monotonic()
    torch.set_num_threads(a.threads)
    inventory = verify_train_dev_only(a.corpus_dir, a.split_metadata)
    model, optimizer, base, comp, metadata = load_unified(a.candidate, a.root, a.threads)
    if (metadata['architecture_round'] != ROUND or metadata['promoted'] is not False or
            len(optimizer.param_groups) != 28):
        raise ValueError('expected genuine unpromoted 28-group R7.48 candidate')
    dev = prepare(a, model, base, comp, 'dev', a.targets, a.seed)
    if int(torch.unique(dev['book_ids']).numel()) != 15:
        raise AssertionError('all 15 DEV books required')
    result = evaluate(model, dev, a.seed + 100)
    with torch.no_grad():
        begin = time.monotonic()
        predicted, extras = forward(model, dev)
        scoring_seconds = time.monotonic() - begin
        logits = extras['byte_logits'].float()
        byte_top5 = logits.topk(5, dim=-1).indices
        first_byte_targets = dev['first_byte_target']
        byte_loss = F.cross_entropy(logits, first_byte_targets)
    # An independent load of R7.47 is a stronger parity check than merely
    # disabling the new gates on the already-loaded R7.48 object.
    with pair_value_control(model), torch.no_grad():
        gated_parent, _ = forward(model, dev)
    # Free the entire recovered candidate before restoring the independent
    # ancestor. Both complete optimizers resident together can exceed a
    # constrained CPU runner's memory allowance.
    del model, optimizer, base, comp, metadata, extras
    gc.collect()
    inherited, old_optimizer, _, _, old_meta = load_unified(a.parent_r747, a.root, a.threads)
    if old_meta['architecture_round'] != R747 or len(old_optimizer.param_groups) != 27:
        raise ValueError('original R7.47 source missing')
    inherited.eval()
    with torch.no_grad():
        original, _ = forward(inherited, dev)
    if not torch.equal(original, gated_parent):
        raise AssertionError('R7.48 gate-disabled scores differ from reloaded R7.47')
    if not bool(torch.isfinite(original).all() and torch.isfinite(predicted).all()):
        raise AssertionError('nonfinite scores')
    if abs(float((original - predicted).mean()) - result['gain']) > 2.e-6:
        raise AssertionError('gated R7.47 control differs from independent checkpoint')
    difference = original - predicted
    books = dev['book_ids']
    per_book = []
    for book in torch.unique(books):
        selected = books == book
        per_book.append({'book_id': int(book), 'positions': int(selected.sum()),
                         'word_nll_gain': float(difference[selected].mean())})
    payload = {
        'round': 'FA1_R7_48_FULL_15_BOOK_DEV_V1', 'branch': 'master',
        'independent_test_opened': False, 'promoted': False,
        'candidate_sha256': digest(a.candidate),
        'r747_ancestor_sha256': digest(a.parent_r747),
        'one_optimizer_groups': 28, 'dev_positions': a.targets,
        'dev_books': 15, 'inventory': inventory,
        'independent_r747_parent_nll': float(original.mean()),
        'independent_r748_candidate_nll': float(predicted.mean()),
        'independent_incremental_word_gain': float(difference.mean()),
        'independent_gain_book_bootstrap': bootstrap_by_book(difference, books, a.seed + 199),
        'first_utf8_byte_auxiliary_nll_nats': float(byte_loss),
        'first_utf8_byte_auxiliary_top1': float((byte_top5[:, 0] == first_byte_targets).float().mean()),
        'first_utf8_byte_auxiliary_top5': float((byte_top5 == first_byte_targets[:, None]).any(-1).float().mean()),
        'raw_byte_bpb': None, 'full_word_top1': None, 'full_word_top5': None,
        'other_prediction_horizons': None,
        'raw_byte_metric_reason': 'FA1 has target-conditioned tree/retrieval scores, not a fully materialized byte-stream distribution',
        'candidate_word_positions_per_second': float(a.targets / max(scoring_seconds, 1.e-8)),
        'matched_counterfactual_metrics': result, 'per_dev_book': per_book,
        'elapsed_seconds': time.monotonic() - start,
        'limits': ['Exploratory repeated DEV sampling, not confirmatory evidence',
                   'TEST sealed', 'Earlier R7 ancestor overlap unknown'],
    }
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, sort_keys=True, indent=2) + '\n')
    print('R748 FULL DEV', json.dumps({'gain': payload['independent_incremental_word_gain'],
          'ci95': payload['independent_gain_book_bootstrap']['book_bootstrap_ci95'],
          'address': result['address'], 'bank': result['pair_value_bank'],
          'sign': result['signed_orientation'], 'strict': result['strict_dev_gate_passed']}), flush=True)
    return payload


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--split-metadata', required=True)
    p.add_argument('--corpus-dir', required=True)
    p.add_argument('--parent-r747', required=True)
    p.add_argument('--candidate', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--targets', type=int, default=2048)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--seed', type=int, default=811)
    run(p.parse_args())
