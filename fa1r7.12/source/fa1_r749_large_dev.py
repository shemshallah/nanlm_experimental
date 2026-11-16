#!/usr/bin/env python3
"""Full independent-reload R7.48 -> R7.49 comparison on TRAIN-disjoint DEV.

Loads one optimizer at a time to stay inside constrained CPU memory. The
independent TEST data are never evaluated; this is a nonpromotional audit.
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
from fa1_r748_mirror_pair_values import ROUND as R748
from fa1_r748_mirror_pair_values import pair_indices
from fa1_r748_cpu_pilot import verify_train_dev_only
from fa1_r749_address_reliability import ROUND
from fa1_r749_cpu_pilot import evaluate


def run(a):
    if Path(a.output).exists():
        raise FileExistsError('R7.49 DEV audit is immutable')
    torch.set_num_threads(a.threads)
    start = time.monotonic()
    inventory = verify_train_dev_only(a.corpus_dir, a.split_metadata)
    model, optimizer, base, comp, meta = load_unified(a.candidate, a.root, a.threads)
    if meta['architecture_round'] != ROUND or len(optimizer.param_groups) != 29 or meta['promoted']:
        raise ValueError('not the full R7.49 unpromoted checkpoint')
    dev = prepare(a, model, base, comp, 'dev', a.targets, a.seed)
    if int(torch.unique(dev['book_ids']).numel()) != 15:
        raise AssertionError('all 15 DEV books required')
    controls = evaluate(model, dev, a.seed + 100)
    with torch.no_grad():
        t = time.monotonic()
        model.eval()
        candidate, extras = forward(model, dev)
        speed = a.targets / max(time.monotonic() - t, 1.e-10)
        first_byte_logits = extras['byte_logits'].float()
        first_byte_target = dev['first_byte_target']
        first_byte_top5 = first_byte_logits.topk(5, -1).indices
        aux = {'nll': float(F.cross_entropy(first_byte_logits, first_byte_target)),
               'top1': float((first_byte_top5[:, 0] == first_byte_target).float().mean()),
               'top5': float((first_byte_top5 == first_byte_target[:, None]).any(-1).float().mean())}
        ids = extras['backoff_candidate_ids']
        copied = ((ids == dev['target'][:, None]) &
                  (extras['backoff_candidate_weights'] > 0)).any(-1)
        active = dev['vertex_ids'].long()
        edges = torch.triu_indices(8, 8, 1)
        pair_ids, _ = pair_indices(active[:, edges[0]], active[:, edges[1]])
        word_heads = model.word_backoff.word_head_lookup[ids.long()]
        candidate_pair, valid = pair_indices(word_heads[..., 0], word_heads[..., 1])
        exact_matches = ((candidate_pair[:, :, None] == pair_ids[:, None, :]).any(-1)
                         & valid & (extras['backoff_candidate_weights'] > 0))
        gold_exact = ((ids == dev['target'][:, None]) & exact_matches).any(-1)
    # The R7.48 checkpoint was NOT mutated in place. Independently reload it
    # after releasing the candidate/optimizer to establish the real baseline.
    del model, optimizer, base, comp, meta, extras
    gc.collect()
    parent, previous_optimizer, _, _, previous = load_unified(a.parent_r748, a.root, a.threads)
    if previous['architecture_round'] != R748 or len(previous_optimizer.param_groups) != 28:
        raise ValueError('R7.48 parent lineage is not authentic')
    parent.eval()
    with torch.no_grad():
        baseline, _ = forward(parent, dev)
    improvement = baseline - candidate
    books = dev['book_ids']
    by_causal_copy = {}
    for name, mask in (('gold_word_in_history', copied),
                       ('gold_word_absent_from_history', ~copied),
                       ('gold_pair_exactly_in_active_28', gold_exact),
                       ('copied_gold_without_exact_pair', copied & ~gold_exact)):
        by_causal_copy[name] = {
            'positions': int(mask.sum()),
            'fraction': float(mask.float().mean()),
            'parent_word_nll': float(baseline[mask].mean()) if bool(mask.any()) else None,
            'candidate_word_nll': float(candidate[mask].mean()) if bool(mask.any()) else None,
            'gain': float(improvement[mask].mean()) if bool(mask.any()) else None}
    report = {
        'round': ROUND + '_INDEPENDENT_RELOAD_DEV', 'branch': 'master',
        'promoted': False, 'independent_test_opened': False,
        'candidate_sha256': digest(a.candidate),
        'parent_r748_sha256': digest(a.parent_r748),
        'optimizer_count': 1, 'optimizer_groups': 29,
        'dev_books': 15, 'dev_positions': a.targets, 'inventory': inventory,
        'independently_reloaded_r748_nll': float(baseline.mean()),
        'independently_reloaded_r749_nll': float(candidate.mean()),
        'real_r748_to_r749_gain': float(improvement.mean()),
        'real_gain_book_bootstrap': bootstrap_by_book(improvement, books, a.seed + 999),
        'first_utf8_byte_auxiliary': aux,
        'true_raw_byte_bpb': None, 'full_word_top1': None, 'full_word_top5': None,
        'next_plus2_plus3_horizons': None,
        'scoring_word_positions_per_second': speed,
        'matched_causal_controls': controls,
        'gold_history_address_subgroups': by_causal_copy,
        'per_dev_book': [{'book': int(book), 'positions': int((books == book).sum()),
                          'gain_nats_per_word': float(improvement[books == book].mean())}
                         for book in torch.unique(books)],
        'elapsed_seconds': time.monotonic() - start,
        'limits': ['same DEV split used for earlier FA1 research',
                   'no sealed independent TEST evaluation',
                   'not a full raw-text byte probability distribution']}
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(json.dumps(report, sort_keys=True, indent=2) + '\n')
    print('R749 RELOADED DEV', json.dumps({
        'r748': report['independently_reloaded_r748_nll'],
        'r749': report['independently_reloaded_r749_nll'],
        'gain': report['real_r748_to_r749_gain'],
        'ci': report['real_gain_book_bootstrap']['book_bootstrap_ci95'],
        'addr': controls['canonical_address_delta'],
        'sign': controls['total_signed_rail_contribution'],
        'strict': controls['strict_dev_gate_passed']}), flush=True)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--split-metadata', required=True)
    p.add_argument('--corpus-dir', required=True)
    p.add_argument('--parent-r748', required=True)
    p.add_argument('--candidate', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--targets', type=int, default=512)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--seed', type=int, default=913)
    run(p.parse_args())
