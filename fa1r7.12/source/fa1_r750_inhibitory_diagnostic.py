#!/usr/bin/env python3
"""TRAIN-only exact-address copy/exclusion diagnostic for R7.50 design.

All pair features are target-free; targets are inspected only for TRAIN
statistics. DEV/TEST are not opened by this diagnostic.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from fa1_r723_train import prepare, forward
from fa1_r727_checkpoint import load_unified, digest
from fa1_r748_mirror_pair_values import pair_indices
from fa1_r748_cpu_pilot import verify_train_dev_only
from fa1_r749_address_reliability import ROUND


def summarized(model, data):
    with torch.no_grad():
        _, extras = forward(model, data)
        ids = extras['backoff_candidate_ids']
        available = extras['backoff_candidate_weights'] > 0
        gold = (ids == data['target'][:, None]) & available
        active = data['vertex_ids'].long()
        edge = torch.triu_indices(8, 8, 1)
        active_pair, _ = pair_indices(active[:, edge[0]], active[:, edge[1]])
        heads = model.word_backoff.word_head_lookup[ids.long()]
        pair, valid = pair_indices(heads[..., 0], heads[..., 1])
        hit = (pair[:, :, None] == active_pair[:, None, :]).any(-1) & valid & available
        partial = ((heads[..., None] == active[:, None, None, :]).any(-1).sum(-1) == 1) & valid & available
        counts = {'candidate_positions': int(available.sum()),
                  'gold_positions': int(gold.sum()),
                  'exact_candidate_positions': int(hit.sum()),
                  'exact_gold_positions': int((hit & gold).sum()),
                  'partial_candidate_positions': int(partial.sum()),
                  'partial_gold_positions': int((partial & gold).sum()),
                  'examples': int(len(ids)),
                  'copy_eligible_examples': int(gold.any(1).sum())}
        for name, mask in [('exact', hit), ('not_exact', available & ~hit),
                           ('partial', partial), ('not_partial', available & ~partial)]:
            eligible = int(mask.sum())
            target = int((gold & mask).sum())
            counts[name+'_gold_fraction'] = (target / eligible if eligible else None)
        q = counts['exact_gold_fraction']
        r = counts['not_exact_gold_fraction']
        counts['exact_vs_nonexact_gold_risk_ratio'] = (q / r if q is not None and r else None)
        counts['smoothed_log_ratio'] = math.log((counts['exact_gold_positions']+1) /
                                            (counts['exact_candidate_positions']+2)) - math.log(
                                            (counts['gold_positions']-counts['exact_gold_positions']+1) /
                                            (counts['candidate_positions']-counts['exact_candidate_positions']+2))
        return counts


def run(a):
    if Path(a.output).exists():
        raise FileExistsError('TRAIN diagnostic report must not be overwritten')
    torch.set_num_threads(a.threads)
    inventory = verify_train_dev_only(a.corpus_dir, a.split_metadata)
    model, optimizer, base, comp, meta = load_unified(a.checkpoint, a.root, a.threads)
    if meta['architecture_round'] != ROUND or len(optimizer.param_groups) != 29:
        raise ValueError('R7.49 current 29-group checkpoint required')
    sample = prepare(a, model, base, comp, 'train', a.targets, a.seed)
    model.eval()
    stats = summarized(model, sample)
    sweep = []
    coefficient = model.word_backoff.mirror_exact_address_gain
    original = coefficient.detach().clone()
    try:
        with torch.no_grad():
            for value in (-.6, -.3, -.1, -.05, 0., .05, .1, .3, .6):
                coefficient.fill_(value)
                word_loss, _ = forward(model, sample)
                sweep.append({'exact_address_gain': value,
                              'train_word_nll':float(word_loss.mean())})
    finally:
        with torch.no_grad():
            coefficient.copy_(original)
    report = {'round':'FA1_R7_50_TRAIN_ONLY_PAIR_INHIBITION_DIAGNOSTIC',
              'r749_source_sha256': digest(a.checkpoint),
              'corpus_inventory': inventory, 'sample_train_positions':a.targets,
              'independent_test_opened':False, 'dev_content_accessed':False,
              'checkpoint_modified':False, 'promoted':False,
              'stats':stats, 'train_coefficient_sweep':sweep}
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    Path(a.output).write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    print('R750 TRAIN',json.dumps(stats),flush=True)
    print('R750 TRAIN SWEEP',json.dumps(sweep),flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',required=True)
    p.add_argument('--split-metadata',required=True)
    p.add_argument('--corpus-dir',required=True)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--targets',type=int,default=4096)
    p.add_argument('--batch',type=int,default=8)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--seed',type=int,default=750)
    run(p.parse_args())
