#!/usr/bin/env python3
"""Larger independent DEV-book R7.44 verification, never reading TEST labels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from fa1_r723_train import prepare, forward
from fa1_r725_controls import different_book_donors
from fa1_r727_checkpoint import digest, load_unified
from fa1_r727_eval import validate_book_corpus
from fa1_r744_cpu_pilot import bootstrap_by_book, foreign_word_history


def run(args):
    if Path(args.output).exists():
        raise FileExistsError('will not overwrite larger validation')
    torch.set_num_threads(args.threads)
    begin=time.monotonic()
    inventory=validate_book_corpus(args.corpus_dir,'dev')
    model,optim,base,comp,meta=load_unified(args.checkpoint,args.root,args.threads)
    if (meta['architecture_round']!='FA1_R7_44_CAUSAL_WORD_BACKOFF_V1' or
            len(optim.param_groups)!=25 or meta.get('promoted') is not False):
        raise ValueError('expected unpromoted R7.44 with one 25-group optimizer')
    data=prepare(args,model,base,comp,'dev',args.targets,args.seed)
    if len(torch.unique(data['book_ids']))!=15:
        raise ValueError('large DEV must include all 15 independent books')
    foreign,donors=foreign_word_history(data,args.seed+10)
    geometry_donors=different_book_donors(data['book_ids'],args.seed+20)
    wrong_geo=dict(data)
    # Exchange *only* the target-free active geometric addresses and context
    # to test whether these condition the word-backoff benefit, not its copy Q.
    for key in ('vertex_ids','byte_ids','sentence_ids'):
        wrong_geo[key]=data[key][geometry_donors]
    model.eval()
    with torch.no_grad():
        good,extras=forward(model,data)
        copied_from_foreign,foreign_ex=forward(model,foreign)
        wrong_geometry,wrong_geometry_ex=forward(model,wrong_geo)
        no_apex,no_apex_ex=forward(model,data,'no_apex')
    parent=extras['backoff_parent_nll']
    gain=parent-good
    history=copied_from_foreign-good
    true_geometry_gain=gain-(wrong_geometry_ex['backoff_parent_nll']-wrong_geometry)
    apex_gain=gain-(no_apex_ex['backoff_parent_nll']-no_apex)
    result={'round':'FA1_R7_44_LARGE_BOOK_DISJOINT_DEV_V1',
            'checkpoint_sha256':digest(args.checkpoint),
            'source_parent_sha256':meta['source_checkpoint_sha256'],
            'optimizer_groups':len(optim.param_groups),
            'accepted_step_protected':14416,
            'independent_test_opened':False,
            'evaluated_split':'dev', 'corpus_inventory':inventory,
            'positions':len(data['target']),'independent_books':int(torch.unique(data['book_ids']).numel()),
            'target_free_geometry_donor_control':True,
            'full_word_nll_parent':float(parent.mean()),
            'full_word_nll_candidate':float(good.mean()),
            'word_gain_nats_per_word':float(gain.mean()),
            'word_gain_book_bootstrap':bootstrap_by_book(gain,data['book_ids'],args.seed+30),
            'correct_vs_foreign_word_history':float(history.mean()),
            'correct_vs_foreign_bootstrap':bootstrap_by_book(history,data['book_ids'],args.seed+40),
            'true_vs_foreign_geometric_addresses_specificity':float(true_geometry_gain.mean()),
            'geometric_addresses_bootstrap':bootstrap_by_book(true_geometry_gain,data['book_ids'],args.seed+50),
            'apex_geometry_specificity':float(apex_gain.mean()),
            'apex_bootstrap':bootstrap_by_book(apex_gain,data['book_ids'],args.seed+60),
            'mean_backoff_weight':float(extras['backoff_alpha'].mean()),
            'history_copy_target_present_fraction':float((extras['backoff_target_probability']>0).float().mean()),
            'raw_text_bits_per_byte':None,
            'whole_model_normalized_top5':None,
            'full_promotion':False,
            'elapsed_seconds':time.monotonic()-begin}
    result['r744_candidate_only_dev_gate_passed']=bool(
        result['word_gain_book_bootstrap']['book_bootstrap_ci95'][0]>0 and
        result['correct_vs_foreign_bootstrap']['book_bootstrap_ci95'][0]>0 and
        result['geometric_addresses_bootstrap']['book_bootstrap_ci95'][0]>0 and
        result['apex_bootstrap']['book_bootstrap_ci95'][0]>0)
    path=Path(args.output)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps(result,indent=2,sort_keys=True),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',required=True)
    p.add_argument('--corpus-dir',required=True)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--targets',type=int,default=2048)
    p.add_argument('--batch',type=int,default=8)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--seed',type=int,default=811)
    run(p.parse_args())
