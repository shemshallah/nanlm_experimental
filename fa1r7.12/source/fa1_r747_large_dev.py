#!/usr/bin/env python3
"""R7.47 matched independent-book DEV evaluation against its R7.45 parent."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from fa1_r723_train import prepare
from fa1_r727_checkpoint import load_unified,digest
from fa1_r727_eval import validate_book_corpus
from fa1_r744_cpu_pilot import foreign_word_history,bootstrap_by_book
from fa1_r746_factorial_dev import perturbed_context
from fa1_r747_cpu_pilot import evaluate
from fa1_r747_head_address_copy import ROUND


def run(a):
    if Path(a.output).exists():
        raise FileExistsError('independent DEV output already exists')
    started=time.monotonic()
    torch.set_num_threads(a.threads)
    inventory=validate_book_corpus(a.corpus_dir,'dev')
    model,opt,base,comp,meta=load_unified(a.candidate,a.root,a.threads)
    if (meta['architecture_round']!=ROUND or len(opt.param_groups)!=27 or
            meta['source_checkpoint_sha256']!=digest(a.parent)):
        raise ValueError('candidate must be a single-lineage R7.47 continuation')
    dev=prepare(a,model,base,comp,'dev',a.targets,a.seed)
    if int(torch.unique(dev['book_ids']).numel())!=15:
        raise AssertionError('independent comparison must include all 15 DEV books')
    bad_history,_=foreign_word_history(dev,a.seed+5)
    wrong_address=perturbed_context(dev,'addresses_only',a.seed+6)
    result=evaluate(model,dev,bad_history,wrong_address,a.seed+21)
    # Candidate has inherited R7.45 weights unchanged, but independently
    # loading that checkpoint still verifies real source recovery and parity.
    parent,parent_optimizer,_,_,parent_meta=load_unified(a.parent,a.root,a.threads)
    if (parent_meta['architecture_round']!='FA1_R7_45_GEOMETRIC_VALUE_READ_V1' or
            len(parent_optimizer.param_groups)!=26):
        raise ValueError('parent must be the protected complete R7.45 model')
    from fa1_r723_train import forward
    with torch.no_grad():
        original,_=forward(parent,dev)
        predicted,_=forward(model,dev)
    word_delta=original-predicted
    if abs(float(word_delta.mean())-result['r747_incremental_word_gain'])>2.e-6:
        raise AssertionError('R7.47 ablation disagrees with reloaded R7.45 parent')
    result['independently_reloaded_parent_nll']=float(original.mean())
    result['independently_reloaded_word_gain']=float(word_delta.mean())
    result['independently_reloaded_word_book_bootstrap']=bootstrap_by_book(
        word_delta,dev['book_ids'],a.seed+50)
    payload={
        'round':'FA1_R7_47_LARGE_BOOK_DISJOINT_DEV_V1',
        'parent_sha256':digest(a.parent),
        'candidate_sha256':digest(a.candidate),
        'model_parameters_updated_during_evaluation':False,
        'independent_test_opened':False,
        'candidate_promoted':False,
        'dev_positions':a.targets,'dev_books':15,
        'corpus_inventory':inventory,'metrics':result,
        'elapsed_seconds':time.monotonic()-started,
    }
    out=Path(a.output)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(payload,sort_keys=True,indent=2)+'\n')
    print('R747 LARGE',json.dumps({
        'word_gain':result['r747_incremental_word_gain'],
        'address_specificity':result['r747_address_specificity'],
        'pair_specificity':result['r747_pair_specificity'],
        'strict_gate':result['r747_strict_dev_gate_passed'],
        'seconds':round(payload['elapsed_seconds'],1)}),flush=True)
    return payload


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',required=True)
    p.add_argument('--corpus-dir',required=True)
    p.add_argument('--parent',required=True)
    p.add_argument('--candidate',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--targets',type=int,default=2048)
    p.add_argument('--batch',type=int,default=8)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--seed',type=int,default=811)
    run(p.parse_args())
