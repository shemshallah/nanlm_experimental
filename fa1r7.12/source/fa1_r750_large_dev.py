#!/usr/bin/env python3
"""Independently reload R7.49/R7.50 on matched 15-book DEV samples."""
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
from fa1_r748_cpu_pilot import verify_train_dev_only
from fa1_r749_address_reliability import ROUND as R749
from fa1_r750_calibrated_inhibition import ROUND, inhibition_control
from fa1_r750_cpu_pilot import evaluate


def run(a):
    if Path(a.output).exists():
        raise FileExistsError('R7.50 audit is immutable')
    torch.set_num_threads(a.threads)
    t0=time.monotonic()
    inventory=verify_train_dev_only(a.corpus_dir,a.split_metadata)
    model,opt,base,comp,meta=load_unified(a.candidate,a.root,a.threads)
    if meta['architecture_round']!=ROUND or len(opt.param_groups)!=30 or meta['promoted']:
        raise ValueError('not an unpromoted R7.50 complete checkpoint')
    dev=prepare(a,model,base,comp,'dev',a.targets,a.seed)
    if int(torch.unique(dev['book_ids']).numel())!=15:
        raise AssertionError('every DEV book required')
    measures=evaluate(model,dev,a.seed+100)
    with torch.no_grad():
        model.eval()
        started=time.monotonic()
        full,extras=forward(model,dev)
        throughput=a.targets/max(time.monotonic()-started,1.e-8)
        first=extras['byte_logits'].float()
        rank=first.topk(5,dim=-1).indices
        gold=dev['first_byte_target']
        byte_aux={'nll':float(F.cross_entropy(first,gold)),
                  'top1':float((rank[:,0]==gold).float().mean()),
                  'top5':float((rank==gold[:,None]).any(-1).float().mean())}
    with inhibition_control(model),torch.no_grad():
        disabled,_=forward(model,dev)
    del model,opt,base,comp,meta,extras
    gc.collect()
    previous,oldopt,_,_,oldmeta=load_unified(a.parent_r749,a.root,a.threads)
    if oldmeta['architecture_round']!=R749 or len(oldopt.param_groups)!=29:
        raise ValueError('genuine R7.49 parent missing')
    previous.eval()
    with torch.no_grad():
        parent,_=forward(previous,dev)
    if not torch.equal(disabled,parent):
        raise AssertionError('R7.50 disabled pathway does not recover exact R7.49 predictions')
    improvement=parent-full
    books=dev['book_ids']
    report={'round':ROUND+'_INDEPENDENT_RELOAD_DEV',
            'branch':'master','promoted':False,'independent_test_opened':False,
            'candidate_sha256':digest(a.candidate),
            'parent_r749_sha256':digest(a.parent_r749),
            'optimizer_count':1,'optimizer_groups':30,
            'dev_books':15,'dev_positions':a.targets,'corpus_inventory':inventory,
            'independent_parent_r749_nll':float(parent.mean()),
            'independent_r750_word_nll':float(full.mean()),
            'independent_word_nll_gain':float(improvement.mean()),
            'word_nll_gain_whole_book_bootstrap':bootstrap_by_book(improvement,books,a.seed+199),
            'first_utf8_byte_auxiliary':byte_aux,
            'raw_byte_bpb':None,'full_word_top1':None,'full_word_top5':None,
            'next_plus2_plus3_horizons':None,
            'scoring_positions_per_second':throughput,
            'matched_causal_controls':measures,
            'positive_dev_books':sum(float(improvement[books==b].mean())>0
                                     for b in torch.unique(books)),
            'elapsed_seconds':time.monotonic()-t0,
            'limitations':['same DEV books used in prior exploratory rounds',
                           'no independent TEST or long-context evaluation',
                           'first UTF-8 byte auxiliary is not raw-text byte BPB']}
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    Path(a.output).write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    print('R750 RELOADED DEV',json.dumps({
        'parent':report['independent_parent_r749_nll'],
        'r750':report['independent_r750_word_nll'],
        'gain':report['independent_word_nll_gain'],
        'ci':report['word_nll_gain_whole_book_bootstrap']['book_bootstrap_ci95'],
        'address':measures['correct_minus_wrong_address_specificity'],
        'bank':measures['learned_inhibition_row_permutation'],
        'strict':measures['strict_dev_gate_passed']}),flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',required=True)
    p.add_argument('--split-metadata',required=True)
    p.add_argument('--corpus-dir',required=True)
    p.add_argument('--parent-r749',required=True)
    p.add_argument('--candidate',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--targets',type=int,default=512)
    p.add_argument('--batch',type=int,default=8)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--seed',type=int,default=913)
    run(p.parse_args())
