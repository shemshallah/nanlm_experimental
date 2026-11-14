#!/usr/bin/env python3
"""One-shot, post-selection independent-book TEST of an R7.37 candidate.

Do not run while choosing hyperparameters or checkpoint steps. The sealed
test is evaluated only after the DEV-selected state has been saved, against
the original R7.33 checkpoint migrated with identity-initialized new routes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from fa1_r723_train import forward, prepare
from fa1_r724_train import boot_books
from fa1_r727_checkpoint import digest, load_unified
from fa1_r727_eval import validate_book_corpus
from fa1_r728_geometric_memory import geometry_diagnostic
from fa1_r732_geometry_calibration import load_calibrated
from fa1_r735_history_values import attach_history_values
from fa1_r736_deep_wiring import attach_deep_wiring
from fa1_r737_feedback_closure import attach_feedback_closure, ROUND


def run(args):
    if Path(args.reference).resolve()==Path(args.candidate).resolve():
        raise ValueError('reference and candidate must be different checkpoints')
    if Path(args.output).resolve() in (Path(args.reference).resolve(),
                                      Path(args.candidate).resolve()):
        raise ValueError('TEST report may not overwrite a checkpoint')
    inventory=validate_book_corpus(args.corpus_dir,'test')
    torch.set_num_threads(args.threads)
    reference_hash=digest(args.reference)
    candidate_hash=digest(args.candidate)

    parent,oldopt,base,comp,reference_meta=load_calibrated(
        args.reference,args.root,args.threads,with_apex_feedback=True)
    if reference_meta.get('architecture_round')!='FA1_R7_32_CALIBRATED_DIAMOND_V1':
        raise ValueError('expected known R7.33/R7.32 reference model')
    attach_history_values(parent,oldopt)
    attach_deep_wiring(parent,oldopt)
    attach_feedback_closure(parent,oldopt)
    candidate,optimizer,base2,comp2,meta=load_unified(
        args.candidate,args.root,args.threads)
    if meta.get('architecture_round')!=ROUND or len(optimizer.param_groups)!=21:
        raise ValueError('not a complete 21-group R7.37 candidate')
    if meta.get('source_checkpoint_sha256') != reference_hash:
        raise ValueError('candidate was not continued from this exact reference')
    if meta.get('promoted') is not False:
        raise ValueError('candidate promotion state was altered')

    sample_args=argparse.Namespace(root=args.root,corpus_dir=args.corpus_dir,
                                   batch=args.batch)

    with torch.no_grad():
        data=prepare(sample_args,candidate,base2,comp2,'test',args.positions,args.seed)
        data['layered_byte_supervision']=True
        old_word,_=forward(parent,data)
        new_word,_=forward(candidate,data)
        initial=geometry_diagnostic(parent,data,args.seed+1,True,base)
        successor=geometry_diagnostic(candidate,data,args.seed+1,True,base2)
        delta=old_word-new_word
        gain_books=boot_books(delta,data['book_ids'],args.seed+11)
    report={
        'format':'FA1_R7_37_ONE_SHOT_SEALED_TEST_V1',
        'reference_sha256':reference_hash,
        'candidate_sha256':candidate_hash,
        'candidate_model_architecture':meta['architecture_round'],
        'candidate_selection':'precommitted development checkpoint; TEST not used for selection',
        'source_architecture':'R7.33 candidate, saved as R7.32 calibrated decoder',
        'book_disjoint_within_this_round':True,
        'prior_checkpoint_corpus_overlap_verified':False,
        'corpus_inventory':inventory,
        'test_positions':args.positions,
        'tested_independent_books':int(data['book_ids'].unique().numel()),
        'tested_far_eligible_books':int(data['book_ids'][
            data['memory_modal_mask'][:,8:].any(1)].unique().numel()),
        'reference':initial,'candidate':successor,
        'full_word_nll_gain_nats_per_word':float(delta.mean()),
        'full_word_nll_gain_book_bootstrap':gain_books,
        'test_used_for_training_or_checkpoint_selection':False,
        'reference_file_unchanged':digest(args.reference)==reference_hash,
        'candidate_file_unchanged':digest(args.candidate)==candidate_hash,
        'promotion_authorized':False,
        'word_top1_top5_scope':'inherited trunk only; full mixture top-k unavailable',
        'byte_bpb_scope':'conditional UTF-8 word spelling, not continuous raw text',
    }
    path=Path(args.output)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2)+'\n')
    print('SEALED_TEST '+json.dumps({
        'word_loss_reference':initial['full_word_nll'],
        'word_loss_candidate':successor['full_word_nll'],
        'word_gain_nats_per_word':report['full_word_nll_gain_nats_per_word'],
        'word_gain_book_ci95':gain_books['book_bootstrap_ci95'],
        'candidate_spelling_bpb':successor['conditional_spelling_bits_per_byte'],
        'test_books':report['tested_independent_books'],
        'far_eligible_books':report['tested_far_eligible_books']}),flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for flag in ('reference','candidate','root','corpus-dir','output'):
        p.add_argument('--'+flag,required=True)
    p.add_argument('--positions',type=int,default=96)
    p.add_argument('--seed',type=int,default=737)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--batch',type=int,default=8)
    run(p.parse_args())
