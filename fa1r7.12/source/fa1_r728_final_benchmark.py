#!/usr/bin/env python3
"""One fixed, read-only book-disjoint R7.27 vs R7.28 comparison.

This evaluator has no optimizer step, no parameter sweep, no promotion and no
candidate selection. A TEST report is valid only after the DEV candidate has
been frozen. Full-model word ranking and raw-byte BPB remain unavailable.
"""
import argparse
import gc
import json
from pathlib import Path

import torch

from fa1_r723_train import forward, prepare
from fa1_r724_train import boot_books
from fa1_r727_checkpoint import digest, load_unified
from fa1_r727_eval import evaluate, validate_book_corpus
from fa1_r728_geometric_memory import geometry_diagnostic


@torch.no_grad()
def run(args):
    if args.split not in ('dev','test'):
        raise ValueError('expected book-disjoint development or final sealed test')
    torch.set_num_threads(args.threads)
    inventory=validate_book_corpus(args.corpus_dir,args.split)
    parent,_,base,comp,parent_meta=load_unified(args.parent,args.root,args.threads)
    if parent_meta.get('candidate_trained'):
        raise ValueError('reference checkpoint must be the untrained R7.27 conversion')
    inputs=prepare(args,parent,base,comp,args.split,args.targets,args.seed)
    parent.eval()
    base_nll,_=forward(parent,inputs)
    base_metrics=evaluate(parent,base,inputs,args.seed+10)
    base_geometry=geometry_diagnostic(parent,inputs,args.seed+11)
    # There must not be a target/head/trunk change between checkpoints.
    parent_trunk={k:v.clone() for k,v in parent.state_dict().items() if not k.startswith('diamond.')}
    del parent
    gc.collect()
    candidate,_,base2,comp2,candidate_meta=load_unified(args.candidate,args.root,args.threads)
    if not candidate_meta.get('candidate_trained') or candidate_meta.get('promoted'):
        raise ValueError('R7.28 comparison requires an explicitly unpromoted trained candidate')
    if any(not torch.equal(candidate.state_dict()[k],v) for k,v in parent_trunk.items()):
        raise ValueError('candidate silently changed inherited R7 model tensors')
    candidate.eval()
    candidate_nll,_=forward(candidate,inputs)
    candidate_metrics=evaluate(candidate,base2,inputs,args.seed+10)
    candidate_geometry=geometry_diagnostic(candidate,inputs,args.seed+11)
    delta=base_nll-candidate_nll
    paired=boot_books(delta,inputs['book_ids'],args.seed+12)
    report={
       'format':'FA1_R7_28_FIXED_BOOK_COMPARISON_V1',
       'split':args.split,'checkpoint_promoted':False,
       'training_or_calibration_on_this_split':False,
       'model_weights_identical_outside_diamond':True,
       'parent_sha256':digest(args.parent),
       'candidate_sha256':digest(args.candidate),
       'candidate_training_step':candidate_meta.get('candidate_training_updates'),
       'single_optimizer_restored':True,'corpus_inventory':inventory,
       'n_positions':len(inputs['target']),
       'baseline':base_metrics,'baseline_geometry':base_geometry,
       'candidate':candidate_metrics,'candidate_geometry':candidate_geometry,
       'paired_full_nll_gain_nats_per_word':float(delta.mean()),
       'paired_book_gain':paired,
       'complete_raw_byte_bpb_measured':False,
       'full_mixture_top5_available':False,
       'historical_ancestor_book_overlap':'not independently established',
    }
    output=Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2)+'\n')
    print('FINAL',json.dumps({k:v for k,v in report.items() if k not in ('baseline','candidate','baseline_geometry','candidate_geometry')}),flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True)
    p.add_argument('--parent',required=True)
    p.add_argument('--candidate',required=True)
    p.add_argument('--corpus-dir',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--split',choices=('dev','test'),default='test')
    p.add_argument('--targets',type=int,default=512)
    p.add_argument('--batch',type=int,default=8)
    p.add_argument('--seed',type=int,default=7288)
    p.add_argument('--threads',type=int,default=2)
    run(p.parse_args())
