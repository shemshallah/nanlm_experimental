#!/usr/bin/env python3
"""Match both experimental R7.45 checkpoints on one larger DEV sample."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from fa1_r723_train import prepare
from fa1_r727_checkpoint import load_unified,digest
from fa1_r727_eval import validate_book_corpus
from fa1_r744_cpu_pilot import foreign_word_history
from fa1_r745_cpu_pilot import foreign_geometry,evaluate
from fa1_r745_geometric_value_read import ROUND


def run(a):
    if Path(a.output).exists():
        raise FileExistsError('large DEV report already exists')
    t=time.monotonic()
    torch.set_num_threads(a.threads)
    inventory=validate_book_corpus(a.corpus_dir,'dev')
    first,opt,base,comp,meta=load_unified(a.first,a.root,a.threads)
    if meta['architecture_round']!=ROUND or len(opt.param_groups)!=26:
        raise ValueError('expected real R7.45 with 26 optimizer groups')
    data=prepare(a,first,base,comp,'dev',a.targets,a.seed)
    if len(torch.unique(data['book_ids']))!=15:
        raise ValueError('DEV sample must span all 15 books')
    wrong_hist,_=foreign_word_history(data,a.seed+5)
    wrong_geom=foreign_geometry(data,a.seed+6)
    runs={}
    for label,path in (('initial_128',a.first),('continued_512_selected_1',a.second)):
        model,optimizer,_,_,meta=load_unified(path,a.root,a.threads)
        if meta['architecture_round']!=ROUND or len(optimizer.param_groups)!=26:
            raise AssertionError('R7.45 failed normal unified loader')
        metric=evaluate(model,data,wrong_hist,wrong_geom,a.seed+20)
        runs[label]={'checkpoint_sha256':digest(path),
                     'source_checkpoint_sha256':meta['source_checkpoint_sha256'],
                     'selected_training_update':meta['candidate_training_updates'],
                     'metrics':metric}
        print('R745 LARGE',label,'gain',metric['r745_incremental_word_gain'],
              'addr',metric['r745_geometric_address_specificity'],
              'apex',metric['r745_apex_specificity'],flush=True)
    out={'round':'FA1_R7_45_MATCHED_LARGE_DEV_V1',
         'corpus_inventory':inventory,
         'split':'dev', 'positions':len(data['target']),
         'books':int(torch.unique(data['book_ids']).numel()),
         'independent_test_opened':False, 'candidate_promoted':False,
         'one_model_lineage':True,
         'runs':runs,'elapsed_seconds':time.monotonic()-t}
    dest=Path(a.output)
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',required=True)
    p.add_argument('--corpus-dir',required=True)
    p.add_argument('--first',required=True)
    p.add_argument('--second',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--targets',type=int,default=2048)
    p.add_argument('--batch',type=int,default=8)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--seed',type=int,default=811)
    run(p.parse_args())
