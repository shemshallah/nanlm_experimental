#!/usr/bin/env python3
"""R53: opt-in exact-address temporal pair retrieval, DEV-only frozen probe."""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from poincare_frontier import file_hash, load_meta
from compare_geometry_books import METRICS, book_starts, measure_books
from probe_r47_selective_emission import book_comparison, literal_byte_count, load_model
from geometry_address_controls import fixed_permutations


CONDITIONS = (('inherited', 0., False),
              ('exact_pos_003', .03, False), ('wrong_pos_003', .03, True),
              ('exact_neg_003', -.03, False),
              ('exact_pos_010', .10, False), ('wrong_pos_010', .10, True))


@contextmanager
def pair_history_intervention(model, gain, query_permutation=None):
    """Modify runtime controls only; even exceptions restore exact semantics."""
    geometries = [module for module in model.modules()
                  if getattr(module,'simplex',None) is not None and
                     hasattr(module,'pair_history_gain')]
    if not geometries:
        raise ValueError('R53 needs the existing sparse simplex modules')
    saved=[(m,m.pair_history_gain,m.pair_history_query_map) for m in geometries]
    try:
        for module in geometries:
            module.pair_history_gain=float(gain)
            module.pair_history_query_map=query_permutation
        yield
    finally:
        for module, old_gain, old_map in saved:
            module.pair_history_gain=old_gain
            module.pair_history_query_map=old_map


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--context',type=int,default=32)
    parser.add_argument('--windows-per-book',type=int,default=2)
    parser.add_argument('--batch',type=int,default=2)
    parser.add_argument('--bootstrap',type=int,default=2000)
    parser.add_argument('--seed',type=int,default=2026)
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if min(args.context,args.windows_per_book,args.batch,args.bootstrap,args.threads)<1:
        parser.error('All workload counts must be positive')
    torch.set_num_threads(args.threads)
    meta=load_meta(args.data_dir)
    if meta.get('split_unit')!='whole_book' or meta.get('unit')!='UTF8_bytes_plus_eos':
        raise ValueError('Requires whole-book byte/EOS NaNROUTER DEV split')
    if file_hash(args.data_dir/'split_manifest.json')!=meta.get('split_manifest_sha256'):
        raise ValueError('Split manifest mismatch')
    dev=args.data_dir/'valid.bin'
    digest=meta.get('valid_sha256') or meta.get('bin_valid_sha256')
    if not digest or file_hash(dev)!=digest:
        raise ValueError('DEV checksum mismatch')
    mem=np.memmap(dev,dtype=meta['token_dtype'],mode='r')
    starts,covered=book_starts(meta['valid_doc_lengths'],args.context,
                               args.windows_per_book,args.seed)
    if len(mem)!=covered:
        raise ValueError('DEV book offsets mismatch')
    model,saved=load_model(args.checkpoint,meta,args.context)
    original={k:v.clone() for k,v in model.state_dict().items()}
    permutation=fixed_permutations(args.seed)['pair_addresses']
    rows, elapsed={}, {}
    for label,gain,wrong in CONDITIONS:
        begin=perf_counter()
        with pair_history_intervention(model,gain,permutation if wrong else None):
            rows[label]=measure_books(model,mem,starts,args.context,args.batch)
        elapsed[label]=perf_counter()-begin
    replay=measure_books(model,mem,starts,args.context,args.batch)
    if any(a!=b for a,b in zip(replay,rows['inherited'])):
        raise AssertionError('R53 inherited replay changed after intervention')
    if any(not torch.equal(original[k],v) for k,v in model.state_dict().items()):
        raise AssertionError('R53 changed checkpoint tensors')
    n=len(starts)*args.windows_per_book*args.context
    raw=literal_byte_count(mem,starts,args.context)
    scores={}
    for label,gain,wrong in CONDITIONS:
        means={key:float(np.mean([row[key] for row in rows[label]])) for key in METRICS}
        scores[label]={'gain':gain,'query_only_scrambled':wrong,
                       'metrics':{**means,'bits_per_literal_byte':means['ce_1']*n/raw/np.log(2)},
                       'targets_per_second':n/max(elapsed[label],1e-9),
                       'vs_inherited':book_comparison(rows['inherited'],rows[label],
                                                      args.bootstrap,args.seed,len(CONDITIONS)-1)}
    specificity={
        gain_label:book_comparison(rows[wrong_label],rows[gain_label],
                                   args.bootstrap,args.seed,2)
        for gain_label,wrong_label in (
            ('exact_pos_003','wrong_pos_003'),
            ('exact_pos_010','wrong_pos_010'))}
    report={'status':'R53_read_only_experimental_DEV_not_promotion',
            'step':saved['step'],'checkpoint_sha256':file_hash(args.checkpoint),
            'dev_sha256':digest,'books':len(starts),'context':args.context,
            'windows_per_book':args.windows_per_book,'targets_per_arm':n,
            'raw_bytes_per_arm':raw,'seed':args.seed,
            'bootstrap_replicates':args.bootstrap,
            'bootstrap_unit':'whole DEV book, Bonferroni over 5 comparisons',
            'recovery':'original state_dict and inherited replay exact',
            'TEST_opened':False,'weights_or_optimizer_updated':False,
            'results':scores,
            'exact_over_wrong_address':specificity}
    output=json.dumps(report,indent=2,allow_nan=False)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(output,encoding='utf-8')
    print(output,end='')


if __name__=='__main__':
    main()
