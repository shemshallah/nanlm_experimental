"""Large, completely unused TRAIN-remainder holdout for Terminal-RSI v102.

The initial BPE, token LMs, and C donor use only the first `max_train_bytes`
of the deterministically ordered TRAIN books. The residual correction uses a
separate 48-book calibration window set. Books in this holdout contributed to
neither process. This is an auxiliary never-trained-on evaluation pool, not a
replacement for the originally partitioned DEV/TEST books.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random

import numpy as np

from . import nancm_v102 as donor
from .controller import assign_books, digest, select_windows, write_windows
from .contextual import ContextualResidual, previous_byte_classes
from .residual import ResidualPolicy, load_scored


def prepare_remainder(*, corpus: Path, work: Path, count: int = 64):
    initial = json.loads((work/'split_manifest.json').read_text())
    residual = json.loads((work/'residual_manifest.json').read_text())
    parts = assign_books(corpus)
    if len(parts['train']) != initial['split_counts']['train']:
        raise ValueError('TRAIN partition differs from original frozen corpus')
    train = sorted(parts['train'], key=lambda x: hashlib.sha256(x[0].encode()).hexdigest())
    original_stream = b'\n\n'.join(data for _, data in train)[:initial['max_train_bytes']]
    if digest(original_stream) != initial['train_sha256']:
        raise ValueError('Original 1 MiB donor training stream changed')
    prefix_names = set()
    consumed = 0
    for name, data in train:
        if consumed >= initial['max_train_bytes']:
            break
        prefix_names.add(name)
        consumed += len(data) + 2
    untouched = [(name, data) for name, data in parts['train']
                 if name not in prefix_names]
    bpe = donor.BPE.load(work/'bpe.txt')
    calibration = select_windows(untouched,bpe,count=48,
                                 width=initial['width'],skip=initial['skip'],seed=3003)
    old_cal_hash = digest('\n'.join(sorted(name for name, _ in calibration)).encode())
    if old_cal_hash != residual['splits']['calibration']['book_list_sha256']:
        raise ValueError('Original residual calibration selection changed')
    calibrated_names = {name for name, _ in calibration}
    fresh = [(name,data) for name,data in untouched if name not in calibrated_names]
    selected = select_windows(fresh,bpe,count=count,width=initial['width'],
                              skip=initial['skip'],seed=9009)
    if {name for name,_ in selected} & (prefix_names | calibrated_names):
        raise ValueError('Training-to-holdout book leakage')
    meta = write_windows(work,'holdout4',selected,bpe,skip=initial['skip'])
    report = {'schema':'terminal-rsi-v102-train-remainder-holdout-v1',
              'scope':'never-trained TRAIN partition remainder; separate from original DEV/TEST',
              'original_manifest_sha256':digest((work/'split_manifest.json').read_bytes()),
              'residual_manifest_sha256':digest((work/'residual_manifest.json').read_bytes()),
              'original_donor_training_book_count':len(prefix_names),
              'excluded_residual_calibration_books':len(calibrated_names),
              'remaining_eligible_books':len(fresh),
              'excluded_book_names_sha256':digest('\n'.join(sorted(prefix_names | calibrated_names)).encode()),
              'split':meta}
    (work/'holdout_manifest.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    return report


def evaluate_remainder(work:Path):
    """Score parent, raw context, and fixed top-k guard without learning."""
    prior=json.loads((work/'context_search.json').read_text())
    selected=prior['best_checkpoint_path']
    if not selected:raise ValueError('No preexisting context candidate')
    contextual=ContextualResidual.from_checkpoint(work/selected)
    parent=ResidualPolicy.load_checkpoint(work/'residual_checkpoint.npz')
    p,y,per=load_scored(work,'holdout4')
    prev=previous_byte_classes(work,'holdout4')
    baseline=parent.score(p,y,per)
    raw=contextual.score(p,y,per,prev)
    guarded=contextual.score_guarded(p,y,per,prev)
    if baseline['top1']!=guarded['top1'] or baseline['top5']!=guarded['top5']:
        raise ValueError('Guaranteed exact top-k preservation failed')
    delta=np.array(baseline['per_book_bpc'])-np.array(guarded['per_book_bpc'])
    rng=random.Random(9010)
    bootstrap=sorted(float(np.mean(rng.choices(delta.tolist(),k=len(delta))))
                     for _ in range(2000))
    report={'schema':'terminal-rsi-v102-train-remainder-holdout-eval-v1',
            'books':len(delta),'scored_bytes':len(y),
            'target_free_topk_guard':True,'training_during_evaluation':False,
            'original_parent':baseline,'raw_contextual':raw,'guarded_contextual':guarded,
            'paired_guard_gain_bpc_ci95':[bootstrap[50],bootstrap[1949]],
            'positive_gain_books':int(np.sum(delta>0)),
            'candidate_promoted':False,
            'caveat':'Auxiliary never-trained TRAIN-remainder books, not original sealed TEST; check full-book and independent corpus before promotion'}
    (work/'holdout_search.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','evaluate'))
    parser.add_argument('--corpus',type=Path)
    parser.add_argument('--work',type=Path,required=True)
    parser.add_argument('--books',type=int,default=64)
    args=parser.parse_args()
    if args.command=='prepare':
        if args.corpus is None:parser.error('--corpus required for prepare')
        report=prepare_remainder(corpus=args.corpus,work=args.work,count=args.books)
    else:report=evaluate_remainder(args.work)
    print(json.dumps(report,sort_keys=True,indent=2))


if __name__=='__main__':main()
