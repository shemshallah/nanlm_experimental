#!/usr/bin/env python3
"""Deterministic 270/15/15 SHA-disjoint long-history book selection.

Selects holdouts using only document length and SHA, never next-token scores.
It does not overwrite inputs; output is a separate symlink tree consumable by
prepare_document_corpus.py --presplit. The resulting test split stays sealed
until the final chosen candidate is frozen.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from infer_unified_now import RGX


def make_split(source, destination, train_books=270, heldout_books=15,
               min_heldout_words=65536):
    source=Path(source).resolve()
    destination=Path(destination).resolve()
    if destination == source or source in destination.parents:
        raise ValueError('split output must not be inside the source collection')
    files=sorted(source.rglob('*.txt'))
    unique={}
    for path in files:
        raw=path.read_bytes()
        sha=hashlib.sha256(raw).hexdigest()
        if sha in unique:
            continue
        count=len(RGX.findall(raw.decode('utf8',errors='replace').lower()))
        unique[sha]={'sha256':sha,'path':path,'words':count}
    candidates=sorted(unique.values(),key=lambda x:x['sha256'])
    long=[r for r in candidates if r['words']>=min_heldout_words]
    if len(long)<2*heldout_books:
        raise ValueError('not enough age-eligible long books for two holdouts')
    # SHA sorting prevents filename order or sampled targets from determining
    # the split, while long-document selection permits independent 128K probes.
    selected={'dev':long[:heldout_books],
              'test':long[heldout_books:2*heldout_books]}
    excluded={r['sha256'] for values in selected.values() for r in values}
    available=[r for r in candidates if r['sha256'] not in excluded]
    if len(available)<train_books:
        raise ValueError('not enough independent training books')
    selected['train']=available[:train_books]
    if any(destination.iterdir()) if destination.exists() else False:
        raise FileExistsError('split output already exists: '+str(destination))
    destination.mkdir(parents=True,exist_ok=True)
    manifest={}
    for split in ('train','dev','test'):
        folder=destination/split
        folder.mkdir(exist_ok=True)
        manifest[split]=[]
        for record in selected[split]:
            name=record['sha256'][:16]+'__'+record['path'].name
            (folder/name).symlink_to(record['path'])
            manifest[split].append({'sha256':record['sha256'],
                                    'name':record['path'].name,'words':record['words']})
    (destination/'split_manifest.json').write_text(json.dumps({
        'selection':'SHA-ordered, length-qualified DEV/TEST, target-blind',
        'min_heldout_words':min_heldout_words,
        'splits':manifest},indent=2)+'\n')
    return {s:{'books':len(v),'words':sum(x['words'] for x in v)}
            for s,v in manifest.items()}


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--input',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--train-books',type=int,default=270)
    p.add_argument('--heldout-books',type=int,default=15)
    p.add_argument('--min-heldout-words',type=int,default=65536)
    a=p.parse_args()
    print(json.dumps(make_split(a.input,a.output,a.train_books,
                               a.heldout_books,a.min_heldout_words),indent=2))
