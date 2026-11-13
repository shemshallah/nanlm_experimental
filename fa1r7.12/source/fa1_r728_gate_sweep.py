#!/usr/bin/env python3
"""Diagnostic sign/scale sweep of four *existing* R7 geometric gates.

No weights are trained, no TEST books consulted, no promotion. This tests
whether learned signed geometry carries anticorrelated predictive information
that a fixed positive gain has been forcing the model to misuse.
"""
import argparse
import json
import pathlib

import torch

from fa1_r723_train import forward, prepare
from fa1_r724_train import GEOMETRY_GATES, boot_books
from fa1_r727_checkpoint import load_unified, digest
from fa1_r727_eval import validate_book_corpus


@torch.no_grad()
def run(args):
    if args.split != 'dev':
        raise ValueError('do not calibrate the sign of geometry on TEST')
    torch.set_num_threads(args.threads)
    validate_book_corpus(args.corpus_dir, 'dev')
    m, optim, base, comp, meta = load_unified(args.checkpoint,args.root,args.threads)
    data = prepare(args,m,base,comp,'dev',args.dev,args.seed)
    m.eval()
    g = m.diamond.geometry
    gates = {k:getattr(g,k).detach().clone() for k in GEOMETRY_GATES}
    rows = []
    try:
        for scale in (-2.,-1.5,-1.,-.5,0.,.5,1.,1.5,2.):
            for name, value in gates.items():
                getattr(g,name).copy_(value*scale)
            nll, ex = forward(m,data)
            rows.append({'scale':scale,'full_nll':float(nll.mean()),
                         'branch_nll':float(ex['branch_nll'].mean()),
                         'book_mean_nll':boot_books(nll,data['book_ids'],args.seed)['book_average'],
                         'nll_per_position':nll.detach().tolist()})
    finally:
        for name,value in gates.items():
            getattr(g,name).copy_(value)
    off = next(x for x in rows if x['scale']==0.)
    off_values=torch.tensor(off['nll_per_position'])
    for row in rows:
        row['geometry_gain_vs_off']=off['full_nll']-row['full_nll']
        row['book_gain_vs_off']=boot_books(
            off_values-torch.tensor(row['nll_per_position']),
            data['book_ids'],args.seed+1)
        del row['nll_per_position']
    report={'round':'FA1 R7.28 geometry sign/scale scan',
            'checkpoint_sha256':digest(args.checkpoint),
            'candidate_promoted':False,'split':'dev','test_books_consulted':False,
            'n_positions':len(data['target']),'n_books':int(data['book_ids'].unique().numel()),
            'original_gates':{k:float(v) for k,v in gates.items()},
            'rows':rows,'best_scale_dev_only':min(rows,key=lambda x:x['full_nll'])['scale']}
    pathlib.Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    pathlib.Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='rows'}),flush=True)
    for row in rows:print('SCALE',json.dumps(row),flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--corpus-dir',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--dev',type=int,default=256)
    p.add_argument('--batch',type=int,default=8)
    p.add_argument('--seed',type=int,default=730)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--split',default='dev')
    run(p.parse_args())
