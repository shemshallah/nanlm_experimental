#!/usr/bin/env python3
"""Independent development-position grade-2/3/4 ablation audit for R7.24.

Do not inspect test books for training/promotion selection. Word-target NLL
is not byte-accurate bits per byte.
"""
from __future__ import annotations
import argparse,json,pathlib
import torch
from r710_checkpoint_integrity import sha256
from fa1_r724_unified import load_geometric
from fa1_r723_train import prepare,forward
from fa1_r724_train import boot_books

@torch.no_grad()
def run(a):
    torch.set_num_threads(a.threads)
    m,opts,base,comp,meta=load_geometric(a.candidate,a.root,threads=a.threads)
    m.eval()
    data=prepare(a,m,base,comp,a.split,a.targets,a.seed)
    g=m.diamond.geometry
    original=[p.detach().clone() for p in (g.edge_gain,g.face_gain,g.tetra_gain)]
    full,trace=forward(m,data)
    unrelated,_=forward(m,data,'unrelated')
    no_pair,_=forward(m,data,'no_pair')
    no_apex,_=forward(m,data,'no_apex')
    results={}
    for name,p,old in zip(['edges','triangles','tetrahedra'],
                           [g.edge_gain,g.face_gain,g.tetra_gain],original):
        p.zero_()
        ablated,_=forward(m,data)
        p.copy_(old)
        delta=ablated-full
        results[name]={'incremental_gain':float(delta.mean()),
            'books':boot_books(delta,data['book_ids'],a.seed+2) if 'book_ids' in data else None}
    for p in (g.edge_gain,g.face_gain,g.tetra_gain):p.zero_()
    old,_=forward(m,data)
    for p,x in zip((g.edge_gain,g.face_gain,g.tetra_gain),original):p.copy_(x)
    gain=old-full
    result={'format':'FA1_R7_24_GEOMETRY_INDEPENDENT_DEV_AUDIT',
      'checkpoint_sha256':sha256(a.candidate),'candidate_promoted':False,
      'split':a.split,'new_positions_seed':a.seed,'n_vocab_word_targets':a.targets,
      'old_unpromoted_r723_nll_nats_per_vocab_word':float(old.mean()),
      'full_r724_nll_nats_per_vocab_word':float(full.mean()),
      'geometry_total_gain':float(gain.mean()),
      'geometry_book_ci':boot_books(gain,data['book_ids'],a.seed) if 'book_ids' in data else None,
      'correct_history_advantage':float((unrelated-full).mean()),
      'down_pair_advantage':float((no_pair-full).mean()),
      'apex_advantage':float((no_apex-full).mean()),
      'geometry_per_level':results,
      'memory_optimizers':len(opts),'test_books_consulted':a.split=='test',
      'bits_per_byte_measured':False}
    pathlib.Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    pathlib.Path(a.output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    return result
if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',default='/mnt/data/diamond_r723/r7_ancestor')
    p.add_argument('--candidate',required=True)
    p.add_argument('--corpus-dir',default='/mnt/data/diamond_r723/nanrouter_verified/processed')
    p.add_argument('--split',choices=['train','dev','test'],default='dev')
    p.add_argument('--targets',type=int,default=256)
    p.add_argument('--seed',type=int,default=19017)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--batch',type=int,default=8)
    p.add_argument('--output',default='/mnt/data/diamond_r724/independent_dev_audit.json')
    run(p.parse_args())