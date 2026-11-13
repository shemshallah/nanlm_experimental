#!/usr/bin/env python3
"""Sealed, one-shot book-cluster R7 Diamond evaluation; no test selection."""
from __future__ import annotations
import argparse,pathlib,json,math
import numpy as np,torch
from r7_core import Corpus
from fa1_r721_joint_train import gather_joint
from fa1_r723_diamond import load_diamond
from fa1_r723_corpus import diamond_context
from r710_checkpoint_integrity import sha256

def run(a):
    torch.set_num_threads(2)
    m,opts,base,comp,meta=load_diamond(a.candidate,a.root)
    m.eval()
    corpus=Corpus(a.root,'test',corpus_dir=a.corpus_dir)
    with torch.no_grad():
        d=gather_joint(a.root,m,base,comp,'test',a.samples,a.seed,batch=a.batch,corpus_dir=a.corpus_dir)
        d.update(diamond_context(a.root,base,'test',d['positions'],d['hidden'],a.corpus_dir))
        original=m.joint_nll(d,diamond_enabled=False)[0]
        selected=m.joint_nll(d,diamond_enabled=True)[0]
        if meta['new_diamond_updates']==0:
            assert torch.equal(original,selected),'identity checkpoint violated exact parity'
        nll,close,extra=m.joint_nll(d,return_extras=True)
        assert torch.equal(nll,selected)
        positions=np.asarray(d['positions'],dtype=np.int64)
        books=np.asarray(corpus.doc_start[positions],dtype=np.int64)
        ids=np.unique(books)
        old=original.detach().cpu().numpy();candidate=selected.detach().cpu().numpy()
        gain=old-candidate
        mean_by_book=np.asarray([gain[books==i].mean() for i in ids])
        rng=np.random.default_rng(a.seed+321)
        samples=rng.choice(mean_by_book,size=(4000,len(ids)),replace=True).mean(1)
        ci=np.percentile(samples,[2.5,97.5])
        sectors=extra['trace']['selected_sectors'][:,0].unique().numel()
        unique_tri=extra['trace']['triangle_addresses'].unique().numel()
        report={'source':'SHA-pinned NaNROUTER corpora-short-stories-0001',
            'split':'sealed whole-book test, never used for model selection',
            'checkpoint_sha256':sha256(a.candidate),'accepted_parent_step':14416,
            'candidate_training_updates':meta['new_diamond_updates'],
            'optimizer_histories_inherited':len(opts),
            'total_book_count':len(np.unique(np.asarray(corpus.doc_start))),
            'eligible_books_in_sample':len(ids),
            'target_samples':len(old),'nats_per_word_target_baseline':float(old.mean()),
            'nats_per_word_target_diamond':float(candidate.mean()),
            'diamond_gain_nats_per_word':float(gain.mean()),
            'book_cluster_bootstrap_95ci_gain':[float(ci[0]),float(ci[1])],
            'bits_per_byte':None,'exact_byte_bpb_measured':False,
            'distinct_apex_sectors':int(sectors),'distinct_active_triangles':int(unique_tri),
            'per_book':[{'doc_start':int(i),'targets':int((books==i).sum()),
                        'parent_nll':float(old[books==i].mean()),
                        'diamond_nll':float(candidate[books==i].mean())} for i in ids],
            'test_used_for_training_or_selection':False,'promoted':False}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='per_book'},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',default='/mnt/data/diamond_r723/r7_ancestor')
    p.add_argument('--corpus-dir',default='/mnt/data/diamond_r723/nanrouter_verified/processed')
    p.add_argument('--candidate',default='/mnt/data/diamond_r723/FA1_R723_DIAMOND_IDENTITY.pt')
    p.add_argument('--output',type=pathlib.Path,default=pathlib.Path('/mnt/data/diamond_r723/r723_nanrouter_sealed_test.json'))
    p.add_argument('--samples',type=int,default=512)
    p.add_argument('--seed',type=int,default=72492)
    p.add_argument('--batch',type=int,default=8)
    run(p.parse_args())