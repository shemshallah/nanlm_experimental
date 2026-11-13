#!/usr/bin/env python3
"""Read-only, whole-book R7.22 held-out test audit; never select or promote."""
from __future__ import annotations
import argparse,json,pathlib
import numpy as np
import torch
from r7_core import Corpus
from fa1_r721_joint_train import gather_joint
from fa1_r722_train import gather_sem
from fa1_r722_unified import load_unified
from r710_checkpoint_integrity import sha256

def run(a):
    torch.set_num_threads(2)
    original=sha256(a.parent)
    parent,parent_opts,base,comp,_=load_unified(a.parent,a.root)
    current,current_opts,_,_,_=load_unified(a.candidate,a.root)
    assert len(parent_opts)==len(current_opts)==5
    for name,t in parent.state_dict().items():
        if not name.startswith('semantic.'):
            assert torch.equal(t,current.state_dict()[name]),'Non-semantic weights changed: '+name
    corpus=Corpus(a.root,'test',corpus_dir=a.corpus_dir)
    assert corpus.doc_start is not None,'book boundaries required'
    data=gather_joint(a.root,parent,base,comp,'test',a.test,a.seed,a.batch,a.corpus_dir)
    sem=gather_sem(a.root,base,'test',data['positions'],a.seed+1,corpus_dir=a.corpus_dir)
    donor=gather_sem(a.root,base,'test',data['positions'],a.seed+2,
                     train_donor=True,corpus_dir=a.corpus_dir)
    with torch.no_grad():
        prior,_=parent.joint_nll(data,semantic_enabled=False)
        correct,_=current.semantic(prior,sem['semantic_features'],
                                   sem['semantic_prob'],sem['semantic_count'])
        frequency,_=current.semantic(prior,sem['semantic_features'],
                                     sem['frequency_prob'],sem['semantic_count'])
        unrelated,_=current.semantic(prior,donor['semantic_features'],
                                     donor['semantic_prob'],donor['semantic_count'])
    p,c,f,u=[x.detach().cpu().numpy() for x in (prior,correct,frequency,unrelated)]
    assert all(np.isfinite(x).all() for x in (p,c,f,u))
    positions=np.asarray(data['positions'],dtype=np.int64)
    book_ids=np.asarray(corpus.doc_start[positions],dtype=np.int64)
    unique=np.unique(book_ids)
    book_gain=np.asarray([np.mean((p-c)[book_ids==x]) for x in unique])
    rng=np.random.default_rng(a.seed+3)
    boot=np.mean(rng.choice(book_gain,size=(2000,len(book_gain)),replace=True),axis=1)
    lo,hi=np.percentile(boot,[2.5,97.5])
    report={'round':'FA1 R7.23 sealed whole-book test (read-only)',
      'source_release':a.source_release,
      'tested_candidate_sha256':sha256(a.candidate),'parent_sha256':original,
      'candidate_model_lineage':'one unified R7.22; accepted ancestry step 14416',
      'optimizer_history_count':5,
      'n_test_word_targets':int(len(p)),'book_clusters':int(len(unique)),
      'base_nll_nats_per_word':float(p.mean()),
      'candidate_nll_nats_per_word':float(c.mean()),
      'candidate_bits_per_word':float(c.mean()/np.log(2)),
      'bits_per_byte':None,
      'note':'NLL is per vocabulary word target, not byte-accurate language-model BPB',
      'candidate_gain_nats_per_word':float(np.mean(p-c)),
      'book_mean_gain_bootstrap_95ci':[float(lo),float(hi)],
      'correct_vs_old_frequency_advantage':float(np.mean(f-c)),
      'correct_vs_train_donor_advantage':float(np.mean(u-c)),
      'retrieval_nonempty':int(sem['nonempty']),
      'retrieval_target_supported':int(sem['target_supported']),
      'per_book':[{'doc_start':int(k),'n':int((book_ids==k).sum()),
                   'gain':float(np.mean((p-c)[book_ids==k]))} for k in unique],
      'heldout_test_used_for_selection':False,
      'requires_overlap_screen':True,'promoted':False}
    assert sha256(a.parent)==original
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='per_book'},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',required=True);p.add_argument('--corpus-dir',required=True)
    p.add_argument('--parent',required=True);p.add_argument('--candidate',required=True)
    p.add_argument('--source-release',default='unverified external corpus')
    p.add_argument('--output',required=True,type=pathlib.Path)
    p.add_argument('--test',type=int,default=256)
    p.add_argument('--seed',type=int,default=72383)
    p.add_argument('--batch',type=int,default=8)
    run(p.parse_args())
