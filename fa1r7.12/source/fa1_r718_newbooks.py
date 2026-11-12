#!/usr/bin/env python3
"""Fail-closed R7.18 new-book comparison. Read-only; never trains or promotes weights.

Requires actual, separately identified, NaNROUTER whole-book files imported with
prepare_document_corpus.py. Toy fixture alone is not evidence of generalization.
"""
import argparse, json, pathlib, time
import numpy as np
import torch
from r7_core import Corpus
from r712_validate import restore as restore_parent
from r710_checkpoint_integrity import sha256
from fa1_r716_router import TrustRouter,PARENT
from fa1_r716_newbooks import audit_documents
from fa1_r716_overlap import screen
from fa1_r718_far_residual import (FarResidual,gather,reroute_far,loss_far)


def book_cluster_ci(data,c,shuffled,train_donor,corpus,bootstrap=2000,seed=718):
    if corpus.doc_start is None:raise ValueError('missing document boundaries')
    pos=np.asarray(data['positions'],dtype=np.int64)
    starts=np.asarray(corpus.doc_start[pos],dtype=np.int64)
    unique=np.unique(starts)
    values=torch.stack([data['base']-c,data['near']-c,shuffled-c,train_donor-c],dim=1).cpu().numpy()
    by_book=np.asarray([values[starts==b].mean(axis=0) for b in unique],dtype=np.float64)
    rng=np.random.default_rng(seed)
    indices=rng.integers(0,len(unique),size=(bootstrap,len(unique)))
    means=by_book[indices].mean(axis=1)
    names=['accepted_gain','incremental_far_gain','shuffled_past_advantage','train_donor_advantage']
    return {'eligible_sampled_books':len(unique),'sampled_positions':len(pos),
       'mean':dict(zip(names,by_book.mean(axis=0).tolist())),
       'ci95':dict(zip(names,np.quantile(means,[.025,.975],axis=0).T.tolist())),
       'books_positive_incremental':int((by_book[:,1]>0).sum()),
       'books_positive_shuffle_control':int((by_book[:,2]>0).sum())}


def main(a):
    torch.set_num_threads(a.threads);t=time.monotonic()
    root=pathlib.Path(a.root);corpus=pathlib.Path(a.corpus_dir)
    parent=root/'checkpoint/pal_active.pt';assert sha256(parent)==PARENT
    book_audit=audit_documents(root,corpus)
    overlap=screen(root,corpus) if not a.skip_overlap else None
    near_blob=torch.load(a.r716_overlay,map_location='cpu',weights_only=False)
    far_blob=torch.load(a.r718_overlay,map_location='cpu',weights_only=False)
    assert near_blob['format']=='FA1_R7_16_32K_TRUST_CANDIDATE_V1'
    assert near_blob['parent_sha256']==PARENT==far_blob['parent_sha256']
    assert far_blob['format']=='FA1_R7_18_FAR_DISJOINT_CANDIDATE'
    assert far_blob['r716_overlay_sha256']==sha256(a.r716_overlay)
    model,opts,base,comp,_,_=restore_parent(root,root/'adaptation/r712_gate_delta.pt')
    assert len(opts)==5;model.eval()
    for p in model.parameters():p.requires_grad_(False)
    near=TrustRouter();near.load_state_dict(near_blob['gate']);near.eval()
    far=FarResidual();far.load_state_dict(far_blob['far_gate']);far.eval()
    d=gather(root,model,base,comp,near,'test',a.test,a.seed,a.batch,corpus)
    w=reroute_far(root,model,base,comp,'test',d,a.seed,corpus,mode='shuffle')
    u=reroute_far(root,model,base,comp,'test',d,a.seed,corpus,mode='train_donor')
    y,_=loss_far(far,d);sh,_=loss_far(far,w);tr,_=loss_far(far,u)
    ci=book_cluster_ci(d,y,sh,tr,Corpus(root,'test',corpus_dir=corpus),a.bootstrap,a.seed)
    overlap_ok=bool(overlap is not None and overlap.get('test_overlap_screen_passed') and overlap.get('new_books',{}).get('test',{}).get('known_token_window_fraction',0)>=.05)
    eligible=(ci['eligible_sampled_books']>=a.min_books and overlap_ok
       and all(v[0]>0 for v in ci['ci95'].values())
       and sha256(parent)==PARENT)
    report={'kind':'FA1 R7.18 independent-whole-book READ-ONLY validation',
       'checkpoint_promoted':False,'candidate_review_eligible':bool(eligible),
       'full_parent_sha256':PARENT,'r716_overlay_sha256':sha256(a.r716_overlay),
       'r718_overlay_sha256':sha256(a.r718_overlay),'documents':book_audit,
       'book_content_hash_splits_disjoint':True,'legacy_fingerprint_overlap_screen':overlap,
       'legacy_overlap_risk_not_fully_eliminated':True,'test_examples':a.test,
       'test_base_nll':float(d['base'].mean()),'test_near_nll':float(d['near'].mean()),
       'test_far_nll':float(y.mean()),'test_shuffled_far_nll':float(sh.mean()),
       'test_train_donor_far_nll':float(tr.mean()),'book_cluster_ci':ci,
       'whole_book_age_eligible':True,'oldest_possible_age':131072,'book_boundary_masked':True,
       'source_overlap_screen_passed':overlap_ok,'elapsed_s':round(time.monotonic()-t,2)}
    path=pathlib.Path(a.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'candidate_review_eligible':eligible,'book_count':ci['eligible_sampled_books'],'report':str(path)}),flush=True)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=pathlib.Path,required=True)
    p.add_argument('--corpus-dir',type=pathlib.Path,required=True)
    p.add_argument('--r716-overlay',type=pathlib.Path,required=True)
    p.add_argument('--r718-overlay',type=pathlib.Path,required=True)
    p.add_argument('--test',type=int,default=2048)
    p.add_argument('--seed',type=int,default=71877)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--batch',type=int,default=8)
    p.add_argument('--bootstrap',type=int,default=2000)
    p.add_argument('--min-books',type=int,default=12)
    p.add_argument('--skip-overlap',action='store_true')
    p.add_argument('--output',type=pathlib.Path,default=pathlib.Path('reports/FA1_R7p18_NEWBOOKS.json'))
    main(p.parse_args())