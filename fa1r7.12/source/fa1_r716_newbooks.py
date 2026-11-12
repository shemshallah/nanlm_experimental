#!/usr/bin/env python3
"""FA1 R7.16 evaluation on newly imported, whole-book NaNROUTER data.

Strictly evaluation only. Does not fit, select, or update the R7.16 adapter.
Gate must be dev-selected BEFORE inspecting any of these test books.
Exact book content-hash partition checking and source-mismatch control.
Original historical corpora are repeatedly used; zero-overlap with historical
books is NOT guaranteed from ISBN/title hashes alone. Report contamination risks.
"""
from __future__ import annotations
import argparse,hashlib,json,pathlib,collections,time
import numpy as np,torch
from r7_core import Corpus
from r712_validate import restore as restore_parent
from r710_checkpoint_integrity import sha256
from fa1_r716_router import (PARENT, TrustRouter, nll, gather, reroute_wrong, SCALES,
                                eval_pair, pick_positions)
from fa1_r716_overlap import screen

def audit_documents(root,corpus_dir):
    p=pathlib.Path(corpus_dir);book={};all_hash=[]
    for split in ('train','dev','test'):
        metap=p/f'{split}_documents.json'
        if not metap.exists():raise FileNotFoundError('missing book metadata '+str(metap))
        ds=json.loads(metap.read_text()).get('documents') or []
        if len(ds)<3:raise ValueError('need at least 3 independently identified books per split')
        for rec in ds:
            digest=rec.get('sha256')
            if not digest or len(digest)!=64:raise ValueError('missing content hash')
            all_hash.append(digest)
        book[split]={'books':len(ds),'unique_sha256':len(set(x['sha256'] for x in ds)),
            'total_words':sum(int(x['words']) for x in ds)}
        c=Corpus(root,split,corpus_dir=p)
        if c.doc_start is None:raise ValueError('no causal book boundary file for '+split)
        if not np.all(c.doc_start[1:]>=c.doc_start[:-1]):raise ValueError('bad book-boundary ordering')
        if int(c.doc_start[-1])>=len(c.seq):raise ValueError('document start outside sequence')
        # Verify document metadata matches the binary boundary at each book start.
        for doc in ds:
            idx=int(doc['first_record'])
            if not 0<=idx<len(c.seq) or int(c.doc_start[idx])!=idx:
                raise ValueError('book metadata/causal boundary mismatch '+str(doc['relative_file']))
    if len(all_hash)!=len(set(all_hash)):
        raise ValueError('same book SHA256 appears across train/dev/test')
    return book

def cluster_ci(data,correct,wrong,corpus,bootstrap=2000,seed=716):
    pos=np.asarray(data['positions'],dtype=np.int64)
    start=np.asarray(corpus.doc_start[pos]);books=np.unique(start)
    gain=(data['base']-correct).detach().cpu().numpy()
    adv=(wrong-correct).detach().cpu().numpy()
    by_book=np.asarray([[float(gain[start==b].mean()),float(adv[start==b].mean()),int((start==b).sum())] for b in books])
    rng=np.random.default_rng(seed)
    samples=rng.integers(0,len(books),size=(bootstrap,len(books)))
    # Equal book-weighted bootstrap, not a misleading independent-query CI.
    means=by_book[samples,:2].mean(axis=1)
    return {'book_count':len(books),'sampled_positions':len(pos),
       'equal_book_mean_gain':float(by_book[:,0].mean()),
       'equal_book_mean_memory_advantage':float(by_book[:,1].mean()),
       'book_bootstrap_gain_ci95':np.quantile(means[:,0],[.025,.975]).tolist(),
       'book_bootstrap_memory_advantage_ci95':np.quantile(means[:,1],[.025,.975]).tolist(),
       'worst_book_gain':float(by_book[:,0].min()),'best_book_gain':float(by_book[:,0].max()),
       'positions_per_book':{'min':int(by_book[:,2].min()),'median':float(np.median(by_book[:,2])),'max':int(by_book[:,2].max())}}

def main(a):
    torch.set_num_threads(a.threads);t=time.monotonic();root=pathlib.Path(a.root);data_dir=pathlib.Path(a.corpus_dir)
    if sha256(root/'checkpoint/pal_active.pt')!=PARENT:raise ValueError('wrong accepted parent')
    books=audit_documents(root,data_dir)
    overlap=screen(root,data_dir) if not a.skip_overlap_screen else None
    if a.audit_only:
        report={'audit_only':True,'documents':books,'all_split_book_hashes_disjoint':True,'legacy_fingerprint_audit':overlap,'source_overlap_with_legacy_unverified':overlap is None}
    else:
        obj=torch.load(a.overlay,map_location='cpu',weights_only=False)
        if obj['format']!='FA1_R7_16_32K_TRUST_CANDIDATE_V1' or obj['parent_sha256']!=PARENT:
            raise ValueError('R7.16 parent/adapter mismatch')
        model,opts,base,comp,_,_=restore_parent(root,root/'adaptation/r712_gate_delta.pt');model.eval()
        for p in model.parameters():p.requires_grad_(False)
        router=TrustRouter();router.load_state_dict(obj['gate'],strict=True);router.eval()
        sample=gather(root,model,base,comp,'test',a.test,a.seed,a.batch,data_dir)
        wrong=reroute_wrong(root,model,base,comp,'test',sample,a.seed,data_dir)
        c,g=nll(router,sample);w,_=nll(router,wrong)
        paired=eval_pair(router,sample,wrong)
        ci=cluster_ci(sample,c,w,Corpus(root,'test',corpus_dir=data_dir),a.bootstrap,a.seed)
        eligible=(ci['book_count']>=a.min_books and ci['book_bootstrap_gain_ci95'][0]>0
          and ci['book_bootstrap_memory_advantage_ci95'][0]>0 and paired['gain']>0 and paired['advantage']>0
          and overlap is not None and overlap['test_overlap_screen_passed']
          and overlap['new_books']['test']['known_token_window_fraction']>=0.05)
        report={'newbooks':'NaNROUTER independent whole-book evaluation','book_hash_disjoint':True,
           'legacy_fingerprint_audit':overlap,
           'source_overlap_with_legacy_unverified':overlap is None,
           'candidate_permitted_for_review_not_automatically_promoted':bool(eligible),
           'accepted_checkpoint_promoted':False,'parent_sha256':PARENT,'accepted_unchanged':True,
           'candidate_adapter_sha256':sha256(a.overlay),'train_dev_test_book_counts':books,
           'sample_count':a.test,'metrics':paired,'book_cluster_bootstrap':ci,
           'meets_new_book_promotion_preconditions':bool(eligible),'elapsed_s':round(time.monotonic()-t,2)}
    out=pathlib.Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'phase':'DONE','audit_only':a.audit_only,'newbook':bool(not a.audit_only),
        'review_ready':report.get('meets_new_book_promotion_preconditions',False),'path':str(out)}),flush=True)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True)
    p.add_argument('--corpus-dir',type=pathlib.Path,required=True)
    p.add_argument('--overlay',type=pathlib.Path,default=pathlib.Path('adaptation/fa1_r716_candidate.pt'))
    p.add_argument('--audit-only',action='store_true');p.add_argument('--skip-overlap-screen',action='store_true');p.add_argument('--test',type=int,default=2048)
    p.add_argument('--seed',type=int,default=71699);p.add_argument('--batch',type=int,default=8)
    p.add_argument('--threads',type=int,default=2);p.add_argument('--bootstrap',type=int,default=2000)
    p.add_argument('--min-books',type=int,default=12)
    p.add_argument('--output',type=pathlib.Path,default=pathlib.Path('reports/FA1_R7p16_NEW_BOOKS.json'))
    main(p.parse_args())