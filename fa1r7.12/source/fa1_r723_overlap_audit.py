#!/usr/bin/env python3
"""32-token exact-overlap audit: new book splits versus inherited R7 train.

Hashing is a candidate search; exact full token arrays verify every match.
No data is changed, and this audit never chooses a checkpoint on TEST.
"""
from __future__ import annotations
import argparse,json,pathlib
import numpy as np
from r7_core import Corpus

BASE=np.uint64(11400714819323198485)

def hashes(seq,window=32):
    seq=np.asarray(seq,dtype=np.uint64)
    if len(seq)<window:return np.empty(0,dtype=np.uint64)
    n=len(seq)-window+1
    h=np.zeros(n,dtype=np.uint64)
    for j in range(window):h=h*BASE+(seq[j:j+n]+np.uint64(0x9e3779b97f4a7c15))
    return h

def run(a):
    root=pathlib.Path(a.root);external=pathlib.Path(a.external)
    accepted=Corpus(root,'train')
    old_ids=accepted.seq
    oldhash=hashes(old_ids,a.window)
    keep=np.argsort(oldhash,kind='stable')
    sorted_h=oldhash[keep]
    report={'window_words':a.window,'old_train_tokens':len(old_ids),
            'new_corpus_root':str(external),'parts':{},
            'all_matches_exactly_verified':True,'whole_book_split_required':True}
    for part in ('train','dev','test'):
        doc=Corpus(root,part,corpus_dir=external)
        h=hashes(doc.seq,a.window)
        starts=np.asarray(doc.doc_start,dtype=np.int64)
        position=np.arange(0,len(h),a.stride,dtype=np.int64)
        position=position[position+a.window-1<len(doc.seq)]
        position=position[starts[position]==starts[position+a.window-1]]
        # Exclude padding: long sequences of BOS create spurious matches.
        position=position[np.asarray(doc.seq[position],dtype=np.int64)>1]
        q=h[position]
        ix=np.searchsorted(sorted_h,q)
        mask=ix<len(sorted_h)
        rows=np.flatnonzero(mask)
        rows=rows[sorted_h[np.minimum(ix[rows],len(sorted_h)-1)]==q[rows]]
        confirmations=[]
        for j in rows:
            newpos=int(position[j]);hashv=q[j]
            lo=int(np.searchsorted(sorted_h,hashv,'left'))
            hi=int(np.searchsorted(sorted_h,hashv,'right'))
            for k in range(lo,min(hi,lo+32)):
                oldpos=int(keep[k])
                if np.array_equal(np.asarray(old_ids[oldpos:oldpos+a.window]),
                                  np.asarray(doc.seq[newpos:newpos+a.window])):
                    confirmations.append({'new_position':newpos,'new_doc_start':int(doc.doc_start[newpos]),'old_train_position':oldpos})
                    break
        unique_doc=len(np.unique(starts))
        contaminated=len(set(x['new_doc_start'] for x in confirmations))
        report['parts'][part]={'whole_books':unique_doc,'tokens':len(doc.seq),
                    'windows_sampled':len(position),'exact_old_train_matches':len(confirmations),
                    'exact_match_fraction':len(confirmations)/max(1,len(position)),
                    'books_with_detected_overlap':contaminated,
                    'first_matches':confirmations[:12]}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',default='/mnt/data/diamond_r723/r7_ancestor')
    p.add_argument('--external',default='/mnt/data/diamond_r723/nanrouter_verified/processed')
    p.add_argument('--window',type=int,default=32)
    p.add_argument('--stride',type=int,default=32)
    p.add_argument('--output',type=pathlib.Path,default=pathlib.Path('/mnt/data/diamond_r723/nanrouter_overlap.json'))
    run(p.parse_args())