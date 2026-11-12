#!/usr/bin/env python3
"""Known-vocabulary 16-gram fingerprint screen for legacy/new-book text overlap.

If any exact 16-token run is observed in old corpora, block promotion.
A clean screen does not guarantee copyright/source/book disjointness: OOV spans
and unusual normalization can evade a fingerprint.
"""
import argparse,json,pathlib
import numpy as np
from r7_core import Corpus

K=16
BASE=np.uint64(11400714819323198485)

def windows(seq,k=K):
    s=np.asarray(seq,dtype=np.int64)
    n=len(s)-k+1
    if n<=0:return np.empty(0,dtype=np.uint64),0
    x=np.zeros(n,dtype=np.uint64)
    for j in range(k):
        with np.errstate(over='ignore'):
            x=x*BASE+(s[j:j+n]+3).astype(np.uint64)
    bad=(s<=1).astype(np.int64)
    prefix=np.empty(len(s)+1,dtype=np.int64)
    prefix[0]=0;np.cumsum(bad,out=prefix[1:])
    valid=prefix[k:]-prefix[:-k]==0
    return x[valid],int(valid.sum())

def screen(root,corpus_dir):
    root=pathlib.Path(root);corpus_dir=pathlib.Path(corpus_dir)
    older=[];counts={}
    for split in ('train','dev','test'):
        corpus=Corpus(root,split)
        h,n=windows(corpus.seq)
        older.append(h);counts['legacy_'+split+'_known_16grams']=n
    legacy=np.unique(np.concatenate(older))
    results={};test_hits=0
    for split in ('train','dev','test'):
        c=Corpus(root,split,corpus_dir=corpus_dir)
        h,n=windows(c.seq)
        hit=np.isin(h,legacy,assume_unique=False)
        num=int(hit.sum())
        results[split]={'known_token_16gram_windows':n,'matched_to_legacy':num,
           'matched_fraction':float(num/max(1,n)),
           'known_token_window_fraction':float(n/max(1,len(c.seq)-K+1))}
        if split=='test':test_hits=num
    result={'method':'64bit_polynomial_fingerprint_exact_known_word_16gram',
       'legacy_windows':counts,'unique_legacy_fingerprints':len(legacy),
       'new_books':results,'test_overlap_screen_passed':test_hits==0,
       'not_a_proof_of_no_document_overlap':True}
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True)
    p.add_argument('--corpus-dir',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True)
    a=p.parse_args();report=screen(a.root,a.corpus_dir)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'test_overlap_screen_passed':report['test_overlap_screen_passed'],
       'matched_newbook_test_16grams':report['new_books']['test']['matched_to_legacy']}))