#!/usr/bin/env python3
"""R7.20 independent-document packed-vs-rescan parity; promotion forbidden.

Requires independently imported and hashed NaNROUTER *whole-book* corpus,
not historical train/dev/test splits or synthetic fixtures. Paired scores on
identical contiguous positions per book. This tests numerical parity only;
the R7.18 `fa1_r718_newbooks.py` confidence-interval and history controls
remain the separate capability-promotion gate.
"""
import argparse,json,pathlib,time
import numpy as np
from r7_core import Corpus
from r710_checkpoint_integrity import sha256
from fa1_r716_router import PARENT
from fa1_r720_runtime import restore_frozen,FA1R720Runtime
from fa1_r716_newbooks import audit_documents
from fa1_r716_overlap import screen
from fa1_r719_streaming_index import EXCLUSION


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=pathlib.Path,required=True)
    p.add_argument('--corpus-dir',type=pathlib.Path,required=True)
    p.add_argument('--r716-overlay',type=pathlib.Path,required=True)
    p.add_argument('--r718-overlay',type=pathlib.Path,required=True)
    p.add_argument('--max-books',type=int,default=12)
    p.add_argument('--window',type=int,default=8)
    p.add_argument('--seed',type=int,default=720)
    p.add_argument('--skip-overlap',action='store_true',help='debug only; disables review eligibility')
    p.add_argument('--output',type=pathlib.Path,required=True)
    a=p.parse_args();beg=time.monotonic()
    if a.max_books<12 or a.window<4:raise ValueError('at least 12 books and 4 positions/book required')
    docs=audit_documents(a.root,a.corpus_dir)
    overlap=None if a.skip_overlap else screen(a.root,a.corpus_dir)
    corpus=Corpus(a.root,'test',corpus_dir=a.corpus_dir)
    if corpus.doc_start is None:raise ValueError('missing whole-book boundaries')
    starts=np.unique(np.asarray(corpus.doc_start,dtype=np.int64))
    ends=np.r_[starts[1:],len(corpus.seq)]
    eligible=[(int(s),int(e)) for s,e in zip(starts,ends) if e-s>EXCLUSION+a.window+16]
    if len(eligible)<a.max_books:raise ValueError('insufficient independently eligible whole books')
    rng=np.random.default_rng(a.seed)
    sampled=rng.choice(len(eligible),a.max_books,replace=False)
    model,opts,base,comp,near,far=restore_frozen(a.root,a.r716_overlay,a.r718_overlay)
    if len(opts)!=5:raise AssertionError('optimizer history lost')
    rt1=FA1R720Runtime(a.root,model,base,comp,near,far,'test',a.corpus_dir)
    rt2=FA1R720Runtime(a.root,model,base,comp,near,far,'test',a.corpus_dir)
    perbook=[];worst=0.
    for i in sampled:
        start,end=eligible[int(i)]
        p=start+EXCLUSION+1+int(rng.integers(0,end-start-EXCLUSION-a.window-1))
        inds=np.arange(p,p+a.window,dtype=np.int64)
        old=rt1.score_positions(inds,mode='exact_reference',batch_size=a.window)
        new=rt2.score_positions(inds,mode='packed_index',batch_size=a.window)
        error=max(abs(x-y) for key in ('baseline_nll','near_nll','far_nll') for x,y in zip(old[key],new[key]))
        if error>2e-6:raise AssertionError(f'book {start} numerical parity failed: {error}')
        worst=max(worst,error)
        perbook.append({'book_start':start,'book_end':end,'position_first':p,
                        'positions':a.window,'max_abs_error':error,
                        'r718_reference_far_nll':float(np.mean(old['far_nll'])),
                        'r720_packed_far_nll':float(np.mean(new['far_nll']))})
    if sha256(a.root/'checkpoint/pal_active.pt')!=PARENT:raise AssertionError('accepted checkpoint mutated')
    report={'round':'FA1 R7.20 independent whole-book numerical parity',
        'corpus_dir':str(a.corpus_dir),'books_available_eligible':len(eligible),
        'books_sampled':len(perbook),'test_positions':a.window*len(perbook),
        'maximum_abs_parity_error':worst,'book_results':perbook,
        'document_integrity':docs,'legacy_overlap_screen':overlap,
        'parity_passed':True,'capability_promotion_gate':'SEPARATE R7.18 newbooks bootstrap and shuffled/donor counterfactuals',
        'not_a_capability_improvement_test':True,'no_checkpoint_write':True,'promoted':False,
        'elapsed_seconds':time.monotonic()-beg}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'round':report['round'],'books_sampled':len(perbook),
                      'max_error':worst,'promoted':False,'report':str(a.output)}))
    return report
if __name__=='__main__':main()