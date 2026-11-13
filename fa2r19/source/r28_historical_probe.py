#!/usr/bin/env python3
"""Frozen R28 vs R27 on R28-unselected but historically used PAL test books.

No fitting/step selection; not an independent new-author benchmark.
"""
import argparse,json,pathlib,torch
from r27_restore_unified import restore as restore_r27
from r28_restore_unified import restore as restore_r28
from r28_recurrent import MirrorRecurrentResidual
from r28_cpu_continue import metrics
from r13_corpus_cache import read_books_cached


def main(a):
    root=pathlib.Path(a.root).resolve();torch.set_num_threads(a.threads)
    parent,_,comp,r6,_,_=restore_r27(root,threads=a.threads)
    # Identity-initialized R28 is bitwise R27 even though the instrumented
    # stream uses the identical R28 dispatch path in both test arms.
    parent.add_module('r28',MirrorRecurrentResidual())
    candidate,_,_,_,_,_,metadata=restore_r28(root,threads=a.threads)
    books,_=read_books_cached(root/'nanrouter',comp,r6.vocab,a.max_books)
    test=books['test'][:a.test_books]
    if not test:raise ValueError('no held historical test books')
    before=metrics(parent,test,a.chunks,a.chunk_span)
    after=metrics(candidate,test,a.chunks,a.chunk_span)
    result={'format':'FA2_R28_FROZEN_HISTORICAL_V1',
      'status':'historically_consulted_test_not_blind',
      'r28_selection_step':metadata.get('r28_step',metadata.get('selected',{}).get('r28_step')),
      'books':[b.book_id for b in test], 'r27':before,'r28':after,
      'gain_bpb':before['bpb']-after['bpb'],
      'book_wins':sum(after['books'][k]<v for k,v in before['books'].items()),
      'no_updates_performed':True}
    (root/'reports/r28_historical_frozen.json').write_text(json.dumps(result,indent=2)+'\n')
    print('R28_FROZEN_HISTORICAL',json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='.')
    p.add_argument('--chunks',type=int,default=16);p.add_argument('--chunk-span',type=int,default=12)
    p.add_argument('--max-books',type=int,default=2);p.add_argument('--test-books',type=int,default=2)
    p.add_argument('--threads',type=int,default=2);main(p.parse_args())
