#!/usr/bin/env python3
"""Exact subword n-gram leakage diagnostic for archived PAL book splits.

This reports overlap, not author-disjointness; it does not use any held-out
labels for training or selection. Ignore first/last 512 tokens of each book
so shared Gutenberg headers/footers do not dominate overlap statistics.
"""
import argparse,collections,json,pathlib, time
import numpy as np


def grams(tok,n,trim):
    tok=tok[trim:len(tok)-trim] if len(tok)>trim*2 else tok[:0]
    return (tuple(int(x) for x in tok[i:i+n]) for i in range(max(0,len(tok)-n+1)))


def run(args):
    root=pathlib.Path(args.root).resolve()
    manifest=json.loads((root/'nanrouter/.pal_preprocessed'/args.cache/'index.json').read_text())
    base=root/'nanrouter/.pal_preprocessed'/args.cache
    books={split:[] for split in ('train','dev','test')}
    for doc in manifest['documents']:
        path=base/(doc['split']+'_'+doc['gutenberg_id']+'.npz')
        with np.load(path,allow_pickle=False) as d:tokens=d['tokens'].copy()
        books[doc['split']].append((doc['gutenberg_id'],tokens))
    results=[]
    for n in args.ngrams:
        trainset=set()
        for doc,tokens in books['train']:
            trainset.update(grams(tokens,n,args.trim))
        for split in ('dev','test'):
            for doc,tokens in books[split]:
                held=list(grams(tokens,n,args.trim)); unique=set(held)
                overlap=sum(g in trainset for g in held)
                unique_overlap=len(unique&trainset)
                results.append({'ngram_length':n,'split':split,'book_id':doc,
                    'sequence_total':len(held),'sequence_train_seen':overlap,
                    'fraction_of_all_ngrams_seen_in_train':overlap/max(1,len(held)),
                    'unique_total':len(unique),'unique_train_seen':unique_overlap,
                    'fraction_unique_seen_in_train':unique_overlap/max(1,len(unique))})
        print('R29_NGRAM '+json.dumps({'n':n,'train_unique':len(trainset),
            'holdout_books':len(books['dev'])+len(books['test'])}),flush=True)
    record={'format':'FA2_R29_CORPUS_LEAKAGE_SCAN_V1','cache_id':args.cache,
        'trim_each_end_tokens':args.trim,'no_author_metadata':True,
        'all_heldout_books_previously_in_pal_corpus':True,'results':results}
    dest=root/'reports/r29_corpus_overlap.json';dest.write_text(json.dumps(record,indent=2)+'\n')
    print('R29_CORPUS_OVERLAP '+json.dumps(results),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='.')
    p.add_argument('--cache',default='65a99f1275430be4a38b5d94')
    p.add_argument('--ngrams',type=int,nargs='+',default=[8,16]);p.add_argument('--trim',type=int,default=512)
    run(p.parse_args())
