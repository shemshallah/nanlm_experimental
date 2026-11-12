#!/usr/bin/env python3
"""Read-only whole-book evaluation on external NaNROUTER Gutenberg release files.

Accepts already extracted book .txt files; no external network required. Runs
content-hash deduplication and an approximate 16-token n-gram contamination
screen against the original historical training corpus. NEVER trains on new
books and NEVER changes tokenizer, vocabulary, historical corpus or checkpoint.
"""
import argparse,json,hashlib,pathlib,tempfile,sys,os,time
import numpy as np,torch
from prepare_document_corpus import prepare
from eval_r77_fast import evaluate_positions
from r77_semantic import restore
from r7_core import Corpus

NGRAM=16
BASE=np.uint64(1315423911)

def fingerprint(seq,n=NGRAM,block=100000):
    """Stable modular polynomial fingerprint for windows with >= 75% known tokens."""
    seq=np.asarray(seq,dtype=np.int64)
    if len(seq)<n:return np.empty(0,dtype=np.uint64)
    powers=np.array([pow(int(BASE),j,2**64) for j in range(n)],dtype=np.uint64)
    from numpy.lib.stride_tricks import sliding_window_view
    pieces=[]
    for start in range(0,len(seq)-n+1,block):
        stop=min(start+block+n-1,len(seq))
        part=np.asarray(seq[start:stop],dtype=np.int64)
        windows=sliding_window_view(part,n)
        eligible=(windows>1).sum(1)>=int(n*.75)
        if not np.any(eligible):continue
        hashvals=((windows[eligible].astype(np.uint64)+np.uint64(1))*powers).sum(1,dtype=np.uint64)
        pieces.append(hashvals)
    return np.concatenate(pieces) if pieces else np.empty(0,dtype=np.uint64)

def screen(train_corpus,new_corpus):
    source_hashes=np.unique(fingerprint(train_corpus.seq))
    records=[]
    documents=json.loads((new_corpus.root/'new_books_corpus'/f'{new_corpus.part}_documents.json').read_text())['documents']
    # Document starts reflect exact offsets, never synthetic 2048-window resets.
    for j,doc in enumerate(documents):
        s=int(doc['first_record']);e=s+int(doc['words'])
        fp=fingerprint(new_corpus.seq[s:e])
        overlap=int(np.isin(fp,source_hashes,assume_unique=False).sum()) if len(fp) else 0
        rate=overlap/max(1,len(fp))
        # Strong signal of book reuse, copied passages, or test contamination.
        status='REJECT_OVERLAP' if overlap>128 or rate>.005 else 'SCREENED_NO_LARGE_OVERLAP'
        records.append({**doc,'eligible_16grams':int(len(fp)),'overlapping_16grams':overlap,
                        'overlap_fraction':rate,'screening_status':status})
    return records

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=pathlib.Path,default=pathlib.Path('.'))
    ap.add_argument('--books-dir',type=pathlib.Path,required=True)
    ap.add_argument('--limit',type=int,default=50)
    ap.add_argument('--max-positions',type=int,default=4096)
    ap.add_argument('--split',choices=('test','dev'),default='test')
    ap.add_argument('--batch',type=int,default=8)
    ap.add_argument('--threads',type=int,default=2)
    ap.add_argument('--output',type=pathlib.Path,default=pathlib.Path('reports/R7p7_newbooks_audit.json'))
    a=ap.parse_args();torch.set_num_threads(a.threads);start=time.monotonic();root=a.root.resolve()
    if not a.books_dir.exists():raise FileNotFoundError(a.books_dir)
    # Stage new prepared corpus into its own directory; never alter 'corpus/'.
    extension=root/'new_books_corpus';extension.mkdir(parents=True,exist_ok=True)
    prepare(root,a.books_dir,extension,a.limit)
    model,opts,base,comp,prior=restore(root,root/'checkpoint/pal_active.pt',torch.device('cpu'))
    old_train=Corpus(root,'train');new_data=Corpus(root,a.split,extension)
    fingerprint_result=screen(old_train,new_data)
    # A full 8,192-token initial context must belong to the SAME book.
    # This is essential: Corpus.get() is unaware of book boundaries.
    passed_docs=[d for d in fingerprint_result if d['screening_status']=='SCREENED_NO_LARGE_OVERLAP']
    valid_docs=[d for d in passed_docs if int(d['words'])>8192]
    if not valid_docs:raise RuntimeError('No screened holdout books have >8192 tokens for leak-free older-memory evaluation')
    positions=[]
    each=max(1,a.max_positions//len(valid_docs))
    for d in valid_docs:
        first=int(d['first_record']);length=int(d['words'])
        available=length-8192
        offsets=np.arange(min(each,available),dtype=np.int64)*available//min(each,available)
        positions.extend(first+8192+offsets)
    positions=np.asarray(sorted(set(positions)),dtype=np.int64)
    assert all(any(int(d['first_record'])+8192<=i<int(d['first_record'])+int(d['words']) for d in valid_docs) for i in positions)
    metrics=evaluate_positions(model,base,comp,new_data,positions,a.batch)
    metadata={'checkpoint_step':prior['joint_steps'],'split':a.split,
              'source_path':str(a.books_dir.resolve()),
              'source_kind':'user-provided external .txt files; provenance not independently verified by this evaluator',
              'new_books_trained':False,'historical_train_corpus_unchanged':True,
              'release_provenance_verified_in_evaluator':False,'claim_independent_holdout':False,
              'holdout_books_available':len(fingerprint_result),'holdout_books_screened':len(passed_docs),'holdout_books_evaluable':len(valid_docs),
              'holdout_books_longer_than_8192_tokens':sum(d['words']>8192 for d in valid_docs),
              'screening':'document SHA deduplication + historical training 16-token rolling hash',
              'screening_is_not_proof_of_zero_leakage':True,'all_eval_contexts_within_one_document':True,'warmup_words_before_first_evaluated_target':8192,
              'documents':fingerprint_result,'metrics':metrics,'elapsed_seconds':round(time.monotonic()-start,2)}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(metadata,indent=2))
if __name__=='__main__':main()
