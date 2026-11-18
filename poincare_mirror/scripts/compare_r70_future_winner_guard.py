#!/usr/bin/env python3
"""R70 opt-in rank guard on genuine book-disjoint NaNROUTER DEV.

The R69 checkpoint remains the only model/optimizer; comparison toggles only
the inference-time future-memory rank guard. TEST is deliberately inaccessible.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from book_stream_eval import evaluate_causal_book
from fa2_active import load_active_fa2


def checksum(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024),b''):
            digest.update(chunk)
    return digest.hexdigest()


def bootstrap(before,after,*,larger_better,seed=70,repetitions=2000):
    a=np.asarray(before,dtype=float)
    b=np.asarray(after,dtype=float)
    if a.shape!=b.shape or len(a)<2:
        raise ValueError('Paired book-level bootstrap requires >=2 books')
    delta=(b-a if larger_better else a-b)
    samples=np.random.default_rng(seed).integers(0,len(a),(repetitions,len(a)))
    bounds=np.quantile(delta[samples].mean(1),[.025,.975])
    return {'mean_gain':float(delta.mean()),'ci95':bounds.tolist(),
            'books_improved':int((delta>0).sum())}


def run(checkpoint,data_dir,*,books=None,warmup=128,positions=128,
        context=64,chunk=16,history=32768,threads=2):
    if min(warmup,positions,context,chunk,history,threads)<1:
        raise ValueError('All R70 DEV experiment quantities must be positive')
    torch.set_num_threads(threads)
    meta=json.loads((data_dir/'meta.json').read_text())
    if meta.get('split_unit')!='whole_book' or meta.get('unit')!='UTF8_bytes_plus_eos':
        raise ValueError('Unverified NaNROUTER whole-book corpus')
    valid=data_dir/'valid.bin'
    if checksum(valid)!=meta['valid_sha256']:
        raise ValueError('DEV corpus SHA-256 mismatch')
    split=json.loads((data_dir/'split_manifest.json').read_text())
    names=[entry['archive_name'] for entry in split['books']
           if entry['split']=='dev']
    lengths=meta['valid_doc_lengths']
    if len(names)!=len(lengths):
        raise ValueError('Book-identity metadata inconsistent')
    mem=np.memmap(valid,dtype=meta['token_dtype'],mode='r')
    if int(sum(lengths))!=len(mem):
        raise ValueError('DEV offsets mismatch')
    model,_,saved=load_active_fa2(checkpoint)
    if saved['step']!=4465 or not model.cfg.pair_successor_future:
        raise ValueError('R70 requires the verified R69 step-4465 model')
    rows=[]
    started=perf_counter()
    offset=0
    for index,(name,length) in enumerate(zip(names,lengths)):
        if books is not None and index>=books:
            break
        token_bytes=np.asarray(mem[offset:offset+length],dtype=np.int64)
        offset+=length
        tokens=torch.tensor(token_bytes,dtype=torch.long)
        kwargs=dict(context=context,chunk=chunk,history=history,
                    score_start=warmup,score_positions=positions,
                    space_rank_guard=True)
        r69=evaluate_causal_book(model,tokens,name,**kwargs)
        r70=evaluate_causal_book(model,tokens,name,**kwargs,
                                 future_winner_guard=True)
        for branch in ('neutral','exact'):
            for metric in ('acc_2','acc_3'):
                if (branch=='neutral' and
                        abs(r69[branch][metric]-r70[branch][metric])>1e-10):
                    raise AssertionError('Neutral future route changed')
        if abs(r70['exact']['acc_2']-r70['neutral']['acc_2'])>1e-10 or abs(
                r70['exact']['acc_3']-r70['neutral']['acc_3'])>1e-10:
            raise AssertionError('R70 winner rank not preserved')
        rows.append({'book_id':name,'r69':r69['exact'],'r70':r70['exact'],
                     'neutral':r69['neutral']})
        print(f'R70 DEV book {index+1}/{books or len(names)} '+
              f'+2 CE {r69["exact"]["ce_2"]:.5f}->{r70["exact"]["ce_2"]:.5f} '+
              f'+3 CE {r69["exact"]["ce_3"]:.5f}->{r70["exact"]["ce_3"]:.5f}',
              flush=True)
    metrics=('ce_1','ce_2','ce_3','acc_1','acc_2','acc_3',
             'top5_1','top5_2','top5_3','bits_per_raw_byte')
    comparison={}
    for metric in metrics:
        old=[x['r69'][metric] for x in rows]
        new=[x['r70'][metric] for x in rows]
        comparison[metric]={'r69':float(np.mean(old)),'r70':float(np.mean(new)),
            **bootstrap(old,new,larger_better=metric.startswith(('acc_','top5_')))}
    return {'status':'DEV-only R70 candidate, no promotion',
            'checkpoint_step':saved['step'],'checkpoint_sha256':checksum(checkpoint),
            'dev_sha256':meta['valid_sha256'],'dev_books':len(rows),
            'dev_positions_per_book':positions,'warmup':warmup,
            'seconds':perf_counter()-started,'comparison':comparison,
            'per_book':rows,'test_opened':False,
            'gate_pass':bool(comparison['ce_2']['ci95'][0]>0 and
                             comparison['ce_3']['ci95'][0]>0 and
                             comparison['acc_2']['mean_gain']>=0 and
                             comparison['acc_3']['mean_gain']>=0)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True,type=Path)
    parser.add_argument('--data-dir',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--books',type=int,default=12)
    parser.add_argument('--warmup',type=int,default=128)
    parser.add_argument('--positions',type=int,default=128)
    parser.add_argument('--threads',type=int,default=2)
    args=parser.parse_args()
    results=run(args.checkpoint,args.data_dir,books=args.books,
                warmup=args.warmup,positions=args.positions,threads=args.threads)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(results,indent=2)+'\n')
    print('R70 comparison '+json.dumps({'books':results['dev_books'],
          'gate_pass':results['gate_pass'], 'metrics':results['comparison']}),flush=True)


if __name__=='__main__':
    main()
