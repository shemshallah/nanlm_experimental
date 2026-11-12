#!/usr/bin/env python3
"""PAL CPU continuation broadening train windows while retaining unified R12 model.
The original optimizer, RNG, evaluation windows, controls and promotion guards remain.
"""
import argparse, json, pathlib, hashlib
import r14_slim_base as r12
from r13_cpu import run_window_fast
from r13_corpus_cache import read_books_cached
from r11_raw import windows as original_windows
r12.run_window = run_window_fast
r12.read_books = read_books_cached

def main(args):
    def broadened_windows(docs, span, evaluation_n, max_tokens):
        pools = original_windows(docs, span, max(evaluation_n,args.train_pool), max_tokens)
        for key in ('dev','test'):
            pools[key] = pools[key][:evaluation_n]
        print(json.dumps({'phase':'BROADENED_SPLIT','training_pool':len(pools['train']),
                          'development_windows':len(pools['dev']),
                          'training_books':len(docs['train']),
                          'unique_training_books':len({i for i,_ in pools['train']}),
                          'training_pool_sha256':hashlib.sha256(json.dumps(pools['train']).encode()).hexdigest()}),flush=True)
        return pools
    r12.windows=broadened_windows
    r12.main(args)
    p=pathlib.Path(args.report)
    if p.exists():
        r=json.loads(p.read_text());r['round_label']='R14_CPU_BROAD_TRAIN';r['train_pool_requested']=args.train_pool
        r['train_source']='NaNROUTER 12 historical train books, broadened positions within same books'
        r['dev_source']='historical development books, 96 fixed windows'
        r['no_new_books']=True
        p.write_text(json.dumps(r,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',default='.');p.add_argument('--books',default='nanrouter')
    p.add_argument('--checkpoint',default='checkpoint/r12_active.pal12')
    p.add_argument('--output',default='');p.add_argument('--report',default='reports/r14_cpu_broadtrain.json')
    p.add_argument('--steps',type=int,default=200);p.add_argument('--batch',type=int,default=2)
    p.add_argument('--span',type=int,default=24);p.add_argument('--eval',type=int,default=96)
    p.add_argument('--train-pool',type=int,default=8000)
    p.add_argument('--max-tokens',type=int,default=160);p.add_argument('--max-books',type=int,default=0)
    p.add_argument('--eval-batch',type=int,default=2);p.add_argument('--capacity',type=int,default=64)
    p.add_argument('--topk',type=int,default=8);p.add_argument('--threads',type=int,default=2)
    p.add_argument('--word-eval',type=int,default=64);p.add_argument('--future-eval',type=int,default=48)
    p.add_argument('--word-margin',type=float,default=.0008)
    p.add_argument('--future-margin',type=float,default=.00045)
    p.add_argument('--min-gain',type=float,default=.0001)
    p.add_argument('--seed',type=int,default=20261111)
    p.add_argument('--promote',action='store_true')
    p.add_argument('--skip-ablations',action='store_true',default=True)
    main(p.parse_args())
