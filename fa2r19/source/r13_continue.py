#!/usr/bin/env python3
"""CPU-optimized equal-update PAL R12 continuation; one active checkpoint.

Reuses the exact R12 model/optimizer/RNG, the R12 matched control and promotion
checks, and swaps in a verified-equivalent cached fast path. No new model.
"""
import argparse
import r12_experiment as r12
from r13_cpu import run_window_fast
from r13_corpus_cache import read_books_cached
r12.run_window=run_window_fast
r12.read_books=read_books_cached

if __name__=='__main__':
 p=argparse.ArgumentParser()
 p.add_argument('--root',default='.');p.add_argument('--books',default='nanrouter')
 p.add_argument('--checkpoint',default='checkpoint/r12_active.pal12')
 p.add_argument('--output',default='');p.add_argument('--report',default='reports/r13_cpu_continuation.json')
 p.add_argument('--steps',type=int,default=200);p.add_argument('--batch',type=int,default=2)
 p.add_argument('--span',type=int,default=24);p.add_argument('--eval',type=int,default=96)
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
 r12.main(p.parse_args())