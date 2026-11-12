#!/usr/bin/env python3
"""Single post-selection book-disjoint evaluation; NEVER update model or choose on test.

Compares exact inherited R11 to active R12, same NaNROUTER books/windows/byte tokenizer.
R12 ablations turn off learned additive paths without changing parameters.
This is an exploratory historical test: books may overlap other pre-R11 PAL work.
"""
import argparse,hashlib,json,pathlib,torch
from r12_experiment import load_for_arm,evaluate
from r11_raw import read_books,windows

def main(a):
    torch.set_num_threads(a.threads)
    r11=pathlib.Path(a.r11);r12=pathlib.Path(a.r12)
    s11,old,comp11,m11,b11,adapter11,opt11,rg11=load_for_arm(a.root,r11,True)
    s12,new,comp12,m12,b12,adapter12,opt12,rg12=load_for_arm(a.root,r12,True)
    assert s11['format']=='PAL11_RAW_SINGLE_V1' and s12['format']=='PAL12_CAUSAL_RESIDUAL_V1'
    assert old.vocab==new.vocab
    docs,manifest=read_books(a.books,comp12,old.vocab,0)
    selections=windows(docs,a.span,a.windows,a.max_tokens)
    samples=selections['test']
    summary={
       'selection':'Frozen hashes and sample algorithm, only evaluated after R12 development selection',
       'test_caveat':'Prior PAL data overlap beyond known Gutenberg IDs has not been exhaustively excluded; exploratory, not independently sealed',
       'book_counts':{k:len(v) for k,v in docs.items()},
       'sample_book_ids':sorted(set(docs['test'][i].book_id for i,k in samples)),
       'sample_windows':len(samples),'span':a.span,'max_tokens':a.max_tokens,
       'checkpoint_SHA256':{'r11':hashlib.sha256(r11.read_bytes()).hexdigest(),'r12':hashlib.sha256(r12.read_bytes()).hexdigest()},
       'r11':evaluate(m11,b11,adapter11,docs['test'],samples,a.span,a.batch),
       'r12_full':evaluate(m12,b12,adapter12,docs['test'],samples,a.span,a.batch),
       'r12_no_attention':evaluate(m12,b12,adapter12,docs['test'],samples,a.span,a.batch,disable_attention=True),
       'r12_no_ffn':evaluate(m12,b12,adapter12,docs['test'],samples,a.span,a.batch,disable_ffn=True),
       'r12_no_both':evaluate(m12,b12,adapter12,docs['test'],samples,a.span,a.batch,disable_ffn=True,disable_attention=True),
       'r12_r10_updates':s12['r10_updates'],'r12_r11_updates':s12['r11_updates'],
       'r12_updates':s12['r12_updates'],
    }
    pathlib.Path(a.output).write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({'R11':summary['r11'],'R12':summary['r12_full'],'status':'completed','path':a.output},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='.');p.add_argument('--books',default='nanrouter');
    p.add_argument('--r11',required=True);p.add_argument('--r12',default='checkpoint/r12_active.pal12');
    p.add_argument('--output',default='reports/r12_heldout_14books.json');p.add_argument('--windows',type=int,default=160);
    p.add_argument('--span',type=int,default=24);p.add_argument('--max-tokens',type=int,default=160)
    p.add_argument('--batch',type=int,default=2);p.add_argument('--threads',type=int,default=2)
    main(p.parse_args())