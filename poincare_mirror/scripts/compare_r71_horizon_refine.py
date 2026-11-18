#!/usr/bin/env python3
"""Book-disjoint, resumable DEV comparison of R69 and R71; TEST stays sealed."""
import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from book_stream_eval import evaluate_causal_book
from fa2_active import load_active_fa2
from poincare_frontier import Config,MirrorLM,file_hash,load_v25_compatible
from compare_r70_future_winner_guard import bootstrap


def run(parent,candidate,data_dir,output,books=12,positions=128,warmup=128,
        context=64,chunk=16,history=32768,threads=2):
    if min(books,positions,warmup,context,chunk,history,threads)<1 or books<2:
        raise ValueError('R71 needs >=2 books and positive bounded controls')
    torch.set_num_threads(threads)
    meta=json.loads((data_dir/'meta.json').read_text())
    if meta.get('split_unit')!='whole_book' or meta.get('unit')!='UTF8_bytes_plus_eos':
        raise ValueError('Expected whole-book NaNROUTER validation data')
    if file_hash(data_dir/'valid.bin')!=meta['valid_sha256'] or file_hash(
            data_dir/'split_manifest.json')!=meta['split_manifest_sha256']:
        raise ValueError('DEV corpus integrity failure')
    split=json.loads((data_dir/'split_manifest.json').read_text())
    names=[x['archive_name'] for x in split['books'] if x['split']=='dev']
    lengths=meta['valid_doc_lengths']
    mem=np.memmap(data_dir/'valid.bin',dtype=meta['token_dtype'],mode='r')
    if len(names)!=len(lengths) or sum(lengths)!=len(mem):
        raise ValueError('Misaligned or incomplete DEV book boundaries')
    root,_,saved=load_active_fa2(parent)
    trial=torch.load(candidate,map_location='cpu',weights_only=False)
    if (trial['step']!=4529 or trial['config']!=saved['config'] or
            trial['data']!=saved['data'] or len(trial['optimizer']['state'])!=198):
        raise AssertionError('R71 candidate structure, corpus or AdamW lineage changed')
    for name,tensor in saved['model'].items():
        if name!='pair_successor.future_weight' and not torch.equal(
                tensor,trial['model'][name]):
            raise AssertionError('R71 candidate mutated inherited tensor '+name)
    trained=MirrorLM(Config(**trial['config'])).eval()
    load_v25_compatible(trained,trial['model'])
    identity={'r69_sha256':file_hash(parent),'r71_sha256':file_hash(candidate),
              'dev_sha256':meta['valid_sha256'],'warmup':warmup,
              'positions':positions,'context':context,'chunk':chunk,
              'history':history,'space_rank_guard':True,'test_opened':False}
    selected=min(books,len(names))
    rows=[]
    if output.is_file():
        old=json.loads(output.read_text())
        if any(old.get(key)!=value for key,value in identity.items()):
            raise ValueError('Refuse resume: candidate/DEV protocol changed')
        rows=old['per_book']
        if len(rows)>selected or [r['book_id'] for r in rows]!=names[:len(rows)]:
            raise ValueError('Refuse resume: book identities reordered')
        print(f'Resuming {len(rows)} checked R71 DEV books',flush=True)
    def save(report):
        output.parent.mkdir(parents=True,exist_ok=True)
        pending=output.with_name(output.name+'.incomplete')
        pending.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        pending.replace(output)
    offset=0
    started=perf_counter()
    for i,(name,length) in enumerate(zip(names,lengths)):
        if i>=selected: break
        if i<len(rows):
            offset+=length
            continue
        tokens=torch.tensor(np.asarray(mem[offset:offset+length],dtype=np.int64),
                            dtype=torch.long)
        offset+=length
        kw={'context':context,'chunk':chunk,'history':history,
            'score_start':warmup,'score_positions':positions,
            'space_rank_guard':True}
        baseline=evaluate_causal_book(root,tokens,name,**kw)
        upgraded=evaluate_causal_book(trained,tokens,name,**kw)
        for key in ('ce_1','acc_1','top5_1','bits_per_raw_byte'):
            if abs(baseline['exact'][key]-upgraded['exact'][key])>1e-8:
                raise AssertionError('R71 mutated next-byte predictor '+key)
        rows.append({'book_id':name,'r69':baseline['exact'],
                     'r71':upgraded['exact'],'neutral':baseline['neutral']})
        save({'status':'incomplete',**identity,'per_book':rows})
        print(f'R71 DEV book {i+1}/{selected}: '+
              f'+2 CE {baseline["exact"]["ce_2"]:.5f}->{upgraded["exact"]["ce_2"]:.5f}, '+
              f'+3 CE {baseline["exact"]["ce_3"]:.5f}->{upgraded["exact"]["ce_3"]:.5f}',flush=True)
    metrics=('ce_1','ce_2','ce_3','acc_1','acc_2','acc_3',
             'top5_1','top5_2','top5_3','bits_per_raw_byte')
    comparison={}
    for metric in metrics:
        previous=[row['r69'][metric] for row in rows]
        current=[row['r71'][metric] for row in rows]
        comparison[metric]={'r69':float(np.mean(previous)),
                            'r71':float(np.mean(current)),
            **bootstrap(previous,current,
                        larger_better=metric.startswith(('acc_','top5_')),
                        seed=71)}
    gate=bool(selected==len(names) and comparison['ce_2']['ci95'][0]>0 and
              comparison['ce_3']['ci95'][0]>0 and
              comparison['acc_2']['mean_gain']>=0 and
              comparison['acc_3']['mean_gain']>=0)
    final={'status':'R71 DEV candidate; no promotion without gate',
           **identity,'dev_books':len(rows),'dev_books_total':len(names),
           'seconds':perf_counter()-started,'comparison':comparison,
           'per_book':rows,'gate_pass':gate,'promoted':False}
    save(final)
    return final


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ('parent','candidate','data-dir','output'):
        p.add_argument('--'+arg,type=Path,required=True)
    p.add_argument('--books',type=int,default=12)
    p.add_argument('--warmup',type=int,default=128)
    p.add_argument('--positions',type=int,default=128)
    p.add_argument('--threads',type=int,default=2)
    a=p.parse_args()
    d=run(a.parent,a.candidate,a.data_dir,a.output,a.books,a.positions,
          a.warmup,threads=a.threads)
    print(json.dumps({'gate':d['gate_pass'],'books':d['dev_books'],
                      'metrics':d['comparison']},indent=2),flush=True)


if __name__=='__main__': main()
