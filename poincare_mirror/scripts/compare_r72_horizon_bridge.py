#!/usr/bin/env python3
"""R72 paired book-disjoint DEV gate of TRAIN-calibrated causal bridge.

Runs the SAME active R69 checkpoint in both arms; R72 only changes +3
inference when +3 geometric history is sparse. No TEST or weight updates.
"""
import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from book_stream_eval import evaluate_causal_book
from compare_r70_future_winner_guard import bootstrap
from fa2_active import load_active_fa2
from poincare_frontier import file_hash


def run(checkpoint,data_dir,report,*,bridge=.75,books=12,warmup=128,
        positions=128,threads=2):
    if not 0<bridge<=1 or books<2 or min(warmup,positions,threads)<1:
        raise ValueError('Invalid R72 DEV configuration')
    torch.set_num_threads(threads)
    meta=json.loads((data_dir/'meta.json').read_text())
    if (meta.get('split_unit')!='whole_book' or
            file_hash(data_dir/'valid.bin')!=meta['valid_sha256'] or
            file_hash(data_dir/'split_manifest.json')!=meta['split_manifest_sha256']):
        raise ValueError('Invalid independent-book NaNROUTER DEV corpus')
    splits=json.loads((data_dir/'split_manifest.json').read_text())
    names=[x['archive_name'] for x in splits['books'] if x['split']=='dev']
    lengths=meta['valid_doc_lengths']
    mem=np.memmap(data_dir/'valid.bin',dtype=meta['token_dtype'],mode='r')
    if len(names)!=len(lengths) or sum(lengths)!=len(mem):
        raise ValueError('DEV identity/offset mismatch')
    model,optimizer,saved=load_active_fa2(checkpoint,include_optimizer=True)
    if saved['step']!=4465 or len(optimizer.state)!=198:
        raise AssertionError('R72 must reuse complete active R69 checkpoint')
    identity={'checkpoint_sha256':file_hash(checkpoint),
              'dev_sha256':meta['valid_sha256'],'bridge_strength':bridge,
              'warmup':warmup,'positions':positions,'context':64,
              'chunk':16,'history':32768,'test_opened':False}
    selected=min(books,len(names))
    rows=[]
    if report.is_file():
        previous=json.loads(report.read_text())
        if any(previous.get(k)!=v for k,v in identity.items()):
            raise ValueError('Refusing mismatched R72 resume state')
        rows=previous['per_book']
        if len(rows)>selected or [r['book_id'] for r in rows]!=names[:len(rows)]:
            raise ValueError('Refusing changed R72 DEV book ordering')
    def save(obj):
        report.parent.mkdir(parents=True,exist_ok=True)
        tmp=report.with_name(report.name+'.incomplete')
        tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
        tmp.replace(report)
    offset=0
    started=perf_counter()
    for i,(name,length) in enumerate(zip(names,lengths)):
        if i>=selected: break
        if i<len(rows):
            offset+=length
            continue
        tokens=torch.from_numpy(np.asarray(mem[offset:offset+length],
                                           dtype=np.int64).copy())
        offset+=length
        kw={'context':64,'chunk':16,'history':32768,
            'score_start':warmup,'score_positions':positions,
            'space_rank_guard':True}
        baseline=evaluate_causal_book(model,tokens,name,**kw)
        candidate=evaluate_causal_book(model,tokens,name,**kw,
                                       future_horizon_bridge=bridge)
        for metric in ('ce_1','ce_2','acc_1','acc_2',
                       'top5_1','top5_2','bits_per_raw_byte'):
            if abs(baseline['exact'][metric]-candidate['exact'][metric])>1e-9:
                raise AssertionError('R72 changed protected +1/+2 path: '+metric)
        rows.append({'book_id':name,'r69':baseline['exact'],
                     'r72':candidate['exact'],'neutral':baseline['neutral']})
        save({'status':'incomplete',**identity,'per_book':rows})
        print(f'R72 DEV book {i+1}/{selected} +3 CE '+
              f'{baseline["exact"]["ce_3"]:.6f}->'+
              f'{candidate["exact"]["ce_3"]:.6f}',flush=True)
    metrics=('ce_1','ce_2','ce_3','acc_1','acc_2','acc_3',
             'top5_1','top5_2','top5_3','bits_per_raw_byte')
    comparison={}
    for key in metrics:
        old=[x['r69'][key] for x in rows]
        new=[x['r72'][key] for x in rows]
        comparison[key]={'r69':float(np.mean(old)),'r72':float(np.mean(new)),
                          **bootstrap(old,new,larger_better=key.startswith(
                              ('acc_','top5_')),seed=72)}
    gate=bool(selected==len(names) and comparison['ce_3']['ci95'][0]>0 and
              comparison['acc_3']['mean_gain']>=0 and
              comparison['top5_3']['mean_gain']>=0)
    outcome={'status':'R72 DEV-only opt-in candidate, not promoted',**identity,
             'dev_books':len(rows),'total_dev_books':len(names),
             'comparison':comparison,'seconds':perf_counter()-started,
             'per_book':rows,'gate_pass':gate,'promoted':False,
             'unchanged_optimizer_state_count':198}
    save(outcome)
    return outcome


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','data-dir','report'):
        p.add_argument('--'+name,required=True,type=Path)
    p.add_argument('--bridge',type=float,default=.75)
    p.add_argument('--books',type=int,default=12)
    p.add_argument('--warmup',type=int,default=128)
    p.add_argument('--positions',type=int,default=128)
    p.add_argument('--threads',type=int,default=2)
    a=p.parse_args()
    result=run(a.checkpoint,a.data_dir,a.report,bridge=a.bridge,
               books=a.books,warmup=a.warmup,positions=a.positions,
               threads=a.threads)
    print('R72 DEV results',json.dumps({'books':result['dev_books'],
       'gate_pass':result['gate_pass'],'metrics':result['comparison']}),flush=True)


if __name__=='__main__': main()
