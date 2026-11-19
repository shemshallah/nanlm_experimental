#!/usr/bin/env python3
"""R76 sealed-TEST, paired book-disjoint DEV promotion gate.

This script only accepts a coefficient already fit on TRAIN and validated
on separately withheld TRAIN books. It may never select a new coefficient
from DEV results. It leaves the active R69 checkpoint untouched.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from calibrate_r76_head_weight import evaluate_book,METRICS
from fa2_active import load_active_fa2
from poincare_frontier import file_hash


def paired_bootstrap(a,b,*,higher=False,seed=76):
    old=np.asarray(a,dtype=np.float64)
    new=np.asarray(b,dtype=np.float64)
    gain=new-old if higher else old-new
    rng=np.random.default_rng(seed)
    indices=rng.integers(0,len(gain),(10000,len(gain)))
    values=gain[indices].mean(1)
    return {'gain':float(gain.mean()),'ci95':np.quantile(values,[.025,.975]).tolist(),
            'books_improved':int((gain>0).sum())}


def run(checkpoint,data_dir,train_report,report,*,books=12,
        warmup=256,positions=128,threads=2):
    trained=json.loads(train_report.read_text())
    if (trained['round']!='FA2_R76_CONFIDENCE_BACKOFF' or
            trained['parent_checkpoint_sha256']!=file_hash(checkpoint) or
            trained['warmup']!=warmup or trained['confirmation_ce1_gain']<=0):
        raise ValueError('R76 requires positive independent TRAIN confirmation')
    coefficient=float(trained['selected_head_backoff_weight'])
    if coefficient==1.:
        raise ValueError('No candidate difference to test')
    torch.set_num_threads(threads)
    model,optimizer,saved=load_active_fa2(checkpoint,include_optimizer=True)
    if saved['step']!=4465 or len(optimizer.state)!=198:
        raise ValueError('R76 DEV requires accepted inherited R69')
    meta=json.loads((data_dir/'meta.json').read_text())
    if (meta.get('split_unit')!='whole_book' or
            file_hash(data_dir/'valid.bin')!=meta['valid_sha256'] or
            file_hash(data_dir/'split_manifest.json')!=meta['split_manifest_sha256']):
        raise ValueError('DEV corpus metadata/checksum mismatch')
    split=json.loads((data_dir/'split_manifest.json').read_text())
    names=[x['archive_name'] for x in split['books'] if x['split']=='dev']
    lengths=meta['valid_doc_lengths']
    mem=np.memmap(data_dir/'valid.bin',dtype=meta['token_dtype'],mode='r')
    if len(names)!=len(lengths) or len(mem)!=sum(lengths):
        raise ValueError('DEV book IDs and lengths mismatch')
    rows=[]
    offset=0
    for i,(identity,length) in enumerate(zip(names,lengths)):
        if i>=books:break
        tokens=torch.from_numpy(np.asarray(mem[offset:offset+warmup+positions+3],
                                          dtype=np.int64).copy())
        offset+=int(length)
        row=evaluate_book(model,tokens,identity,warmup=warmup,
                          positions=positions,weights=(1.,coefficient))
        rows.append({'book_id':identity,'parent':row['metrics']['1.0'],
                     'candidate':row['metrics'][str(coefficient)],
                     'changed_positions':row['changed_positions_against_low_head_weight']})
        print(f'R76 DEV book {i+1}/{books}: CE1 '+
              f'{rows[-1]["parent"]["ce_1"]:.6f}->'+
              f'{rows[-1]["candidate"]["ce_1"]:.6f}',flush=True)
    comparison={}
    for metric in (*METRICS,'bpb_1'):
        old=[r['parent'][metric] for r in rows]
        new=[r['candidate'][metric] for r in rows]
        comparison[metric]={'parent':float(np.mean(old)),
            'candidate':float(np.mean(new)),
            **paired_bootstrap(old,new,higher=metric.startswith(('acc_','top5_')))}
    gate=bool(len(rows)==len(names) and
       all(comparison[k]['ci95'][0]>0 for k in ('ce_1','ce_2','ce_3')) and
       all(comparison[k]['gain']>=0 for k in
           ('acc_1','acc_2','acc_3','top5_1','top5_2','top5_3')))
    result={'round':'FA2_R76_DEV_GATE','status':'DEV-only comparison',
        'checkpoint_sha256':file_hash(checkpoint),'dev_sha256':meta['valid_sha256'],
        'frozen_train_coefficient':coefficient,'books':len(rows),
        'warmup':warmup,'positions':positions,'comparison':comparison,
        'gate_pass':gate,'promoted':False,'test_opened':False,'per_book':rows}
    report.parent.mkdir(parents=True,exist_ok=True)
    report.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('R76 DEV complete gate:',gate,flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('checkpoint','data-dir','train-report','report'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--books',type=int,default=12)
    p.add_argument('--warmup',type=int,default=256)
    p.add_argument('--positions',type=int,default=128)
    p.add_argument('--threads',type=int,default=2)
    a=p.parse_args()
    run(a.checkpoint,a.data_dir,a.train_report,a.report,
        books=a.books,warmup=a.warmup,positions=a.positions,threads=a.threads)


if __name__=='__main__':main()
