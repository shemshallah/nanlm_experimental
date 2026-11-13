#!/usr/bin/env python3
"""Compare the SAME held-out target bytes at different causal context lengths.

R30 uses the existing checkpoint and whole-book validation split; no weights or
optimizer are changed. Each sampled 4096-byte window has 256 aligned final
predictions evaluated both with all 4096 preceding input tokens and with the
last 256 only. Validation windows never cross book boundaries.
"""
import argparse
import json
import math
from pathlib import Path
import sys
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from poincare_frontier import Config, MirrorLM, DocumentWindowSampler, legacy_document_lengths, load_v25_compatible, file_hash


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--split', choices=['valid','test'], default='valid')
    parser.add_argument('--long', type=int, default=4096)
    parser.add_argument('--short', type=int, default=256)
    parser.add_argument('--windows', type=int, default=16)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--threads', type=int, default=4)
    args=parser.parse_args()
    torch.set_num_threads(args.threads)
    ck=torch.load(Path(args.run_dir)/'latest.pt',map_location='cpu',weights_only=False)
    cfg=Config(**ck['config'])
    assert 1 <= args.short < args.long <= cfg.context
    meta=ck['data']
    dtype=meta.get('token_dtype','<u4')
    splitpath=Path(args.data_dir)/(args.split+'.bin')
    expected=meta.get('bin_'+args.split+'_sha256') or meta.get(args.split+'_sha256')
    assert expected and file_hash(splitpath)==expected, 'dataset checksum mismatch'
    mem=np.memmap(splitpath,dtype=dtype,mode='r')
    sampler=DocumentWindowSampler(mem,args.long,legacy_document_lengths(mem,meta,args.split))
    model=MirrorLM(cfg)
    load_v25_compatible(model,ck['model'])
    model.eval()
    rng=np.random.default_rng(args.seed)
    scores={str(args.short):{'ce':0.0,'correct':0},str(args.long):{'ce':0.0,'correct':0}}
    with torch.no_grad():
        for _ in range(args.windows):
            ids=sampler.windows(1,rng,torch.device('cpu'))
            x=ids[:,:-3]
            targets=ids[:,1:-2][:,-args.short:]
            for context in (args.short,args.long):
                logits,_,_=model(x[:,-context:])
                logits=logits[:,-args.short:]
                key=str(context)
                scores[key]['ce'] += float(F.cross_entropy(logits.reshape(-1,logits.shape[-1]),targets.reshape(-1),reduction='sum'))
                scores[key]['correct'] += int((logits.argmax(-1)==targets).sum())
    n=args.windows*args.short
    for d in scores.values():
        d['ce_nats_byte']=round(d.pop('ce')/n,6)
        d['bpb']=round(d['ce_nats_byte']/math.log(2),6)
        d['accuracy']=round(d.pop('correct')/n,6)
    print(json.dumps({'step':ck['step'],'split':args.split,'same_target_positions':n,'windows':args.windows,
                      'seed':args.seed,'short':args.short,'long':args.long,
                      'data_split_sha256':expected,
                      'context_comparison':scores,'long_minus_short_bpb':round(scores[str(args.long)]['bpb']-scores[str(args.short)]['bpb'],6)},indent=2))

if __name__=='__main__':main()
