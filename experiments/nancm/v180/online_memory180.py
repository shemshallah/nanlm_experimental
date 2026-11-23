#!/usr/bin/env python3
"""Leak-free prequential within-book word memory.
At query for position t, index only observed positions <t (provided 48-word history
plus previous revealed targets), NEVER current Y or later records.
"""
from pathlib import Path
import argparse,json
from collections import Counter,defaultdict
import numpy as np
ROOT=Path(__file__).resolve().parents[3];RES=ROOT/'results/v180'

def run(split='val'):
    files=sorted(RES.glob(f'{split}_neural_*.npz'),key=lambda x:int(x.stem.split('_')[-1]));d=np.load(files[-1]);assert len(d['target'])==13200 if split=='val' else True
    orig=np.load(ROOT/f'results/v175_release3/{split}.npz')
    X=orig['X'][d['rows']];book=d['book'];pos=d['pos'];target=d['target'];ng=d['ngram'];neural=d['neural_top']
    assert len(book)==len(target)
    ix=np.lexsort((pos,book));
    output=np.full((len(ix),4,2),-1,dtype=np.int32)
    totals=dict(rows=len(ix),prequential_matches={str(n):0 for n in range(1,5)},
                prequential_correct={str(n):0 for n in range(1,5)},
                combined_oracle64=0,combined_oracle64_plus_local=0,
                combined_oracle5=0,combined_oracle5_plus_local=0,
                local_8_oracle=0,local8_with_global8=0,
                baseline_top1=0,local_top1_backoff=0,overlaps_checked=0,observed=0)
    lastid=None;data=None;maps=None;first=0
    for z in ix:
        b=int(book[z]);p=int(pos[z]);gold=int(target[z]);
        if b!=lastid:
            lastid=b;data=np.full(p+1,-2,dtype='int16');maps=[None]+[defaultdict(Counter) for _ in range(4)];first=0
        if len(data)<=p:data=np.pad(data,(0,p+1-len(data)),constant_values=-2)
        # All additions must have word positions strictly BEFORE the current target.
        for jj,w in enumerate(X[z]):
            j=p-48+jj;assert j<p
            if data[j]!=-2:
                assert data[j]==w,(b,p,j);totals['overlaps_checked']+=1
                continue
            data[j]=w;totals['observed']+=1
            if w<0:continue
            for n in range(1,5):
                if j-n<0:break
                keys=data[j-n:j]
                if np.any(keys<0):break
                maps[n][tuple(int(t) for t in keys)][int(w)]+=1
        predicted=[]
        for n in range(1,5):
            keys=X[z,48-n:48]
            c=maps[n].get(tuple(int(t) for t in keys),None) if np.all(keys>=0) else None
            if c:
                common=sorted(c.items(),key=lambda kv:(-kv[1],kv[0]))[:2]
                for k,(tok,cnt) in enumerate(common):output[z,n-1,k]=tok
                totals['prequential_matches'][str(n)]+=1
                totals['prequential_correct'][str(n)]+=int(common[0][0]==gold)
                predicted.append(common[0][0])
            else:predicted.append(-1)
        own=output[z].reshape(-1)
        global_cand=ng[z,:,:2].ravel()
        localhit=gold in own
        totals['local_8_oracle']+=int(localhit)
        totals['local8_with_global8']+=int(localhit or gold in global_cand)
        totals['baseline_top1']+=int(neural[z,0]==gold)
        totals['local_top1_backoff']+=int(next((p for p in reversed(predicted) if p!=-1),-1)==gold)
        for n in (5,64):
            parent=gold in neural[z,:n];union=parent or gold in global_cand
            totals[f'combined_oracle{n}']+=int(union)
            totals[f'combined_oracle{n}_plus_local']+=int(union or localhit)
        # Reveal target ONLY AFTER prediction. This is teacher-forced, legal prequential.
        if data[p]==-2:
            data[p]=gold
            for n in range(1,5):
                if p-n<0:break
                h=data[p-n:p]
                if np.any(h<0):break
                maps[n][tuple(int(t) for t in h)][gold]+=1
        else:
            assert data[p]==gold,(b,p)
    path=RES/f'{split}_prequential_local_top2.npz';np.savez_compressed(path,local=output)
    (RES/f'{split}_prequential_metrics.json').write_text(json.dumps(totals,indent=2)+'\n')
    print('PREQUENTIAL',split,json.dumps(totals,indent=2))

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--split',choices=['train','val'],default='val');a=ap.parse_args();run(a.split)
