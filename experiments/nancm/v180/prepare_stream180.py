#!/usr/bin/env python3
"""Reconstruct only TRAIN book tokens from overlapping, already observed 48-word contexts.
Val/test histories are NEVER written into training data. Book/position gaps remain -2.
"""
from pathlib import Path
import struct, hashlib, json, argparse, numpy as np
ROOT=Path(__file__).resolve().parents[3]
SRC=ROOT/'results/v175_release3'
OUT=ROOT/'results/v180'

def build(split='train'):
    data=np.load(SRC/f'{split}.npz')
    x=data['X'];y=data['Y'];books=data['book'];pos=data['word_pos']; assert len(y)==len(pos)
    report=dict(split=split,samples=len(y),books=0,total_positions=0,observed_positions=0,unseen_gaps=0,oov_positions=0,overlap_asserts=0,conflicts=0,valid_ngrams={str(n):0 for n in range(1,6)})
    target=OUT/f'{split}_observed_stream.bin';OUT.mkdir(parents=True,exist_ok=True)
    with target.open('wb') as f:
        f.write(b'NC180OBS');f.write(struct.pack('<I',len(np.unique(books))))
        for book in np.unique(books):
            idx=np.flatnonzero(books==book)
            length=int(pos[idx].max()+1)
            out=np.full(length,-2,dtype='<i2')
            for j in idx:
                p=int(pos[j]);start=p-48
                assert start>=0
                spans=np.arange(start,p,dtype=np.int32)
                old=out[spans];now=x[j].astype('int16')
                collision=(old!=-2)
                mismatches=collision&(old!=now)
                if np.any(mismatches):
                    report['conflicts']+=int(mismatches.sum())
                    raise ValueError((split,book,p,'conflict',int(mismatches.sum())))
                report['overlap_asserts']+=int(collision.sum())
                out[spans]=now
                if out[p]!=-2 and out[p]!=y[j]:
                    raise ValueError((split,book,p,'target conflict'))
                out[p]=y[j]
            report['books']+=1;report['total_positions']+=len(out)
            report['observed_positions']+=int((out!=-2).sum())
            report['unseen_gaps']+=int((out==-2).sum())
            report['oov_positions']+=int((out==-1).sum())
            valid=(out>=0)
            for n in range(1,6):
                b=valid.copy()
                for lag in range(1,n+1):b[lag:] &= valid[:-lag];b[:lag]=False
                report['valid_ngrams'][str(n)]+=int(b.sum())
            f.write(struct.pack('<iI',int(book),len(out)))
            f.write(out.tobytes())
    report['file']=str(target.relative_to(ROOT));report['sha256']=hashlib.sha256(target.read_bytes()).hexdigest();report['bytes']=target.stat().st_size
    (OUT/f'{split}_reconstruction_stats.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--split',choices=['train','val','test'],default='train');a=p.parse_args();
    if a.split!='train':raise SystemExit('FORBIDDEN: only TRAIN books may build prediction index; validation is evaluated by supplied X directly')
    build(a.split)
