#!/usr/bin/env python3
"""Train v133-style byte-pair tokens on v175 TRAIN-only word examples.

Source basis: user-supplied v133_iterate.zip / v133/nancm.py train_bpe,
BPE enc_pre. All merges are retrained exclusively on v175 Release3 TRAIN.
These BPE blocks are speculative *proposal spans*, not a replacement trained LM.
"""
from __future__ import annotations
from pathlib import Path
import argparse,collections,hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[3]


def train_bpe(counts,nmerge):
    words=[list(w) for w in counts];freq=[counts[w] for w in counts]
    pc=collections.Counter();where=collections.defaultdict(set)
    for i,w in enumerate(words):
        for p in zip(w,w[1:]):pc[p]+=freq[i];where[p].add(i)
    merges=[]
    for r in range(nmerge):
        if not pc:break
        pair=max(pc,key=pc.get)
        if pc[pair]<2:break
        nid=256+r;merges.append(pair)
        for i in list(where[pair]):
            w=words[i];f=freq[i]
            for p in zip(w,w[1:]):
                pc[p]-=f
                if pc[p]<=0:del pc[p]
            nw=[];j=0
            while j<len(w):
                if j<len(w)-1 and (w[j],w[j+1])==pair:nw.append(nid);j+=2
                else:nw.append(w[j]);j+=1
            words[i]=nw
            for p in zip(nw,nw[1:]):pc[p]+=f;where[p].add(i)
        where.pop(pair,None)
    return merges


class BPE:
    def __init__(self,merges):
        self.merges=merges;self.rank={m:i for i,m in enumerate(merges)}
        self.vocab=[bytes([i]) for i in range(256)]
        for a,b in merges:self.vocab.append(self.vocab[a]+self.vocab[b])
    def encode(self,word:bytes):
        seq=list(word)
        while len(seq)>1:
            best=None;br=1<<30
            for i in range(len(seq)-1):
                k=self.rank.get((seq[i],seq[i+1]))
                if k is not None and k<br:br=k;best=i
            if best is None:break
            seq[best:best+2]=[256+br]
        return seq
    def bytes(self,ids):return b''.join(self.vocab[i] for i in ids)
    def block_spans(self,word:str):
        if not word.isascii() or not word.isalpha():raise ValueError('letters only')
        ids=self.encode(word.encode('ascii'))
        return [self.vocab[i] for i in ids]+[b' ']


def main():
    p=argparse.ArgumentParser();p.add_argument('--merges',type=int,default=1600)
    args=p.parse_args()
    from collections import Counter
    names=(ROOT/'experiments/nancm/v175/vocab175.txt').read_text().splitlines()
    with np.load(ROOT/'results/v175_release3/train.npz',allow_pickle=False) as z:
        labels=z['Y']
    f=Counter(names[int(i)].encode('ascii') for i in labels)
    merges=train_bpe(f,args.merges)
    bpe=BPE(merges)
    for word in list(f)[:5000]:assert bpe.bytes(bpe.encode(word))==word
    token_lengths=np.asarray([len(bpe.encode(w)) for w in f])
    model=dict(version='v176',algorithm='v133 BPE retrained on 396-book v175 TRAIN only',requested_merges=args.merges,
               learned_merges=len(merges),vocab_size=256+len(merges),train_examples=int(len(labels)),
               training_cohort='corpora-short-stories-0003 TRAIN',word_count=len(f),
               merges=merges)
    side=ROOT/'checkpoints/v176_v133_style_bpe.json';side.write_text(json.dumps(model,separators=(',',':')))
    info=dict(status='PASS',source='user v133/nancm.py BPE concept; new TRAIN-only merge weights',
       merges=len(merges),vocab_size=len(bpe.vocab),unique_train_words=len(f),
       mean_BPE_tokens_unweighted=float(token_lengths.mean()),
       mean_letters_unweighted=float(np.mean([len(w) for w in f])),
       exact_reconstruct_checked=5000,sha256=hashlib.sha256(side.read_bytes()).hexdigest())
    (ROOT/'results/v176_bpe_train.json').write_text(json.dumps(info,indent=2))
    print('V176 BPE',json.dumps(info),flush=True)
if __name__=='__main__':main()