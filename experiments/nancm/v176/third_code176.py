#!/usr/bin/env python3
"""NaNCM v176: reversible level-3 speculative prefix coding.

The trained 12k-word top-five proposes a tiny probabilistic trie; a byte (or
BPE-derived byte) oracle *verifies* every edge. Escape mass is retained, so
no in-vocabulary/OOV byte is ever ruled out. The target distribution is the
verifier and cannot be altered by the draft. Mathematical operations here are
model-independent, enabling exhaustive tests with a synthetic oracle.
"""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
import math
import random
from typing import Callable, Sequence

ALPHABET = 256
END = 256  # symbolic word boundary; ordinary byte oracle still has 256 outputs
EPS = 1.e-12


def normalize(seq):
    seq = [float(v) for v in seq]
    if not seq or any(not math.isfinite(v) or v < 0 for v in seq):
        raise ValueError('invalid mass')
    z = sum(seq)
    if z <= 0: raise ValueError('zero mass')
    return [v / z for v in seq]


class TopKTrie:
    """Conditional prefix proposals with an explicit lower-model escape path.

    Words are letters only. For a boundary byte, each proposal includes the
    trailing separator b' ' so that speculative acceptance can cover a whole
    word. On any unmatched prefix the escape expert is exactly the byte oracle.
    The escape path receives a floor (epsilon), preventing false certainty.
    """
    def __init__(self, candidates: Sequence[tuple[str, float]], *, eos: bytes=b' ', escape: float=.30):
        if not 0 < escape < 1: raise ValueError('escape must be in (0,1)')
        if not isinstance(eos, bytes) or len(eos) != 1: raise ValueError('one byte delimiter')
        if not candidates:raise ValueError('empty proposals')
        self.escape = escape
        self.candidates = []
        agg = defaultdict(float)
        for word, p in candidates:
            if not isinstance(word,str) or not word.isascii() or not word.isalpha() or not word.islower() or len(word)>40:
                raise ValueError('invalid word')
            if not isinstance(p,(int,float)) or not math.isfinite(p) or p<0:raise ValueError('invalid candidate')
            agg[word.encode('ascii')+eos] += p
        probs = normalize(agg.values())
        self.candidates = list(zip(agg, probs))
        self.trie: dict[bytes,dict[int,float]] = defaultdict(lambda: defaultdict(float))
        self.mass: dict[bytes,float] = defaultdict(float)
        for word,p in self.candidates:
            for k,b in enumerate(word):
                self.trie[word[:k]][b] += p
                self.mass[word[:k]] += p

    def proposal(self,prefix:bytes,byte_oracle:Sequence[float]):
        p = normalize(byte_oracle)
        if len(p)!=256:raise ValueError('byte oracle must be 256-way')
        if prefix not in self.mass or self.mass[prefix]<=0:return p
        out = [self.escape*q for q in p]
        m = self.mass[prefix]
        for b,weight in self.trie[prefix].items():out[b]+=(1-self.escape)*weight/m
        assert abs(sum(out)-1)<1e-8
        return out

    def match(self,word:str):
        return any(w[:-1].decode('ascii')==word for w,p in self.candidates)


def residual(target:Sequence[float],proposal:Sequence[float]):
    p=normalize(target);q=normalize(proposal)
    if len(p)!=len(q):raise ValueError('shape mismatch')
    diff=[max(0.,a-b) for a,b in zip(p,q)]
    return normalize(diff) if sum(diff)>1e-12 else p


def sample(p,rng:random.Random):
    u=rng.random();cum=0.
    for k,x in enumerate(p):
        cum+=x
        if u<cum:return k
    return len(p)-1


def speculative_one(target:Sequence[float],proposal:Sequence[float],rng:random.Random):
    """Exact distribution-preserving draft/accept/reject for a single byte.

    Acceptance is min(1,P(b)/Q(b)). After rejection, draw from the positive
    residual proportional to max(P-Q,0). This returns exactly the P law.
    """
    p=normalize(target);q=normalize(proposal)
    if len(p)!=len(q):raise ValueError('shape mismatch')
    b=sample(q,rng)
    if q[b]>0 and rng.random()<min(1.,p[b]/q[b]):return b,True
    return sample(residual(p,q),rng),False


def speculative_word(trie:TopKTrie,history:bytes,
                     oracle:Callable[[bytes],Sequence[float]],rng:random.Random,
                     *,maxbytes:int=40,delimiter:int=32):
    """Exact target-law speculative sampling of a *prefix* through word trie.

    Crucial: each drafted byte is sampled from the actual Q(prefix) conditional,
    not selected deterministically from the word top-five. Q is a true Markov
    proposal distribution; a pre-chosen word is NOT generally a Q sample and
    would invalidate distribution-preservation. After rejection, the sampled
    residual byte is emitted and the caller continues with target-only draws.
    No speedup is asserted without a batched verifier implementation.
    """
    if maxbytes<1:raise ValueError('invalid max length')
    emitted=bytearray();accepted=0;rejected=False
    for _ in range(maxbytes):
        p=normalize(oracle(history+emitted))
        q=trie.proposal(bytes(emitted),p)
        b,ok=speculative_one(p,q,rng)
        emitted.append(b)
        accepted+=int(ok)
        if not ok:
            rejected=True
            break
        if b==delimiter:break
    return bytes(emitted),accepted,rejected


def fixed_draft_accept_probabilities(trie:TopKTrie,history:bytes,draft:str,
                                     oracle:Callable[[bytes],Sequence[float]]):
    """Diagnostic only: acceptance chances for a *fixed* top-five word.

    This is NOT an exact standalone speculative sampler. Its proposed word was
    not drawn from the trie's bytewise Q distribution. Use speculative_word to
    sample without bias; fixed drafts need correction for their true proposal
    law before claiming distribution preservation.
    """
    prefix=b'';rates=[]
    for b in (draft+' ').encode('ascii'):
        p=normalize(oracle(history+prefix));q=trie.proposal(prefix,p)
        rates.append(min(1.,p[b]/q[b]) if q[b]>0 else 0.)
        prefix+=bytes([b])
    return rates

def prefix_beam(trie:TopKTrie,history:bytes,oracle:Callable[[bytes],Sequence[float]],
                *,width:int=5,maxdepth:int=20,sep:int=32):
    """Byte-verified word beam, shared prefix cache, length-normalized ranking.

    This is a proposal/verification algorithm, not exact beam probability.
    Cache avoids duplicated prefix verification; EOS is a real separator byte.
    If a word is not in the teacher's proposal set the byte oracle still
    retains that branch through escape, but beam width limits exploration.
    """
    if width<1 or maxdepth<1:raise ValueError('invalid beam')
    beams=[(b'',0.)];done=[];calls=0;cache={}
    for depth in range(maxdepth):
        pool=[]
        for prefix,score in beams:
            if prefix not in cache:
                cache[prefix]=normalize(oracle(history+prefix));calls+=1
            p=cache[prefix]
            q=trie.proposal(prefix,p)
            # Beam prediction is explicitly a mixture; target coding remains p.
            use=normalize([.5*a+.5*b for a,b in zip(p,q)])
            for b in sorted(range(256),key=lambda j:-use[j])[:min(width*2,256)]:
                val=score+math.log(max(use[b],EPS))
                word=prefix+bytes([b])
                if b==sep and prefix:done.append((word,val))
                elif len(word)<maxdepth:pool.append((word,val))
        pool.sort(key=lambda x:-x[1])
        beams=pool[:width]
        if not beams:break
    done.sort(key=lambda x:-x[1])
    return done[:width],{'oracle_calls':calls,'shared_prefixes':len(cache)}


def full_word_top5(word_probs:Sequence[float],vocab:Sequence[str]):
    if len(vocab)!=len(word_probs):raise ValueError('shape mismatch')
    if len(vocab)<5:raise ValueError('fewer than five words')
    ids=sorted(range(len(word_probs)),key=lambda k:-word_probs[k])[:5]
    return [(vocab[i],float(word_probs[i])) for i in ids]


def analytic_acceptance(target:Sequence[float],proposal:Sequence[float]):
    """Exact expected single-step accept fraction = sum_b min(P(b),Q(b))."""
    p=normalize(target);q=normalize(proposal)
    return sum(min(a,b) for a,b in zip(p,q))