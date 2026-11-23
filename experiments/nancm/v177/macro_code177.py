#!/usr/bin/env python3
"""NaNCM v177 third coding: exact top-five macro symbols with compensating escape.

The math is a lossless change of *factorization* of the verifier P, NOT an
entropy improvement. For prefix s and candidate word events W_i, let
M(s)=sum_i P(W_i) 1[s prefix-of W_i]. Conditioned on escape E (none of W_i):
  P(b | s,E) = (P(s)*P(b|s)-M(sb))/(P(s)-M(s)).
This preserves the full 256-byte verifier distribution for EVERY word, not
only top-five words. The macro channel can encode a whole frequent word as a
single arithmetic symbol. Finite precision quantization is separately tested.
"""
from __future__ import annotations
import math
from collections import defaultdict
from typing import Callable, Sequence
from range_code176 import RangeEncoder,RangeDecoder,quantize,normalize


def quantize_n(probs:Sequence[float]):
    p=normalize(probs)
    n=len(p)
    if not 2<=n<=256:raise ValueError('invalid macro alphabet')
    f=[1+int(x*(65536-n)) for x in p]
    cum=[0]
    for x in f:cum.append(cum[-1]+x)
    return cum

class MacroByteModel:
    def __init__(self,words:Sequence[bytes],oracle:Callable[[bytes],Sequence[float]]):
        if not 1<=len(words)<=5 or len(set(words))!=len(words):raise ValueError('unique 1–5 words')
        if any(not isinstance(w,bytes) or len(w)<2 or len(w)>40 or w[-1]!=32 or
               any(c<97 or c>122 for c in w[:-1]) for w in words):raise ValueError('invalid literal word')
        self.words=tuple(words);self.oracle=oracle
        self.cache={};self.pm={b'':1.};self.subtree=defaultdict(float)
        self.events=[]
        for w in self.words:
            p=1.;pref=b''
            for b in w:
                p*=self.byte(pref)[b]
                pref+=bytes([b]);self.pm.setdefault(pref,p)
            self.events.append(p)
            for k in range(len(w)+1):self.subtree[w[:k]]+=p
        self.escape=1.-sum(self.events)
        if not 1e-14<self.escape<1:raise ValueError('ill-conditioned or non-prefix-free candidates')
        # With each candidate ending in space, distinct words are prefix-free.
        self.root=quantize_n([*self.events,self.escape])
    def byte(self,prefix):
        if prefix not in self.cache:
            p=normalize(self.oracle(prefix))
            if len(p)!=256:raise ValueError('byte oracle must be 256-way')
            self.cache[prefix]=p
        return self.cache[prefix]
    def conditional(self,prefix:bytes):
        p=self.byte(prefix)
        if prefix not in self.subtree:return p
        prefix_mass=self.pm.get(prefix)
        if prefix_mass is None:
            prefix_mass=1.;s=b''
            for b in prefix:prefix_mass*=self.byte(s)[b];s+=bytes([b])
            self.pm[prefix]=prefix_mass
        remaining=prefix_mass-self.subtree[prefix]
        if remaining<=0:raise ValueError('event is impossible after escape')
        out=[]
        for b in range(256):
            excluded=self.subtree.get(prefix+bytes([b]),0.)
            val=(prefix_mass*p[b]-excluded)/remaining
            if val < -5e-11:raise ArithmeticError('negative conditional escape mass')
            out.append(max(0.,val))
        return normalize(out)
    def oracle_probability(self,data):
        s=b'';p=1.
        for b in data:p*=self.byte(s)[b];s+=bytes([b])
        return p
    def macro_probability(self,data):
        if data in self.words:return self.events[self.words.index(data)]
        s=b'';p=self.escape
        for b in data:p*=self.conditional(s)[b];s+=bytes([b])
        return p
    def encode(self,data:bytes):
        if not data or data[-1]!=32 or len(data)>40:raise ValueError('must encode one word with trailing space')
        coder=RangeEncoder()
        if data in self.words:
            coder.encode(self.words.index(data),self.root)
        else:
            coder.encode(len(self.words),self.root)
            s=b''
            for b in data:
                coder.encode(b,quantize(self.conditional(s)))
                s+=bytes([b])
        return coder.finish()
    def decode(self,packed:bytes,bits:int):
        coder=RangeDecoder(packed,bits)
        first=coder.decode(self.root)
        if first<len(self.words):return self.words[first]
        out=bytearray()
        while len(out)<40:
            sym=coder.decode(quantize(self.conditional(bytes(out))))
            out.append(sym)
            if sym==32:return bytes(out)
        raise ValueError('decoder exceeded max word length')