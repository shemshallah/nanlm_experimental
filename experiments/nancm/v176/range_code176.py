#!/usr/bin/env python3
"""Lossless arithmetic/range layer for the exact native byte verifier.

The word top-5 trie is a speculative accelerator, but *the verifier* defines
coding frequencies. Encoding under Q instead of P would change the modeled
bitstream and invalidate exact speculative sampling. This implementation
checks reversible 32-bit integer arithmetic under time-varying byte oracles.
"""
from __future__ import annotations
import bisect
from typing import Callable,Sequence
from third_code176 import normalize

BITS=32
TOP=(1<<BITS)-1
HALF=1<<(BITS-1)
QTR=1<<(BITS-2)
THREE=QTR*3


def quantize(p:Sequence[float]):
    """Portable 16-bit positive integer coding frequencies, all 256 supported."""
    a=normalize(p)
    f=[1+int(x*(65536-256)) for x in a]
    assert len(f)==256 and all(x>0 for x in f)
    cumul=[0]
    for x in f:cumul.append(cumul[-1]+x)
    return cumul


class _Bits:
    def __init__(self):self.data=[]
    def put(self,x):self.data.append(int(x))
    def pack(self):
        b=bytearray((len(self.data)+7)//8)
        for j,x in enumerate(self.data):b[j//8]|=x<<(7-(j%8))
        return bytes(b),len(self.data)


class _BitsIn:
    def __init__(self,data,n):self.data=data;self.n=n;self.i=0
    def get(self):
        i=self.i;self.i+=1
        if i>=self.n:return 0
        return (self.data[i//8]>>(7-i%8))&1


class RangeEncoder:
    def __init__(self):
        self.lo=0;self.hi=TOP;self.underflow=0;self.bits=_Bits()
    def _emit(self,b):
        self.bits.put(b)
        for _ in range(self.underflow):self.bits.put(1-b)
        self.underflow=0
    def encode(self,symbol:int,cum):
        if not 0<=symbol<256:raise ValueError('bad byte')
        total=cum[-1];width=self.hi-self.lo+1
        if total<=0 or cum[symbol+1]<=cum[symbol]:raise ValueError('invalid probability')
        low=self.lo
        self.hi=low+(width*cum[symbol+1]//total)-1
        self.lo=low+(width*cum[symbol]//total)
        while True:
            if self.hi<HALF:self._emit(0)
            elif self.lo>=HALF:
                self._emit(1);self.lo-=HALF;self.hi-=HALF
            elif self.lo>=QTR and self.hi<THREE:
                self.underflow+=1;self.lo-=QTR;self.hi-=QTR
            else:break
            self.lo*=2;self.hi=self.hi*2+1
    def finish(self):
        self.underflow+=1
        self._emit(0 if self.lo<QTR else 1)
        return self.bits.pack()


class RangeDecoder:
    def __init__(self,data,bits):
        self.stream=_BitsIn(data,bits);self.lo=0;self.hi=TOP;self.value=0
        for _ in range(BITS):self.value=(self.value<<1)|self.stream.get()
    def decode(self,cum):
        total=cum[-1];width=self.hi-self.lo+1
        target=((self.value-self.lo+1)*total-1)//width
        sym=bisect.bisect_right(cum,target)-1
        if sym<0 or sym>=256:raise ValueError('broken compressed data')
        low=self.lo;self.hi=low+(width*cum[sym+1]//total)-1
        self.lo=low+(width*cum[sym]//total)
        while True:
            if self.hi<HALF:pass
            elif self.lo>=HALF:
                self.lo-=HALF;self.hi-=HALF;self.value-=HALF
            elif self.lo>=QTR and self.hi<THREE:
                self.lo-=QTR;self.hi-=QTR;self.value-=QTR
            else:break
            self.lo*=2;self.hi=self.hi*2+1;self.value=(self.value<<1)|self.stream.get()
        return sym


def encode(data:bytes,oracle:Callable[[bytes],Sequence[float]],history:bytes=b''):
    """Actual lossless codec, conditional on deterministic verifier P_byte."""
    c=RangeEncoder();p=bytearray(history)
    for x in data:
        c.encode(x,quantize(oracle(bytes(p))))
        p.append(x)
    return c.finish()


def decode(data:bytes,nbits:int,n:int,oracle:Callable[[bytes],Sequence[float]],history:bytes=b''):
    if n<0 or nbits<0:raise ValueError('invalid length')
    c=RangeDecoder(data,nbits);p=bytearray(history);out=bytearray()
    for _ in range(n):
        x=c.decode(quantize(oracle(bytes(p))))
        out.append(x);p.append(x)
    return bytes(out)