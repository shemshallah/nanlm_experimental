#!/usr/bin/env python3
"""R7.20 packed, exact (15-bit/token) causal n-gram inverted index.

Native NumPy arrays, not per-event Python objects. Four 1-4 gram lanes use
uint64 prefix keys + int32 occurrence positions. Indexed keys were already
observed at or before prediction's past. A bounded pending block is merged
by causal rebuild, never using later tokens. No weights or embeddings change.
"""
from __future__ import annotations
import math
import time
import numpy as np
from fa1_r719_streaming_index import HORIZON, EXCLUSION, ORDERS, FAR_LEN, ALPHA

_MASK=32767
_EMPTY=dict(n=0, weight_total=0.,peak=0.,entropy=0.,recent=0.,modal=1,target_probability=0.)


def _keys(seq,ps,order):
    """Collision-free packing for original 32,000-token vocabulary."""
    key=np.zeros(len(ps),dtype=np.uint64)
    good=np.ones(len(ps),dtype=bool)
    for offset in range(order):
        v=seq[ps-order+1+offset]
        if np.any((v<0)|(v>_MASK)):
            raise ValueError('R7.20 packed index requires 0..32767 token IDs')
        key=(key << np.uint64(15)) | v.astype(np.uint64)
        good &= (v>1)
    valid_next=seq[ps+1]>1
    return key,good&valid_next


def _query_key(q,order):
    key=0
    for x in q[-order:]:
        x=int(x)
        if x<=1:return None
        if x>_MASK:raise ValueError('token exceeds packed 15-bit vocabulary')
        key=(key<<15)|x
    return np.uint64(key)


class PackedFarIndex:
    def __init__(self,sequence,position,document_start=0,merge_stride=4096):
        seq=np.asarray(sequence)
        if seq.dtype.kind not in 'ui': raise TypeError('integer token sequence required')
        self.sequence=seq
        self.document_start=int(document_start)
        self.position=int(position)
        if self.document_start<0 or self.position<self.document_start or self.position>len(seq):
            raise ValueError('invalid document/index position')
        if len(seq)>2**31-1:raise ValueError('int32 occurrence index limit')
        self.merge_stride=int(merge_stride)
        if self.merge_stride<16:raise ValueError('merge_stride must be >=16')
        self.pending_keys=[{} for _ in range(ORDERS)]
        self.pending_events=0
        self.rebuilds=0
        self._build()

    def _build(self):
        low=max(self.document_start,self.position-HORIZON)
        high=min(self.position-EXCLUSION-2,len(self.sequence)-2)
        self.sorted_keys=[];self.sorted_pos=[]
        for order in range(1,ORDERS+1):
            ps=np.arange(low+order-1,high+1,dtype=np.int32) if high>=low+order-1 else np.empty(0,dtype=np.int32)
            if len(ps):
                keys,valid=_keys(self.sequence,ps,order)
                ps=ps[valid];keys=keys[valid]
                idx=np.argsort(keys,kind='stable')
                keys=keys[idx];ps=ps[idx]
            else: keys=np.empty(0,dtype=np.uint64)
            self.sorted_keys.append(keys);self.sorted_pos.append(ps)
        self.pending_keys=[{} for _ in range(ORDERS)]
        self.pending_events=0;self.rebuilds+=1

    def advance(self,position):
        t=int(position)
        if t<self.position or t>len(self.sequence):
            raise ValueError('only causal forward advancing supported')
        for now in range(self.position+1,t+1):
            p=now-EXCLUSION-2
            low=max(self.document_start,now-HORIZON)
            if p>=low and p+1<len(self.sequence):
                outcome=int(self.sequence[p+1])
                if outcome>1:
                    key=0
                    for order in range(1,ORDERS+1):
                        if p-order+1<low:break
                        token=int(self.sequence[p-order+1])
                        if token<=1:break
                        if token>_MASK:raise ValueError('token exceeds packed 15-bit vocabulary')
                        key |= token << (15*(order-1))
                        self.pending_keys[order-1].setdefault(key,[]).append(p)
            self.pending_events+=1
            self.position=now
            if self.pending_events >= self.merge_stride:
                self._build()
        return self

    def _positions(self,key,order):
        a=self.sorted_keys[order-1];p=self.sorted_pos[order-1]
        left=np.searchsorted(a,key,side='left');right=np.searchsorted(a,key,side='right')
        base=p[left:right]
        extras=self.pending_keys[order-1].get(int(key))
        if extras:
            other=np.asarray(extras,dtype=np.int32)
            return np.concatenate((base,other)) if len(base) else other
        return base

    def query(self,context,target=None):
        q=np.asarray(context,dtype=np.int64)[-ORDERS:]
        if len(q)!=ORDERS:raise ValueError('need 4 past query words')
        cutoff=self.position-EXCLUSION-2
        start=self.position-HORIZON
        low=max(self.document_start,start)
        out=[]
        for order in range(1,ORDERS+1):
            key=_query_key(q,order)
            if key is None:
                out.append(_EMPTY.copy());continue
            ps=self._positions(key,order)
            if len(ps):ps=ps[(ps>=low+order-1)&(ps<=cutoff)]
            if not len(ps):
                out.append(_EMPTY.copy());continue
            # The following recurrence matches R7.18's causal window and
            # target-independent route statistics; target only scores CE.
            w=np.exp(ALPHA*(ps.astype(np.float64)-start-(FAR_LEN-1))).astype(np.float32)
            values=self.sequence[ps+1]
            unique,inv=np.unique(values,return_inverse=True)
            weighted=np.bincount(inv,weights=w,minlength=len(unique))
            total=float(w.sum())
            frac=weighted/max(total,1e-30)
            imax=int(weighted.argmax())
            entropy=float(-(frac*np.log(frac+1e-30)).sum()/np.log1p(len(unique)))
            target_weight=float(w[values==target].sum()) if target is not None else 0.
            out.append({'n':len(ps),'weight_total':total,
                        'peak':float(frac[imax]),'entropy':entropy,
                        'recent':float((int(ps.max())-start)/FAR_LEN),
                        'modal':int(unique[imax]),'target_probability':target_weight/total})
        return out

    def stats(self):
        bytes_used=sum(a.nbytes+b.nbytes for a,b in zip(self.sorted_keys,self.sorted_pos))
        pending=sum(sum(len(v) for v in d.values()) for d in self.pending_keys)
        return dict(position=self.position,keys_sorted=int(sum(len(a) for a in self.sorted_keys)),
                    pending_events=self.pending_events,pending_records=pending,
                    index_array_bytes=bytes_used,merge_rebuilds=self.rebuilds)