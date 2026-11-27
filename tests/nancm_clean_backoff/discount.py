"""Interpolated absolute-discount continuation law, exact tuple addresses."""
import numpy as np

def compress(a):
    if not len(a):return a,np.empty(0,np.int64)
    ix=np.lexsort(tuple(a[:,j] for j in reversed(range(a.shape[1]))));a=a[ix]
    s=np.flatnonzero(np.r_[True,np.any(a[1:]!=a[:-1],axis=1)])
    return a[s],np.diff(np.r_[s,len(a)])

class Discount:
    def __init__(self,docs,prior,order=5,discount=.75):
        assert 1<=order<=8 and 0<discount<1
        self.V=len(prior);self.D=discount;self.order=order;self.tables={}
        raw={}
        for n in range(1,order+1):
            a=[np.lib.stride_tricks.sliding_window_view(np.asarray(d,np.int32),n+1) for d in docs if len(d)>n]
            raw[n]=compress(np.concatenate(a)) if a else (np.empty((0,n+1),np.int32),np.empty(0,np.int64))
        unigram=np.bincount(raw[1][0][:,1],minlength=self.V).astype(np.float64)
        self.base=.99*unigram/unigram.sum()+.01*np.asarray(prior,np.float64) if unigram.sum() else np.asarray(prior,np.float64).copy()
        for n in range(1,order+1):
            rows,c=raw[n] if n==order else compress(raw[n+1][0][:,1:])
            dt=np.dtype([('k'+str(j),'<i4') for j in range(n)])
            keys=np.ascontiguousarray(rows[:,:n],dtype=np.int32).view(dt).ravel()
            self.tables[n]=(keys,rows[:,n].copy(),c,dt)
    def query(self,h,order=None):
        q=self.base.copy();order=self.order if order is None else order
        assert 0<=order<=self.order
        for n in range(1,min(order,len(h))+1):
            ctx=tuple(h[-n:])
            if not all(ctx):continue
            keys,v,c,dt=self.tables[n];k=np.array([ctx],dtype=dt)[0]
            a,b=np.searchsorted(keys,k,'left'),np.searchsorted(keys,k,'right')
            if a==b:continue
            count=c[a:b];total=float(count.sum());d=self.D
            q*=d*(b-a)/total;q[v[a:b]]+=(count-d)/total
        assert abs(q.sum()-1)<1e-9 and np.min(q)>0
        return q
