"""Raw-byte count backoff and exact rational interval codec; no word normalization."""
import collections as C
import numpy as np
import re
TOTAL=65536
def context(prefix,depth=8):
    h=prefix[-depth:][::-1]
    return list(h)+[256]*(depth-len(h))
def words(prefix):
    x=re.findall(rb'[A-Za-z]+',prefix)
    if prefix and ((65<=prefix[-1]<=90) or (97<=prefix[-1]<=122)):x=x[:-1]
    return [s.lower().decode('ascii') for s in x[-128:]]
class Backoff:
    def __init__(self,docs):
        self.rows=[C.defaultdict(C.Counter) for _ in range(4)]
        for doc in docs:
            for i,b in enumerate(doc):
                for n in range(min(i,3)+1):self.rows[n][doc[i-n:i]][b]+=1
    def prob(self,prefix):
        p=np.full(256,1/256,dtype=np.float64)
        for n in range(min(len(prefix),3)+1):
            row=self.rows[n].get(prefix[len(prefix)-n:])
            if not row:continue
            k=np.fromiter(row.keys(),dtype=np.int64);v=np.fromiter(row.values(),dtype=np.float64)
            a=8.+len(row);p*=a/(a+v.sum());p[k]+=v/(a+v.sum())
        p/=p.sum();return p
def cdf(prob):
    p=np.asarray(prob,dtype=np.float64);assert p.shape==(256,) and np.isfinite(p).all() and p.min()>0
    p=p/p.sum();z=p*(TOTAL-256);f=np.floor(z).astype(np.int64)+1
    k=TOTAL-int(f.sum());f[np.argsort(-(z-np.floor(z)),kind='stable')[:k]]+=1
    a=np.r_[0,np.cumsum(f)];assert a[-1]==TOTAL and np.diff(a).min()>0
    return a
def encode(prefix,source,predict):
    lo=0;width=1;den=1;history=prefix
    for b in source:
        a=cdf(predict(history));lo=lo*TOTAL+width*int(a[b]);width*=int(a[b+1]-a[b]);den*=TOTAL;history+=bytes([b])
    k=0
    while True:
        m=(lo*(1<<k)+den-1)//den
        if (m+1)*den<=(lo+width)*(1<<k):return m,k
        k+=1
def decode(prefix,code,length,predict):
    m,k=code;num=2*m+1;den=1<<(k+1);out=bytearray()
    for _ in range(length):
        a=cdf(predict(prefix+bytes(out)));v=num*TOTAL//den
        b=int(np.searchsorted(a,v,side='right')-1);assert 0<=b<256
        num=num*TOTAL-int(a[b])*den;den*=int(a[b+1]-a[b]);out.append(b)
        assert 0<=num<den
    return bytes(out)
def preflight():
    raw=bytes(range(256))+b'\r\nMixed CASE\x00\xff!'
    model=Backoff([raw]);p=model.prob
    c=encode(b'',raw,p);assert decode(b'',c,len(raw),p)==raw
    assert words(b'One TWO thr')==['one','two'] and context(b'\x00\xff')[:3]==[255,0,256]
    print(dict(all_256_roundtrip=True,bits=c[1],normalization=float(p(b'xyz').sum()),raw_length=len(raw)))
if __name__=='__main__':preflight()
