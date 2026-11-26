"""Eight bounded causal cross-half edge rails; no future writes."""
import collections as C
import hashlib
import math
import numpy as np
WINDOWS=(64,1024,64,1024,64,1024,64,1024)
DEPTHS=(1,1,2,2,4,4,3,3)
def address(key):
    return int.from_bytes(hashlib.blake2s(repr(key).encode(),digest_size=4).digest(),'little')%128
class Graph:
    def __init__(self,code,ids,role):
        self.code=code;self.ids=ids;self.role=role
        self.proj=np.random.default_rng(4101).normal(0,1/4,(16,128)).astype(np.float32)
        self.cache={};self.h=[];self.t=0
        self.edges=np.zeros((2,8,128,128),np.float32)
        self.count=np.zeros((8,128),np.int32)
        self.rows=[{} for _ in WINDOWS];self.ring=[C.deque() for _ in WINDOWS]
        self.doc=C.Counter();self.doc_ring=C.deque();self.background=np.zeros(128,np.float32)
    def vec(self,w):
        if w not in self.cache:
            v=self.code[self.ids.get(w,0)]@self.proj
            v=v.copy();v[address(('word',w))]+=.75
            v[address(('role',self.role(w)))]+=.35
            v/=max(float(np.linalg.norm(v)),1e-8)
            if len(self.cache)>=4096:self.cache.clear()
            self.cache[w]=v
        return self.cache[w]
    def keys(self,h):
        out=[]
        for r,d in enumerate(DEPTHS):
            if len(h)<d:out.append(None);continue
            if r>=6:key=(self.role(h[-3]),h[-1])
            else:key=tuple(h[-d:])
            out.append(key)
        return out
    def change(self,r,key,w,sign):
        i=address((r,key));v=self.vec(w)
        self.edges[0,r,i]+=sign*np.maximum(v,0)
        self.edges[1,r,i]+=sign*np.maximum(-v,0)
        self.count[r,i]+=sign
        row=self.rows[r].setdefault(key,C.Counter());row[w]+=sign
        if row[w]==0:del row[w]
        if not row:del self.rows[r][key]
        if self.count[r,i]==0:self.edges[:,r,i].fill(0)
    def observe(self,w):
        for r,key in enumerate(self.keys(self.h)):
            ring=self.ring[r]
            while ring and self.t-ring[0][0]>=WINDOWS[r]:
                _,k,v=ring.popleft();self.change(r,k,v,-1)
            if key is not None:
                self.change(r,key,w,1);ring.append((self.t,key,w))
        if len(self.doc_ring)>=1024:
            v=self.doc_ring.popleft();self.doc[v]-=1;self.background-=self.vec(v)
            if self.doc[v]==0:del self.doc[v]
        self.doc_ring.append(w);self.doc[w]+=1;self.background+=self.vec(w)
        self.h=(self.h+[w])[-128:];self.t+=1
    def query(self,h=None):
        h=self.h if h is None else h;rows=[];vectors=[];counts=[]
        for r,key in enumerate(self.keys(h)):
            if key is None:rows.append({});vectors.append(np.zeros(128,np.float32));counts.append(0);continue
            i=address((r,key));n=int(self.count[r,i]);rows.append(self.rows[r].get(key,{}));counts.append(n)
            vectors.append((self.edges[0,r,i]-self.edges[1,r,i])/max(n,1) if n else np.zeros(128,np.float32))
        return rows,np.stack(vectors),np.asarray(counts)
    def propose(self,h=None,k=32):
        rows,_,_=self.query(h);scores=C.Counter()
        for r,row in enumerate(rows):
            n=sum(row.values())
            for w,c in row.items():scores[w]+=math.sqrt(DEPTHS[r])*c/(n+4)
        return sorted(scores,key=lambda w:(-scores[w],w))[:k]
    def features(self,words,h=None):
        rows,vec,n=self.query(h);bg=self.background/max(1,len(self.doc_ring));out=[]
        for w in words:
            p=(self.doc.get(w,0)+.1)/(len(self.doc_ring)+10.)
            exact=[math.log((row.get(w,0)+4*p)/(sum(row.values())+4)/p) for row in rows]
            residual=(vec-bg)@self.vec(w);residual*=n/(n+4)
            out.append([*np.clip(exact,-4,4),*residual,float(w in self.doc),math.log1p(self.doc.get(w,0))/math.log(1025)])
        return np.asarray(out,np.float64).reshape(-1,18)
    def snapshot(self):return list(self.h),self.t,[list(r) for r in self.ring],list(self.doc_ring)
    def restore(self,s):
        self.h=list(s[0]);self.t=s[1];self.edges.fill(0);self.count.fill(0);self.rows=[{} for _ in WINDOWS];self.ring=[C.deque(r) for r in s[2]]
        for r,ring in enumerate(self.ring):
            for _,k,w in ring:self.change(r,k,w,1)
        self.doc_ring=C.deque(s[3]);self.doc=C.Counter(s[3]);self.background.fill(0)
        for w in self.doc_ring:self.background+=self.vec(w)
def preflight():
    words=['alpha','beta','answer','noise','other'];ids={w:i+1 for i,w in enumerate(words)};code=np.random.default_rng(41).normal(size=(6,16)).astype(np.float32);code[0]=0
    g=Graph(code,ids,lambda w:ids.get(w,0));assert not g.propose()
    for w in ['alpha','beta','answer']+['noise']*150+['alpha','beta']:g.observe(w)
    assert g.propose()[0]=='answer',g.propose()
    f=g.features(['answer','other']);assert f[0,3]>f[1,3]
    b=Graph(code,ids,lambda w:ids.get(w,0));b.restore(g.snapshot());assert b.propose()==g.propose();assert np.allclose(b.features(words),g.features(words),atol=1e-5)
    for _ in range(1100):g.observe('noise')
    assert not g.propose(['alpha','beta']);assert all(len(r)<=w for r,w in zip(g.ring,WINDOWS))
    b.restore(g.snapshot());err=float(np.max(np.abs(b.edges-g.edges)));assert err<.01 and np.array_equal(b.count,g.count)
    print(dict(rails=8,edge_addresses=8*128*128,sign_channels=2,recall_after_intervening_words=150,restore_edge_max_error=err,eviction=True,empty=True))
if __name__=='__main__':preflight()
