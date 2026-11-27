"""Mixed lexical/concept address lattice; normalized class and word transport."""
import numpy as np
import torch
from torch import nn

SCHEMAS=(((1,1),(2,1)),((1,1),(2,1),(3,1)),((1,0),(2,1)),((1,1),(2,0)),((1,0),(2,1),(3,1)),((1,0),(3,0)),((1,0),(3,0),(4,1)))
GROUPS=((0,1),(2,3,4),(2,3,4),(5,6))
NAMES=('abstract_class','mixed_class','mixed_word','skip_word')

class Lattice:
    def __init__(self,docs,cls):
        self.cls=np.asarray(cls,np.int64);self.V=len(cls);self.K=int(self.cls.max())+1
        assert self.cls[0]==0 and self.V**3<2**63
        self.tables=[]
        for schema in SCHEMAS:
            lag=max(l for l,r in schema);keys=[];values=[]
            for doc in docs:
                d=np.asarray(doc,np.int64)
                if len(d)<=lag:continue
                k=np.zeros(len(d)-lag,np.int64)
                for l,r in schema:
                    a=d[lag-l:len(d)-l]
                    k=k*self.V+(self.cls[a] if r else a)
                keys.append(k);values.append(d[lag:])
            if not keys:
                self.tables.append(tuple(np.empty(0,np.int64) for _ in range(3)));continue
            k=np.concatenate(keys);v=np.concatenate(values);ix=np.lexsort((v,k));k=k[ix];v=v[ix]
            s=np.flatnonzero(np.r_[True,(k[1:]!=k[:-1])|(v[1:]!=v[:-1])])
            self.tables.append((k[s],v[s],np.diff(np.r_[s,len(k)])))
    def row(self,h,i):
        schema=SCHEMAS[i];k=0
        if len(h)<max(l for l,r in schema) or any(h[-l]==0 for l,r in schema):return None
        for l,r in schema:k=k*self.V+int(self.cls[h[-l]] if r else h[-l])
        keys,v,c=self.tables[i];a,b=np.searchsorted(keys,k,side='left'),np.searchsorted(keys,k,side='right')
        return (v[a:b],c[a:b]) if b>a else None
    def evidence(self,h,group):
        counts=np.zeros(self.V,np.float64);trust=0.;peak=0.;nrows=0
        for i in group:
            row=self.row(h,i)
            if row is None:continue
            v,c=row;n=float(c.sum());d=len(c)
            w=n/(n+8+d);z=np.maximum(c-.5,0.);z/=z.sum()
            counts[v]+=w*z;trust+=w;peak=max(peak,float(c.max()/n));nrows+=1
        if trust==0:return None
        counts/=trust
        return counts,trust,peak,nrows
    def donors(self,h,parent):
        assert parent.shape==(self.V,) and np.all(parent>0) and abs(parent.sum()-1)<1e-8
        mass=np.bincount(self.cls,weights=parent,minlength=self.K)
        out=[];features=[];valid=[]
        for j,group in enumerate(GROUPS):
            e=self.evidence(h,group)
            if e is None:
                out.append(parent);features.append(np.zeros(5,np.float32));valid.append(False);continue
            q,t,peak,nrows=e;strength=t/(t+.75)
            if j<2:
                cm=np.bincount(self.cls,weights=q,minlength=self.K)
                target=(1-strength)*mass+strength*cm
                d=parent*(target/mass)[self.cls]
            else:d=(1-strength)*parent+strength*q
            assert abs(d.sum()-1)<1e-9 and np.min(d)>0
            out.append(d);features.append(np.array([t/len(group),peak,nrows/len(group),np.log1p(np.count_nonzero(q))/10,float(q.max())],np.float32));valid.append(True)
        return out,np.stack(features),np.asarray(valid,bool)

class Credit(nn.Module):
    def __init__(self):
        super().__init__();self.gain=nn.Parameter(torch.zeros(4,6));self.gain.data[:,0]=-2.5
    def weights(self,f,valid,scale=1.):
        z=(self.gain[:,0]+(f*self.gain[:,1:]).sum(-1)).double().masked_fill(~valid,-torch.inf)
        p=torch.cat((torch.zeros_like(z[:,:1]),z),-1).softmax(-1)*scale
        return torch.cat((1-scale+p[:,:1],p[:,1:]),-1)
    def forward(self,p,f,v,scale=1.):return (p.double()*self.weights(f,v,scale)).sum(-1)

def fuse(parent,donors,w):
    if not np.any(w[1:]):return parent
    q=w[0]*parent+sum(a*p for a,p in zip(w[1:],donors))
    assert abs(q.sum()-1)<1e-8 and np.min(q)>0
    return q
