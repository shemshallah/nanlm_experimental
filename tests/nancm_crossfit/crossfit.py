"""Exact leave-one-document-out continuation counts and target-free gate features."""
import collections as C
import numpy as np
import torch
from torch import nn

class Exclusion:
    def __init__(self,seq,V):
        self.freq=np.bincount(seq,minlength=V);self.rows=[C.defaultdict(C.Counter) for _ in range(3)]
        for i,t in enumerate(seq):
            for order in range(1,min(3,i)+1):
                k=0
                for v in seq[i-order:i]:k=k*V+int(v)
                self.rows[order-1][k][int(t)]+=1

class Crossfit:
    def __init__(self,index):
        self.index=index;self.V=index.V
        self.freq=np.zeros(self.V,np.int64)
        for seq in index.seq:self.freq+=np.bincount(seq,minlength=self.V)
    def query(self,h,target=None,exclude=None):
        """Scalar target probability (training) or complete vector (deployment)."""
        f=self.freq if exclude is None else self.freq-exclude.freq
        assert f.min()>=0
        den=float(f.sum()+.1*self.V)
        q=(f.astype(np.float64)+.1)/den if target is None else (float(f[target])+.1)/den
        features=[]
        for order,(keys,ids,counts) in enumerate(self.index.tables,1):
            if len(h)<order or not all(h[-order:]):features.extend([0.,0.,0.,0.]);continue
            k=0
            for v in h[-order:]:k=k*self.V+int(v)
            a=int(np.searchsorted(keys,k,'left'));b=int(np.searchsorted(keys,k,'right'))
            v=ids[a:b];c=counts[a:b].astype(np.int64,copy=True)
            if exclude is not None:
                local=exclude.rows[order-1].get(k,{})
                if local:c-=np.asarray([local.get(int(w),0) for w in v],np.int64)
            assert (c>=0).all()
            ok=c>0;v=v[ok];c=c[ok];n=float(c.sum());d=len(c)
            if not n:features.extend([0.,0.,0.,0.]);continue
            trust=n/(n+8+d);features.extend([np.log1p(n)/12,np.log1p(d)/12,float(c.max()/n),trust])
            q*=1-trust
            if target is None:q[v]+=trust*c/n
            else:q+=trust*float(c[v==target].sum())/n
        if target is None:assert abs(q.sum()-1)<1e-10
        return q,np.asarray(features,np.float32)

class Gate(nn.Module):
    def __init__(self):
        super().__init__();self.linear=nn.Linear(12,1);nn.init.zeros_(self.linear.weight);nn.init.zeros_(self.linear.bias)
    def forward(self,features,base,strength=1.):
        if strength==0 or base==0 or base==1:return torch.full(features.shape[:-1],base,dtype=torch.float64,device=features.device)
        center=torch.logit(torch.as_tensor(base,dtype=torch.float64,device=features.device))
        return (center+4*strength*self.linear(features).squeeze(-1).double().tanh()).sigmoid()

def loss(field_logprob,count_prob,gate):
    return -torch.log((1-gate)*count_prob.double()+gate*field_logprob.double().exp()).mean()
