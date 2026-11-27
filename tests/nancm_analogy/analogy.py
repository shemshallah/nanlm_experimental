"""Causal signed analogical memory. Reference lookup is target-independent."""
import numpy as np
import torch
from torch import nn

class Analogy:
    def __init__(self,code,docs):
        self.code=np.asarray(code,np.float32)
        self.code=self.code/np.maximum(np.linalg.norm(self.code,axis=1,keepdims=True),1e-8)
        self.role=(1+2*np.argmax(abs(self.code),axis=1)+(self.code[np.arange(len(code)),np.argmax(abs(self.code),axis=1)]<0)).astype(np.int32);self.role[0]=0
        rows=[];books=[]
        for i,doc in enumerate(docs):
            d=np.asarray(doc,np.int32)
            if len(d)<5:continue
            rows.append(np.column_stack([d[3:-1],d[2:-2],d[1:-3],d[:-4],d[4:]]));books.append(np.full(len(d)-4,i,np.int32))
        self.rows=np.concatenate(rows);self.books=np.concatenate(books)
        self.orders=[]
        for keys in (self.rows[:,0],self.role[self.rows[:,0]]):
            ix=np.argsort(keys,kind='stable');self.orders.append((keys[ix],ix))
    def retrieve(self,h,role=False,k=64,exclude=None):
        if len(h)<4 or h[-1]==0:return {},np.zeros(4,np.float32)
        key=self.role[h[-1]] if role else h[-1];keys,ix=self.orders[int(role)]
        a,b=np.searchsorted(keys,key,side='left'),np.searchsorted(keys,key,side='right');rows=ix[a:b]
        if exclude is not None:rows=rows[self.books[rows]!=exclude]
        if not len(rows):return {},np.zeros(4,np.float32)
        ids=np.array(h[-4:][::-1],np.int64);sim=self.code@self.code[ids].T
        r=self.rows[rows];weights=np.array([.7,1.,.65,.35],np.float32)
        if not role:weights[0]=0
        score=sum(weights[j]*sim[r[:,j],j] for j in range(4))/weights.sum()
        score+=.12*np.sum((r[:,1:4]==ids[None,1:4])&(ids[None,1:4]!=0),axis=1)
        n=min(k,len(rows));top=np.argpartition(-score,n-1)[:n]
        top=top[np.lexsort((rows[top],-score[top]))];chosen=rows[top];s=score[top]
        _,inv,cnt=np.unique(self.books[chosen],return_inverse=True,return_counts=True)
        w=np.exp(5*(s-s.max())).astype(np.float64)/np.sqrt(cnt[inv]);w/=w.sum()
        out={}
        for t,p in zip(self.rows[chosen,4],w):out[int(t)]=out.get(int(t),0.)+float(p)
        peak=max(out.values());f=np.array([np.log1p(len(rows))/14,float(s[0])/1.4,len(cnt)/n,peak],np.float32)
        return out,f
    def local(self,h,k=64):
        if len(h)<5:return {},np.zeros(4,np.float32)
        d=np.asarray(h,np.int32);t=np.arange(4,len(d));ids=d[-4:][::-1]
        r=d[t[:,None]-np.arange(1,5)[None,:]];sim=self.code@self.code[ids].T
        score=(sim[r,np.arange(4)]*np.array([1.,.65,.35,.2])).sum(1)/2.2
        score+=.15*np.sum(r==ids,axis=1);score-=.08*np.log1p(len(d)-t)
        keep=(d[t]!=0)&(r[:,0]==ids[0])&(ids[0]!=0)
        t=t[keep];score=score[keep]
        if not len(t):return {},np.zeros(4,np.float32)
        n=min(k,len(t));ix=np.argsort(-score,kind='stable')[:n];s=score[ix];w=np.exp(5*(s-s[0]));w/=w.sum();out={}
        for target,p in zip(d[t[ix]],w):out[int(target)]=out.get(int(target),0.)+float(p)
        return out,np.array([np.log1p(len(t))/8,s[0]/1.6,len(out)/n,max(out.values())],np.float32)
    def experts(self,h,exclude=None):
        return (self.retrieve(h,False,exclude=exclude),self.retrieve(h,True,exclude=exclude),self.local(h))

class Credit(nn.Module):
    def __init__(self):
        super().__init__();self.gain=nn.Parameter(torch.zeros(3,5));self.gain.data[:,0]=-2
    def weights(self,f,valid,strength=1.):
        z=(self.gain[:,0]+(f*self.gain[:,1:]).sum(-1)).double().masked_fill(~valid,-torch.inf)
        q=torch.cat((torch.zeros_like(z[:,:1]),z),-1).softmax(-1)*strength
        return torch.cat((1-strength+q[:,:1],q[:,1:]),-1)
    def forward(self,p,f,valid,strength=1.):return (p.double()*self.weights(f,valid,strength)).sum(-1)

def mix(parent,experts,weights):
    if not np.any(weights[1:]):return parent
    out=weights[0]*parent
    for law,w in zip(experts,weights[1:]):
        if w:
            assert abs(sum(law.values())-1)<1e-7
            for t,p in law.items():out[t]+=w*p
    assert abs(out.sum()-1)<1e-7
    return out
