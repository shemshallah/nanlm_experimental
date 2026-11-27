"""Causal geometry-conditioned transition with exact event and lexical credit."""
import torch
from torch import nn
from transition import Transition

class Joint(nn.Module):
    def __init__(self,prior,hazard,d=128):
        super().__init__();self.core=Transition(prior,hazard)
        self.signed=nn.Parameter(torch.zeros(2,3,3,d,dtype=torch.float64))
    def matrix(self,f,h,scale=1.):
        h=h.double();delta=torch.einsum('ijd,d->ij',self.signed[0],h.relu())-torch.einsum('ijd,d->ij',self.signed[1],(-h).relu())
        base=self.core.base+3*torch.einsum('d,ijd->ij',f,self.core.theta).tanh()
        return (base+2*scale*delta.tanh()).softmax(-1)
    def predict(self,features,h,state=None,prev=None,scale=1.):
        if state is None:return self.core.prior
        v=prev.clamp_min(1e-300).log();f=torch.cat((features,((v[1:]-v[0])/8).clamp(-1,1)))
        return state@self.matrix(f,h,scale)
    def advance(self,w,lik):
        a=w*lik;z=a.sum();return a/z,z
    def trajectory(self,a,b,raw,e,s,features,hidden,target,state=None,prev=None,scale=1.,word_weight=.25):
        event=a.new_zeros(());word=a.new_zeros(());count=0
        for t in range(len(a)):
            w=self.predict(features[t],hidden[t],state,prev,scale)
            lik=torch.stack((raw[t],(a[t]*e[t]).sum()*s[t],(b[t]*e[t]).sum()*s[t]))
            state,z=self.advance(w,lik);event=event-z.log();prev=lik
            if int(target[t])>=0:
                y=target[t];p=(w[1]*a[t,y]+w[2]*b[t,y])/(w[1]+w[2]);word=word-p.log();count+=1
        return event/len(a)+word_weight*word/max(1,count),state,prev,event,word,count
