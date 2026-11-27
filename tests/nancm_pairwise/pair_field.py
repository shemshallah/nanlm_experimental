"""Signed pairwise rail interaction, optional residual on v50 field."""
import torch
from torch import nn
from hierarchical_field import Field

class PairInteraction(nn.Module):
    def __init__(self,d=128,rank=32):
        super().__init__()
        self.register_buffer('pairs',torch.triu_indices(8,8,offset=1))
        self.left=nn.Parameter(torch.randn(8,d,rank)*.05)
        self.right=nn.Parameter(torch.randn(8,d,rank)*.05)
        self.support=nn.Parameter(torch.zeros(28,rank))
        self.opposition=nn.Parameter(torch.zeros(28,rank))
        self.route=nn.Parameter(torch.zeros(28))
        self.read=nn.Parameter(torch.randn(rank,d)*.05)
    def forward(self,x):
        a=torch.einsum('bri,rij->brj',x.tanh(),self.left).tanh()
        b=torch.einsum('bri,rij->brj',x.tanh(),self.right).tanh()
        i,j=self.pairs
        p=a[:,i]*b[:,j]
        rel=p.relu()*(1+self.support.tanh())-(-p).relu()*(1+self.opposition.tanh())
        return .5*((rel*self.route.softmax(0)[None,:,None]).sum(1)@self.read).tanh()

class PairField(Field):
    def __init__(self,code,prior,**kwargs):
        super().__init__(code,prior,**kwargs)
        self.pair=PairInteraction(d=self.code.shape[1]);self.pair_scale=1.
    def hidden(self,x,ids,coupled=True):
        h=super().hidden(x,ids,coupled)
        if not self.pair_scale:return h
        xx=torch.cat((x[:,:4]+self.id_in(self.identity(ids)),x[:,4:]),1)
        return h+self.pair_scale*self.pair(xx)
