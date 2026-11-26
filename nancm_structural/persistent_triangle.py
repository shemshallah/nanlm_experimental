"""Persistent triangle anchor, with fixed-point-safe convex edge blending."""
import torch
from torch import nn

class TriangleAnchor(nn.Module):
    def __init__(self, rails=8, points=128, triangles=1024):
        super().__init__()
        self.lower=nn.Parameter(torch.zeros(rails,points))
        self.support=nn.Parameter(torch.zeros(rails,points))
        self.opposition=nn.Parameter(torch.zeros(rails,points))
        self.gate=nn.Parameter(torch.zeros(rails,2))
        self.tri_plus=nn.Parameter(torch.zeros(rails,triangles,3))
        self.tri_minus=nn.Parameter(torch.zeros(rails,triangles,3))
        self.query=nn.Parameter(torch.zeros(points,16))
        nn.init.normal_(self.query,std=.05)
    def forward(self,s,z,residual,counts,pairs,scale=1.):
        assert 0<=scale<=1
        strength=counts/(counts+4)
        lower=s.tanh()*(1+self.lower.tanh())*.5
        u=residual*(1+self.support.tanh())*.5
        v=residual*(1+self.opposition.tanh())*.5
        a=torch.cat((lower,u),-1);b=torch.cat((lower,v),-1)
        i,j=pairs[:,0],pairs[:,1]
        seed=torch.stack(((a[...,i]+a[...,j]).mul(.5).tanh(),(b[...,i]-b[...,j]).mul(.5).tanh()),-1)
        gate=.5*scale*strength[...,None,None]*self.gate.sigmoid()[None,:,None,:]
        factor=scale*strength[...,None,None]
        return seed,gate,factor*self.tri_plus.tanh(),factor*self.tri_minus.tanh()
    def lift(self,delta,code):
        return 4*((delta@self.query)[:,None,:]*code).sum(-1).div(4).tanh()

def blend(edges,seed,gate):
    return torch.where(gate==0,edges,torch.lerp(edges,seed,gate))
