"""v93: independently route boundary, punctuation, raw-word and lexical mass.

All routing is computed from causal context before the first byte of an
event. The two raw word donors are normalized conditional on starting
with a letter, independently of their punctuation distributions.
"""
import math
import numpy as np
import torch
from torch import nn
import folded_event as FE
from field82 import LongEvent
from field81 import Hurdle

class RoutedEvent:
    def __init__(self,law,raw,long,root,surface,share,route):
        self.word=law;self.surface=surface;self.weight=share;self.cap=64
        self.old=FE.Event(law,raw,root,surface,0.)
        self.new=LongEvent(law,long,root,surface,0.)
        self.q0=self.old.donor(self.old.suffix(root))
        self.q1=self.new.donor(self.new.suffix(root))
        self.r0=float(self.q0[FE.IS_LETTER].sum());self.r1=float(self.q1[FE.IS_LETTER].sum())
        g,h,k,c=map(float,route)
        assert all(0<=v<=1 for v in (g,h,k,c)) and c<1
        self.m=share+(1-share)*((1-k)*self.r0+k*self.r1)
        self.a0=self.m*(1-c)*(1-g)/self.r0
        self.a1=self.m*(1-c)*g/self.r1
        self.b=self.m*c
        self.n0=(1-self.m)*(1-h)/(1-self.r0)
        self.n1=(1-self.m)*h/(1-self.r1)
    def done(self,p):return self.old.done(p)
    def components(self,p):
        a,b=self.old.literal(p);c,_=self.new.literal(p)
        return a,c,b
    def mass(self,p):
        if not p:return 1.
        a,c,b=self.components(p)
        if not FE.IS_LETTER[p[0]]:return self.n0*a+self.n1*c
        return self.a0*a+self.a1*c+self.b*b
    def prefix_mass(self,p):
        if not p:return 1.
        return self.a0*self.old.prefix_mass(p)+self.a1*self.new.prefix_mass(p)+self.b*self.word.mass(p.decode('ascii'))
    def complete(self,p):
        if not p or len(p)>=self.cap:return 0.
        return self.a0*self.old.complete(p)+self.a1*self.new.complete(p)+self.b*self.word.complete(p.decode('ascii'))
    def next(self,p):
        assert not self.done(p)
        a,c,b=self.components(p)
        a*=self.a0;c*=self.a1;b*=self.b
        q=a*self.old.donor(self.old.suffix(self.old.root+p))+c*self.new.donor(self.new.suffix(self.new.root+p))
        if b:
            w=FE.folded(p);den=self.word.mass(w)
            assert den>0
            for v in FE.LOWER:
                m=b*self.word.mass(w+chr(v))/den
                case=self.surface.letter(p,v)
                q[v]+=m*case[0];q[v-32]+=m*case[1]
            if p:q+=b*self.word.complete(w)/den*self.surface.delimiter(p)
        if not p:q[FE.STOP]=self.n0*self.q0[FE.STOP]+self.n1*self.q1[FE.STOP]
        else:q/=a+c+b
        assert q.min()>0 and abs(q.sum()-1)<1e-8
        return q
    def posterior_components(self,q,newq,event):
        # Three inherited states: combined raw, lexical A, lexical B.
        if FE.IS_LETTER[event[0]]:
            return np.array([(self.a0*q[0]+self.a1*newq)/(1-self.weight),
                             q[1]*self.b/self.weight,q[2]*self.b/self.weight])
        return np.array([(self.n0*q[0]+self.n1*newq)/(1-self.weight),0.,0.])
    rank=FE.Event.rank
    tokens=FE.Event.tokens

def event93(law,raw,long,root,surface,share,route):
    g,h,k,c=route
    if g==h==k==0:return Hurdle(law,raw,root,surface,share,c)
    return RoutedEvent(law,raw,long,root,surface,share,route)

def features93(prefix,q0,q1,long,w,cond,ls):
    r0=float(q0[FE.IS_LETTER].sum());r1=float(q1[FE.IS_LETTER].sum())
    depth,support=long.support(prefix)
    H=lambda q:-float(q@np.log(q))/8.
    return np.array([1.,r0,r1,w[0],w[1],w[2],cond,H(q0),H(q1),float(np.abs(q0-q1).sum())/2.,
        depth/12.,math.log1p(support)/12.,float(prefix[-1:]==b' '),float(prefix[-1:] in (b'\n',b'\r')),
        math.log1p(len(prefix))/10.,max((float(v[2]) for v in ls[4:]),default=0.)],np.float64)

class Route93(nn.Module):
    def __init__(self):
        super().__init__()
        self.theta=nn.Parameter(torch.zeros(4,16,dtype=torch.float64))
        # Initial exploration: half raw-word, half punctuation, half boundary
        # adaptation; the conditional lexical gate starts exactly inherited.
    def forward(self,f,c,scale=1.):
        v=f@self.theta.T
        g=scale*torch.sigmoid(v[...,:3])
        if scale==0:return torch.cat((g,c[...,None]),-1)
        lex=(c.clamp(1e-12,1-1e-12).logit()+scale*3*v[...,3].tanh()).sigmoid()
        return torch.cat((g,lex[...,None]),-1)

def probabilities(route,r0,r1,share,components,isletter):
    # components: short raw literal, long raw literal, integrated lexical literal.
    g,h,k,c=route.unbind(-1)
    m=share+(1-share)*((1-k)*r0+k*r1)
    a,b,l=components.unbind(-1)
    word=m*((1-c)*((1-g)*a/r0+g*b/r1)+c*l)
    stop=(1-m)*((1-h)*a/(1-r0)+h*b/(1-r1))
    return torch.where(isletter,word,stop)

def selftest():
    from trajectory69 import Sparse
    from field82 import LongByte
    import itertools
    docs=[b'The cat ran. The Cat sat.',b'The dog ran. A cat sat.']
    long=LongByte(docs,12);surface=FE.Surface(docs)
    law=Sparse({'a':.2,'cat':.5,'dog':.3})
    raw=lambda h:np.arange(1,257,dtype=np.float64)/32896.
    share=.3;root=b'The ';c=.65
    old=Hurdle(law,raw,root,surface,share,c)
    ident=event93(law,raw,long,root,surface,share,(0.,0.,0.,c))
    for p in (b'',b'c',b'Ca',b'dog'):
        assert np.array_equal(old.next(p),ident.next(p))
    for g,h,k in ((.5,.5,.5),(.1,.9,.2),(1.,0.,1.),(0.,1.,0.)):
        route=(g,h,k,c);e=event93(law,raw,long,root,surface,share,route)
        for p in (b'',b'c',b'Ca',b'cat',b'Do'):
            q=e.next(p);z=np.array([e.mass(p+bytes([i]))/e.mass(p) for i in range(256)])
            assert np.max(np.abs(q-z))<1e-12
        assert abs(e.next(b'')[FE.IS_LETTER].sum()-e.m)<1e-12
        for p in (b'a',b'cat',b'dog'):
            variants=[bytes(v) for v in itertools.product(*[(b,b-32) for b in p])]
            want=math.fsum(e.mass(v+bytes([i])) for v in variants for i in FE.STOP)
            assert abs(e.complete(p)-want)<1e-12
        for p in (b'Cat ',b' ',b'dog.'):
            a,b,l=e.components(p);q=np.array([a,l,l]);w=np.array([1-share,.2,.1])
            assert abs(float(w@e.posterior_components(q,b,p))-e.mass(p))<1e-12
            t=lambda v:torch.as_tensor(v,dtype=torch.float64)
            got=probabilities(t(route),t(e.r0),t(e.r1),t(share),t([a,b,l]),torch.tensor(bool(FE.IS_LETTER[p[0]])))
            assert abs(float(got)-e.mass(p))<1e-12
    m=Route93();f=torch.randn(12,16,dtype=torch.float64);c=torch.full((12,),.6,dtype=torch.float64)
    v=m(f,c);r=torch.full((12,),.5,dtype=torch.float64)
    comp=torch.rand(12,3,dtype=torch.float64)*.2
    mask=torch.arange(12)%2==0
    p=probabilities(v,r,r*.8,r*.5,comp,mask)
    (-p.log().mean()).backward()
    assert (m.theta.grad.abs().sum(-1)>0).all()
    assert torch.equal(m(f,c,0.)[:,3],c)
    print('v93 independent boundary, punctuation, raw-word and lexical routing; exact posterior and byte/word law passed',flush=True)
if __name__=='__main__':selftest()
