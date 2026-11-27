"""v81: conserve nonletter event mass, learn conditional word-vs-byte allocation."""
import math
import numpy as np
import torch
from torch import nn
import folded_event as FE

class Hurdle(FE.Event):
    def __init__(self,word,raw,root,surface,weight,conditional,order=3,cap=64):
        super().__init__(word,raw,root,surface,weight,order,cap)
        q=self.donor(self.suffix(self.root))
        self.r=float(q[FE.IS_LETTER].sum())
        self.m=weight+(1-weight)*self.r
        self.original=weight/self.m
        assert 0<=conditional<1 and self.r>0 and self.m>0
        self.a=self.m*(1-conditional)/self.r
        self.b=self.m*conditional
        self.identity=conditional==self.original
    def coefficients(self,p):
        return (1-self.weight,0.) if p and not FE.IS_LETTER[p[0]] else (self.a,self.b)
    def mass(self,p):
        if self.identity:return super().mass(p)
        if not p:return 1.
        a,b=self.literal(p);x,y=self.coefficients(p)
        return x*a+y*b
    def prefix_mass(self,p):
        if self.identity:return super().prefix_mass(p)
        if not p:return 1.
        return self.a*math.fsum(self.states(p).values())+self.b*self.word.mass(p.decode('ascii'))
    def complete(self,p):
        if self.identity:return super().complete(p)
        if not p or len(p)>=self.cap:return 0.
        raw=math.fsum(m*float(self.donor(s)[FE.STOP].sum()) for s,m in self.states(p).items())
        return self.a*raw+self.b*self.word.complete(p.decode('ascii'))
    def next(self,p):
        if self.identity:return super().next(p)
        assert not self.done(p)
        if not p:
            q=self.donor(self.suffix(self.root)).copy()
            q[FE.IS_LETTER]*=self.a;q[FE.STOP]*=1-self.weight
            for c in FE.LOWER:
                v=self.b*self.word.mass(chr(c));case=self.surface.letter(b'',c)
                q[c]+=v*case[0];q[c-32]+=v*case[1]
        else:
            a,b=self.literal(p);a*=self.a;b*=self.b
            q=a*self.donor(self.suffix(self.root+p)).copy()
            if b:
                s=FE.folded(p);den=self.word.mass(s)
                for c in FE.LOWER:
                    v=b*self.word.mass(s+chr(c))/den;case=self.surface.letter(p,c)
                    q[c]+=v*case[0];q[c-32]+=v*case[1]
                q+=b*self.word.complete(s)/den*self.surface.delimiter(p)
            q/=(a+b)
        assert q.min()>0 and abs(q.sum()-1)<1e-8
        return q

class Gate81(nn.Module):
    def __init__(self,d=8):
        super().__init__();self.theta=nn.Parameter(torch.zeros(d,dtype=torch.float64))
    def forward(self,f,s,scale=1.):
        if scale==0:return s
        return (s.clamp(1e-12,1-1e-12).logit()+scale*3*(f*self.theta).sum(-1).tanh()).sigmoid()

def event_prob(s,r,m,raw,lex):
    return m*((1-s)*raw/r+s*lex)

def selftest():
    from trajectory69 import Sparse
    law=Sparse({'a':.3,'ab':.2,'b':.5})
    raw=lambda h:np.arange(1,257,dtype=np.float64)/32896.
    surface=FE.Surface([b'A ab b a.'])
    old=FE.Event(law,raw,b' ',surface,.4)
    q=Hurdle(law,raw,b' ',surface,.4,.75)
    ident=Hurdle(law,raw,b' ',surface,.4,q.original)
    for p in (b'',b'a',b'A',b'b',b'ab'):
        assert np.array_equal(ident.next(p),old.next(p))
        assert abs(q.mass(p)-sum(q.mass(p+bytes([i])) for i in range(256)))<1e-12
        assert np.max(np.abs(q.next(p)-np.array([q.mass(p+bytes([i]))/q.mass(p) for i in range(256)])))<1e-12
    assert np.array_equal(q.next(b'')[FE.STOP],old.next(b'')[FE.STOP])
    for w in (b'a',b'ab',b'b'):
        assert abs(q.complete(w)-sum(q.mass(w+bytes([i])) for i in FE.STOP)-sum(q.mass(w.upper()+bytes([i])) for i in FE.STOP)- (0 if len(w)==1 else sum(q.mass(v+bytes([i])) for v in (b'Ab',b'aB') for i in FE.STOP)))<1e-12
    gate=Gate81();f=torch.ones(2,8,dtype=torch.float64);s=torch.tensor([.2,.4],dtype=torch.float64)
    assert torch.equal(gate(f,s,0.),s)
    (-gate(f,s).log().mean()).backward();assert gate.theta.grad.abs().sum()>0
    print('v81 exact zero identity, first-byte punctuation conservation, complete word mass, literal parity and gradient passed')
if __name__=='__main__':selftest()
