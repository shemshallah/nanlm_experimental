"""Bounded write-time admission metadata and differentiable retention basis."""
import sys,pathlib
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'nancm_graph'))
from persistent_edges import Graph,WINDOWS,address
from adaptive_retention import AGES,Retention
import numpy as np
import collections as C
import torch
class AdaptiveGraph(Graph):
    def __init__(self,*args):
        super().__init__(*args);self.meta=[C.deque() for _ in WINDOWS]
    def observe(self,w):
        for r,key in enumerate(self.keys(self.h)):
            q=self.meta[r];expired=0
            while q and self.t-q[0][0]>=WINDOWS[r]:
                _,k,v,_,_=q.popleft();expired+=int(k==key and v==w)
            if key is not None:
                repeated=int(self.rows[r].get(key,{}).get(w,0)>expired)
                q.append((self.t,key,w,repeated,address((r,key))))
        super().observe(w)
        assert all(len(a)==len(b) for a,b in zip(self.meta,self.ring))
    def basis(self,h=None):
        h=self.h if h is None else h
        moments=np.zeros((8,4,2,128),np.float32);counts=np.zeros((8,4,2),np.float32)
        for r,key in enumerate(self.keys(h)):
            if key is None:continue
            a=address((r,key))
            for t,k,w,repeat,addr in self.meta[r]:
                if addr!=a:continue
                age=self.t-t;idx=int(np.searchsorted(AGES,age,side='left'))
                assert 0<=idx<4 and age>=1
                moments[r,idx,repeat]+=self.vec(w);counts[r,idx,repeat]+=1
        bg=self.background/max(1,len(self.doc_ring))
        return moments,counts,bg.copy()
    def snapshot(self):return super().snapshot(),[list(q) for q in self.meta]
    def restore(self,s):
        super().restore(s[0]);self.meta=[C.deque(q) for q in s[1]]
        assert all(len(a)==len(b) for a,b in zip(self.meta,self.ring))
def preflight():
    rng=np.random.default_rng(44);words=['alpha','beta','answer','noise'];ids={w:i+1 for i,w in enumerate(words)};code=rng.normal(size=(5,16)).astype(np.float32);code[0]=0
    g=AdaptiveGraph(code,ids,lambda w:ids.get(w,0))
    assert not g.basis()[1].any()
    for w in ['alpha','beta','answer','alpha','beta','answer']+['noise']*150+['alpha','beta']:g.observe(w)
    m,c,b=g.basis();_,vec,n=g.query();assert np.array_equal(c.sum((1,2)),n)
    with torch.no_grad():res,count=Retention()(torch.tensor(m[None]),torch.tensor(c[None]),torch.tensor(b[None]))
    target=(vec-b)*(n>0)[:,None];assert np.allclose(res[0].numpy(),target,atol=1e-5)
    copy=AdaptiveGraph(code,ids,lambda w:ids.get(w,0));copy.restore(g.snapshot())
    for x,y in zip(copy.basis(),g.basis()):assert np.allclose(x,y,atol=1e-5)
    assert g.propose()[0]=='answer';assert c[:,2,1].sum()>0
    for _ in range(1025):g.observe('noise')
    assert not g.propose(['alpha','beta'])
    assert all(len(q)<=n for q,n in zip(g.meta,WINDOWS))
    print({'basis_count_parity':True,'fixed_residual_max_error':float(np.max(np.abs(res[0].numpy()-target))),'repeat_age_signal':True,'snapshot':True,'eviction':True})
if __name__=='__main__':preflight()
