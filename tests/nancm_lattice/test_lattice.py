import numpy as np,torch
from lattice import Lattice,Credit,fuse

def main():
    torch.set_num_threads(2);torch.manual_seed(56)
    cls=np.array([0,1,1,2,2,3,3,4,4,5]);parent=np.ones(len(cls))/len(cls)
    docs=[[3,1,5,9]*40,[4,1,5,9]*40,[7,8,9,9]*40]
    bank=Lattice(docs,cls);a,f,v=bank.donors([9,4,2],parent);b,_,_=bank.donors([9,7,8],parent)
    assert a[0][6]>parent[6] and a[0][6]>b[0][6]
    assert a[0][5]/a[0][6]==parent[5]/parent[6]
    assert all(abs(p.sum()-1)<1e-12 for p in a+b)
    empty,_,vv=bank.donors([],parent);assert not vv.any() and all(p is parent for p in empty)
    m=Credit().double();f=torch.tensor(f[None],dtype=torch.float64);v=torch.tensor(v[None]);w=m.weights(f,v,0)[0].detach().numpy();assert fuse(parent,a,w) is parent
    q=torch.rand(64,5,dtype=torch.float64)*.8+.01;x=torch.rand(64,4,5,dtype=torch.float64);valid=torch.rand(64,4)>.15
    loss=lambda:-m(q,x,valid).log().mean();loss().backward();g=m.gain.grad.clone();fd=0
    for ix in np.ndindex(tuple(g.shape)):
        with torch.no_grad():m.gain[ix]+=1e-5;hi=float(loss());m.gain[ix]-=2e-5;lo=float(loss());m.gain[ix]+=1e-5
        fd=max(fd,abs((hi-lo)/2e-5-float(g[ix])))
    assert fd<1e-8 and torch.count_nonzero(g)==g.numel()
    print(dict(abstract_unseen_word_before=float(parent[6]),correct=float(a[0][6]),unrelated=float(b[0][6]),max_mass_error=float(max(abs(p.sum()-1) for p in a+b)),gradient_max_error=fd,gradient_coordinates=g.numel(),exact_zero=True,empty_exact=True,class_ratio_exact=True))
if __name__=='__main__':main()
