"""Sparse triangle incidence, signed edge states, bounded relational messages."""
import torch
import itertools

SCALES=(.020,.015)
def topology():
    triples=sorted({tuple(sorted((i,(i+a)%256,(i+b)%256))) for i in range(256) for a,b in ((1,17),(3,61),(31,127),(64,129))})
    edges=sorted({e for t in triples for e in itertools.combinations(t,2)})
    lookup={e:i for i,e in enumerate(edges)}
    incident=[[[lookup[tuple(sorted((t[v],t[j]))) ] for j in range(3) if j!=v] for v in range(3)] for t in triples]
    vertices=torch.tensor(triples,dtype=torch.long);pairs=torch.tensor(edges,dtype=torch.long)
    degree=torch.bincount(vertices.flatten(),minlength=256).clamp_min(1)
    assert len(set(vertices.flatten().tolist()))==256
    return vertices,pairs,torch.tensor(incident,dtype=torch.long),degree

def edge_update(points,old,pairs):
    a=points[:,:,pairs[:,0]];b=points[:,:,pairs[:,1]]
    instant=torch.stack(((a+b)*.5,(a-b)*.5),-1).tanh()
    return instant if old is None else .5*old+.5*instant

def message(edges,positive,negative,topo,mode='triangle'):
    vertices,pairs,inc,degree=topo
    a=edges[:,:,inc[:,:,0],:];b=edges[:,:,inc[:,:,1],:]
    if mode=='pair':
        support=(a[...,0]+b[...,0])*.5
        opposition=(a[...,1].square()+b[...,1].square())*.5
    else:
        support=a[...,0]*b[...,0]
        opposition=((a[...,1]-b[...,1])*.5).square()
    m=SCALES[0]*positive.tanh()*support-SCALES[1]*negative.tanh()*opposition
    idx=vertices.flatten()[None,None,:].expand(edges.shape[0],edges.shape[1],-1)
    return edges.new_zeros((edges.shape[0],edges.shape[1],256)).scatter_add(2,idx,m.flatten(2))/degree

def preflight():
    torch.set_num_threads(2);torch.manual_seed(3101)
    topo=topology();v,p,inc,d=topo
    x=torch.rand(2,8,256,dtype=torch.float64)*2-1;y=torch.rand_like(x)*2-1
    w=torch.zeros(8,len(v),3,dtype=x.dtype,requires_grad=True);n=torch.zeros_like(w,requires_grad=True)
    e=edge_update(x,None,p);assert torch.equal(message(e,w,n,topo),torch.zeros_like(x))
    with torch.no_grad():w.uniform_(.1,.5);n.uniform_(.1,.5)
    def f():return message(e,w,n,topo).square().mean()
    f().backward();errs=[]
    for a in (w,n):
        assert a.grad.norm()>0
        idx=tuple(torch.unravel_index(a.grad.abs().argmax(),a.shape));g=a.grad[idx].item()
        with torch.no_grad():
            a[idx]+=1e-5;hi=f().item();a[idx]-=2e-5;lo=f().item();a[idx]+=1e-5
        errs.append(abs(g-(hi-lo)/2e-5))
    with torch.no_grad():
        delta=(message(e,w,n,topo)-message(edge_update(y,None,p),w,n,topo)).abs().max()/(x-y).abs().max()
        assert delta<=.07+1e-12
        assert not torch.allclose(message(e,w,n,topo),message(e,w,n,topo,'pair'))
        assert not torch.allclose(edge_update(y,e,p),edge_update(y,None,p))
    assert max(errs)<1e-10
    print(dict(triangles=len(v),edges=len(p),parameters=w.numel()+n.numel(),gradient_errors=errs,observed_lipschitz=float(delta),joint_bound=.975,zero_parity=True,pair_control_distinct=True,edge_memory=True))
if __name__=='__main__':preflight()
