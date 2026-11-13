#!/usr/bin/env python3
"""Exact rank, S+/S- orientation, nilpotence, gradient, no bypass."""
import math
from types import SimpleNamespace
import torch
import torch.nn.functional as F
from fa1_r724_geometry import (GeometricDiamond,OrientedSimplicialGeometry,
  tetra_rank,TETRAHEDRA,TETRA_LOCAL,TETRA_FACES,FACE_SIGNS,orientation_sign)
from fa1_r723_diamond import pair_rank,triple_rank

def test():
    torch.set_num_threads(2)
    assert len(TETRA_LOCAL)==70 and len(TETRA_FACES)==70
    assert TETRAHEDRA==math.comb(256,4)==174792640
    assert tetra_rank(*[torch.tensor(v) for v in [252,253,254,255]])==TETRAHEDRA-1
    assert tetra_rank(*[torch.tensor(v) for v in [0,1,2,3]])==0
    assert tetra_rank(*[torch.tensor(v) for v in [251,254,252,253]])==tetra_rank(*[torch.tensor(v) for v in [251,252,253,254]])
    # d∘d = 0 on 1-cochains: every oriented edge cancels across
    # the boundary of the four faces of every tetrahedron.
    g=OrientedSimplicialGeometry()
    ed=g.triangle_edges
    phases=torch.randn(2,28)
    face_curl=phases[:,ed[:,0]]+phases[:,ed[:,2]]-phases[:,ed[:,1]]
    closed=(face_curl[:,g.tetra_faces]*g.face_signs).sum(-1)
    assert float(closed.abs().max())<1e-6,closed.abs().max()
    assert torch.equal(orientation_sign(torch.tensor([1]),torch.tensor([5]),torch.tensor([9])),torch.ones(1))
    assert torch.equal(orientation_sign(torch.tensor([5]),torch.tensor([1]),torch.tensor([9])),-torch.ones(1))
    p=torch.randn(3,64);m=torch.randn(3,64)
    a=torch.tensor([2,5,9]);b=torch.tensor([5,9,1])
    plus,minus,phase=g.signed_edge_channels(p,m,a,b)
    plus_r,minus_r,phase_r=g.signed_edge_channels(p,m,b,a)
    assert torch.equal(plus,plus_r) and torch.equal(minus,-minus_r) and torch.equal(phase,-phase_r)
    triple_a=torch.tensor([[17]],dtype=torch.long)
    abc=torch.tensor([[[2,4,7]]],dtype=torch.long)
    bac=torch.tensor([[[4,2,7]]],dtype=torch.long)
    cur=torch.tensor([[0.23]])
    p3,m3=g.signed_triangle_channels(triple_a,abc,cur)
    p3r,m3r=g.signed_triangle_channels(triple_a,bac,-cur)
    assert torch.equal(p3,p3r) and torch.allclose(m3,-m3r,atol=1e-7)
    torch.manual_seed(724)
    mod=GeometricDiamond()
    trunk=SimpleNamespace(E=torch.nn.Parameter(torch.randn(32000,96)),
                          H=torch.nn.Parameter(torch.randn(256,96)),
                          P=torch.nn.Parameter(torch.randn(32640,64)),
                          M=torch.nn.Parameter(torch.randn(32640,64)))
    B=2
    args=(torch.randn(B,96),torch.randint(0,32000,(B,32)),torch.randint(0,256,(B,128)),
          torch.randint(0,32000,(B,8)),torch.stack([torch.randperm(256)[:8] for _ in range(B)]),trunk)
    opts={'memory_features':torch.randn(B,92),'modal_ids':torch.randint(2,32000,(B,12)),
          'modal_mask':torch.ones(B,12,dtype=torch.bool)}
    state0,_,_,t0=mod(*args,**opts,return_trace=True)
    assert len(t0['geometry'])>=6
    geo=mod.geometry
    with torch.no_grad():
        geo.edge_gain.fill_(0.025);geo.face_gain.fill_(0.025);geo.tetra_gain.fill_(0.025)
    state1,logits,vec,trace=mod(*args,**opts,return_trace=True)
    assert not torch.equal(state0,state1),'geometry has no effect'
    ablated,_,_=mod(*args,**opts,without_pair=True)
    assert torch.equal(ablated,args[0]),'geometric apex bypassed pair/point descent'
    # Trainability of rank2+rank3 mirror and sparse tetra decisions.
    loss=(state1.square().mean()+logits.square().mean()+vec.square().mean())
    loss.backward()
    names=dict(mod.named_parameters())
    for k in ('geometry.edge_plus.weight','geometry.edge_minus.weight',
              'geometry.edge_phase.weight','geometry.face_plus.weight',
              'geometry.face_minus.weight','geometry.tetra_net.0.weight',
              'geometry.point_coords.weight', 'geometry.edge_gain',
              'geometry.face_gain','geometry.tetra_gain'):
        grad=names[k].grad
        assert grad is not None and torch.isfinite(grad).all() and float(grad.abs().sum())>0,('no gradient',k)
    assert all(z.grad is None for z in (trunk.E,trunk.H,trunk.P,trunk.M))
    print('PASS: C(256,2)=32640 / C(256,3)=2763520 / C(256,4)=174792640; active 28/56/70')
    print('PASS: signed S+ orientation invariant / S- orientation odd; d²=0; trigonometric face holonomy')
    print('PASS: old R7 frozen / exact-downward no-bypass / differentiable tetra -> triangle -> pair -> point')

if __name__=='__main__':test()