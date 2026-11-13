#!/usr/bin/env python3
"""Mathematical parity, sparse tetra closures, gradient and no-bypass tests."""
from __future__ import annotations
import torch,itertools
from types import SimpleNamespace
from fa1_r724_curvature import CurvatureDiamond,rank4,triangle_parity,TETRAHEDRA
from fa1_r723_diamond import CombinatorialDiamond,pair_rank,triple_rank

def run():
    torch.manual_seed(724);torch.set_num_threads(2)
    assert TETRAHEDRA==174792640
    g=CurvatureDiamond();g.train()
    # Combinadic 4-simplex indexing exact on all active 70 tetra.
    V=torch.tensor([[3,7,11,12,18,22,35,55]],dtype=torch.long)
    n=torch.randn(1,8,32)
    E=g.edges;F=g.faces;T=g.tetrahedra
    pe=pair_rank(V[:,E[:,0]],V[:,E[:,1]])
    tf=triple_rank(V[:,F[:,0]],V[:,F[:,1]],V[:,F[:,2]])
    tc=torch.randn(1,56,32)
    trunk=SimpleNamespace(P=torch.nn.Parameter(torch.randn(32640,64)),
                          M=torch.nn.Parameter(torch.randn(32640,64)))
    dx,dy,z=g(n,V,pe,tf,tc,trunk,trace=True)
    assert dx.shape==(1,28,32) and dy.shape==(1,56,32)
    ids=z['tetra_addresses'];assert ids.numel()==70 and ids.unique().numel()==70
    assert ids.min()>=0 and ids.max()<TETRAHEDRA
    assert g.tetra_faces.shape==(70,4)
    assert torch.bincount(g.tetra_faces.flatten(),minlength=56).unique().tolist()==[5]
    assert torch.isfinite(z['closure_loss']) and z['closure_loss']>1e-7
    # Antisymmetric S- mirror edge and triangle, symmetric S+ under a swap.
    Vp=V.clone();Vp[:,0],Vp[:,1]=V[:,1],V[:,0]
    np=n.clone();np[:,0],np[:,1]=n[:,1],n[:,0]
    pep=pair_rank(Vp[:,E[:,0]],Vp[:,E[:,1]])
    tfp=triple_rank(Vp[:,F[:,0]],Vp[:,F[:,1]],Vp[:,F[:,2]])
    # Use same global triangle codes for exactly matching triangle IDs.
    tcm=tc.clone(); mp={int(x):tc[0,i].clone() for i,x in enumerate(tf[0])}
    for i,j in enumerate(tfp[0]):tcm[0,i]=mp[int(j)]
    _,_,zp=g(np,Vp,pep,tfp,tcm,trunk,trace=True)
    assert torch.allclose(z['even_edge'][:,0],zp['even_edge'][:,0],atol=1e-5)
    assert torch.allclose(z['odd_edge'][:,0],-zp['odd_edge'][:,0],atol=1e-5)
    assert torch.allclose(z['face_even'][:,0],zp['face_even'][:,0],atol=1e-5)
    assert torch.allclose(z['face_odd'][:,0],-zp['face_odd'][:,0],atol=1e-5)
    assert torch.allclose(z['holonomy'][:,0],-zp['holonomy'][:,0],atol=1e-5)
    assert torch.allclose(z['solid_angle'][:,0],-zp['solid_angle'][:,0],atol=1e-5)
    assert torch.allclose(z['tetra_boundary'][:,0],-zp['tetra_boundary'][:,0],atol=1e-5)
    # Exterior derivative squared is zero for the connection field by
    # construction, but not for the separately trained face curvature.
    hc=z['holonomy'][:,g.tetra_faces]*g.tetra_signs.view(1,1,4)
    assert hc.sum(-1).abs().max()<5e-5
    assert z['tetra_boundary'].abs().mean()>1e-4
    (dx.square().mean()+dy.square().mean()+.01*z['closure_loss']).backward()
    for name in ('mirror_P.weight','mirror_M.weight','edge_connection.0.weight',
                 'independent_curvature.weight','point_sphere.weight',
                 'tetra_message.0.weight','face_tetra_fuse.0.weight'):
        p=dict(g.named_parameters())[name]
        assert p.grad is not None and p.grad.isfinite().all() and p.grad.abs().sum()>1e-10,name
    assert trunk.P.grad is None and trunk.M.grad is None
    # Mandatory strict pair descent with active geometry.
    full=CombinatorialDiamond();full.geometry=g
    with torch.no_grad():full.geometry.gate.fill_(.5)
    t=SimpleNamespace(E=torch.nn.Parameter(torch.randn(32000,96)),
        H=torch.nn.Parameter(torch.randn(256,96)),P=trunk.P,M=trunk.M)
    w=torch.randint(0,32000,(1,32));bb=torch.randint(0,256,(1,128))
    ss=torch.randint(0,32000,(1,8));h=torch.randn(1,96)
    with torch.no_grad():
        args=(h,w,bb,ss,V,t)
        base=full(*args,without_geometry=True)[0]
        use=full(*args)[0]
        pair_off=full(*args,without_pair=True)[0]
        assert torch.equal(pair_off,h),'geometry bypassed pair->point descent'
        assert not torch.equal(base,use),'geometry does not reach trunk decoder'
    print('PASS R7.24 exact C(256,4) 174,792,640 addresses, 70 active tetra, 5 tetra/face')
    print('PASS oriented S-/even S+ edges and triangles; holonomy, spherical curvature, tetra parity')
    print('PASS nontrivial learned curvature closure and d^2(connection)=0')
    print('PASS gradients: mirror P/M, edge connection, triangle curvature, tetra feedback; inherited frozen')
    print('PASS geometry cannot bypass same R7 triple->pair->point->trunk decoder')

if __name__=='__main__':run()