#!/usr/bin/env python3
"""Exact combinatorial topology test; no truncated pseudo-C(256,3) indexing."""
import itertools, numpy as np, torch
from fa1_r723_diamond import VERTICES, PAIRS, TRIPLES, pair_rank, triple_rank, CombinatorialDiamond

def run():
    assert VERTICES==256 and PAIRS==32640 and TRIPLES==2763520
    aa=[];bb=[];cc=[];pairs=[]
    for i in range(256):
        for j in range(i+1,256):
            aa.append(i);bb.append(j)
    a=torch.tensor(aa);b=torch.tensor(bb)
    v=pair_rank(a,b)
    assert torch.equal(v,torch.arange(PAIRS)), 'nonbijective inherited pair coordinate'
    # Every triple with a fixed maximum c occupies a distinct contiguous
    # C(c,2) block. Check the full space without 2.76M Python loops.
    encountered=0
    for c in range(2,256):
        a,b=torch.triu_indices(c,c,offset=1)
        cc=torch.full_like(a,c)
        ids=triple_rank(a,b,cc).sort().values
        expect=torch.arange(encountered,encountered+len(ids))
        assert torch.equal(ids,expect),('bad rank',c)
        encountered+=len(ids)
    assert encountered==TRIPLES
    m=CombinatorialDiamond()
    assert m.triple_code.num_embeddings==TRIPLES
    assert m.sector_keys.shape[0]==256
    assert len(m.triangles)==56 and len(m.edges)==28
    assert len(m.tri_edges)==56
    assert m.tri_edges.unique().numel()==28
    assert torch.bincount(m.tri_edges.flatten(),minlength=28).min()==6
    assert torch.bincount(m.tri_edges.flatten(),minlength=28).max()==6
    # Verify gradients flow through the apex into the C(256,3) code and
    # down through pair/point reconstruction. No target enters the route.
    from types import SimpleNamespace
    torch.manual_seed(123)
    trunk=SimpleNamespace(E=torch.nn.Parameter(torch.randn(32000,96)),
                          H=torch.nn.Parameter(torch.randn(256,96)),
                          P=torch.nn.Parameter(torch.randn(32640,64)),
                          M=torch.nn.Parameter(torch.randn(32640,64)))
    batch=3
    h=torch.randn(batch,96)
    words=torch.randint(0,32000,(batch,32))
    by=torch.randint(0,256,(batch,128))
    sen=torch.randint(0,32000,(batch,8))
    verts=torch.stack([torch.randperm(256)[:8] for _ in range(batch)])
    mem=torch.randn(batch,92)
    state,logits,vec=m(h,words,by,sen,verts,trunk,memory_features=mem)
    with torch.no_grad():
        other,_,_=m(h,words,by,sen,verts,trunk,memory_features=torch.zeros_like(mem))
        assert not torch.equal(state,other),'legacy causal reader feature path is disconnected' 
    (torch.nn.functional.cross_entropy(logits,torch.randint(0,256,(batch,)))+vec.square().mean()).backward()
    for name in ('triple_code.weight','triple_in.weight','pair_down.0.weight',
                 'point_down.0.weight','sector_keys','sector_values','legacy_memory_in.weight'):
        param=dict(m.named_parameters())[name]
        assert param.grad is not None and torch.isfinite(param.grad).all(),('gradient absent',name)
        assert float(param.grad.abs().sum())>0,('gradient zero',name)
    assert all(x.grad is None for x in (trunk.H,trunk.P,trunk.M,trunk.E))
    print('PASS gradient reachability: indexed triples, specialist meta-mind, pair and point reconstruction')
    print('PASS exact 256 points / 32,640 inherited pairs / 2,763,520 triples / 256 apex sectors; 56->28 incidence consistent')

if __name__=='__main__':run()
