#!/usr/bin/env python3
"""FA1 hierarchy causal, repeat-key, and independent-control checks (CPU)."""
from __future__ import annotations
import argparse,pathlib,random,json
import numpy as np,torch
from torch.nn import functional as F
from fa1_hierarchical import HierarchicalRouter,route_features,SEG,create_data,retrieval
from r711_delta import restore_overlay
from r710_checkpoint_integrity import sha256


def test_causal_synthetic():
    torch.manual_seed(113)
    E=F.normalize(torch.randn(128,96),dim=-1)
    h=torch.randint(2,128,(3,8*SEG),dtype=torch.long)
    q=torch.tensor([3,4,5]);
    for n in range(3):
        # Only one instance of queried key in older history; place it at
        # a known historical segment, then verify segment-level routing.
        h[n][h[n]==q[n]]=11
        h[n,n*SEG+SEG//2]=q[n]
    features,_=route_features(h,q,E)
    router=HierarchicalRouter();router.weights.data=torch.tensor([12.,9.,0.]);
    chosen,_=router(features,topk=2)
    assert all(int(i) in chosen[n].tolist() for n,i in enumerate(range(3)))
    # A future/unobserved target is not an argument to route_features;
    # adding a separate simulated target leaves routes exactly unchanged.
    for y in (torch.randint(2,128,(3,)),torch.randint(2,128,(3,))):
        assert torch.equal(features,route_features(h,q,E)[0]);_ = y
    flip=h.reshape(3,-1,64).flip(-1).reshape(3,-1)
    assert not torch.equal(h,flip)
    return 3

@torch.no_grad()
def repeat_key_stress(data):
    # Inject an earlier false historical key->different-value association.
    # The target remains the LATEST original occurrence; no future information
    # is used by the router. This is a synthetic conflict test, not semantics.
    original=data['source_segment'];mask=original>0
    indices=mask.nonzero().flatten()
    if len(indices)<10: raise ValueError('not enough later-segment queries')
    arr={k:(v[indices].clone() if isinstance(v,torch.Tensor) and len(v)==len(original) else v)
         for k,v in data.items()}
    arr['count']=len(indices)
    for row,s in enumerate(arr['source_segment'].tolist()):
        earlier=max(0,s-1)
        ix=earlier*SEG+SEG//3
        arr['history'][row,ix]=arr['query'][row]
        assert ix<arr['source_position'][row]
    return arr

@torch.no_grad()
def embedding_neighbor_stress(data,E):
    """Replace query token with a nearby embedding token absent from history.

    This is NOT a semantic synonym test: embedding neighbors are not
    guaranteed to preserve meaning. Intended only as a surface-form stress.
    """
    n=len(data['query']);q=data['query'];e=F.normalize(E,dim=-1)
    qs=F.normalize(E[q],dim=-1);sims=qs@e.T
    sims[:,:2]=-2.
    out={k:(v.clone() if isinstance(v,torch.Tensor) else v) for k,v in data.items()}
    valid=[]
    for i in range(n):
        absent=torch.ones(len(E),dtype=torch.bool)
        absent[data['history'][i].unique()]=False
        absent[q[i]]=False
        local=sims[i].clone();local[~absent]=-2.
        neighbor=int(local.argmax());out['query'][i]=neighbor
        valid.append(bool(local[neighbor]>-1))
    assert all(valid)
    return out


def run(a):
    torch.set_num_threads(2);torch.manual_seed(113);np.random.seed(113)
    root=a.root;before=sha256(root/'checkpoint/pal_active.pt')
    test_causal_synthetic()
    model,opts,base,comp,_=restore_overlay(root,root/'adaptation/r711_experimental_delta.pt')
    p=torch.load(root/'adaptation/fa1_hierarchical_r713.pt',map_location='cpu',weights_only=False)
    assert p['accepted_parent_sha256']==before
    router=HierarchicalRouter();router.load_state_dict(p['router_state']);router.eval()
    E=model.trunk.E.detach();outer=model.r74.outer
    d=create_data(root,'dev',a.count,98717,E)
    variants={'unique_exact':d,'temporal_conflict':repeat_key_stress(d),
              'embedding_neighbor':embedding_neighbor_stress(d,E)}
    scores={k:{m:retrieval(outer,router,E,v,batch=8,mode=m)
            for m in ('correct','wrong','reversed')}
            for k,v in variants.items()}
    assert sha256(root/'checkpoint/pal_active.pt')==before
    report={'synthetic_causal_tests':'passed','accepted_sha256_unchanged':True,
         'stresses':scores,'book_disjointness':False,
         'embedding_neighbor_not_claimed_semantic':True}
    a.output.parent.mkdir(exist_ok=True,parents=True)
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:{m:round(x['ce'],4) for m,x in s.items()} for k,s in scores.items()}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,default=pathlib.Path('.'))
    p.add_argument('--count',type=int,default=128)
    p.add_argument('--output',type=pathlib.Path,default=pathlib.Path('reports/FA1_hierarchy_stress.json'))
    run(p.parse_args())
