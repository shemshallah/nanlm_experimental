#!/usr/bin/env python3
"""R7.15 CPU structural tests: gradients, causality, normalization, rollback."""
import json,pathlib,torch
from torch import nn
from torch.nn import functional as F
from fa1_r715_mirror import QueryMirrorLM,wrong_history

def run(root):
    torch.set_num_threads(2);torch.manual_seed(71599)
    V=384;D=96;R=64;N=12;K=48;H=6144
    E=torch.randn(V,D)
    oldkey=nn.Linear(D,R,bias=False);oldval=nn.Linear(D,R,bias=False)
    model=QueryMirrorLM(E,oldkey,oldval)
    hist=torch.randint(2,V,(N,H));q=torch.randint(2,V,(N,));prev=torch.randint(2,V,(N,))
    ids=torch.randint(3,H-3,(N,))
    targets=torch.randint(2,V,(N,))
    for i in range(N):
        hist[i,ids[i]-1]=prev[i];hist[i,ids[i]]=q[i];hist[i,ids[i]+1]=targets[i]
    # Construct valid source pairs with a deterministic query-matching prefix.
    ix=ids[:,None].repeat(1,K)
    ix[:,1:]=torch.randint(2,H-2,(N,K-1))
    src={'history':hist,'query':q,'prev_query':prev,'target':targets,
       'base':torch.full((N,),float(torch.log(torch.tensor(V,dtype=torch.float32)))),
       'key':hist.gather(1,ix),'val':hist.gather(1,ix+1),
       'prev_key':hist.gather(1,ix-1), 'age':ix.float()/H-1,
       'valid':torch.ones(N,K,dtype=torch.bool),'n':N}
    src['exact']=(src['key']==q[:,None]).float()
    src['exact2']=((src['key']==q[:,None])&(src['prev_key']==prev[:,None])).float()
    loss=model(src)
    assert torch.isfinite(loss).all()
    loss.mean().backward()
    grad={n:float(v.grad.norm()) if v.grad is not None else 0.0
         for n,v in model.named_parameters()}
    for n in ('key.weight','value.weight','context.weight','scores','gate.2.weight','gate.2.bias'):
        assert grad[n]>0, (n,grad[n])
    wrong=wrong_history(src,N//3,E,k=K,batch=4)
    assert torch.equal(src['base'],wrong['base'])
    # Targets are absent from the selector; changing targets cannot change any
    # wrong-memory key, value, age, exact or exact-two contextual features.
    changed=dict(src);changed['target']=torch.randint(2,V,(N,))
    wr2=wrong_history(changed,N//3,E,k=K,batch=4)
    for n in ('key','val','age','valid','exact','exact2'):
        assert torch.equal(wrong[n],wr2[n]),n
    with torch.no_grad():
        model.gate[2].weight.zero_();model.gate[2].bias.fill_(-80.)
        assert torch.equal(model(src),src['base'])
    tests={'finite_likelihood':True,'all_writer_reader_gate_gradients':True,
        'wrong_memory_baseline_unchanged':True,'retrieval_selection_independent_of_target':True,
        'zero_trust_exactly_recovers_inherited_likelihood':True,
        'grad_norm':grad}
    root=pathlib.Path(root);root.mkdir(exist_ok=True,parents=True)
    (root/'FA1_R7p15_structural_tests.json').write_text(json.dumps(tests,indent=2)+'\n')
    print(json.dumps({'status':'PASS','structural_tests':5,'writer_grad_norm':grad['value.weight']}))
if __name__=='__main__':run(pathlib.Path(__file__).parents[1]/'reports')