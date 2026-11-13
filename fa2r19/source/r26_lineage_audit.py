#!/usr/bin/env python3
"""Bitwise preservation of every inherited FA2 R25 tensor and AdamW moment.
R25 binary is optional: needed only for this lineage-diff audit, not recovery.
"""
import argparse,json,pathlib,torch

def run(root):
    root=pathlib.Path(root)
    a=torch.load(root/'reports/fa2_r25_unified_candidate.pth',map_location='cpu',weights_only=False)
    b=torch.load(root/'reports/fa2_r26_unified_candidate.pth',map_location='cpu',weights_only=False)
    assert a['format']=='FA2_R25_UNIFIED_CANDIDATE_V1'
    assert b['format']=='FA2_R26_UNIFIED_CANDIDATE_V1'
    assert b['accepted_model_promoted'] is False
    assert b['selected']['r25_step']==a['selected']['r25_step']==120
    n=0
    for k,v in a['selected']['model'].items():
        assert k in b['selected']['model'] and torch.equal(v,b['selected']['model'][k]),('model',k)
        n+=1
    original=a['selected']['optimizer'];extended=b['selected']['optimizer']
    assert len(original['param_groups'])==10 and len(extended['param_groups'])==11
    assert len(original['state'])==130 and len(extended['state'])==137
    for x,y in zip(original['param_groups'],extended['param_groups'][:10]):
        assert x==y,'optimizer group metadata changed'
    cnt=0
    for k,vals in original['state'].items():
        assert k in extended['state']
        got=extended['state'][k]
        assert vals.keys()==got.keys(),('state keys',k)
        for name,v in vals.items():
            assert torch.equal(v,got[name]) if torch.is_tensor(v) else v==got[name],(k,name)
            cnt+=1
    assert len(a['selected']['carry'])==len(b['selected']['carry'])==12
    # Exact recursive comparison of the inherited long-document cursors and
    # recurrent hidden/history states.
    def same(x,y):
        if torch.is_tensor(x):return torch.equal(x,y)
        if isinstance(x,dict):return x.keys()==y.keys() and all(same(x[k],y[k]) for k in x)
        if isinstance(x,(list,tuple)):return len(x)==len(y) and all(same(a,b) for a,b in zip(x,y))
        return x==y
    assert same(a['selected']['carry'],b['selected']['carry'])
    result={'status':'PASS','inherited_model_tensors_exact':n,
       'inherited_optimizer_states_exact':len(original['state']),
       'inherited_optimizer_fields_exact':cnt,'inherited_streams_exact':12,
       'optimizer_groups':11,'optimizer_states':137,
       'r25_step':120,'selected_r26_step':b['selected']['r26_step']}
    (root/'reports/r26_lineage_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='.')
    run(p.parse_args().root)
