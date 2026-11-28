"""Continue only a verified v99 checkpoint; evaluate ordered semantic completion.

No inference answer injection. The same byte generator must recover both
requested values in the requested order and emit its own EOT.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import numpy as np
import torch
from torch.nn import functional as F
from field96 import EOT, batch, load_data
from field97 import nll
from field99 import Chat99
from field100 import Chat100
from train99 import paired, flatten, crossed, evaluate, digest
from invariance100 import make_groups, paired_views, equivalence_loss, selftest as invariant_selftest

def ordered_pairs(n,seed,heldout):
    rng=np.random.default_rng(seed)
    # Exclude the earlier v99 held-out values even from this new holdout.
    used={r[1][:-1].decode() for r in flatten(paired(64,9905,True))}
    def value():
        while True:
            s=''.join(rng.choice(list('abcdefghjkmnpqrstuvwxyz'),6))
            h=hashlib.sha256(s.encode()).hexdigest()
            if (int(h[:8],16)%10==0)==heldout and s not in used:
                used.add(s);return s
    result=[]
    for i in range(n):
        values={k:value() for k in 'ABC'}
        order=list(rng.permutation(list('ABC')))
        a,b=order[:2]
        if heldout and i%2:
            tmpl='User: Item C contains {C}; item A contains {A}; item B contains {B}. Give item {first} followed by item {second}, separated by one space.\nAssistant: '
        else:
            tmpl='User: A={A}; B={B}; C={C}. Return {first} then {second}, separated by a space.\nAssistant: '
        pair=[]
        for first,second in ((a,b),(b,a)):
            p=tmpl.format(**values,first=first,second=second).encode()
            y=(values[first]+' '+values[second]).encode()+bytes([EOT])
            key=hashlib.sha256(p).hexdigest()
            pair.append((p,y,key,'ordered_binding'))
        result.append(tuple(pair))
    return result

def values(pairs):
    return {s for row in flatten(pairs) for s in re.findall(rb'\b[a-z]{6}\b',row[0])
            if all(c in b'abcdefghjkmnpqrstuvwxyz' for c in s) and s not in (b'return',b'space')}

def selftest():
    train=ordered_pairs(32,10004,False)
    dev=ordered_pairs(16,10005,True)
    assert not values(train)&values(dev)
    for a,b in train+dev:
        assert a[0]!=b[0]
        x=a[1][:-1].split();y=b[1][:-1].split()
        assert x==y[::-1] and len(x)==2 and x[0]!=x[1]
        assert all(v in a[0] and v in b[0] for v in x)
        assert a[1].endswith(bytes([EOT])) and b[1].endswith(bytes([EOT]))
    print('v100 ordered-pair curriculum: same source, reversed target order, disjoint values and exact termination passed',flush=True)

def main():
    ap=argparse.ArgumentParser()
    for k in ('parent','parent-metrics','parent-integration','data','out'):
        ap.add_argument('--'+k,type=Path)
    ap.add_argument('--steps',type=int,default=512)
    ap.add_argument('--selftest',action='store_true')
    a=ap.parse_args()
    if a.selftest:
        selftest();invariant_selftest();return
    if any(getattr(a,k) is None for k in ('parent','parent_metrics','parent_integration','data','out')) or a.steps<=0:
        ap.error('all checkpoint, metric, integration, corpus and output paths plus positive steps required')
    torch.set_num_threads(4);torch.manual_seed(10001)
    sha=digest(a.parent)
    ck=torch.load(a.parent,map_location='cpu',weights_only=True)
    metrics=json.loads(a.parent_metrics.read_text())
    integration=json.loads(a.parent_integration.read_text())
    assert metrics['report_stage']==integration['report_stage']=='final'
    assert integration['comparison']['v99_sha256']==sha
    assert metrics['parent_sha256']==ck['parent_sha256']==integration['comparison']['v97_sha256']
    assert integration['parent_artifacts_unchanged']
    bpb=integration['matched_holdout']['bpb']
    if not bpb['v99'] < bpb['v97']:
        raise RuntimeError('v99 full-system holdout failed; do not extend unpromoted checkpoint')
    if metrics['transport_enabled']['both_pair_exact'] <= metrics['baseline']['both_pair_exact']:
        raise RuntimeError('v99 paired exact generation failed; do not extend unpromoted checkpoint')
    if metrics['transport_enabled']['human_bpb'] > metrics['baseline']['human_bpb']:
        raise RuntimeError('v99 human DEV regression; do not extend unpromoted checkpoint')
    train,dev,manifest=load_data(a.data)
    assert manifest['sha256']==ck['data_sha256']
    from pilot96 import PROMPTS
    fixed={hashlib.sha256(' '.join(s.lower().split()).encode()).hexdigest()
           for _,p in PROMPTS for s in re.findall(r'(?:^|\n)User: (.*?)\nAssistant:',p)}
    train=[r for r in train if r[2] not in fixed]
    dev=[r for r in dev if r[2] not in fixed]
    human=[dev[i] for i in np.random.default_rng(10003).permutation(len(dev))[:128]]
    old_train=paired(2048,9904,False)
    old_probe=paired(64,9905,True)
    new_train=ordered_pairs(2048,10004,False)
    new_probe=ordered_pairs(64,10005,True)
    assert not values(new_train)&values(new_probe)
    exclude={v.decode() for v in values(new_train+new_probe)} | {r[1][:-1].decode() for r in flatten(old_train+old_probe)}
    eq_train,et=make_groups(2048,10021,exclude=exclude)
    eq_probe,ep=make_groups(48,10022,True,exclude=exclude|et)
    assert not et&ep and not (et|ep)&exclude
    parent=Chat99().eval().requires_grad_(False)
    parent.load_state_dict(ck['chat99'])
    model=Chat100().eval().requires_grad_(False)
    missing=model.load_state_dict(ck['chat99'],strict=False)
    assert not missing.unexpected_keys and all('100' in k for k in missing.missing_keys)
    params=dict(model.named_parameters())
    inherited=[[n for n,_ in parent.named_parameters() if '97' not in n and '99' not in n],
               [n for n,_ in parent.named_parameters() if '97' in n],
               [n for n,_ in parent.named_parameters() if '99' in n]]
    opt=torch.optim.AdamW([params[n] for n in inherited[0]],lr=.0005)
    for names in inherited[1:]:
        opt.add_param_group({'params':[params[n] for n in names]})
    opt.load_state_dict(ck['optimizer99'])
    expected=[2304+ck['additional_steps'],1536+ck['additional_steps'],ck['additional_steps']]
    for names,count in zip(inherited,expected):
        assert all(int(opt.state[params[n]]['step'])==count for n in names)
    for group in opt.param_groups:group['lr']=.0003
    fresh=[params[n] for n in missing.missing_keys]
    opt.add_param_group({'params':fresh,'lr':.001,'weight_decay':.01})
    model.scale100=0.
    with torch.no_grad():
        x,y,p=batch(human[:4])
        assert torch.equal(model(x,p),parent(x,p))
        assert all(torch.equal(t,model.state_dict()[n]) for n,t in parent.state_dict().items())
    model.scale100=1.
    with torch.no_grad():assert torch.equal(model(x,p),parent(x,p))
    def measure(m,generation=False):
        eq=evaluate(m,human,paired_views(eq_probe),generation)
        if generation:
            categories={r[0].decode():r[3] for g in eq_probe for r in g}
            by_form={}
            for row in eq['responses']:
                name=categories[row['prompt']]
                stats=by_form.setdefault(name,{'exact':0,'total':0})
                stats['exact']+=int(row['exact']);stats['total']+=1
            eq['by_unseen_form']=by_form
        return {'ordered':evaluate(m,human,new_probe,generation),
                'retention':evaluate(m,human,old_probe,generation),
                'unseen_wording':eq}
    baseline=measure(parent,True)
    trace=[{'step':0,**measure(model)}]
    model.requires_grad_(True)
    rng=np.random.default_rng(10006)
    for i in range(a.steps):
        model.train()
        pp=[old_train[int(rng.integers(len(old_train)))],new_train[int(rng.integers(len(new_train)))]]
        gg=[eq_train[int(rng.integers(len(eq_train)))]]
        pp.extend(paired_views(gg))
        good=flatten(pp)
        rows=[train[j] for j in rng.choice(len(train),12,replace=False)]+good
        x,y,p=batch(rows)
        opt.zero_grad(set_to_none=True)
        ce=F.cross_entropy(model(x,p).transpose(1,2),y,ignore_index=-100)
        right,_=nll(model,good);wrong,_=nll(model,crossed(pp))
        js=equivalence_loss(model,gg)
        loss=ce+.3*F.softplus(.4+right-wrong).mean()+.2*right.mean()+.25*js
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        opt.step()
        if (i+1)%128==0 or i+1==a.steps:
            row={'step':i+1,**measure(model)}
            trace.append(row);print('CHAT100_TRACE '+json.dumps(row),flush=True)
    for names,count in zip(inherited,expected):
        assert all(int(opt.state[params[n]]['step'])==count+a.steps for n in names)
    assert all(int(opt.state[p]['step'])==a.steps for p in fresh)
    final=measure(model,True)
    model.scale100=0.
    ablation=measure(model,True)
    model.scale100=1.
    assert digest(a.parent)==sha
    a.out.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'chat100':model.state_dict(),'optimizer100':opt.state_dict(),
                'parent_sha256':sha,'additional_steps':a.steps,'selected_chat_gain':ck['selected_chat_gain'],
                'data_sha256':manifest['sha256']},a.out.with_suffix('.pt'))
    result={'report_stage':'final','parent_sha256':sha,'baseline':baseline,
            'enabled':final,'disabled_after_training':ablation,'trace':trace,'steps':a.steps,
            'promotion':False,'equivalence_train_groups':len(eq_train),'unseen_wording_groups':len(eq_probe),
            'equivalence_weight':.25,'scope':'chat field; requires matched full-system generation before promotion'}
    a.out.write_text(json.dumps(result,indent=2,allow_nan=False))
    print('CHAT100_FINAL '+json.dumps({k:{m:{i:j for i,j in v.items() if i!='responses'} for m,v in result[k].items()} for k in ('baseline','enabled','disabled_after_training')}),flush=True)

if __name__=='__main__':
    main()
