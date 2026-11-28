"""Experimental on-policy continuation of v103; no new model weights or decoder rules.

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
from field103 import Chat103
from recovery104 import rollout,repeat_loss,selftest as recovery_selftest
import invariance100 as inv
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
    for k in ('parent','parent-metrics','data','out'):
        ap.add_argument('--'+k,type=Path)
    ap.add_argument('--steps',type=int,default=512)
    ap.add_argument('--selftest',action='store_true')
    a=ap.parse_args()
    if a.selftest:
        selftest();invariant_selftest();recovery_selftest();return
    if any(getattr(a,k) is None for k in ('parent','parent_metrics','data','out')) or a.steps<=0:
        ap.error('all checkpoint, metric, integration, corpus and output paths plus positive steps required')
    torch.set_num_threads(4);torch.manual_seed(10401)
    sha=digest(a.parent)
    ck=torch.load(a.parent,map_location='cpu',weights_only=True)
    metrics=json.loads(a.parent_metrics.read_text())
    assert metrics['report_stage']=='final' and metrics['parent_sha256']==ck['parent_sha256']
    assert metrics['enabled']['ordered']['exact']>=metrics['baseline']['ordered']['exact']
    assert metrics['enabled']['retention']['exact']>=metrics['baseline']['retention']['exact']
    assert metrics['enabled']['ordered']['human_bpb']<=metrics['baseline']['ordered']['human_bpb']+.02
    # Research continuation only; full-system promotion remains separate.
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
    # Reproduce the v100 frozen evaluation first, then add novel syntax.
    original_forms=inv.TRAIN_FORMS
    original_facts=inv.TRAIN_FACTS
    inv.TRAIN_FORMS=original_forms+(
        "Retrieve the value indexed by {a}, followed by the value indexed by {b}. Records: {facts}.",
        "Records: {facts}. Put {a}'s value ahead of {b}'s value.",
        "Extract two entries from {facts}, in this sequence: {a}, {b}.",
        "Output the contents of {a}, then those of {b}, from the mapping {facts}.")
    inv.TRAIN_FACTS=original_facts+("{v} is assigned to {k}","{v} (key {k})","entry {k} stores {v}")
    diverse_train,dt=make_groups(2048,10321,exclude=exclude|et|ep)
    inv.TRAIN_FORMS=original_forms;inv.TRAIN_FACTS=original_facts
    fresh_probe,fp=make_groups(48,10322,True,exclude=exclude|et|ep|dt)
    assert not fp&(exclude|et|ep|dt)
    parent=Chat103().eval().requires_grad_(False)
    parent.load_state_dict(ck['chat103'])
    model=Chat103().eval().requires_grad_(False)
    model.load_state_dict(ck['chat103'])
    params=dict(model.named_parameters())
    inherited=[[n for n in params if not any(k in n for k in ('97','99','100','103'))]]
    inherited.extend([[n for n in params if k in n] for k in ('97','99','100','103')])
    opt=torch.optim.AdamW([params[n] for n in inherited[0]],lr=.0001)
    for names in inherited[1:]:
        opt.add_param_group({'params':[params[n] for n in names]})
    opt.load_state_dict(ck['optimizer103'])
    expected=[3328+ck['additional_steps'],2560+ck['additional_steps'],1024+ck['additional_steps'],512+ck['additional_steps'],ck['additional_steps']]
    for names,count in zip(inherited,expected):
        assert all(int(opt.state[params[n]]['step'])==count for n in names)
    for group in opt.param_groups:group['lr']=.0001
    with torch.no_grad():
        x,y,p=batch(human[:4]);assert torch.equal(model(x,p),parent(x,p))
        assert all(torch.equal(t,model.state_dict()[n]) for n,t in parent.state_dict().items())
    # Heldout instruction IDs distinct from all previous 128-record probes.
    forbidden_ids={r[2] for r in human}
    for seed in (9903,9908):
        forbidden_ids.update(dev[i][2] for i in np.random.default_rng(seed).permutation(len(dev))[:128 if seed==9903 else 12])
    forbidden_ids.update(dev[i][2] for i in np.random.default_rng(9703).choice(len(dev),128,replace=False))
    genprobe=[dev[i] for i in np.random.default_rng(10403).permutation(len(dev)) if dev[i][2] not in forbidden_ids][:24]
    assert len(genprobe)==24
    def generation_probe(m):
        m.eval();res=[]
        for row in genprobe:
            for temperature in (0.,.8):
                rng=np.random.default_rng(10409)
                # Retain stop status explicitly instead of inferring from text length.
                from train99 import generate
                if temperature==0.:
                    raw=generate(m,row[0],96)
                    ended=raw.endswith(bytes([EOT]))
                    raw=raw.removesuffix(bytes([EOT]))
                else:
                    state,anchor,q=m.prefill(row[0]);seq=bytearray();ended=False
                    for j in range(96):
                        probs=np.maximum(q,1e-300)**(1/.8)
                        byte=int(rng.choice(256,p=probs/probs.sum()))
                        if byte==EOT:ended=True;break
                        seq.append(byte);state,q=m.advance(state,anchor,byte)
                    raw=bytes(seq)
                words=re.findall(rb'[a-zA-Z]+',raw.lower())
                tri=list(zip(words,words[1:],words[2:]))
                res.append({'prompt':row[0].decode(),'reference':row[1].removesuffix(bytes([EOT])).decode(errors='replace'),
                            'response':raw.decode(errors='replace'),'response_hex':raw.hex(),'temperature':temperature,
                            'eot':ended,'bytes':len(raw),'repeated_trigram_fraction':1-len(set(tri))/len(tri) if tri else 0.})
        greedy=[r for r in res if r['temperature']==0.]
        return {'responses':res,'eot':sum(r['eot'] for r in res),'total':len(res),
                'greedy_repetition':float(np.mean([r['repeated_trigram_fraction'] for r in greedy]))}
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
                'unseen_wording':eq,'fresh_unseen_wording':evaluate(m,human,paired_views(fresh_probe),generation)}
    baseline=measure(parent,True)
    baseline_generation=generation_probe(parent)
    rollout_positions=0
    trace=[{'step':0,**measure(model)}]
    model.requires_grad_(True)
    rng=np.random.default_rng(10406)
    for i in range(a.steps):
        model.train()
        pp=[old_train[int(rng.integers(len(old_train)))],new_train[int(rng.integers(len(new_train)))]]
        pool=diverse_train if i%2==0 else eq_train
        gg=[pool[int(rng.integers(len(pool)))]]
        pp.extend(paired_views(gg))
        good=flatten(pp)
        natural=[train[j] for j in rng.choice(len(train),12,replace=False)]
        rows=natural+good
        examples=[]
        if i%4==0:
            model.eval()
            for j,r in enumerate(natural[:2]):
                examples.append((r[0],r[1],rollout(model,r[0],rng,64,0. if j==0 else .8)))
            model.train()
        x,y,p=batch(rows)
        opt.zero_grad(set_to_none=True)
        ce=F.cross_entropy(model(x,p).transpose(1,2),y,ignore_index=-100)
        right,_=nll(model,good);wrong,_=nll(model,crossed(pp))
        js=equivalence_loss(model,gg)
        alignment=model.alignment103(good)
        ul,n=repeat_loss(model,examples)
        rollout_positions+=n
        loss=ce+.3*F.softplus(.4+right-wrong).mean()+.2*right.mean()+.25*js+.5*alignment+.15*ul
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        opt.step()
        if (i+1)%128==0 or i+1==a.steps:
            row={'step':i+1,**measure(model)}
            trace.append(row);print('CHAT104_TRACE '+json.dumps(row),flush=True)
    for names,count in zip(inherited,expected):
        assert all(int(opt.state[params[n]]['step'])==count+a.steps for n in names)
    final=measure(model,True)
    final_generation=generation_probe(model)
    assert digest(a.parent)==sha
    a.out.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'chat103':model.state_dict(),'optimizer103':opt.state_dict(),
                'parent_sha256':sha,'additional_steps':a.steps,'selected_chat_gain':ck['selected_chat_gain'],
                'data_sha256':manifest['sha256']},a.out.with_suffix('.pt'))
    result={'report_stage':'final','parent_sha256':sha,'baseline':baseline,
            'enabled':final,'trace':trace,'steps':a.steps,
            'baseline_generation':baseline_generation,'final_generation':final_generation,
            'rollout_penalty_positions':rollout_positions,'unlikelihood_weight':.15,
            'promotion':False,'equivalence_train_groups':len(eq_train),'unseen_wording_groups':len(eq_probe),
            'equivalence_weight':.25,'alignment_weight':.5,'learned_gain':float(model.gain103.detach().clamp(0,.8)),'diverse_groups':len(diverse_train),'scope':'chat field; requires matched full-system generation before promotion'}
    a.out.write_text(json.dumps(result,indent=2,allow_nan=False))
    print('RECOVERY104 '+json.dumps({k:{i:j for i,j in v.items() if i!='responses'} for k,v in [('baseline',baseline_generation),('final',final_generation)]}|{'rollout_penalty_positions':rollout_positions}),flush=True)
    print('CHAT104_FINAL '+json.dumps({k:{m:{i:j for i,j in v.items() if i!='responses'} for m,v in result[k].items()} for k in ('baseline','enabled')}),flush=True)

if __name__=='__main__':
    main()
