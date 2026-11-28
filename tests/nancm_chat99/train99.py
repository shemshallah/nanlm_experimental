"""v99 continuation, matched counterfactual supervision and free generation."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from field96 import load_data, batch, EOT
from field97 import Chat97, nll
from field99 import Chat99

def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def paired(n, seed, heldout):
    """Paired prompts differ only in which value is requested. Both values
    occur in both prompts, so bag-of-bytes copying cannot solve the task."""
    rng = np.random.default_rng(seed)
    used = set()
    pairs = []
    def value():
        while True:
            s = ''.join(rng.choice(list('abcdefghjkmnpqrstuvwxyz'), 6))
            key = hashlib.sha256(s.encode()).hexdigest()
            if (int(key[:8],16)%10 == 0) == heldout and s not in used:
                used.add(s)
                return s, key
    for i in range(n):
        a, ka = value()
        b, kb = value()
        if heldout and i % 2:
            template = 'User: Item B has {b}; item A has {a}. Which is item {key}? Reply with its value only.\nAssistant: '
        else:
            template = 'User: A={a}; B={b}. Return only the value of {key}.\nAssistant: '
        pairs.append(tuple((template.format(a=a,b=b,key=k).encode(),
                            v.encode()+bytes([EOT]), h, 'paired_address')
                           for k,v,h in [('A',a,ka),('B',b,kb)]))
    return pairs

def crossed(pairs):
    return [(b[0],a[1],a[2],a[3]) for a,b in pairs]+[(a[0],b[1],b[2],b[3]) for a,b in pairs]

def flatten(pairs):
    return [p[0] for p in pairs]+[p[1] for p in pairs]

@torch.no_grad()
def generate(model, prompt, limit=96):
    state, anchor, q = model.prefill(prompt)
    out = bytearray()
    for _ in range(limit):
        b = int(np.argmax(q))
        out.append(b)
        if b == EOT:
            break
        state, q = model.advance(state, anchor, b)
    return bytes(out)

@torch.no_grad()
def evaluate(model, human, pairs, generation=False):
    model.eval()
    def bpb(rows):
        bits = 0.
        count = 0
        for i in range(0,len(rows),16):
            x,y,p = batch(rows[i:i+16])
            bits += float(F.cross_entropy(model(x,p).transpose(1,2),y,ignore_index=-100,reduction='sum'))/math.log(2)
            count += int((y!=-100).sum())
        return bits/count
    good = flatten(pairs)
    wrong = crossed(pairs)
    report = {'human_bpb':bpb(human),'paired_bpb':bpb(good),
              'swapped_prompt_bpb':bpb(wrong)}
    report['prompt_advantage_bpb'] = report['swapped_prompt_bpb']-report['paired_bpb']
    if generation:
        records = []
        for row in good:
            raw = generate(model,row[0],32)
            records.append({'prompt':row[0].decode(),'expected_hex':row[1].hex(),
                            'response_hex':raw.hex(),'response':raw.removesuffix(bytes([EOT])).decode(errors='replace'),
                            'exact':raw==row[1],'eot':raw.endswith(bytes([EOT]))})
        report.update(exact=sum(r['exact'] for r in records),total=len(records),responses=records,
                      both_pair_exact=sum(records[i]['exact'] and records[i+len(pairs)]['exact'] for i in range(len(pairs))))
    return report

def main():
    ap=argparse.ArgumentParser()
    for key in ('parent','parent-metrics','data','out'):
        ap.add_argument('--'+key,type=Path,required=True)
    ap.add_argument('--steps',type=int,default=512)
    a=ap.parse_args()
    if a.steps<=0: raise ValueError('steps must be positive')
    torch.set_num_threads(4)
    torch.manual_seed(9901)
    parent_sha=digest(a.parent)
    ck=torch.load(a.parent,map_location='cpu',weights_only=True)
    report=json.loads(a.parent_metrics.read_text())
    assert report['report_stage']=='final' and ck['selected_chat_gain'] is not None
    assert ck['additional_steps']==1536
    assert report['v96_sha256']==ck['v96_sha256']
    train,dev,manifest=load_data(a.data)
    assert manifest['sha256']==ck['chat_data']['sha256']
    # Reproduce the v97 fixed-prompt exclusion rather than broaden its corpus.
    from pilot96 import PROMPTS
    import re
    fixed={hashlib.sha256(' '.join(s.lower().split()).encode()).hexdigest()
           for _,p in PROMPTS for s in re.findall(r'(?:^|\n)User: (.*?)\nAssistant:',p)}
    train=[r for r in train if r[2] not in fixed]
    dev=[r for r in dev if r[2] not in fixed]
    assert len(train)==ck['chat_data']['train_records_after_exclusion']
    assert len(dev)==ck['chat_data']['dev_records_after_exclusion']
    # Fixed fresh cohort, never used for optimizer updates.
    order=np.random.default_rng(9903).permutation(len(dev))
    human=[dev[i] for i in order[:128]]
    pairs=paired(2048,9904,False)
    probe=paired(64,9905,True)
    assert not {r[2] for r in flatten(pairs)} & {r[2] for r in flatten(probe)}
    parent=Chat97()
    parent.load_state_dict(ck['chat97'])
    parent.eval().requires_grad_(False)
    model=Chat99()
    missing=model.load_state_dict(ck['chat97'],strict=False)
    assert set(missing.missing_keys)=={'transport99.weight','transport99.bias'} and not missing.unexpected_keys
    params=dict(model.named_parameters())
    oldnames=[n for n,_ in parent.named_parameters() if '97' not in n]
    newnames=[n for n,_ in parent.named_parameters() if '97' in n]
    opt=torch.optim.AdamW([params[n] for n in oldnames],lr=.001)
    opt.add_param_group({'params':[params[n] for n in newnames],'lr':.002})
    opt.load_state_dict(ck['optimizer97'])
    for names,steps in ((oldnames,2304),(newnames,1536)):
        assert all(int(opt.state[params[n]]['step'])==steps for n in names)
    # Preserve inherited Adam moments; use a lower continuation LR.
    for group in opt.param_groups: group['lr']=.0005
    fresh=[params[n] for n in missing.missing_keys]
    opt.add_param_group({'params':fresh,'lr':.002,'weight_decay':.01})
    model.scale99=0.
    with torch.no_grad():
        x,y,p=batch(human[:4]);assert torch.equal(model(x,p),parent(x,p))
    model.scale99=1.
    baseline=evaluate(parent,human,probe,True)
    trace=[{'step':0,**evaluate(model,human,probe)}]
    rng=np.random.default_rng(9906)
    seen=set()
    for i in range(a.steps):
        model.train()
        indices=rng.choice(len(train),12,replace=False)
        seen.update(map(int,indices))
        pp=[pairs[j] for j in rng.choice(len(pairs),2,replace=False)]
        good=flatten(pp)
        x,y,p=batch([train[j] for j in indices]+good)
        opt.zero_grad(set_to_none=True)
        logits=model(x,p)
        ce=F.cross_entropy(logits.transpose(1,2),y,ignore_index=-100)
        right,_=nll(model,good)
        wrong,_=nll(model,crossed(pp))
        loss=ce+.3*F.softplus(.4+right-wrong).mean()+.2*right.mean()
        if not torch.isfinite(loss): raise FloatingPointError('nonfinite loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        opt.step()
        if (i+1)%128==0 or i+1==a.steps:
            row={'step':i+1,**evaluate(model,human,probe)}
            trace.append(row)
            print('CHAT99_TRACE '+json.dumps(row),flush=True)
    for names,steps in ((oldnames,2304),(newnames,1536)):
        assert all(int(opt.state[params[n]]['step'])==steps+a.steps for n in names)
    assert all(int(opt.state[p]['step'])==a.steps for p in fresh)
    final=evaluate(model,human,probe,True)
    model.scale99=0.
    ablation=evaluate(model,human,probe,True)
    model.scale99=1.
    assert digest(a.parent)==parent_sha
    a.out.parent.mkdir(parents=True,exist_ok=True)
    result={'report_stage':'final','parent_sha256':parent_sha,'data_sha256':manifest['sha256'],
            'steps':a.steps,'human_train_records_seen':len(seen),'trace':trace,'baseline':baseline,
            'transport_enabled':final,'transport_disabled_after_training':ablation,
            'promotion':False,'scope':'chat field diagnostic; full-system integration evaluated separately'}
    payload={'chat99':model.state_dict(),'optimizer99':opt.state_dict(),'parent_sha256':parent_sha,
             'additional_steps':a.steps,'selected_chat_gain':ck['selected_chat_gain'],
             'data_sha256':manifest['sha256']}
    torch.save(payload,a.out.with_suffix('.pt'))
    a.out.write_text(json.dumps(result,indent=2,allow_nan=False))
    print('CHAT99_FINAL '+json.dumps({k:{i:j for i,j in result[k].items() if i!='responses'} for k in ('baseline','transport_enabled','transport_disabled_after_training')}),flush=True)

if __name__=='__main__':
    main()
