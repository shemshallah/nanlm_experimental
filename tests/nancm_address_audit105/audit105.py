"""Audit v103's closed source gate without modifying its stored weights.

Choose a gain using only a fresh train-side calibration cohort. Report all
fixed-gain likelihoods there, then run autonomous generation at the chosen
gain on the original and fresh held-out wording families. Audit addresses
directly, independently of the decoder's gate.
"""
import argparse, hashlib, json, math, re
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from field96 import batch, EOT, load_data
from field103 import Chat103
from train99 import paired,flatten,evaluate,digest
from train103 import ordered_pairs,values
from invariance100 import make_groups,paired_views
import invariance100 as inv

@torch.no_grad()
def audit(model,rows):
    counts={}
    for start in range(0,len(rows),16):
        part=rows[start:start+16]
        x,y,p=batch(part)
        h,_=model.gru(model.embed(x))
        anchor=h[torch.arange(len(x)),p-1]
        mem=model.memory(h,x,p)
        q=model.address103(h,anchor,mem).softmax(-1)
        for i,(prompt,answer,*_) in enumerate(part):
            j=len(prompt)-1;offset=0
            for widx,w in enumerate(answer[:-1].split(b' ')):
                where=prompt.find(w)
                if where>=0 and prompt.find(w,where+1)<0:
                    for k in range(len(w)):
                        name=('first_value_start' if widx==0 else 'second_value_start') if k==0 else 'inside_value'
                        target=where+k
                        z=q[i,j+offset+k]
                        row=counts.setdefault(name,{'n':0,'top1':0,'top5':0,'bits':0.})
                        row['n']+=1;row['top1']+=int(z.argmax()==target)
                        row['top5']+=int((z.topk(5).indices==target).any())
                        row['bits']-=float(torch.log2(z[target].clamp_min(1e-30)))
                offset+=len(w)+1
    for row in counts.values():row['mean_bits']=row.pop('bits')/row['n']
    return counts

@torch.no_grad()
def human_bpb(model,rows):
    bits=n=0
    for start in range(0,len(rows),16):
        x,y,p=batch(rows[start:start+16])
        bits+=float(F.cross_entropy(model(x,p).transpose(1,2),y,ignore_index=-100,reduction='sum'))/math.log(2)
        n+=int((y!=-100).sum())
    return bits/n

def main():
    ap=argparse.ArgumentParser()
    for name in ('checkpoint','metrics','data','out'):ap.add_argument('--'+name,type=Path,required=True)
    a=ap.parse_args();torch.set_num_threads(4);torch.manual_seed(105)
    sha=digest(a.checkpoint)
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    meta=json.loads(a.metrics.read_text())
    assert meta['report_stage']=='final' and ck['parent_sha256']==meta['parent_sha256']
    model=Chat103().eval().requires_grad_(False);model.load_state_dict(ck['chat103'])
    original=float(model.gain103)
    print('ADDRESS105_RAW_GAIN '+str(original),flush=True)
    train,dev,manifest=load_data(a.data)
    assert manifest['sha256']==ck['data_sha256']
    from pilot96 import PROMPTS
    fixed={hashlib.sha256(' '.join(s.lower().split()).encode()).hexdigest() for _,p in PROMPTS for s in re.findall(r'(?:^|\n)User: (.*?)\nAssistant:',p)}
    train=[r for r in train if r[2] not in fixed];dev=[r for r in dev if r[2] not in fixed]
    hcal=[train[i] for i in np.random.default_rng(10501).permutation(len(train))[:128]]
    hdev=[dev[i] for i in np.random.default_rng(10003).permutation(len(dev))[:128]]
    assert not {r[2] for r in hcal}&{r[2] for r in hdev}
    old_train=paired(2048,9904,False);old_probe=paired(64,9905,True)
    new_train=ordered_pairs(2048,10004,False);new_probe=ordered_pairs(64,10005,True)
    exclude={v.decode() for v in values(new_train+new_probe)}|{r[1][:-1].decode() for r in flatten(old_train+old_probe)}
    eq_train,et=make_groups(2048,10021,exclude=exclude)
    eq_probe,ep=make_groups(48,10022,True,exclude=exclude|et)
    of,ofacts=inv.TRAIN_FORMS,inv.TRAIN_FACTS
    inv.TRAIN_FORMS=of+(
        "Retrieve the value indexed by {a}, followed by the value indexed by {b}. Records: {facts}.",
        "Records: {facts}. Put {a}'s value ahead of {b}'s value.",
        "Extract two entries from {facts}, in this sequence: {a}, {b}.",
        "Output the contents of {a}, then those of {b}, from the mapping {facts}.")
    inv.TRAIN_FACTS=ofacts+("{v} is assigned to {k}","{v} (key {k})","entry {k} stores {v}")
    _,dt=make_groups(2048,10321,exclude=exclude|et|ep)
    calibration,ct=make_groups(48,10521,exclude=exclude|et|ep|dt)
    inv.TRAIN_FORMS=of;inv.TRAIN_FACTS=ofacts
    fresh,ft=make_groups(48,10522,True,exclude=exclude|et|ep|dt|ct)
    assert not ct&ft
    cal_pairs=paired_views(calibration)
    suites={'binding':old_probe,'ordered':new_probe,'unseen_wording':paired_views(eq_probe),'fresh_unseen_wording':paired_views(fresh)}
    addresses={'calibration':audit(model,flatten(cal_pairs))}
    for name,rows in suites.items():addresses[name]=audit(model,flatten(rows))
    print('ADDRESS105_AUDIT '+json.dumps(addresses),flush=True)
    # Objective and guard fixed in advance; no heldout feedback in gain selection.
    sweep=[]
    for gain in (0.,.025,.05,.1,.2,.4,.6,.8):
        with torch.no_grad():model.gain103.fill_(gain)
        rec=evaluate(model,hcal,cal_pairs,False)
        rec['gain']=gain;rec['objective']=rec['human_bpb']+rec['paired_bpb']
        sweep.append(rec)
    baseline=sweep[0]
    allowed=[r for r in sweep if r['human_bpb']<=baseline['human_bpb']+.02 and r['prompt_advantage_bpb']>=baseline['prompt_advantage_bpb']-.02]
    selected=min(allowed,key=lambda r:(r['objective'],r['gain']))['gain']
    print('ADDRESS105_SWEEP '+json.dumps({'sweep':sweep,'selected':selected}),flush=True)
    results={}
    for label,g in [('closed',0.),('selected',selected),('forced_20pct',.2)]:
        with torch.no_grad():model.gain103.fill_(g)
        results[label]={'gain':g,'human_bpb':human_bpb(model,hdev),
                        'suites':{name:evaluate(model,hdev,rows,True) for name,rows in suites.items()}}
        print('ADDRESS105_RESULT '+json.dumps({'label':label,'gain':g,'human_bpb':results[label]['human_bpb'],
              'suites':{k:{n:v for n,v in r.items() if n!='responses'} for k,r in results[label]['suites'].items()}}),flush=True)
    with torch.no_grad():model.gain103.fill_(original)
    assert all(torch.equal(v,model.state_dict()[n]) for n,v in ck['chat103'].items())
    assert digest(a.checkpoint)==sha
    report={'report_stage':'final','parent_sha256':sha,'raw_gain':original,'addresses':addresses,
            'calibration_sweep':sweep,'selected_gain':selected,'results':results,'parent_unchanged':True,'promotion':False}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2,allow_nan=False))
if __name__=='__main__':main()
