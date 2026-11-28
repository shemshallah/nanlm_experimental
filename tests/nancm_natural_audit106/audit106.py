"""Frozen natural-answer audit. No training, no decoding heuristics.
Replays archived v104 prompts and samples a separate short-reference cohort.
Token overlap and normalized exactness are lexical proxies, not semantic grading.
"""
import argparse,collections,hashlib,itertools,json,re
from pathlib import Path
import numpy as np
import torch
from field96 import EOT,load_data
from field103 import Chat103
from field106 import Chat106

def words(s):
    return re.findall(r'[a-z0-9]+',s.lower())

def metrics(raw,ref,ended):
    s=raw.decode('utf-8',errors='replace')
    a,b=words(s),words(ref)
    common=sum((collections.Counter(a)&collections.Counter(b)).values())
    def rep(seq,n):
        chunks=[tuple(seq[i:i+n]) for i in range(len(seq)-n+1)]
        return 1-len(set(chunks))/len(chunks) if chunks else 0.
    return dict(token_f1=2*common/max(1,len(a)+len(b)),
                exact=bool(b and a==b),exact_and_eot=bool(b and a==b and ended),
                contains_reference=bool(b and any(a[i:i+len(b)]==b for i in range(len(a)-len(b)+1))),
                eot=ended,empty=not raw,word3=rep(a,3),byte4=rep(raw,4),
                byte8=rep(raw,8),byte12=rep(raw,12),
                max_run=max((sum(1 for _ in z) for _,z in itertools.groupby(raw)),default=0))

@torch.no_grad()
def generate(model,prompt,temp,seed,limit):
    rng=np.random.default_rng(seed)
    state,anchor,q=model.prefill(prompt);out=bytearray()
    for _ in range(limit):
        if temp==0.:byte=int(q.argmax())
        else:
            p=np.power(np.maximum(q,1e-300),1./temp)
            byte=int(rng.choice(256,p=p/p.sum()))
        if byte==EOT:return bytes(out),True
        out.append(byte);state,q=model.advance(state,anchor,byte)
    return bytes(out),False

def summarize(rows):
    return {k:(float(np.mean([r[k] for r in rows])) if rows else None)
            for k in ('token_f1','exact','exact_and_eot','contains_reference','eot','empty','word3','byte4','byte8','byte12')}|{'n':len(rows),'long_runs':sum(r['max_run']>=8 for r in rows),'max_run':max((r['max_run'] for r in rows),default=0)}

def selftest():
    a=metrics(b'King George VI','King George VI',True)
    assert a['exact_and_eot'] and a['token_f1']==1.
    a=metrics(b'I'*72,'King George VI',False)
    assert a['word3']==0. and a['max_run']==72 and a['byte4']>.9 and not a['exact']
    assert metrics(b'', 'yes',True)['empty']
    assert metrics(b'Answer: Paris.','Paris',True)['contains_reference']
    print('natural answer audit: exactness, lexical overlap, completion and byte collapse passed',flush=True)

def main():
    p=argparse.ArgumentParser()
    for k in ('v104','v104-metrics','v106','v106-metrics','data','out'):p.add_argument('--'+k,type=Path)
    p.add_argument('--selftest',action='store_true');a=p.parse_args()
    selftest()
    if a.selftest:return
    torch.set_num_threads(4)
    def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
    s104,s106=sha(a.v104),sha(a.v106)
    c104=torch.load(a.v104,map_location='cpu',weights_only=True)
    c106=torch.load(a.v106,map_location='cpu',weights_only=True)
    m104=json.loads(a.v104_metrics.read_text());m106=json.loads(a.v106_metrics.read_text())
    assert m104['report_stage']==m106['report_stage']=='final'
    assert c106['parent_sha256']==m106['parent_sha256']==s104
    train,dev,manifest=load_data(a.data)
    assert manifest['sha256']==c104['data_sha256']==c106['data_sha256']
    archive=m104['final_generation']['responses']
    seen={x['prompt'] for x in archive}
    assert len(seen)==24
    # A new heldout lexical answer panel: deterministic selection based on
    # reference length, not model quality. Exclude the archived and earlier
    # training-probe instructions. This is not a claim of semantic correctness.
    from pilot96 import PROMPTS
    fixed={hashlib.sha256(' '.join(i.lower().split()).encode()).hexdigest() for _,p in PROMPTS for i in re.findall(r'(?:^|\n)User: (.*?)\nAssistant:',p)}
    dev=[r for r in dev if r[2] not in fixed]
    excluded=set()
    for seed in (9903,10003,9908):
        excluded.update(dev[i][2] for i in np.random.default_rng(seed).permutation(len(dev))[:128 if seed!=9908 else 12])
    excluded.update(dev[i][2] for i in np.random.default_rng(9703).choice(len(dev),128,replace=False))
    short=[]
    for i in np.random.default_rng(10677).permutation(len(dev)):
        row=dev[i];prompt=row[0].decode();ref=row[1][:-1].decode()
        if prompt in seen or row[2] in excluded or not 1<=len(words(ref))<=8:continue
        short.append((prompt,ref,row[2],row[3]))
        if len(short)==32:break
    assert len(short)>=8,'insufficient distinct short references'
    old=Chat103().eval().requires_grad_(False);old.load_state_dict(c104['chat103'])
    new=Chat106().eval().requires_grad_(False);new.load_state_dict(c106['chat106'])
    ab=Chat106().eval().requires_grad_(False);ab.load_state_dict(c106['chat106']);ab.scale106=0.
    arms={'v104':old,'v106':new,'v106_disabled':ab}
    rows=[]
    for item in archive:
        p,ref,temp=item['prompt'],item['reference'],item['temperature']
        for label,model in arms.items():
            raw,ended=generate(model,p.encode(),temp,10409,96)
            if label=='v104':
                assert raw.hex()==item['response_hex'] and ended==item['eot'],'archived v104 replay mismatch'
            rows.append(dict(arm=label,panel='archive',prompt=p,reference=ref,temperature=temp,
                             response=raw.decode(errors='replace'),response_hex=raw.hex(),**metrics(raw,ref,ended)))
    for p,ref,identity,category in short:
        for label,model in arms.items():
            for temp in (0.,.8):
                raw,ended=generate(model,p.encode(),temp,10679,96)
                rows.append(dict(arm=label,panel='fresh_short',prompt=p,reference=ref,identity=identity,category=category,
                                 temperature=temp,response=raw.decode(errors='replace'),response_hex=raw.hex(),**metrics(raw,ref,ended)))
    summary={}
    for panel in ('archive','fresh_short'):
        summary[panel]={}
        for arm in arms:
            summary[panel][arm]={str(t):summarize([r for r in rows if r['panel']==panel and r['arm']==arm and r['temperature']==t]) for t in (0.,.8)}
    print('NATURAL106_FINAL '+json.dumps(summary,allow_nan=False),flush=True)
    assert sha(a.v104)==s104 and sha(a.v106)==s106
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps({'report_stage':'final','parent_hashes':{'v104':s104,'v106':s106},'summary':summary,'responses':rows,'scope':'chat-only lexical proxies; not factual or semantic correctness; no promotion'},indent=2,allow_nan=False))
if __name__=='__main__':main()
