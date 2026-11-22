#!/usr/bin/env python3
"""Raw TRAIN-book native-lexicon adaptation of v166 with exact AdamW state resume.

This uses only v163 calibration windows drawn from 100 TRAIN books; it does not
train on the book-disjoint DEV or reused TEST windows. Validation split is at
whole-book granularity. Successor weights are saved separately (never overwrite).
"""
import argparse, copy, hashlib, json, math, re, sys, time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
R=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(R/'experiments/nancm/v166'), str(R/'scripts')]
from train_birth166 import Birth166,fnv
from train_gate163 import winread

def read_books(vocab):
    cohort=json.loads((R/'external_v163/COHORTS.json').read_text())
    vid={w:i+1 for i,w in enumerate(vocab)}
    groups={};last={}
    for part in range(3):
        windows=list(winread(R/f'external_v163/v163_calibration_{part}.bin'))
        infos=cohort['calibration_windows'][part]
        assert len(windows)==len(infos)
        for (raw,skip),info in zip(windows,infos):
            book=info['name']
            assert book in set(cohort['calibration_books'])
            # word-order from byte text; last 3 prior complete words only
            tokens=[(m.start(),m.group().lower().decode('ascii')) for m in re.finditer(rb'[A-Za-z]+',raw)]
            events=[]
            for i,(p,w) in enumerate(tokens):
                if p<skip or i<3 or len(w)>20 or w not in vid: continue
                prev=[z for _,z in tokens[i-3:i]]
                c=[vid.get(z,0) for z in prev]
                h=[fnv(z)+1 for z in prev]
                events.append((c,h,vid[w]))
            groups.setdefault(book,[]).extend(events)
    assert len(groups)==100
    names=sorted(groups,key=lambda x:hashlib.sha256(('NaNCM_v168_book_gate:'+x).encode()).digest())
    train=set(names[:80]);valid=set(names[80:])
    assert not train.intersection(valid)
    def encode(selection):
        all_rows=[r for book in names if book in selection for r in groups[book]]
        return tuple(torch.tensor([r[i] for r in all_rows],dtype=torch.long) for i in range(3))
    return encode(train),encode(valid),{'train_books':len(train),'val_books':len(valid),'train_events':sum(len(groups[b]) for b in train),'val_events':sum(len(groups[b]) for b in valid),'unseen_truth_lexicon_pct':None,'sources':['v163_calibration_0.bin','v163_calibration_1.bin','v163_calibration_2.bin']}

def metrics(model,data,batch=128):
    model.eval();c,h,y=data;n=len(y);nll=0.;acc1=0;acc5=0
    with torch.no_grad():
        for k in range(0,n,batch):
            p=model(c[k:k+batch],h[k:k+batch]);target=y[k:k+batch]
            lp=F.log_softmax(p,dim=-1)
            nll+=float(-lp.gather(1,target[:,None]).sum())
            order=p.topk(5,dim=1).indices
            acc1+=int((order[:,0]==target).sum());acc5+=int((order==target[:,None]).any(1).sum())
    return {'n':n,'nll_nats':nll/n,'top1_pct':100*acc1/n,'top5_pct':100*acc5/n}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--steps',type=int,default=250);ap.add_argument('--batch',type=int,default=96);ap.add_argument('--seed',type=int,default=168);ap.add_argument('--lr-ratio',type=float,default=.1);ap.add_argument('--threads',type=int,default=3)
    a=ap.parse_args();t0=time.monotonic()
    torch.set_num_threads(a.threads);torch.manual_seed(a.seed);rng=np.random.default_rng(a.seed)
    pack=torch.load(R/'checkpoints/v166_lexical_birth.pt',map_location='cpu',weights_only=False)
    vocab=pack['vocab'];m=Birth166(len(vocab),np.zeros(len(vocab)+1,np.float32),pack['model']);m.load_state_dict(pack['model'])
    # Resume the exact model weights and AdamW moments; no new model lineage
    cloud=[m.plus,m.minus,m.lower,m.upper];lex=[m.ctx.weight,m.out.weight,m.bias,m.loglex,m.logcloud]
    opt=torch.optim.AdamW([{'params':cloud,'lr':.0007,'weight_decay':.001}, {'params':lex,'lr':.003,'weight_decay':.01}]);opt.load_state_dict(pack['optimizer'])
    base_model={k:v.clone() for k,v in m.state_dict().items()}
    train,val,info=read_books(vocab)
    print('COHORTS',json.dumps(info),flush=True)
    baseline=metrics(m,val)
    print('BASE_VALID',json.dumps(baseline),flush=True)
    # Scale inherited per-parameter-group learning rates without losing Adam state.
    for g in opt.param_groups:g['lr']*=a.lr_ratio
    best=baseline['nll_nats'];best_state=copy.deepcopy(m.state_dict());best_opt=copy.deepcopy(opt.state_dict());best_step=0
    history=[]
    trc,trh,try_=train
    for step in range(1,a.steps+1):
        ids=torch.as_tensor(rng.integers(0,len(try_),size=a.batch),dtype=torch.long)
        wc=trc[ids].clone();hc=trh[ids].clone();target=try_[ids]
        # Causal context dropout makes the head robust to previously unseen words
        mask=torch.tensor(rng.random((a.batch,3))<.08,dtype=torch.bool)
        wc[mask]=0;hc[mask]=0
        m.train();opt.zero_grad(set_to_none=True)
        logits=m(wc,hc)
        loss=F.cross_entropy(logits,target)
        # Anchor large pretrained tensors to their exact inherited checkpoint.
        anchor=.003*((m.ctx.weight-base_model['ctx.weight']).square().mean()+(m.out.weight-base_model['out.weight']).square().mean())
        (loss+anchor).backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.5);opt.step()
        if step%25==0 or step==1 or step==a.steps:
            stat=metrics(m,val);stat.update(step=step,train_nll=float(loss.detach()))
            history.append(stat)
            print('EVAL',json.dumps(stat),flush=True)
            if stat['nll_nats']<best:
                best=stat['nll_nats'];best_step=step;best_state=copy.deepcopy(m.state_dict());best_opt=copy.deepcopy(opt.state_dict())
    m.load_state_dict(best_state);opt.load_state_dict(best_opt)
    out=R/'checkpoints/v168_raw_candidate.pt'
    new={**pack,'model':m.state_dict(),'optimizer':opt.state_dict(),'source_v166_checkpoint_sha256':hashlib.sha256((R/'checkpoints/v166_lexical_birth.pt').read_bytes()).hexdigest(),
         'raw_book_history':history,'raw_book_best_step':best_step,'raw_train_details':info}
    torch.save(new,out)
    native=R/'checkpoints/v168_raw_candidate.bin'
    # Write exact wire format for the native model; check invariant fingerprints.
    import struct
    index=R/'checkpoints/v164_native_distribution.tsv'
    with index.open() as f:head=f.readline().strip().split('\t')
    with native.open('wb') as f:
        f.write(b'N166BRT1');f.write(struct.pack('<3Q',*map(int,head[1:])));f.write(struct.pack('<II',len(vocab),32))
        for name in ('plus','minus','lower','upper'):f.write(m.state_dict()[name].cpu().contiguous().numpy().astype('<f4').tobytes())
        f.write(struct.pack('<ff',m.loglex.exp().item(),m.logcloud.exp().item()))
        for i,w in enumerate(vocab,1):
            f.write(struct.pack('<B',len(w))+w.encode('ascii'))
            f.write(struct.pack('<f',m.bias[i].item()))
            f.write(m.ctx.weight[i].detach().numpy().astype('<f4').tobytes())
            f.write(m.out.weight[i].detach().numpy().astype('<f4').tobytes())
    report={'status':'EXPERIMENTAL_DO_NOT_PROMOTE','source':'3 raw 384-byte windows from each of 100 TRAIN books','split':info,
       'baseline':baseline,'best':metrics(m,val),'best_step':best_step,'history':history,
       'resumed_all_adamw_param_groups':len(opt.param_groups),'original_checkpoint_unchanged':True,
       'training_seconds':round(time.monotonic()-t0,2),'checkpoint_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),
       'native_sha256':hashlib.sha256(native.read_bytes()).hexdigest(),
       'caveat':'small book-disjoint raw TRAIN calibration split; need independent DEV native test; baseline unchanged'}
    (R/'results/v168_raw_train.json').write_text(json.dumps(report,indent=2))
    print('FINAL',json.dumps(report),flush=True)
if __name__=='__main__':main()