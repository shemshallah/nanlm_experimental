#!/usr/bin/env python3
"""NaNCM v166 lexical-birth head: TRAIN-index next-word supervision.

No DEV/TEST gradients or targets enter training. Full-vocabulary next-word
cross-entropy rather than within-candidate reranking. Inherited v165 signed
cloud and optimizer moments are carried into the new experiment; the original
models/checkpoints are never overwritten. The v164 TRAIN index is a lossy top-8
conditional-count representation: this is not a raw-book end-to-end finetune.
"""
import argparse, collections, copy, hashlib, json, math, random, struct, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
R=Path(__file__).resolve().parents[3]
D=32

def fnv(word):
    h=2166136261
    for b in word.encode('ascii'):
        h=((h^b)*16777619)&0xffffffff
    return h & 127

def parse_train_index(path,max_vocab=12000):
    groups=[];counts=collections.Counter();allctx=set();cur=None;rows=[]
    def flush():
        if cur is not None and rows:
            groups.append((cur[0],cur[1],rows[:]))
    with open(path) as f:
        header=f.readline().strip().split('\t')
        assert len(header)==4 and header[0]=='N164DIST1'
        for line in f:
            k,ctx,w,n,tot=line.rstrip('\n').split('\t');k=int(k);n=int(n)
            if k==1:counts[w]+=n
            key=(k,ctx)
            if key!=cur:flush();cur=key;rows=[]
            rows.append((w,n))
        flush()
    # Use distinct input and output id semantics, one <unk> 0 id.
    # Keep higher-frequency targets plus every context word that fits.
    train_words={w for _,ctx,rs in groups for w in ctx.split()} | set(counts)
    extra=collections.Counter()
    for _,ctx,rs in groups:
        for w,n in rs:
            if not counts[w]:extra[w]+=n
        for w in ctx.split():extra[w]+=1
    ordered=sorted(train_words,key=lambda w:(-(counts[w]*10+extra[w]),w))
    vocab=ordered[:max_vocab];vid={w:i+1 for i,w in enumerate(vocab)}
    cidx=np.zeros((len(groups),3),dtype=np.int32)
    hidx=np.zeros((len(groups),3),dtype=np.int32)
    tgt=np.zeros((len(groups),8),dtype=np.int32)
    weights=np.zeros((len(groups),8),dtype=np.float32)
    ords=np.zeros(len(groups),dtype=np.int8)
    retained=0
    for i,(k,ctx,rs) in enumerate(groups):
        ps=ctx.split();ords[i]=k
        assert len(ps)==k
        for j,w in enumerate(ps):
            ix=3-k+j;cidx[i,ix]=vid.get(w,0);hidx[i,ix]=fnv(w)+1
        for j,(w,n) in enumerate(rs[:8]):
            y=vid.get(w,0)
            if y:
                tgt[i,j]=y;weights[i,j]=n;retained+=n
        if weights[i].sum():weights[i]/=weights[i].sum()
    keep=weights.sum(1)>0
    frequencies=np.array([1]+[counts[w]+.2*extra[w]+1 for w in vocab],dtype=np.float64)
    priors=np.log(frequencies/frequencies[1:].sum()).astype(np.float32)
    return header,vocab,(cidx[keep],hidx[keep],tgt[keep],weights[keep],ords[keep]),priors,{'groups':len(groups),'usable_groups':int(keep.sum()),'distinct_vocab':len(vocab),'weighted_retained_train_count':retained}

class Birth166(nn.Module):
    def __init__(self,vocab,prior,old):
        super().__init__();self.vocab=vocab
        self.ctx=nn.Embedding(vocab+1,D,padding_idx=0)
        self.out=nn.Embedding(vocab+1,D,padding_idx=0)
        nn.init.normal_(self.ctx.weight,std=.13)
        nn.init.normal_(self.out.weight,std=.13)
        with torch.no_grad():self.ctx.weight[0].zero_();self.out.weight[0].zero_()
        self.bias=nn.Parameter(torch.from_numpy(prior.copy()))
        # Exact resumed v165 S+ / S- tensors and 8D point coordinates.
        for name in ('plus','minus','lower','upper'):
            setattr(self,name,nn.Parameter(old[name].clone().float()))
        self.register_buffer('hashes',torch.zeros(vocab+1,dtype=torch.long))
        self.register_buffer('recency',torch.tensor([.2,.3,.5]))
        self.loglex=nn.Parameter(torch.tensor(math.log(2.0)))
        self.logcloud=nn.Parameter(torch.tensor(math.log(.08)))
    def concept(self,ha):
        # ha in 1..128, 0 indicates missing previous word.
        bs=ha.shape[0];h=ha.clamp(min=1)-1
        valid=(ha!=0).float()
        li=self.lower[h] # B,3,8
        # plus[t,b,d], advanced indexing -> upper index, batch, temporal, dim
        p=self.plus[:,h].permute(1,0,2,3)
        m=self.minus[:,h].permute(1,0,2,3)
        ui=self.upper[None,:,None,:]
        lv=li[:,None,:,:]
        e=(p*torch.tanh(lv+ui)-m*torch.tanh(lv-ui)+lv*ui/math.sqrt(8)).sum(-1)
        rec=(self.recency[None,:]*valid)
        return (e*rec[:,None,:]).sum(-1)/rec.sum(-1).clamp_min(1e-8)[:,None]
    def forward(self,words,hashes):
        mask=(words>0).float();rec=mask*self.recency[None,:]
        q=(self.ctx(words)*rec[:,:,None]).sum(1)/rec.sum(1).clamp_min(1e-8)[:,None]
        lex=torch.mm(q,self.out.weight.T)*(self.loglex.exp()/math.sqrt(D))
        con=self.concept(hashes).index_select(1,self.hashes)
        y=lex+self.bias[None,:]+con*self.logcloud.exp()
        # Explicitly prohibit the unknown-output token.
        y[:,0]=-1.e6
        return y

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--steps',type=int,default=600);ap.add_argument('--batch',type=int,default=128);ap.add_argument('--vocab',type=int,default=12000);ap.add_argument('--lr',type=float,default=.015);ap.add_argument('--threads',type=int,default=4);a=ap.parse_args()
    torch.set_num_threads(a.threads);torch.manual_seed(166);np.random.seed(166);random.seed(166)
    t0=time.monotonic();index=R/'checkpoints/v164_native_distribution.tsv'
    head,vocab,raw,prior,meta=parse_train_index(index,a.vocab)
    print('DATA',json.dumps(meta),flush=True)
    old=torch.load(R/'checkpoints/v165_signed_cloud_train.pt',weights_only=False,map_location='cpu')
    assert old['training_index_sha']==hashlib.sha256(index.read_bytes()).hexdigest()
    model=Birth166(len(vocab),prior,old['cloud'])
    with torch.no_grad():model.hashes[1:]=torch.tensor([fnv(w) for w in vocab])
    cloudpar=[model.plus,model.minus,model.lower,model.upper]
    freshpar=[model.ctx.weight,model.out.weight,model.bias,model.loglex,model.logcloud]
    opt=torch.optim.AdamW([{'params':cloudpar,'lr':.0007,'weight_decay':.001}, {'params':freshpar,'lr':a.lr,'weight_decay':.01}])
    # Transfer every AdamW moment from trained v165 to its matching tensors.
    ancestral=old['optimizer'];oldids=ancestral['param_groups'][0]['params']
    for p,oid in zip(cloudpar,oldids):
        src=ancestral['state'].get(oid)
        if src:
            assert all(not torch.is_tensor(v) or v.ndim==0 or v.shape==p.shape for v in src.values())
            opt.state[p]={k:v.clone() if torch.is_tensor(v) else v for k,v in src.items()}
    c,h,t,w,orders=raw
    datasets={k:np.flatnonzero(orders==k) for k in (1,2,3)}
    # TRUE next-word label selected from the count-weighted TRAIN-only event group,
    # not a v164 list-reordering objective. Full-vocabulary normalization in loss.
    rng=np.random.default_rng(166);history=[];best=1e9;best_params=None;best_optimizer=None
    # This diagnostic validation is TRAIN-index context-disjoint only; NOT book-disjoint.
    idx=np.arange(len(orders));rng.shuffle(idx);hold=set(idx[-min(8000,len(idx)//10):]);tr={k:np.array([i for i in datasets[k] if i not in hold],dtype=np.int32) for k in (1,2,3)}
    vix=np.array(sorted(hold),dtype=np.int32)
    # Tiny held-out matrix of context groups, never used for training.
    valc=torch.from_numpy(c[vix[:512]].astype(np.int64));valh=torch.from_numpy(h[vix[:512]].astype(np.int64))
    valt=torch.from_numpy(t[vix[:512]].astype(np.int64));valw=torch.from_numpy(w[vix[:512]])
    for step in range(1,a.steps+1):
        ks=rng.choice([1,2,3],size=a.batch,p=[.15,.25,.6]);inds=np.zeros(a.batch,dtype=np.int64)
        for k in (1,2,3):
            m=ks==k;inds[m]=rng.choice(tr[k],int(m.sum()))
        cs=torch.from_numpy(c[inds].astype(np.int64));hs=torch.from_numpy(h[inds].astype(np.int64));tt=t[inds];ww=w[inds]
        u=rng.random(a.batch);picked=(u[:,None]>np.cumsum(ww,axis=1)).sum(1).clip(0,7)
        yy=torch.from_numpy(tt[np.arange(a.batch),picked].astype(np.int64))
        assert (yy>0).all()
        model.train();opt.zero_grad(set_to_none=True)
        logits=model(cs,hs);loss=F.cross_entropy(logits,yy)
        # Constrain cloud drift away from the inherited geometry while allowing
        # actual next-word gradients to shape both oriented channels.
        reg=2.e-5*((model.plus-old['cloud']['plus']).square().mean()+(model.minus-old['cloud']['minus']).square().mean())
        (loss+reg).backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2.0);opt.step()
        if step==1 or step%50==0 or step==a.steps:
            model.eval()
            with torch.no_grad():
                vpred=model(valc,valh);lp=F.log_softmax(vpred,dim=-1)
                vloss=-(lp.gather(1,valt)*(valw)).sum(1).mean().item()
            history.append({'step':step,'train_nll':round(float(loss.item()),5),'context_split_val_nll':round(vloss,5)})
            print('STEP',json.dumps(history[-1]),flush=True)
            if vloss<best:
                best=vloss;best_params={k:v.detach().clone() for k,v in model.state_dict().items()};best_optimizer=copy.deepcopy(opt.state_dict())
    model.load_state_dict(best_params);opt.load_state_dict(best_optimizer)
    out=R/'checkpoints/v166_lexical_birth.pt'
    torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'vocab':vocab,'source_v165_optimizer':old['optimizer'],
                'train_index_sha':hashlib.sha256(index.read_bytes()).hexdigest(),'history':history,'source_v165_cloud_sha':hashlib.sha256((R/'checkpoints/v165_signed_cloud.bin').read_bytes()).hexdigest()},out)
    native=R/'checkpoints/v166_lexical_birth.bin'
    with native.open('wb') as f:
        f.write(b'N166BRT1');f.write(struct.pack('<3Q',*map(int,head[1:])));f.write(struct.pack('<II',len(vocab),D))
        for name in ('plus','minus','lower','upper'):
            f.write(model.state_dict()[name].cpu().numpy().astype('<f4').tobytes())
        f.write(struct.pack('<ff',model.loglex.exp().item(),model.logcloud.exp().item()))
        for i,word in enumerate(vocab,1):
            assert 0<len(word)<=20 and word.isascii() and word.islower() and word.isalpha()
            f.write(struct.pack('<B',len(word))+word.encode('ascii'))
            f.write(struct.pack('<f',model.bias[i].item()))
            f.write(model.ctx.weight[i].detach().cpu().numpy().astype('<f4').tobytes())
            f.write(model.out.weight[i].detach().cpu().numpy().astype('<f4').tobytes())
    report={'status':'experimental','name':'NaNCM v166 lexical birth','train_source':'v164 TRAIN index, lossy top-8 distributional source','data':meta,
        'steps':a.steps,'best_train_index_context_split_val_nll':best,'history':history,
        'v165_signed_optimizer_state_inherited':True,'train_index_sha256':hashlib.sha256(index.read_bytes()).hexdigest(),
        'native_sha256':hashlib.sha256(native.read_bytes()).hexdigest(),'checkpoint_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),
        'elapsed_sec':round(time.monotonic()-t0,2),'caveat':'No full native DEV inference evaluation yet; context split is NOT book-disjoint and is not promotion evidence.'}
    (R/'results/v166_train.json').write_text(json.dumps(report,indent=2));print('FINAL',json.dumps(report),flush=True)
if __name__=='__main__':main()