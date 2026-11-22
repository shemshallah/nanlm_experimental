#!/usr/bin/env python3
"""v167 continuation of v166: full-vocabulary count-weighted NLL + modal
margin/hard-negative guidance, while preserving inherited 128+128 signed
cloud AND all AdamW moments. Uses ONLY the existing TRAIN-only 500-book
TRUNCATED index; the split here is not book-disjoint. Never overwrite v166.
"""
import argparse,copy,hashlib,json,math,sys,time,struct
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'experiments/nancm/v166'))
from train_birth166 import parse_train_index,Birth166,fnv,D

def build(pack,original):
    m=Birth166(len(pack['vocab']),np.zeros(len(pack['vocab'])+1,dtype=np.float32),pack['model'])
    m.load_state_dict(pack['model'],strict=True)
    return m

def validate(m,raw,indices,batch=64):
    c,h,t,w,o=raw
    m.eval();total=0.;ok=0;count=0
    with torch.no_grad():
        for start in range(0,len(indices),batch):
            ids=indices[start:start+batch]
            cc=torch.from_numpy(c[ids].astype(np.int64));hh=torch.from_numpy(h[ids].astype(np.int64))
            y=torch.from_numpy(t[ids].astype(np.int64));p=torch.from_numpy(w[ids])
            logits=m(cc,hh);logp=F.log_softmax(logits,dim=-1)
            loss=-(logp.gather(1,y)*p).sum(1)
            total+=float(loss.sum());ok+=int((logits.argmax(1)==y.gather(1,p.argmax(1)[:,None]).squeeze(1)).sum());count+=len(ids)
    return {'train_index_context_split_nll':round(total/count,7),'modal_top1':round(ok/count,5),'contexts':count}

def export_native(m,head,vocab,dst):
    with dst.open('wb') as f:
        f.write(b'N166BRT1');f.write(struct.pack('<3Q',*map(int,head[1:])));f.write(struct.pack('<II',len(vocab),D))
        st=m.state_dict()
        for name in ('plus','minus','lower','upper'):
            f.write(st[name].cpu().contiguous().numpy().astype('<f4').tobytes())
        f.write(struct.pack('<ff',m.loglex.exp().item(),m.logcloud.exp().item()))
        for i,word in enumerate(vocab,1):
            assert 0<len(word)<=20 and word.isascii() and word.islower() and word.isalpha()
            f.write(struct.pack('<B',len(word))+word.encode('ascii'))
            f.write(struct.pack('<f',m.bias[i].item()))
            for which in ('ctx','out'):
                f.write(st[which+'.weight'][i].cpu().contiguous().numpy().astype('<f4').tobytes())

def main():
    p=argparse.ArgumentParser();p.add_argument('--steps',type=int,default=400);p.add_argument('--threads',type=int,default=4)
    p.add_argument('--batch',type=int,default=128);p.add_argument('--margin-weight',type=float,default=.08);p.add_argument('--distill',type=float,default=.1)
    args=p.parse_args();torch.set_num_threads(args.threads);torch.manual_seed(167);rng=np.random.default_rng(167);start=time.monotonic()
    original=ROOT/'checkpoints/v166_lexical_birth.pt';pack=torch.load(original,map_location='cpu',weights_only=False)
    train_index=ROOT/'checkpoints/v164_native_distribution.tsv'
    assert pack['train_index_sha']==hashlib.sha256(train_index.read_bytes()).hexdigest(), 'TRAIN provenance failure'
    head,vocab,raw,prior,meta=parse_train_index(train_index,len(pack['vocab']))
    assert vocab==pack['vocab'],'vocabulary drift forbidden'
    model=build(pack,original);teacher=build(pack,original);teacher.eval();teacher.requires_grad_(False)
    cloud=[model.plus,model.minus,model.lower,model.upper]
    fresh=[model.ctx.weight,model.out.weight,model.bias,model.loglex,model.logcloud]
    opt=torch.optim.AdamW([{'params':cloud,'lr':.00018,'weight_decay':.001},{'params':fresh,'lr':.0025,'weight_decay':.01}])
    opt.load_state_dict(pack['optimizer'])
    opt.param_groups[0]['lr']=.00018;opt.param_groups[1]['lr']=.0025
    # Validate that every inherited tensor has both Adam moments from v166.
    for name,p in zip(('plus','minus','lower','upper'),cloud):
        assert 'exp_avg' in opt.state[p] and 'exp_avg_sq' in opt.state[p],name
    c,h,t,w,orders=raw
    inds=np.arange(len(orders));np.random.default_rng(166).shuffle(inds)
    nval=min(8000,len(inds)//10);hold=set(inds[-nval:]);train=np.array([i for i in range(len(orders)) if i not in hold],dtype=np.int32)
    val=np.array(sorted(hold)[:768],dtype=np.int32)
    byorder={k:train[orders[train]==k] for k in (1,2,3)}
    base=validate(model,raw,val)
    print('BASE',json.dumps(base),'data',json.dumps(meta),flush=True)
    best=(-1,base['train_index_context_split_nll'],base['modal_top1'])
    history=[];best_model=None;best_optimizer=None
    for step in range(1,args.steps+1):
        ks=rng.choice([1,2,3],size=args.batch,p=[.15,.25,.6]);ids=np.empty(args.batch,dtype=np.int64)
        for k in (1,2,3):
            mask=ks==k
            if mask.any():ids[mask]=rng.choice(byorder[k],int(mask.sum()),replace=True)
        cc=torch.from_numpy(c[ids].astype(np.int64));hh=torch.from_numpy(h[ids].astype(np.int64))
        tt=torch.from_numpy(t[ids].astype(np.int64));ww=torch.from_numpy(w[ids]);modal=tt.gather(1,ww.argmax(1)[:,None]).squeeze(1)
        model.train();opt.zero_grad(set_to_none=True)
        logits=model(cc,hh);logp=F.log_softmax(logits,dim=-1)
        nll=-(ww*logp.gather(1,tt)).sum(1).mean()
        wrong=logits.detach().clone();wrong.scatter_(1,modal[:,None],-1.e6)
        neg=wrong.argmax(1)
        gap=logits.gather(1,neg[:,None]).squeeze(1)-logits.gather(1,modal[:,None]).squeeze(1)
        hinge=F.softplus((gap+.25)*2.).mean()/2.
        with torch.no_grad():ref=F.softmax(teacher(cc,hh),dim=-1)
        distill=F.kl_div(logp,ref,reduction='batchmean')
        drift=(model.plus-teacher.plus).square().mean()+(model.minus-teacher.minus).square().mean()
        loss=nll+args.margin_weight*hinge+args.distill*distill+2e-5*drift
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2.0);opt.step()
        if step%50==0 or step==1 or step==args.steps:
            report=validate(model,raw,val)
            report.update({'step':step,'batch_nll':round(float(nll.detach()),5),'margin':round(float(hinge.detach()),5)})
            history.append(report);print('STEP',json.dumps(report),flush=True)
            # Select by loss but require nondecreasing training-index modal agreement.
            if report['train_index_context_split_nll']<best[1] and report['modal_top1']>=base['modal_top1']:
                best=(step,report['train_index_context_split_nll'],report['modal_top1'])
                best_model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                best_optimizer=copy.deepcopy(opt.state_dict())
    if best_model is None:
        # Fail closed, leave predecessor active.
        result={'status':'rejected','reason':'no qualifying candidate','base':base,'history':history}
        (ROOT/'results/v167_train.json').write_text(json.dumps(result,indent=2));print('FINAL',json.dumps(result));return
    model.load_state_dict(best_model);opt.load_state_dict(best_optimizer)
    ckp=ROOT/'checkpoints/v167_margin_candidate.pt';native=ROOT/'checkpoints/v167_margin_candidate.bin'
    torch.save({'model':best_model,'optimizer':best_optimizer,'vocab':vocab,'train_index_sha':pack['train_index_sha'],
                'v166_checkpoint_sha':hashlib.sha256(original.read_bytes()).hexdigest(),'history':history,'best_step':best[0],
                'source_v165_optimizer':pack['source_v165_optimizer']},ckp)
    export_native(model,head,vocab,native)
    result={'status':'candidate_not_promoted','source_v166':str(original),'parent_sha':hashlib.sha256(original.read_bytes()).hexdigest(),
            'mode':'TRAIN-index full-distribution NLL + hard negative margin and KL to v166','train_index_loss_before':base,
            'best_train_index_loss_after':validate(model,raw,val),'steps_run':args.steps,'best_step':best[0],
            'signed_adam_moments_preserved':True,'native_sha256':hashlib.sha256(native.read_bytes()).hexdigest(),
            'checkpoint_sha256':hashlib.sha256(ckp.read_bytes()).hexdigest(),'history':history,'seconds':round(time.monotonic()-start,2),
            'warning':'TRAIN-index context split not book-disjoint, DEV native gate and strict top-1 are required before promotion'}
    (ROOT/'results/v167_train.json').write_text(json.dumps(result,indent=2));print('FINAL',json.dumps(result),flush=True)
if __name__=='__main__':main()