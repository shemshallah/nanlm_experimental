#!/usr/bin/env python3
"""R7.9 experimental CPU continuation: norm-calibrated causal read of ONE rank64 matrix.

Migrate from accepted R7.7 without altering pre-existing tensors/Adam moments.
Includes the R7.8 order/pair writers zero-gated, and a new zero-gated
RMS-calibrated read path. No separate model, no new independent memory bank.
UNPROMOTED until development AND test gates pass.
"""
import argparse, json, os, pathlib, random, time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from r77_semantic import build_model, restore as restore77
from r78_ordered import upgrade as order_upgrade, LRS
from r78_bigram import upgrade_pair
from r75_retrieval import OPT_NAMES
from r74_stream import PALR74, OUTER_CONTEXT, OLDER
from r7_core import Corpus
from r7_memory import WORD_CONTEXT, LOCAL
from train_r7 import batch_data, future_labels, stream_targets, sha

FORMAT='PAL_R7_9_NORMALIZED_RANK64_READER_V1'
PARENT_FORMAT='PAL_R7_7_SEMANTIC_ASSOCIATIVE_V1'

class NormReaderPAL74(PALR74):
    """A calibrated read from the existing physical matrix; migration gain=0."""
    def forward(self,ex,long,ctx,pieces,lens,state=None,outer_off=False,mirror_off=False):
        now,features=self.core(ex,long,ctx,pieces,lens,mirror_off=mirror_off)
        if state is not None and not outer_off:
            read=self.outer.read_rails(state,now)
            # Detached NOW amplitude prevents a model shortcut through the scale.
            now_rms=now.detach().square().mean(-1,keepdim=True).sqrt().clamp_min(.05)
            read_rms=read.square().mean(-1,keepdim=True).sqrt().clamp_min(.02)
            calibrated=(read/read_rms).clamp(-3.,3.)*now_rms
            residual=self.outer_gain*read+torch.tanh(self.normal_gain)*calibrated
            now=torch.tanh(torch.atanh(now.clamp(-.999999,.999999))+residual)
        return now,features

def upgrade_norm(model, opts):
    assert type(model.r74) is PALR74
    model.r74.__class__=NormReaderPAL74
    model.r74.normal_gain=nn.Parameter(model.r74.outer_gain.detach().new_zeros(()))
    opts[3].add_param_group({'params':[model.r74.normal_gain],'lr':.002,'weight_decay':0.})
    return model, opts

def restore(root,checkpoint,device='cpu'):
    checkpoint=pathlib.Path(checkpoint);root=pathlib.Path(root)
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if ck['format']==PARENT_FORMAT:
        model,opts,base,comp,prior=restore77(root,checkpoint,torch.device(device),LRS)
        order_upgrade(model,opts)
        upgrade_pair(model,opts)
        upgrade_norm(model,opts)
        torch.set_rng_state(prior['torch_rng']);np.random.set_state(prior['numpy_rng']);random.setstate(prior['python_rng'])
        return model,opts,base,comp,prior
    if ck['format']!=FORMAT:raise ValueError('unexpected source checkpoint '+str(ck['format']))
    if sha(root/'tokenizer/gutenberg_r7.model')!=ck['tokenizer_sha256'] or sha(root/'corpus/train_temporal.i32')!=ck['train_split_sha256']:
        raise RuntimeError('R7.9 source corpus/tokenizer mismatch')
    model,opts,base,comp=build_model(root,torch.device(device),LRS)
    order_upgrade(model,opts);upgrade_pair(model,opts);upgrade_norm(model,opts)
    model.load_state_dict(ck['model'],strict=True)
    for name,opt,lr in zip(OPT_NAMES,opts,LRS):
        opt.load_state_dict(ck['optimizers'][name])
        for i,g in enumerate(opt.param_groups):
            g['lr']=.004 if name=='outer_adamw' and i==2 else .002 if name=='outer_adamw' and i==3 else lr
    torch.set_rng_state(ck['torch_rng']);np.random.set_state(ck['numpy_rng']);random.setstate(ck['python_rng'])
    return model,opts,base,comp,ck

@torch.no_grad()
def eval_causal(model,base,comp,corpus,count=128,batch=8,offset=0):
    model.eval();positions=offset+np.arange(count,dtype=np.int64)*(len(corpus.seq)-offset-1)//count
    totals={x:0. for x in ('correct','normalized_off','outer_off','reversed','unrelated')}
    for p in np.array_split(positions,int(np.ceil(count/batch))):
        _,ctx,pc,ln,y,_,_=batch_data(base,corpus,comp,p)
        hist=torch.as_tensor(corpus.get(p,OUTER_CONTEXT)[0][:,:OLDER].copy())
        st=model.initial(hist)
        reversed_hist=hist.reshape(-1,OLDER//64,64).flip(-1).reshape(-1,OLDER)
        strev=model.initial(reversed_hist)
        ex=torch.as_tensor(corpus.data[p].copy());lg=torch.as_tensor(corpus.long[p].copy())
        vals={'correct':st,'normalized_off':st,'outer_off':st,'reversed':strev,'unrelated':st.roll(1,0)}
        old_norm=model.r74.normal_gain.detach().clone()
        for k in totals:
            if k=='normalized_off':model.r74.normal_gain.zero_()
            _,loss,_=model(ex,lg,ctx,pc,ln,vals[k],hist,y,outer_off=k=='outer_off')
            totals[k]+=float(loss.sum())
            model.r74.normal_gain.copy_(old_norm)
    n=len(positions)
    r={k+'_nll':v/n for k,v in totals.items()}
    r.update(n=n,normalized_advantage=r['normalized_off_nll']-r['correct_nll'],
             correct_over_unrelated=r['unrelated_nll']-r['correct_nll'],
             correct_over_reversed=r['reversed_nll']-r['correct_nll'],
             normal_gain=float(model.r74.normal_gain),order_gain=float(model.r74.outer.order_gain),pair_gain=float(model.r74.outer.pair_gain))
    return r

def save(out,model,opts,prior,root,step,rng,stats):
    out=pathlib.Path(out);out.parent.mkdir(exist_ok=True,parents=True)
    payload={k:prior[k] for k in ('base_legacy_steps','adapter_legacy_steps','legacy_r6_sha256','legacy_adapter_sha256','legacy_aux')}
    payload.update(format=FORMAT,model=model.state_dict(),optimizers={k:opt.state_dict() for k,opt in zip(OPT_NAMES,opts)},
      joint_steps=int(step),tokenizer_sha256=sha(pathlib.Path(root)/'tokenizer/gutenberg_r7.model'),
      train_split_sha256=sha(pathlib.Path(root)/'corpus/train_temporal.i32'),torch_rng=torch.get_rng_state(),
      numpy_rng=np.random.get_state(),python_rng=random.getstate(),batch_rng_state=rng.bit_generator.state,
      architecture='PAL R7.7 unified; R7.8 chronological and directed writers; RMS-calibrated read from one rank64 matrix',metrics=stats)
    tmp=out.with_suffix('.tmp');torch.save(payload,tmp);os.replace(tmp,out)
    return sha(out)

def train(args):
    torch.set_num_threads(args.threads);t0=time.monotonic();root=args.root
    model,opts,base,comp,prior=restore(root,args.resume)
    step=int(prior['joint_steps']);rng=np.random.default_rng();rng.bit_generator.state=prior['batch_rng_state']
    corp=Corpus(root,'train');metrics={'source_step':step,'source_hash':sha(args.resume),'updates':args.steps,'five_optimizer_states_retained':True}
    for i in range(args.steps):
        model.train()
        p=rng.integers(WORD_CONTEXT,len(corp.seq)-args.seq_len,size=args.streams,dtype=np.int64)
        h=torch.as_tensor(corp.get(p,OUTER_CONTEXT)[0][:,:OLDER].copy())
        state=model.initial(h);words=[];aux=[]
        for j in range(args.seq_len):
            pos=p+j
            _,ctx,pc,ln,y,tp,tl=batch_data(base,corp,comp,pos)
            ex=torch.as_tensor(corp.data[pos].copy());long=torch.as_tensor(corp.long[pos].copy())
            now, nll,_=model(ex,long,ctx,pc,ln,state,h,y)
            words.append(nll.mean())
            if args.subword:aux.append(args.subword*model.adapter.subword_loss(now,tp,tl))
            if args.future:
                fy,valid=future_labels(corp,pos);aux.append(args.future*model.r74.core.future_loss(now,fy,valid))
            if args.align:
                known=ctx[:,-LOCAL:]>1;v=model.adapter.compose(pc,ln)
                target=F.embedding(ctx[:,-LOCAL:],model.trunk.E)
                if bool(known.any()):aux.append(args.align*F.mse_loss(v[known],target[known]))
            if args.stream:
                st,sl=stream_targets(corp,comp,base.vocab,pos,words=args.stream_words)
                aux.append(args.stream*model.adapter.stream_loss(now,st,sl))
            if j+1<args.seq_len:
                leaving=ctx[:,0].long();state=model.advance(state,leaving);h=torch.cat((h[:,1:],leaving[:,None]),1)
        word=torch.stack(words).mean();loss=word+torch.stack(aux).sum()/args.seq_len if aux else word
        for opt in opts:opt.zero_grad(set_to_none=True)
        loss.backward()
        assert model.r74.normal_gain.grad is not None and torch.isfinite(model.r74.normal_gain.grad)
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        for opt in opts:opt.step()
        if not torch.isfinite(loss):raise FloatingPointError('diverged')
        if (i+1)%args.log_every==0 or i+1==args.steps:
            print(json.dumps(dict(step=step+i+1,loss=float(loss),train_nll=float(word),normal_gain=float(model.r74.normal_gain),
              order_gain=float(model.r74.outer.order_gain),pair_gain=float(model.r74.outer.pair_gain),
              elapsed_s=round(time.monotonic()-t0,1))),flush=True)
    metrics['final_step']=step+args.steps;metrics['elapsed_seconds']=round(time.monotonic()-t0,2)
    digest=save(args.output,model,opts,prior,root,step+args.steps,rng,metrics)
    metrics['candidate_sha256']=digest
    if args.report: pathlib.Path(args.report).write_text(json.dumps(metrics,indent=2)+'\n')
    print(json.dumps({'SAVED':str(args.output),'step':step+args.steps,'sha256':digest}),flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--root',type=pathlib.Path,default=pathlib.Path('.'))
    a.add_argument('--resume',type=pathlib.Path,default=pathlib.Path('checkpoint/pal_active.pt'))
    a.add_argument('--output',type=pathlib.Path,default=pathlib.Path('checkpoint/pal_r79_normalized_candidate.pt'))
    a.add_argument('--report',type=pathlib.Path);a.add_argument('--steps',type=int,default=128)
    a.add_argument('--streams',type=int,default=2);a.add_argument('--seq-len',type=int,default=2)
    a.add_argument('--threads',type=int,default=2);a.add_argument('--log-every',type=int,default=32)
    a.add_argument('--subword',type=float,default=.08);a.add_argument('--future',type=float,default=.03)
    a.add_argument('--align',type=float,default=.1);a.add_argument('--stream',type=float,default=.2)
    a.add_argument('--stream-words',type=int,default=6)
    train(a.parse_args())
