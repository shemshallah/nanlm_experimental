#!/usr/bin/env python3
"""PAL R7.3: ONE end-to-end differentiable R6+R7 network.

Load R6 weights + accumulated squared-gradient statistics without resetting them;
load the trained R7 adapter and its AdamW moments; attach a causal low-rank
S+/S- mirror-write memory. Consolidate into one atomic .pt checkpoint.

The historical C++ optimizer uses acc += grad**2, weight -= lr*grad/sqrt(acc).
This is *not* Adam and should not be labeled as AdamW optimizer history.
"""
from __future__ import annotations
import argparse, hashlib, json, os, pathlib, random, time, weakref
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from r7_core import R6, Corpus, D, R, V, LONGF, PAIRS
from r7_memory import R7Memory, TokenCompositor, WORD_CONTEXT, LOCAL, NCHUNKS
from train_r7 import batch_data, future_labels, stream_targets, load_extension, sha

TRUNK_NAMES=('E','H','P','M','Q','NOW','W','B','L','T')

class TrainableR6(nn.Module):
    def __init__(self,base:R6,root:pathlib.Path):
        super().__init__()
        for k in TRUNK_NAMES[:-1]:
            setattr(self,k,nn.Parameter(torch.from_numpy(np.asarray(getattr(base,k)).copy())))
        self.T=nn.Parameter(torch.from_numpy(np.asarray(base.T).copy()))
        self.register_buffer('head_indices',torch.from_numpy(base.heads.astype(np.int64)),persistent=False)
        self.register_buffer('path_nodes',torch.from_numpy(base.pnodes.copy()),persistent=False)
        self.register_buffer('path_bits',torch.from_numpy(base.pbits.copy()),persistent=False)
        self.register_buffer('path_mask',torch.from_numpy(base.pmask.copy()),persistent=False)
    def hidden(self,examples:torch.Tensor,longctx:torch.Tensor):
        # DO NOT read target or future columns [3:7]. Both are supervision only.
        x0,x1,x2=examples[:,0].long(),examples[:,1].long(),examples[:,2].long()
        ha=self.head_indices[x2,0];hb=self.head_indices[x1,1]
        hb=torch.where(ha==hb,(hb+1)&255,hb)
        lo=torch.minimum(ha,hb);hi=torch.maximum(ha,hb)
        pair=lo*(511-lo)//2+hi-lo-1
        sign=torch.where(ha<hb,1.,-1.).to(self.E.dtype)
        f=.4*sign[:,None]*(self.P[pair]-.2*self.M[pair])
        z=(self.NOW[None,:]+f@self.Q.T+longctx.to(self.E.dtype)@self.L.T/32.)
        z=z+.14*self.E[x0]+.28*self.E[x1]+.58*self.E[x2]
        z=z+.18*(self.H[ha]-self.H[hb])
        z=z+.04*(self.H[(ha+1)&255]+self.H[(ha+2)&255]-self.H[(hb+255)&255]-self.H[(hb+254)&255])
        return torch.tanh(z)
    def nll(self,h,targets):
        y=targets.long();nodes=self.path_nodes[y]
        logits=(self.W[nodes]*h[:,None,:]).sum(-1)+self.B[nodes]
        bits=self.path_bits[y];mask=self.path_mask[y]
        return (F.binary_cross_entropy_with_logits(logits,bits,reduction='none')*mask).sum(-1)

class LegacyAccumulator(torch.optim.Optimizer):
    """Exact mathematical PAL6 AdaGrad-style accumulators, not a fresh optimizer."""
    def __init__(self,trunk:TrainableR6,root,lr=0.0001,init_from_legacy=True):
        super().__init__(list(trunk.parameters()),dict(lr=lr))
        if not init_from_legacy:return
        raw=np.memmap(root/'checkpoint/optimizer.raw',dtype='<f4',offset=24)
        weights=np.memmap(root/'checkpoint/weights.raw',dtype='<f4',offset=24)
        assert len(raw)==len(weights)
        offset=0
        for name in ('E','H','P','M','Q','NOW','W','B'):
            p=getattr(trunk,name);n=p.numel()
            a=torch.from_numpy(np.asarray(raw[offset:offset+n].reshape(p.shape)).copy());offset+=n
            self.state[p]['sum']=a.to(p.device)
        assert offset==len(raw),(offset,len(raw))
        longfile=np.memmap(root/'checkpoint/weights.raw.ctx2048',dtype='<f4',mode='r',offset=24,shape=(2,D,LONGF))
        self.state[trunk.L]['sum']=torch.from_numpy(np.asarray(longfile[1]).copy()).to(trunk.L.device)
        futurefile=np.memmap(root/'checkpoint/weights.raw.future64',dtype='<f4',mode='r',offset=20,shape=(2,4,R))
        self.state[trunk.T]['sum']=torch.from_numpy(np.asarray(futurefile[1]).copy()).to(trunk.T.device)
    @torch.no_grad()
    def step(self,closure=None):
        if closure is not None:raise ValueError('no closure support')
        for group in self.param_groups:
            lr=group['lr']
            for p in group['params']:
                if p.grad is None:continue
                grad=p.grad
                if grad.is_sparse:grad=grad.to_dense()
                acc=self.state[p]['sum']
                acc.addcmul_(grad,grad)
                p.addcdiv_(grad,acc.sqrt().clamp_min_(1.e-15),value=-lr)

class CausalMirrorWriter(nn.Module):
    """ONE S+/S- rank-64 memory: opposite rails are transpose projections.

    Writes only earlier 31 chronological block representations plus 64 local
    words. The target/current or any future labels are never writer inputs.
    Currently recomputes over the 2,048-word window: not infinite persistence.
    """
    def __init__(self):
        super().__init__()
        self.key=nn.Linear(D,R,bias=False)
        self.value=nn.Linear(D,R,bias=False)
        self.query=nn.Linear(D,R,bias=False)
        self.write=nn.Linear(D,1)
        self.keep=nn.Linear(D,1)
        self.read=nn.Linear(2*R,D,bias=False)
        # Zero only final residual gain: initialized candidate exactly matches R7.2.
    def forward(self,h,ctx,embed):
        B=ctx.shape[0]
        past=F.embedding(ctx,embed)
        older=past[:,:-LOCAL].reshape(B,NCHUNKS,64,D).mean(2)
        events=torch.cat((older,past[:,-LOCAL:]),dim=1)
        k=F.normalize(self.key(events),dim=-1)
        v=torch.tanh(self.value(events))
        wg=.04*torch.sigmoid(self.write(events))
        keep=.995 + .004*torch.sigmoid(self.keep(events))
        mem=torch.zeros((B,R,R),dtype=h.dtype,device=h.device)
        for j in range(events.shape[1]):
            mem=keep[:,j,:,None]*mem+wg[:,j,:,None]*(v[:,j,:,None]*k[:,j,None,:])
        q=F.normalize(self.query(h),dim=-1)
        # Symmetric S+ and antisymmetric S- are *views of the same memory*.
        mp=torch.bmm(mem,q.unsqueeze(-1)).squeeze(-1)
        mn=torch.bmm(mem.transpose(-1,-2),q.unsqueeze(-1)).squeeze(-1)
        plus=(mp+mn)*.5;minus=(mp-mn)*.5
        return self.read(torch.cat((plus,minus),dim=-1))

class UnifiedPAL(nn.Module):
    def __init__(self,base,comp,root):
        super().__init__()
        self.trunk=TrainableR6(base,root)
        self.adapter=R7Memory(base,comp)
        # This binds only live read references; no extra duplicate state dicts.
        self.adapter.__dict__['_r73_trunk_ref']=weakref.ref(self.trunk)
        self.mirror=CausalMirrorWriter()
        self.mirror_gain=nn.Parameter(torch.zeros(()))
        self.legacy_future_gain=nn.Parameter(torch.zeros(()))
    def forward(self,examples,longctx,context,pieces,lens,mirror_off=False,adapter_off=False):
        h=self.trunk.hidden(examples,longctx)
        now,features=self.adapter(h,context,pieces,lens,ablate=adapter_off)
        if not mirror_off:
            read=self.mirror(now,context,self.trunk.E)
            now=torch.tanh(torch.atanh(now.clamp(-.999999,.999999))+self.mirror_gain*read)
        return now,features
    def future_loss(self,now,targets,valid):
        # Both sets of horizon offsets share trained R6 rank-64 Q, never targets.
        basis=self.trunk.Q
        offsets=(self.adapter.future_rail + self.legacy_future_gain*self.trunk.T[1:4]) @ basis.T
        future=torch.tanh(torch.atanh(now.clamp(-.999999,.999999))[:,None,:]
                          + self.adapter.future_rail_gain*offsets[None,:,:])
        losses=torch.stack([self.trunk.nll(future[:,j],targets[:,j]) for j in range(3)],dim=1)
        valid=valid.to(losses.dtype)
        return (losses*valid).sum()/valid.sum().clamp_min(1.)


def load_original(root,device,lr_adapter=0.00012,lr_trunk=0.000005,lr_mirror=0.00012):
    root=pathlib.Path(root);base=R6(root);comp=TokenCompositor(root,base)
    model=UnifiedPAL(base,comp,root).to(device)
    old=torch.load(root/'checkpoint/r7_adapter.pt',map_location='cpu',weights_only=False)
    if old['base_sha256']!=sha(root/'checkpoint/weights.raw'):raise ValueError('R6 base mismatch')
    if old['tokenizer_sha256']!=sha(root/'tokenizer/gutenberg_r7.model'):raise ValueError('tokenizer mismatch')
    adapter_optim=torch.optim.AdamW(model.adapter.parameters(),lr=lr_adapter,weight_decay=.0005)
    load_extension(model.adapter,adapter_optim,old)
    for g in adapter_optim.param_groups:g['lr']=lr_adapter
    trunk_optim=LegacyAccumulator(model.trunk,root,lr=lr_trunk)
    mirror_optim=torch.optim.AdamW([*model.mirror.parameters(),model.mirror_gain,model.legacy_future_gain],lr=lr_mirror,weight_decay=.0005)
    return model,(trunk_optim,adapter_optim,mirror_optim),base,comp,old


def save_joint(out,model,optims,old,root,global_steps,rg,metrics=None,train_split_sha256=None):
    root=pathlib.Path(root);out=pathlib.Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    extra_paths=['checkpoint/weights.raw.charv1','checkpoint/weights.raw.charv1.rng',
                 'checkpoint/optimizer.raw.rng']
    legacy_aux=dict(old.get('legacy_aux',{}))
    for path in extra_paths:
        src=root/path
        if src.exists():legacy_aux[path.split('/')[-1]]=src.read_bytes()
    payload={
      'format':'PAL_R7_3_JOINT_V1',
      'model':model.state_dict(),
      'optimizers':{k:opt.state_dict() for k,opt in zip(('legacy_accum','adapter_adamw','mirror_adamw'),optims)},
      'base_legacy_steps':int(old['base_step']),'adapter_legacy_steps':int(old['adapter_updates']),
      'joint_steps':int(global_steps),
      'legacy_r6_sha256':sha(root/'checkpoint/weights.raw') if (root/'checkpoint/weights.raw').exists() else old['base_sha256'],
      'legacy_adapter_sha256':sha(root/'checkpoint/r7_adapter.pt') if (root/'checkpoint/r7_adapter.pt').exists() else old['legacy_adapter_sha256'],
      'tokenizer_sha256':sha(root/'tokenizer/gutenberg_r7.model'),
      'train_split_sha256':train_split_sha256 or sha(root/'corpus/train_temporal.i32'),
      'torch_rng':torch.get_rng_state(),'numpy_rng':np.random.get_state(),
      'python_rng':random.getstate(),'batch_rng_state':rg.bit_generator.state,
      'legacy_aux':legacy_aux,'metrics':metrics or {},
      'architecture':'one R6+R7.2 joint causal graph; rank64 S+/S- mirror write/read, retained R6 accumulator and R7 AdamW state'
    }
    tmp=out.with_suffix('.tmp');torch.save(payload,tmp);os.replace(tmp,out)
    return sha(out)

def restore_joint(root,ckpt,device,lr_adapter,lr_trunk,lr_mirror,data_dir=None,allow_new_corpus=False):
    data=torch.load(ckpt,map_location=device,weights_only=False)
    if (root/'checkpoint/weights.raw').exists():
        model,optims,base,comp,old=load_original(root,device,lr_adapter,lr_trunk,lr_mirror)
    else:
        base=R6(root);comp=TokenCompositor(root,base)
        model=UnifiedPAL(base,comp,root).to(device)
        trunk_optim=LegacyAccumulator(model.trunk,root,lr=lr_trunk,init_from_legacy=False)
        adapter_optim=torch.optim.AdamW(model.adapter.parameters(),lr=lr_adapter,weight_decay=.0005)
        mirror_optim=torch.optim.AdamW([*model.mirror.parameters(),model.mirror_gain,model.legacy_future_gain],lr=lr_mirror,weight_decay=.0005)
        optims=(trunk_optim,adapter_optim,mirror_optim)
        old={'base_step':data['base_legacy_steps'],'adapter_updates':data['adapter_legacy_steps'],
             'base_sha256':data['legacy_r6_sha256'],'legacy_adapter_sha256':data['legacy_adapter_sha256'],
             'legacy_aux':data.get('legacy_aux',{})}
    if data['format']!='PAL_R7_3_JOINT_V1':raise ValueError('checkpoint version')
    if data['tokenizer_sha256']!=sha(root/'tokenizer/gutenberg_r7.model'):raise ValueError('tokenizer changed')
    train_path=pathlib.Path(data_dir or root/'corpus')/'train_temporal.i32'
    if data['train_split_sha256']!=sha(train_path) and not allow_new_corpus:raise ValueError('train split changed; pass --allow-new-corpus for audited continuation')
    model.load_state_dict(data['model'])
    for n,opt in zip(('legacy_accum','adapter_adamw','mirror_adamw'),optims):
        opt.load_state_dict(data['optimizers'][n])
    for opt,lr in zip(optims,(lr_trunk,lr_adapter,lr_mirror)):
        for group in opt.param_groups:group['lr']=lr
    torch.set_rng_state(data['torch_rng'].cpu());np.random.set_state(data['numpy_rng']);random.setstate(data['python_rng'])
    return model,optims,base,comp,old,data

def eval_positions(model,base,comp,corpus,count=128,batch=8):
    """Same deterministic document/split positions before/after joint update."""
    positions=(np.arange(count,dtype=np.int64)*(len(corpus.seq)-1)//count)
    loss=0.;model.eval()
    dev=next(model.parameters()).device
    with torch.no_grad():
        for chunk in np.array_split(positions,int(np.ceil(count/batch))):
            _,ctx,lp,ll,y,tp,tl=batch_data(base,corpus,comp,chunk)
            e=torch.as_tensor(corpus.data[chunk].copy(),device=dev)
            c=torch.as_tensor(corpus.long[chunk].copy(),device=dev)
            pred,_=model(e,c,ctx.to(dev),lp.to(dev),ll.to(dev))
            loss+=float(model.trunk.nll(pred,y.to(dev)).sum())
    return loss/count

def train(args):
    torch.set_num_threads(args.threads)
    device=torch.device('cuda' if (args.device=='auto' and torch.cuda.is_available()) else ('cpu' if args.device=='auto' else args.device))
    if device.type=='cuda' and not torch.cuda.is_available():raise ValueError('CUDA unavailable')
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    root=args.root;start=time.time()
    if args.resume:
        model,opts,base,comp,old,previous=restore_joint(root,args.resume,device,args.adapter_lr,args.trunk_lr,args.mirror_lr,args.data_dir,args.allow_new_corpus)
        updates=int(previous['joint_steps'])
        rng=np.random.default_rng();rng.bit_generator.state=previous['batch_rng_state']
    else:
        model,opts,base,comp,old=load_original(root,device,args.adapter_lr,args.trunk_lr,args.mirror_lr)
        updates=0;rng=np.random.default_rng(args.seed+old['adapter_updates'])
    traincor=Corpus(root,'train',args.data_dir);devcor=Corpus(root,'dev',args.data_dir);testcor=Corpus(root,'test',args.data_dir)
    measured={}
    if args.eval:
        measured['before_dev']=eval_positions(model,base,comp,devcor,args.eval,args.batch)
        measured['before_test']=eval_positions(model,base,comp,testcor,args.eval,args.batch)
        print(json.dumps({'phase':'BASE','joint_step':updates,**measured}),flush=True)
    allopts=list(opts);last={}
    for i in range(args.steps):
        model.train()
        # Strictly causal sampled streams. No document crossing if doc_start exists.
        nseq=max(1,args.batch//args.seq_len)
        starts=rng.integers(WORD_CONTEXT,len(traincor.seq)-args.seq_len,size=nseq,dtype=np.int64)
        if traincor.doc_start is not None:
            for j,s in enumerate(starts):
                if traincor.doc_start[s]!=traincor.doc_start[s+args.seq_len-1]:
                    for tries in range(2048):
                        candidate=int(rng.integers(WORD_CONTEXT,len(traincor.seq)-args.seq_len))
                        if traincor.doc_start[candidate]==traincor.doc_start[candidate+args.seq_len-1]:
                            starts[j]=candidate;break
                    else:raise RuntimeError('unable to sample valid document span')
        positions=(starts[:,None]+np.arange(args.seq_len)[None,:]).reshape(-1)
        _,ctx,lp,ll,y,tp,tl=batch_data(base,traincor,comp,positions)
        e=torch.as_tensor(traincor.data[positions].copy(),device=device)
        long=torch.as_tensor(traincor.long[positions].copy(),device=device)
        ctx,lp,ll,y,tp,tl=(x.to(device) for x in (ctx,lp,ll,y,tp,tl))
        now,_=model(e,long,ctx,lp,ll)
        word=model.trunk.nll(now,y).mean()
        sub=model.adapter.subword_loss(now,tp,tl) if args.subword_weight else word*0
        fy,fmask=future_labels(traincor,positions)
        fut=model.future_loss(now,fy.to(device),fmask.to(device)) if args.future_weight else word*0
        tail=ctx[:,-LOCAL:];known=tail>1
        # Do not detach: alignment is also an R6<->R7 differentiable bridge.
        composed=model.adapter.compose(lp,ll)
        target=F.embedding(tail,model.trunk.E)
        align=F.mse_loss(composed[known],target[known]) if bool(known.any()) else word*0
        if args.stream_weight:
            st,sl=stream_targets(traincor,comp,base.vocab,positions[:args.stream_batch],words=args.stream_words)
            stream=model.adapter.stream_loss(now[:args.stream_batch],st.to(device),sl.to(device))
        else:stream=word*0
        loss=word+args.subword_weight*sub+args.future_weight*fut+args.align_weight*align+args.stream_weight*stream
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite loss')
        for opt in allopts:opt.zero_grad(set_to_none=True)
        loss.backward()
        norms={name:float(getattr(model.trunk,name).grad.norm()) if getattr(model.trunk,name).grad is not None else 0.
               for name in ('E','H','P','M','Q','NOW','W','B','L')}
        if not any(norms[k]>0 for k in ('P','M','Q')):raise AssertionError('no mirror gradient')
        torch.nn.utils.clip_grad_norm_(model.parameters(),args.clip)
        for opt in allopts:opt.step()
        last={'loss':float(loss.detach()),'word':float(word.detach()),'future':float(fut.detach()),'subword':float(sub.detach()),'stream':float(stream.detach()),'grad_norms':norms}
        if (i+1)%args.log_every==0 or i+1==args.steps:
            print(json.dumps({'phase':'TRAIN','joint_step':updates+i+1,**last,'elapsed_s':round(time.time()-start,1)}),flush=True)
    if args.eval:
        measured['after_dev']=eval_positions(model,base,comp,devcor,args.eval,args.batch)
        measured['after_test']=eval_positions(model,base,comp,testcor,args.eval,args.batch)
        print(json.dumps({'phase':'EVAL',**measured}),flush=True)
    result={'steps_executed':args.steps,'joint_step':updates+args.steps,'metrics':measured,'last_train':last,
            'trunk_lr':args.trunk_lr,'adapter_lr':args.adapter_lr,'mirror_lr':args.mirror_lr,
            'elapsed_s':round(time.time()-start,2),'device':str(device),'cuda_verified':device.type=='cuda',
            'legacy_optimizer':'PAL6 squared-gradient accumulator, loaded exact',
            'pretrained_legacy_updates':old['adapter_updates'],
            'source_corpus':str(args.data_dir or root/'corpus'),
            'train_split_sha256':sha(pathlib.Path(args.data_dir or root/'corpus')/'train_temporal.i32')}
    ck_hash=save_joint(args.output,model,allopts,old,root,updates+args.steps,rng,metrics=result,train_split_sha256=result['train_split_sha256'])
    result['active_checkpoint_sha256']=ck_hash
    report=root/'reports/R7p3_joint.json';report.parent.mkdir(exist_ok=True);report.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'phase':'DONE',**result}),flush=True)
    return result

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=pathlib.Path,default=pathlib.Path('.'))
    ap.add_argument('--resume',type=pathlib.Path)
    ap.add_argument('--data-dir',type=pathlib.Path)
    ap.add_argument('--allow-new-corpus',action='store_true')
    ap.add_argument('--output',type=pathlib.Path,default=pathlib.Path('checkpoint/pal_active.pt'))
    ap.add_argument('--steps',type=int,default=8)
    ap.add_argument('--batch',type=int,default=8)
    ap.add_argument('--seq-len',type=int,default=4)
    ap.add_argument('--eval',type=int,default=128)
    ap.add_argument('--device',choices=('auto','cpu','cuda'),default='auto')
    ap.add_argument('--trunk-lr',type=float,default=0.000005)
    ap.add_argument('--adapter-lr',type=float,default=0.00012)
    ap.add_argument('--mirror-lr',type=float,default=0.00012)
    ap.add_argument('--subword-weight',type=float,default=.08)
    ap.add_argument('--future-weight',type=float,default=.03)
    ap.add_argument('--align-weight',type=float,default=.1)
    ap.add_argument('--stream-weight',type=float,default=.2)
    ap.add_argument('--stream-batch',type=int,default=4)
    ap.add_argument('--stream-words',type=int,default=6)
    ap.add_argument('--clip',type=float,default=1.)
    ap.add_argument('--seed',type=int,default=20261111)
    ap.add_argument('--threads',type=int,default=2)
    ap.add_argument('--log-every',type=int,default=1)
    args=ap.parse_args()
    train(args)
