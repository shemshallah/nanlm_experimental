#!/usr/bin/env python3
"""PAL R7.4: single-lineage, causal cross-window recurrent S+/S- memory.

Restores the complete R7.3 trained core (including ALL optimizer states). Adds a
rank64 memory for positions -8192..-2049, then carries it forward as words
leave the existing 2048-token exact window. No future/current targets enter.
An outer-memory state is ephemeral runtime state, NOT a second checkpoint.
"""
from __future__ import annotations
import argparse, hashlib, json, os, pathlib, random, time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from r7_core import R6, Corpus, D, R
from r7_memory import TokenCompositor, WORD_CONTEXT, LOCAL
from r73_joint import UnifiedPAL, LegacyAccumulator, restore_joint, eval_positions
from train_r7 import batch_data, future_labels, stream_targets, sha

OUTER_CONTEXT = 8192
OLDER = OUTER_CONTEXT-WORD_CONTEXT
BLOCK = 64
assert OLDER % BLOCK == 0

class CrossWindowMirror(nn.Module):
    """One rank64 physical matrix; symmetric/antisymmetric reads share it.

    For the initial cache, process the 96 non-overlapping blocks *older* than
    the exact 2048-token window. The cached tensor is carried forward via a
    differentiable update of only the token leaving the exact window.
    """
    def __init__(self):
        super().__init__()
        self.key=nn.Linear(D,R,bias=False)
        self.value=nn.Linear(D,R,bias=False)
        self.write=nn.Linear(D,1)
        self.keep=nn.Linear(D,1)
        self.query=nn.Linear(D,R,bias=False)
        self.read=nn.Linear(2*R,D,bias=False)

    def _update(self, state, event, strength=1.):
        key=F.normalize(self.key(event),dim=-1)
        value=torch.tanh(self.value(event))
        write=(.035*strength)*torch.sigmoid(self.write(event))
        keep=.965 + .034*torch.sigmoid(self.keep(event))
        return keep[...,None]*state + write[...,None]*(value.unsqueeze(-1)*key.unsqueeze(-2))

    def initial(self, history, embeddings):
        """[B,6144] causal prior words, excluding all words in exact window."""
        if history.ndim!=2 or history.shape[1]!=OLDER:
            raise ValueError(f'expected [batch,{OLDER}] historical token IDs')
        b=history.shape[0]
        mask=(history!=1).reshape(b,OLDER//BLOCK,BLOCK).to(embeddings.dtype)
        tokens=F.embedding(history.long(),embeddings).reshape(b,OLDER//BLOCK,BLOCK,D)
        chunks=(tokens*mask.unsqueeze(-1)).sum(2)/mask.sum(2).clamp_min(1).unsqueeze(-1)
        state=embeddings.new_zeros(b,R,R)
        for j in range(chunks.shape[1]):
            # Before a document's start, BOS padding is NOT a real memory event.
            candidate=self._update(state,chunks[:,j])
            active=mask[:,j].sum(1)>0
            state=torch.where(active[:,None,None],candidate,state)
        return state

    def advance(self, state, leaving, embeddings):
        """One token crosses from the exact 2048 window into older memory."""
        e=F.embedding(leaving.long(),embeddings)
        candidate=self._update(state,e,strength=1./BLOCK)
        return torch.where((leaving!=1)[:,None,None],candidate,state)

    def read_rails(self,state,now):
        q=F.normalize(self.query(now),dim=-1)
        forward=torch.bmm(state,q.unsqueeze(-1)).squeeze(-1)
        reverse=torch.bmm(state.transpose(-1,-2),q.unsqueeze(-1)).squeeze(-1)
        return self.read(torch.cat(((forward+reverse)*.5,(forward-reverse)*.5),-1))

class PALR74(nn.Module):
    """The R7.3 core is unchanged; one trainable older-memory residual is added."""
    def __init__(self, core):
        super().__init__()
        self.core=core
        self.outer=CrossWindowMirror()
        # Zero-gain migration matches original R7.3 at every NOW prediction.
        self.outer_gain=nn.Parameter(torch.zeros(()))
    @property
    def trunk(self):return self.core.trunk
    @property
    def adapter(self):return self.core.adapter
    def initial(self,history):return self.outer.initial(history,self.trunk.E)
    def advance(self,state,leaving):return self.outer.advance(state,leaving,self.trunk.E)
    def forward(self,ex,long,ctx,pieces,lens,state=None,outer_off=False,mirror_off=False):
        now,features=self.core(ex,long,ctx,pieces,lens,mirror_off=mirror_off)
        if state is not None and not outer_off:
            read=self.outer.read_rails(state,now)
            now=torch.tanh(torch.atanh(now.clamp(-.999999,.999999))+self.outer_gain*read)
        return now,features


def build_optimizers(model, lr_trunk, lr_adapter, lr_mirror, lr_outer, init_trunk=False, root=None):
    tr=LegacyAccumulator(model.trunk,root,lr=lr_trunk,init_from_legacy=init_trunk)
    ad=torch.optim.AdamW(model.adapter.parameters(),lr=lr_adapter,weight_decay=.0005)
    mi=torch.optim.AdamW([*model.core.mirror.parameters(),model.core.mirror_gain,model.core.legacy_future_gain],lr=lr_mirror,weight_decay=.0005)
    ou=torch.optim.AdamW([*model.outer.parameters(),model.outer_gain],lr=lr_outer,weight_decay=.0005)
    return tr,ad,mi,ou

OPT_NAMES=('legacy_accum','adapter_adamw','mirror_adamw','outer_adamw')

def restore(root, checkpoint, device, lr_trunk=2e-6, lr_adapter=6e-5,lr_mirror=1e-4,lr_outer=8e-5):
    root=pathlib.Path(root); checkpoint=pathlib.Path(checkpoint)
    data=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if data['format']=='PAL_R7_3_JOINT_V1':
        core,previous_opt,base,comp,old,previous=restore_joint(root,checkpoint,device,lr_adapter,lr_trunk,lr_mirror)
        model=PALR74(core).to(device)
        outer_opt=torch.optim.AdamW([*model.outer.parameters(),model.outer_gain],lr=lr_outer,weight_decay=.0005)
        return model,(*previous_opt,outer_opt),base,comp,old,previous
    if data['format']!='PAL_R7_4_CROSSWINDOW_V1':
        raise ValueError('unknown checkpoint format '+str(data.get('format')))
    if sha(root/'tokenizer/gutenberg_r7.model')!=data['tokenizer_sha256']:
        raise ValueError('tokenizer changed since checkpoint')
    base=R6(root); comp=TokenCompositor(root,base)
    core=UnifiedPAL(base,comp,root)
    model=PALR74(core).to(device)
    optimizers=build_optimizers(model,lr_trunk,lr_adapter,lr_mirror,lr_outer)
    model.load_state_dict(data['model'],strict=True)
    for n,opt in zip(OPT_NAMES,optimizers):opt.load_state_dict(data['optimizers'][n])
    for opt,lr in zip(optimizers,(lr_trunk,lr_adapter,lr_mirror,lr_outer)):
        for group in opt.param_groups:group['lr']=lr
    torch.set_rng_state(data['torch_rng']);np.random.set_state(data['numpy_rng']);random.setstate(data['python_rng'])
    old={'base_step':data['base_legacy_steps'],'adapter_updates':data['adapter_legacy_steps'],
         'base_sha256':data['legacy_r6_sha256'],'legacy_adapter_sha256':data['legacy_adapter_sha256'],
         'legacy_aux':data.get('legacy_aux',{})}
    return model,optimizers,base,comp,old,data


def evaluate(model,base,comp,corpus,count=256,batch=4,ablate=False):
    positions=np.arange(count,dtype=np.int64)*(len(corpus.seq)-1)//count
    device=next(model.parameters()).device
    total=0.
    model.eval()
    with torch.no_grad():
        for p in np.array_split(positions,int(np.ceil(count/batch))):
            _,ctx,pieces,lens,targets,_,_=batch_data(base,corpus,comp,p)
            historical=corpus.get(p,OUTER_CONTEXT)[0][:,:OLDER].copy()
            past=model.initial(torch.from_numpy(historical).to(device))
            now,_=model(torch.as_tensor(corpus.data[p].copy(),device=device),
                        torch.as_tensor(corpus.long[p].copy(),device=device),
                        ctx.to(device),pieces.to(device),lens.to(device),
                        state=past,outer_off=ablate)
            total+=float(model.trunk.nll(now,targets.to(device)).sum())
    return total/count


def save_checkpoint(out,model,optimizers,prior,old,root,updates,rng,metrics):
    root=pathlib.Path(root);out=pathlib.Path(out)
    payload={
       'format':'PAL_R7_4_CROSSWINDOW_V1','model':model.state_dict(),
       'optimizers':{k:o.state_dict() for k,o in zip(OPT_NAMES,optimizers)},
       'base_legacy_steps':int(old['base_step']), 'adapter_legacy_steps':int(old['adapter_updates']),
       'joint_steps':int(updates),'legacy_r6_sha256':prior['legacy_r6_sha256'],
       'legacy_adapter_sha256':prior['legacy_adapter_sha256'],
       'tokenizer_sha256':sha(root/'tokenizer/gutenberg_r7.model'),
       'train_split_sha256':sha(root/'corpus/train_temporal.i32'),
       'torch_rng':torch.get_rng_state(),'numpy_rng':np.random.get_state(),
       'python_rng':random.getstate(),'batch_rng_state':rng.bit_generator.state,
       'legacy_aux':prior.get('legacy_aux',old.get('legacy_aux',{})),
       'architecture':'R7.3 full joint core + one causal rank64 S+/S- 6144-token older-memory recurrent carry',
       'metrics':metrics}
    out.parent.mkdir(parents=True,exist_ok=True)
    tmp=out.with_suffix('.tmp');torch.save(payload,tmp);os.replace(tmp,out)
    return sha(out)


def train(args):
    torch.set_num_threads(args.threads)
    device=torch.device('cuda' if args.device=='auto' and torch.cuda.is_available() else ('cpu' if args.device=='auto' else args.device))
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    root=args.root;start=time.monotonic()
    model,opts,base,comp,old,prior=restore(root,args.resume or root/'checkpoint/pal_active.pt',device,
        args.trunk_lr,args.adapter_lr,args.mirror_lr,args.outer_lr)
    if sha(root/'corpus/train_temporal.i32')!=prior['train_split_sha256']:
        raise ValueError('train corpus changed without approved migration')
    current=int(prior['joint_steps'])
    rng=np.random.default_rng();rng.bit_generator.state=prior['batch_rng_state']
    tc=Corpus(root,'train');dev=Corpus(root,'dev');test=Corpus(root,'test')
    measurements={'input_joint_step':current,'R73_pretrained_step':2400}
    if args.eval:
        for split,cor in (('dev',dev),('test',test)):
            measurements[f'before_{split}']=evaluate(model,base,comp,cor,args.eval,args.eval_batch)
            measurements[f'before_{split}_outer_off']=evaluate(model,base,comp,cor,args.eval,args.eval_batch,True)
        print(json.dumps({'phase':'BASE',**measurements}),flush=True)
    for update in range(args.steps):
        model.train()
        # B streams, each S predictions, all from the same document.
        starts=rng.integers(WORD_CONTEXT,len(tc.seq)-args.seq_len,size=args.streams,dtype=np.int64)
        if tc.doc_start is not None:
            for j,s in enumerate(starts):
                if tc.doc_start[s]!=tc.doc_start[s+args.seq_len-1]:
                    for trial in range(512):
                        nxt=int(rng.integers(WORD_CONTEXT,len(tc.seq)-args.seq_len))
                        if tc.doc_start[nxt]==tc.doc_start[nxt+args.seq_len-1]:
                            starts[j]=nxt;break
                    else:raise RuntimeError('no single-document stream available')
        history=tc.get(starts,OUTER_CONTEXT)[0][:,:OLDER].copy()
        state=model.initial(torch.as_tensor(history,device=device))
        words=[];subs=[];futures=[];streams=[];aligns=[]
        for t in range(args.seq_len):
            p=starts+t
            _,ctx,pieces,lens,targets,tp,tl=batch_data(base,tc,comp,p)
            e=torch.as_tensor(tc.data[p].copy(),device=device)
            long=torch.as_tensor(tc.long[p].copy(),device=device)
            ctx,pieces,lens,targets,tp,tl=(x.to(device) for x in (ctx,pieces,lens,targets,tp,tl))
            now,_=model(e,long,ctx,pieces,lens,state)
            words.append(model.trunk.nll(now,targets).mean())
            if args.subword_weight:subs.append(model.adapter.subword_loss(now,tp,tl))
            if args.future_weight:
                fy,valid=future_labels(tc,p)
                futures.append(model.core.future_loss(now,fy.to(device),valid.to(device)))
            if args.align_weight:
                known=ctx[:,-LOCAL:]>1
                composed=model.adapter.compose(pieces,lens)
                target=F.embedding(ctx[:,-LOCAL:],model.trunk.E)
                aligns.append(F.mse_loss(composed[known],target[known]) if bool(known.any()) else now.sum()*0)
            if args.stream_weight:
                st,sl=stream_targets(tc,comp,base.vocab,p,words=args.stream_words)
                streams.append(model.adapter.stream_loss(now,st.to(device),sl.to(device)))
            # At t+1, only the OLDEST word in exact window leaves it.
            # That word is at position p-2048, strictly BEFORE target p.
            if t+1<args.seq_len:state=model.advance(state,ctx[:,0])
        word=torch.stack(words).mean()
        sub=torch.stack(subs).mean() if subs else word*0
        future=torch.stack(futures).mean() if futures else word*0
        align=torch.stack(aligns).mean() if aligns else word*0
        stream=torch.stack(streams).mean() if streams else word*0
        loss=word+args.subword_weight*sub+args.future_weight*future+args.align_weight*align+args.stream_weight*stream
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite training objective')
        for op in opts:op.zero_grad(set_to_none=True)
        loss.backward()
        assert model.trunk.P.grad is not None and model.trunk.P.grad.abs().sum()>0
        outer_gain_grad=float(model.outer_gain.grad.abs().item())
        assert outer_gain_grad>0, 'outer memory has no causal learning path'
        torch.nn.utils.clip_grad_norm_(model.parameters(),args.clip)
        for op in opts:op.step()
        last={'loss':float(loss.detach()),'word':float(word.detach()),'stream':float(stream.detach()),
              'outer_gain':float(model.outer_gain.detach()),'outer_gain_grad':outer_gain_grad}
        if (update+1)%args.log_every==0 or update+1==args.steps:
            print(json.dumps({'phase':'TRAIN','joint_step':current+update+1,**last,
                     'elapsed_s':round(time.monotonic()-start,1)}),flush=True)
    if args.eval:
        for split,cor in (('dev',dev),('test',test)):
            measurements[f'after_{split}']=evaluate(model,base,comp,cor,args.eval,args.eval_batch)
            measurements[f'after_{split}_outer_off']=evaluate(model,base,comp,cor,args.eval,args.eval_batch,True)
    summary={'steps_executed':args.steps,'joint_step':current+args.steps,
             'metrics':measurements,'last_train':last if args.steps else {},
             'wall_time_s':round(time.monotonic()-start,2), 'device':str(device),
             'cuda_verified':device.type=='cuda', 'older_window_words':OLDER,
             'exact_context_words':WORD_CONTEXT, 'optimizer_histories':list(OPT_NAMES)}
    checksum=save_checkpoint(args.output,model,opts,prior,old,root,current+args.steps,rng,summary)
    summary['checkpoint_sha256']=checksum
    (root/'reports').mkdir(exist_ok=True)
    (root/'reports/R7p4_crosswindow.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({'phase':'DONE',**summary}),flush=True)
    return summary

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=pathlib.Path,default=pathlib.Path('.'))
    ap.add_argument('--resume',type=pathlib.Path)
    ap.add_argument('--output',type=pathlib.Path,default=pathlib.Path('checkpoint/pal_active.pt'))
    ap.add_argument('--steps',type=int,default=8)
    ap.add_argument('--streams',type=int,default=2)
    ap.add_argument('--seq-len',type=int,default=4)
    ap.add_argument('--eval',type=int,default=256)
    ap.add_argument('--eval-batch',type=int,default=4)
    ap.add_argument('--device',choices=('auto','cpu','cuda'),default='auto')
    ap.add_argument('--trunk-lr',type=float,default=2e-6)
    ap.add_argument('--adapter-lr',type=float,default=6e-5)
    ap.add_argument('--mirror-lr',type=float,default=1e-4)
    ap.add_argument('--outer-lr',type=float,default=8e-5)
    ap.add_argument('--subword-weight',type=float,default=.08)
    ap.add_argument('--future-weight',type=float,default=.03)
    ap.add_argument('--align-weight',type=float,default=.1)
    ap.add_argument('--stream-weight',type=float,default=.2)
    ap.add_argument('--stream-words',type=int,default=6)
    ap.add_argument('--clip',type=float,default=1.)
    ap.add_argument('--seed',type=int,default=20261111)
    ap.add_argument('--threads',type=int,default=2)
    ap.add_argument('--log-every',type=int,default=10)
    train(ap.parse_args())
