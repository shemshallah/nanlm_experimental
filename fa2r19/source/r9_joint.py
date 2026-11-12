#!/usr/bin/env python3
"""R9 joint fine-tuning of the ONE R7 rank64 causal LM, with exact warm-start.

No new prediction model: original mirrored temporal vectors, word embeddings,
32k hierarchical decoder, and R7 ordered temporal context all share gradients.
Token targets are strictly causal; lookahead targets are not inputs.
"""
import argparse,copy,hashlib,json,os,pathlib,random,time
import numpy as np, torch
from torch import nn
import torch.nn.functional as F
from r7_core import R6,Corpus,D,R,LONGF
from r7_memory import R7Memory,TokenCompositor
from train_r7 import batch_data,sha
from r9_sequence import R9Sequence

class JointCore(nn.Module):
    def __init__(self,r6):
        super().__init__()
        for name in ('E','H','P','M','Q','NOW','L','W','B','T'):
            self.register_parameter(name,nn.Parameter(torch.from_numpy(np.array(getattr(r6,name),copy=True)).float()))
        self.register_buffer('heads',torch.from_numpy(r6.heads.astype('int64')))
        self.register_buffer('pnodes',r6.pnodest)
        self.register_buffer('pbits',r6.pbitst)
        self.register_buffer('pmask',r6.pmaskt)
    def hidden(self,examples,longctx):
        # Same shared NOW representation as the original R6; gradient flows into
        # ALL original tensors, including the mirror pair rails P and M.
        e=examples.long();ha=self.heads[e[:,2],0];hb=self.heads[e[:,1],1]
        hb=torch.where(ha==hb,(hb+1)&255,hb)
        lo=torch.minimum(ha,hb);hi=torch.maximum(ha,hb)
        pair=(lo*(511-lo)//2+hi-lo-1).long()
        sign=torch.where(ha<hb,1.,-1.).to(self.E.dtype)
        f=.4*sign[:,None]*(self.P[pair]-.2*self.M[pair])
        z=self.NOW+f@self.Q.T+longctx.float()@self.L.T/32.
        z=z+.14*self.E[e[:,0]]+.28*self.E[e[:,1]]+.58*self.E[e[:,2]]
        z=z+.18*(self.H[ha]-self.H[hb])
        z=z+.04*(self.H[(ha+1)&255]+self.H[(ha+2)&255]-self.H[(hb+255)&255]-self.H[(hb+254)&255])
        return torch.tanh(z)
    def hidden_horizon(self,examples,longctx,horizon):
        assert horizon in (1,2,3)
        e=examples.long();ha=self.heads[e[:,2],0];hb=self.heads[e[:,1],1]
        hb=torch.where(ha==hb,(hb+1)&255,hb)
        lo=torch.minimum(ha,hb);hi=torch.maximum(ha,hb)
        pair=(lo*(511-lo)//2+hi-lo-1).long()
        sign=torch.where(ha<hb,1.,-1.).to(self.E.dtype)
        beta=0.25*horizon
        # Same canonical P/M rails and the original learned 4x64 temporal offsets.
        f=.4*(sign[:,None]*((1-beta)*(self.P[pair]-.2*self.M[pair])+
                             beta*(self.M[pair]-.2*self.P[pair]))+self.T[horizon])
        z=self.NOW+f@self.Q.T+longctx.float()@self.L.T/32.
        z=z+.14*self.E[e[:,0]]+.28*self.E[e[:,1]]+.58*self.E[e[:,2]]
        z=z+.18*(self.H[ha]-self.H[hb])
        z=z+.04*(self.H[(ha+1)&255]+self.H[(ha+2)&255]-self.H[(hb+255)&255]-self.H[(hb+254)&255])
        return torch.tanh(z)
    def nll(self,h,y):
        ix=self.pnodes[y.long()];bits=self.pbits[y.long()];mask=self.pmask[y.long()]
        logits=(self.W[ix]*h[:,None,:]).sum(dim=-1)+self.B[ix]
        return (F.binary_cross_entropy_with_logits(logits,bits,reduction='none')*mask).sum(-1)

class JointR9(nn.Module):
    def __init__(self,r6,comp):
        super().__init__();self.core=JointCore(r6);self.memory=R7Memory(r6,comp);self.sequence=R9Sequence()
    def forward(self,e,lc,ctx,lp,ll,ablate=False):
        h=self.core.hidden(e,lc)
        # Use the live SAME original embeddings in temporal memory.
        new,_=self.memory(h,ctx,lp,ll,ablate=ablate,word_embedding_override=self.core.E)
        if not ablate:new=self.sequence(new,ctx,lp,ll,self.core.E,self.memory.compose)
        return new

def batch(r6,cor,comp,indices):
    idx=np.asarray(indices,dtype=np.int64)
    past,past_indices=cor.get(idx)
    lp,ll=comp.local_pieces(cor,past_indices[:,-64:],past[:,-64:],r6.vocab)
    words=comp.strings(cor,idx,r6.vocab)
    tp,tl=comp.encode_words(words)
    return (torch.from_numpy(np.array(cor.data[idx],copy=True).astype('int64')),
            torch.from_numpy(np.array(cor.long[idx],copy=True).astype('float32')),
            torch.from_numpy(past.copy().astype('int64')),lp,ll,tp,tl)

def eval_exact(model,r6,cor,comp,n=512,batchsize=24,base=False):
    pos=(np.arange(n,dtype=np.int64)*(len(cor.seq)-1)//n)
    total=0.;accuracy=0;sum_loss=[]
    model.eval()
    with torch.no_grad():
        for ids in np.array_split(pos,max(1,(len(pos)+batchsize-1)//batchsize)):
            e,lc,ctx,lp,ll,tp,tl=batch(r6,cor,comp,ids)
            new=model(e,lc,ctx,lp,ll,ablate=base)
            loss=model.core.nll(new,e[:,3]);total+=loss.sum().item()
            sum_loss.extend(loss.cpu().tolist())
    ids=cor.seq[pos]
    losses=np.asarray(sum_loss);is_added=ids>=4501;is_unk=ids==0
    return {'nll':total/n,'n':n,'per_word_loss':sum_loss,'original_nll':float(losses[(ids>0)&(ids<4501)].mean()),'added_nll':float(losses[is_added].mean()),'unknown_nll':float(losses[is_unk].mean()),'n_added':int(is_added.sum())}

def eval_temporal(model,cor,n=512,batchsize=24):
    model.eval();positions=(np.arange(n,dtype=np.int64)*(len(cor.seq)-1)//n)
    results={}
    with torch.no_grad():
        for horizon in (1,2,3):
            total=0.;count=0
            for ix in np.array_split(positions,max(1,(len(positions)+batchsize-1)//batchsize)):
                e=torch.from_numpy(np.array(cor.data[ix],copy=True).astype('int64'))
                lc=torch.from_numpy(np.array(cor.long[ix],copy=True).astype('float32'))
                mask=e[:,3+horizon]!=1
                if not mask.any():continue
                h=model.core.hidden_horizon(e,lc,horizon)
                losses=model.core.nll(h[mask],e[mask,3+horizon])
                total+=float(losses.sum());count+=int(mask.sum())
            results[str(horizon)]={'nll':total/max(1,count),'n':count}
    return results

def main(args):
    t=time.time();torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
    root=pathlib.Path(args.root)
    active_path=root/'checkpoint/r9_active.pal9'
    initial_path=pathlib.Path(args.resume) if args.resume else active_path
    saved=torch.load(initial_path,map_location='cpu',weights_only=False) if initial_path.exists() else None
    if saved is not None:
        assert saved['format']=='PAL9_JOINT_SINGLE_V1'
        frozen={name:saved['model'][f'core.{name}'] for name in ('E','H','P','M','Q','NOW','W','B','L')}
        if 'core.T' in saved['model']:
            frozen['T']=saved['model']['core.T']
        else:
            blob=saved['legacy_aux_bytes']['weights.raw.future64']
            frozen['T']=np.frombuffer(blob,dtype='<f4',offset=20,count=4*R).copy().reshape(4,R)
        r6=R6(root,weights_state=frozen);comp=TokenCompositor(root,r6)
        model=JointR9(r6,comp);missing=model.load_state_dict(saved['model'],strict=False)
        assert set(missing.missing_keys).issubset({'core.T'}) and not missing.unexpected_keys
        source_sha=saved['source_sha256'];source_adapter_sha=saved['source_adapter_sha256']
        source_core_steps=saved['source_core_steps'];source_adapter_steps=saved['source_adapter_steps']
    else:
        r6=R6(root);comp=TokenCompositor(root,r6)
        model=JointR9(r6,comp)
        prev=torch.load(root/'checkpoint/r7_adapter.pt',map_location='cpu',weights_only=False)
        assert prev['base_sha256']==sha(root/'checkpoint/weights.raw')
        assert prev['tokenizer_sha256']==sha(root/'tokenizer/gutenberg_r7.model')
        model.memory.load_state_dict(prev['adapter'])
        source_sha=prev['base_sha256'];source_adapter_sha=sha(root/'checkpoint/r7_adapter.pt')
        source_core_steps=int(prev['base_step']);source_adapter_steps=int(prev['adapter_updates'])
    dev=Corpus(root,'dev');train=Corpus(root,'train');test=Corpus(root,'test')
    e,lc,ctx,lp,ll,tp,tl=batch(r6,dev,comp,np.array([2048,4096,8192],dtype=np.int64))
    with torch.no_grad():
        delta=(model.core.hidden(e,lc)-torch.from_numpy(r6.hidden(e.numpy(),lc.numpy()))).abs().max().item()
    if delta>2e-5:raise ValueError(f'Core migration not equivalent: {delta}')
    print(json.dumps({'phase':'MIGRATE','max_hidden_absolute_error':delta,
                      'R7_steps':source_adapter_steps,'params':sum(x.numel() for x in model.parameters()),
                      'R9_source_updates':0 if saved is None else saved['r9_updates']}),flush=True)
    # One unified optimizer: all original temporal and NOW tensors + shared ordered memory.
    opt=torch.optim.AdamW([
        {'params':[model.core.E,model.core.H,model.core.P,model.core.M], 'lr':args.lr_core},
        {'params':[model.core.Q,model.core.NOW,model.core.L,model.core.T], 'lr':args.lr_core*2},
        {'params':[model.core.W,model.core.B], 'lr':args.lr_decoder},
        {'params':list(model.memory.parameters()), 'lr':args.lr_adapter},
        {'params':list(model.sequence.parameters()), 'lr':args.lr_sequence},
    ],weight_decay=0.00005)
    # Strong causal dataset invariants.
    assert np.all(ctx.numpy()[:,-1]==dev.seq[np.array([2048,4096,8192])-1]);
    dev_before=eval_exact(model,r6,dev,comp,args.eval,args.eval_batch)
    temporal_before=eval_temporal(model,dev,args.eval,args.eval_batch)
    print(json.dumps({'phase':'BASE_DEV','nll':dev_before['nll'],'eval_n':args.eval}),flush=True)
    if saved is not None:
        # Earlier PAL9 beta had no core.T optimizer group; it had zero R9 updates.
        if 'core.T' in saved['model']:
            opt.load_state_dict(saved['optimizer'])
        elif int(saved['r9_updates'])!=0:
            raise ValueError('nonzero-step checkpoint missing temporal offset optimizer')
        start=int(saved['r9_updates'])
        torch.set_rng_state(saved['torch_rng']);np.random.set_state(saved['numpy_rng'])
        random.setstate(saved['python_rng']);rg=np.random.default_rng()
        rg.bit_generator.state=saved['batch_rng']
    else:start=0;rg=np.random.default_rng(args.seed)
    new_pos=np.flatnonzero(train.seq>=4501)
    if args.rare_fraction>0:print(json.dumps({'phase':'CURRICULUM','rare_fraction':args.rare_fraction,'new_positions':len(new_pos),'total_positions':len(train.seq)}),flush=True)
    order=list(range(args.steps));train_trace=[]
    for k in order:
        model.train()
        indices=rg.integers(2048,len(train.seq),size=args.batch,dtype=np.int64)
        if args.rare_fraction>0:
            selection=rg.random(args.batch)<args.rare_fraction
            indices[selection]=rg.choice(new_pos,size=int(selection.sum()),replace=True)
        e,lc,ctx,lp,ll,tp,tl=batch(r6,train,comp,indices)
        h=model(e,lc,ctx,lp,ll)
        individual=model.core.nll(h,e[:,3])
        if args.rare_fraction and args.importance_weights:
            qrare=len(new_pos)/len(train.seq); frac=args.rare_fraction
            w=torch.where(e[:,3]>=4501,qrare/(frac+(1-frac)*qrare),1./(1-frac))
            l_now=(individual*w).mean()
        else: l_now=individual.mean()
        l_char=model.memory.subword_loss(h,tp,tl)
        # Maintain original auxiliary spelling path, on the SAME trained state.
        loss=l_now+args.subword_weight*l_char
        if args.future_weight:
            fsum=0.
            for horizon in (1,2,3):
                mask=e[:,3+horizon]!=1
                if mask.any():
                    fh=model.core.hidden_horizon(e,lc,horizon)
                    fsum=fsum+model.core.nll(fh[mask],e[mask,3+horizon]).mean()/3
            loss=loss+args.future_weight*fsum
        if not torch.isfinite(loss):raise FloatingPointError(f'nonfinite loss at {start+k}')
        opt.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
        train_trace.append((float(l_now.detach()),float(l_char.detach())))
        if (k+1)%args.report_every==0 or k==args.steps-1:
            last=np.array(train_trace[-args.report_every:]);
            report={'phase':'TRAIN','r9_updates':start+k+1,'word_loss':float(last[:,0].mean()),'subword_loss':float(last[:,1].mean()),'elapsed_s':round(time.time()-t,2)}
            if args.dev_every and (k+1)%args.dev_every==0:
                report['dev_nll']=eval_exact(model,r6,dev,comp,args.eval,args.eval_batch)['nll']
            print(json.dumps(report),flush=True)
    dev_after=eval_exact(model,r6,dev,comp,args.eval,args.eval_batch)
    temporal_after=eval_temporal(model,dev,args.eval,args.eval_batch)
    # Test only after architecture/hyperparameters have been selected, not to tune.
    test_before=None;test_after=None
    if args.test:
        test_after=eval_exact(model,r6,test,comp,args.eval,args.eval_batch)
    out=active_path
    info={'format':'PAL9_JOINT_SINGLE_V1','source_sha256':source_sha,
          'source_adapter_sha256':source_adapter_sha,
          'source_core_steps':source_core_steps,'source_adapter_steps':source_adapter_steps,
          'r9_updates':start+args.steps,'model':model.state_dict(),
          'optimizer':opt.state_dict(),'torch_rng':torch.get_rng_state(),
          'numpy_rng':np.random.get_state(),'python_rng':random.getstate(),
          'batch_rng':rg.bit_generator.state,'hyperparameters':vars(args),
          'tokenizer_sha256':sha(root/'tokenizer/gutenberg_r7.model'),
          'train_sha256':sha(root/'corpus/train_temporal.i32'),
          'legacy_aux_bytes':{} if saved is None else saved.get('legacy_aux_bytes',{}),
          'legacy_optimizer_bytes':b'' if saved is None else saved.get('legacy_optimizer_bytes',b'')}
    temporal_guard=all(temporal_after[str(h)]['nll']<=temporal_before[str(h)]['nll']+args.future_margin for h in (1,2,3))
    promoted=bool(args.steps==0 or (dev_after['nll']<=dev_before['nll']-args.promotion_margin and temporal_guard))
    # The old future-rail tensor is now represented by the live core.T parameter.
    # Preserve original auxiliary data but remove this duplicate frozen weight blob.
    info['legacy_aux_bytes'].pop('weights.raw.future64',None)
    if promoted:
        tmp=str(out)+'.tmp';torch.save(info,tmp);os.replace(tmp,out)
    metrics={'dev_before':{'nll':dev_before['nll'],'n':args.eval},'dev_after':{'nll':dev_after['nll'],'n':args.eval},
             'test_after':None if test_after is None else {'nll':test_after['nll'],'n':args.eval},
             'r9_updates_candidate':start+args.steps,'r9_updates_active':start+args.steps if promoted else start,
             'promoted':promoted,'temporal_guard':temporal_guard,
             'future_before':temporal_before,'future_after':temporal_after,'runtime_s':round(time.time()-t,2),'migration_max_abs_err':delta,
             'checkpoint_sha256':sha(out) if out.exists() else None,
             'param_count':sum(x.numel() for x in model.parameters())}
    (root/'reports/R9_joint_metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    print(json.dumps({'phase':'FINAL','metrics':metrics}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='.');p.add_argument('--resume',default=None)
    p.add_argument('--steps',type=int,default=200);p.add_argument('--batch',type=int,default=16)
    p.add_argument('--eval',type=int,default=512);p.add_argument('--eval-batch',type=int,default=24)
    p.add_argument('--lr-core',type=float,default=0.000002);p.add_argument('--lr-decoder',type=float,default=0.000025)
    p.add_argument('--lr-adapter',type=float,default=0.0001);p.add_argument('--lr-sequence',type=float,default=0.0003);p.add_argument('--subword-weight',type=float,default=0.06)
    p.add_argument('--dev-every',type=int,default=0);p.add_argument('--report-every',type=int,default=50)
    p.add_argument('--seed',type=int,default=20261112);p.add_argument('--threads',type=int,default=2)
    p.add_argument('--rare-fraction',type=float,default=0.0);p.add_argument('--importance-weights',action='store_true')
    p.add_argument('--promotion-margin',type=float,default=0.00005)
    p.add_argument('--future-margin',type=float,default=0.0002)
    p.add_argument('--future-weight',type=float,default=0.025)
    p.add_argument('--test',action='store_true');a=p.parse_args();main(a)
