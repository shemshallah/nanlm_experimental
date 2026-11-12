#!/usr/bin/env python3
"""R10: causal continuous next-subword training in the *same* PAL rank-64 LM.

Uses strictly past whole-word IDs for core.hidden at every new word; predicts
all pieces AND literal word spaces without resetting the subword recurrent
state. Reuses the existing R7 piece embedding, GRUCell and output projection.
This normalized-token stream is a controlled bridge experiment: original
source punctuation/spacing is not fully reproduced by the word records.

One checkpoint holds all weights, optimizer, RNG and training position. The
original R9 is read only; dev gating prevents promotion of regressions.
"""
import argparse,copy,hashlib,json,os,pathlib,random,time,sys
import numpy as np,torch
from torch import nn
import torch.nn.functional as F
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from r7_core import R6,Corpus
from r7_memory import TokenCompositor
from r9_joint import JointR9,eval_exact,eval_temporal

def sha(path):
    d=hashlib.sha256()
    with open(path,'rb') as f:
        for buf in iter(lambda:f.read(2*1024*1024),b''):d.update(buf)
    return d.hexdigest()

class StreamingBridge(nn.Module):
    """Boundary conditioning, not a second model or an independent head."""
    def __init__(self):
        super().__init__()
        self.retain_logit=nn.Parameter(torch.tensor(0.8))
        self.context_gain=nn.Parameter(torch.tensor(0.3))
    def forward(self,model,word_h,pieces,word_index):
        # pieces[B,T] contains stream targets, with -100 padding.
        B,T=pieces.shape;current=torch.full((B,),model.memory.start_piece,dtype=torch.long)
        state=word_h[:,0,:];total=state.new_zeros(());n=0
        last_index=word_index[:,0]
        retain=torch.sigmoid(self.retain_logit)
        for t in range(T):
            valid=pieces[:,t]!=-100
            ix=word_index[:,t].clamp(0,word_h.shape[1]-1)
            change=(ix!=last_index)&valid
            h=word_h.gather(1,ix[:,None,None].expand(-1,1,word_h.shape[-1]))[:,0,:]
            blended=retain*state+(1.-retain)*self.context_gain*h
            state=torch.where(change[:,None],blended,state)
            state=model.memory.subword_cell(model.memory.piece_emb(current),state)
            logits=model.memory.subword_out(state)
            if valid.any():
                total=total+F.cross_entropy(logits[valid],pieces[:,t][valid],reduction='sum')
                n+=int(valid.sum())
            current=torch.where(valid,pieces[:,t],torch.zeros_like(current))
            last_index=torch.where(valid,ix,last_index)
        return total/max(1,n),n

def compatible_starts(cor,window):
    # Every window is a continuous sequence in the original word records.
    # A reset/BOS boundary anywhere in the window makes it ineligible.
    a=cor.data
    good=np.asarray(a[1:,2]==a[:-1,3],dtype=np.int8)
    # Prefix-sum contiguous edges: correct for arbitrary window length.
    ps=np.r_[0,np.cumsum(1-good,dtype=np.int64)]
    allstarts=np.arange(3,len(cor.seq)-window,dtype=np.int64)
    return allstarts[(ps[allstarts+window-1]-ps[allstarts])==0]

def build_batch(cor,comp,vocab,starts,window=6):
    B=len(starts);W=window
    indexes=np.asarray(starts,dtype=np.int64)[:,None]+np.arange(W,dtype=np.int64)[None,:]
    e=torch.from_numpy(np.array(cor.data[indexes],copy=True).astype('int64'))
    lc=torch.from_numpy(np.array(cor.long[indexes],copy=True).astype('float32'))
    words=[cor.spelling(i,vocab) for i in indexes.reshape(-1)]
    pcs,plen=comp.encode_words(words)
    pcs=pcs.numpy().reshape(B,W,-1);plen=plen.numpy().reshape(B,W)
    # Consecutive piece sequence: no EOS, no hidden reset, literal 0x20 separator.
    streams=[];maps=[];bytecounts=[]
    for b in range(B):
        seq=[];tags=[];nbytes=0
        for j in range(W):
            word=words[b*W+j]; nbytes+=len(word.encode('utf8'))+1
            seq.extend(pcs[b,j,:plen[b,j]].tolist());tags.extend([j]*int(plen[b,j]))
            seq.append(32);tags.append(j)
        streams.append(seq);maps.append(tags);bytecounts.append(nbytes)
    T=max(len(s) for s in streams)
    targets=torch.full((B,T),-100,dtype=torch.long)
    tags=torch.zeros((B,T),dtype=torch.long)
    for b,(s,m) in enumerate(zip(streams,maps)):
        targets[b,:len(s)]=torch.tensor(s);tags[b,:len(m)]=torch.tensor(m)
    return e,lc,targets,tags,sum(bytecounts)

def load_model(root,checkpoint_path):
    saved=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
    assert saved['format'] in ('PAL9_JOINT_SINGLE_V1','PAL10_STREAM_SINGLE_V1','PAL11_RAW_SINGLE_V1','PAL12_CAUSAL_RESIDUAL_V1')
    frozen={name:saved['model'][f'core.{name}'] for name in ('E','H','P','M','Q','NOW','W','B','L','T')}
    r6=R6(root,weights_state=frozen);comp=TokenCompositor(root,r6)
    model=JointR9(r6,comp)
    model.load_state_dict(saved['model'],strict=True)
    return saved,r6,comp,model

def stream_eval(model,bridge,cor,comp,starts,window,batchsize=4):
    model.eval();bridge.eval();total=0.;count=0;bytes_=0
    with torch.no_grad():
        for ix in np.array_split(starts, max(1,(len(starts)+batchsize-1)//batchsize)):
            e,lc,pieces,tags,nbytes=build_batch(cor,comp,model._r10_vocab,ix,window)
            B,W=e.shape[:2]
            h=model.core.hidden(e.reshape(-1,7),lc.reshape(-1,72)).reshape(B,W,-1)
            nll,n=bridge(model,h,pieces,tags)
            total+=float(nll)*n;count+=n;bytes_+=nbytes
    return {'subword_nll':total/count,'subwords':count,'windows':len(starts),
            'normalized_bytes':bytes_,'nats_per_byte':total/bytes_,'bits_per_byte':total/bytes_/np.log(2)}

def train(args):
    t0=time.time();root=pathlib.Path(args.root);torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
    orig=root/'checkpoint/r9_active.pal9'
    active=pathlib.Path(args.output)
    resumed=active.exists()
    saved,r6,comp,model=load_model(root,active if resumed else orig)
    bridge=StreamingBridge()
    if resumed:bridge.load_state_dict(saved['stream_bridge'])
    model._r10_vocab=r6.vocab
    train_c=Corpus(root,'train',args.data_dir);dev_c=Corpus(root,'dev',args.data_dir);test_c=Corpus(root,'test',args.data_dir)
    candidates={part:compatible_starts(c,args.window) for part,c in [('train',train_c),('dev',dev_c),('test',test_c)]}
    assert all(len(a)>args.eval for a in candidates.values())
    # Evenly spaced, deterministic evaluation windows; no test-driven tuning.
    dev_pos=candidates['dev'][np.linspace(0,len(candidates['dev'])-1,args.eval,dtype=np.int64)]
    test_pos=candidates['test'][np.linspace(0,len(candidates['test'])-1,args.eval,dtype=np.int64)]
    before=stream_eval(model,bridge,dev_c,comp,dev_pos,args.window,args.eval_batch)
    # Fast matched word evaluation gate, with exact R7 memory and temporal rails.
    word_before=eval_exact(model,r6,dev_c,comp,args.word_eval,args.eval_batch)['nll']
    future_before=eval_temporal(model,dev_c,args.future_eval,args.eval_batch)
    if args.steps==0:
        print(json.dumps({'phase':'BASE','stream_dev':before,'word_nll_dev':word_before,'future':future_before,'num_parameters':sum(x.numel() for x in model.parameters())+sum(x.numel() for x in bridge.parameters())}),flush=True)
        return
    # Update the SAME previously learned GRU, piece embedding, output layer,
    # and temporal NOW. No independent language-model output head is created.
    optim=torch.optim.AdamW([
        {'params':[model.core.E,model.core.H,model.core.P,model.core.M,model.core.Q,model.core.NOW,model.core.L],'lr':args.lr_core},
        {'params':[model.memory.piece_emb.weight,model.memory.subword_cell.weight_ih,model.memory.subword_cell.weight_hh,model.memory.subword_cell.bias_ih,model.memory.subword_cell.bias_hh,model.memory.subword_out.weight,model.memory.subword_out.bias], 'lr':args.lr_subword},
        {'params':list(bridge.parameters()), 'lr':args.lr_bridge}
    ],weight_decay=0.00001)
    if resumed:
        optim.load_state_dict(saved['stream_optimizer'])
        # Explicit stage learning rates override old optimizer-group LR values.
        for group,rate in zip(optim.param_groups,(args.lr_core,args.lr_subword,args.lr_bridge)):
            group['lr']=rate
        torch.set_rng_state(saved['torch_rng']);np.random.set_state(saved['numpy_rng']);random.setstate(saved['python_rng']);rng=np.random.default_rng();rng.bit_generator.state=saved['stream_rng']
        start=saved['r10_updates']
    else:start=0;rng=np.random.default_rng(args.seed)
    grad_check=None;recent=[]
    for k in range(args.steps):
        model.train();bridge.train()
        p=rng.choice(candidates['train'],size=args.batch,replace=False)
        e,lc,pieces,tags,_=build_batch(train_c,comp,r6.vocab,p,args.window)
        B,W=e.shape[:2]
        h=model.core.hidden(e.reshape(-1,7),lc.reshape(-1,72)).reshape(B,W,-1)
        subloss,count=bridge(model,h,pieces,tags)
        # Regularize the original causal whole-word predictor. Target word is
        # only a supervised label, never seen by the hidden state.
        word_loss=model.core.nll(h.reshape(-1,h.shape[-1]),e[:,:,3].reshape(-1)).mean()
        loss=subloss+args.word_weight*word_loss
        if not torch.isfinite(loss):raise ValueError('nonfinite R10 training loss')
        optim.zero_grad(set_to_none=True);loss.backward()
        if grad_check is None:
            grad_check={name:float(getattr(model.core,name).grad.norm().item()) if getattr(model.core,name).grad is not None else 0. for name in ['P','M','Q','E','NOW']}
        torch.nn.utils.clip_grad_norm_(list(model.parameters())+list(bridge.parameters()),1.)
        optim.step();recent.append([float(subloss.detach()),float(word_loss.detach())])
        if (k+1)%args.log_every==0 or k==args.steps-1:
            rr=np.mean(recent,axis=0);print(json.dumps({'phase':'TRAIN','r10_updates_candidate':start+k+1,'subword_loss':float(rr[0]),'word_anchor':float(rr[1]),'elapsed':round(time.time()-t0,2)}),flush=True);recent=[]
    after=stream_eval(model,bridge,dev_c,comp,dev_pos,args.window,args.eval_batch)
    word_after=eval_exact(model,r6,dev_c,comp,args.word_eval,args.eval_batch)['nll']
    future_after=eval_temporal(model,dev_c,args.future_eval,args.eval_batch)
    # Warm start from inherited trained spelling weights; no uninitialized
    # baseline is counted as a language-model performance claim.
    guard=all(future_after[str(h)]['nll']<=future_before[str(h)]['nll']+args.future_margin for h in (1,2,3))
    promoted=bool(after['subword_nll']+args.min_gain<before['subword_nll'] and word_after<=word_before+args.word_margin and guard)
    result={'phase':'FINAL','source_core_updates':saved['source_core_steps'],'source_adapter_updates':saved['source_adapter_steps'],
            'r9_updates':saved['r9_updates'],'r10_updates_candidate':start+args.steps,
            'r10_updates_active':start+args.steps if promoted else start,'promoted':promoted,'stream_dev_before':before,'stream_dev_after':after,
            'word_dev_before':word_before,'word_dev_after':word_after,'future_dev_before':future_before,'future_dev_after':future_after,
            'mirrored_core_gradient_norms_initial':grad_check,'runtime_seconds':round(time.time()-t0,2),
            'validation':'same fixed development windows; held-out test not used for model promotion',
            'lr_effective':[float(group['lr']) for group in optim.param_groups]}
    if promoted:
        # Single model, single optimizer+RNG state for R10 continuation.
        info=dict(saved)
        # R9 had zero accepted updates: its empty torch optimizer and immutable
        # legacy C++ optimizer snapshot aren't used by the active stream trainer.
        # Retain provenance hashes but keep exactly one active optimizer state.
        info.pop('optimizer',None);info.pop('legacy_optimizer_bytes',None)
        info.update({'format':'PAL10_STREAM_SINGLE_V1','model':model.state_dict(),
           'stream_bridge':bridge.state_dict(),'stream_optimizer':optim.state_dict(),
           'r10_updates':start+args.steps,'stream_rng':rng.bit_generator.state,
           'torch_rng':torch.get_rng_state(),'numpy_rng':np.random.get_state(),'python_rng':random.getstate(),
           'stream_data_policy':'normalized word sequence, BPE plus literal 32 spaces; no target word to NOW',
           'r10_hparams':vars(args),'parent_sha256':sha(active if resumed else orig)})
        active.parent.mkdir(parents=True,exist_ok=True)
        tmp=active.with_suffix(active.suffix+'.tmp');torch.save(info,tmp);os.replace(tmp,active)
        # Test is evaluated only after development selection.
        result['stream_test']=stream_eval(model,bridge,test_c,comp,test_pos,args.window,args.eval_batch)
        result['word_test']=eval_exact(model,r6,test_c,comp,args.word_eval,args.eval_batch)['nll']
        result['checkpoint_sha256']=sha(active)
    output=pathlib.Path(args.report);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',default='.');p.add_argument('--data-dir',default=None);p.add_argument('--output',default='./checkpoint/r10_active.pal10')
    p.add_argument('--report',default='./reports/r10_training.json')
    p.add_argument('--steps',type=int,default=120);p.add_argument('--batch',type=int,default=3)
    p.add_argument('--window',type=int,default=6);p.add_argument('--eval',type=int,default=64);p.add_argument('--eval-batch',type=int,default=4)
    p.add_argument('--word-eval',type=int,default=256);p.add_argument('--future-eval',type=int,default=128)
    p.add_argument('--lr-core',type=float,default=1e-7);p.add_argument('--lr-subword',type=float,default=0.00008)
    p.add_argument('--lr-bridge',type=float,default=0.002);p.add_argument('--word-weight',type=float,default=.06)
    p.add_argument('--future-margin',type=float,default=0.00035);p.add_argument('--word-margin',type=float,default=0.0006)
    p.add_argument('--min-gain',type=float,default=0.0001);p.add_argument('--log-every',type=int,default=30)
    p.add_argument('--threads',type=int,default=2);p.add_argument('--seed',type=int,default=20261112)
    train(p.parse_args())