#!/usr/bin/env python3
"""R12 CPU controlled experiment: one inherited PAL R11 model, no reset.

Compares equal-update raw-stream continuation vs a zero-init causal sparse-token
attention + residual FF upgrade. Ephemeral control arms are NEVER accepted as
independent permanent models. Optionally promote only winning/dev-guarded
continuation into the SAME original checkpoint file, atomically.
"""
import argparse, hashlib, json, math, os, pathlib, random, sys, time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
sys.path.insert(0,str(pathlib.Path(__file__).parent))
from r10_stream import load_model, StreamingBridge, sha
from r11_raw import read_books, windows, batch, eval_raw
from r7_core import Corpus
from r9_joint import eval_temporal

class CausalRecall(nn.Module):
    """Add learned exact access to prior subword hidden states; no lookahead.

    Inactive gates ensure bit-exact initial predictions. Shared downstream
    decoder, trunk, mirrored NOW state, and optimizer moments are inherited.
    """
    def __init__(self, dim=96, capacity=64, topk=8, ff_multiplier=2):
        super().__init__()
        self.capacity=capacity; self.topk=topk
        self.query=nn.Linear(dim,dim,bias=False)
        self.key=nn.Linear(dim,dim,bias=False)
        self.value=nn.Linear(dim,dim,bias=False)
        self.attn_out=nn.Linear(dim,dim,bias=False)
        self.ffn=nn.Sequential(nn.Linear(dim,dim*ff_multiplier),nn.SiLU(),nn.Linear(dim*ff_multiplier,dim))
        self.attn_gain=nn.Parameter(torch.zeros(()))
        self.ffn_gain=nn.Parameter(torch.zeros(()))
    def forward(self,state,history,disable_attention=False,disable_ffn=False):
        ans=state
        if history and not disable_attention:
            past=torch.stack(history[-self.capacity:],dim=1)
            key=self.key(past)
            q=self.query(state).unsqueeze(1)
            scores=(q*key).sum(-1)/(state.shape[-1]**.5)
            k=min(self.topk,scores.shape[1])
            indices=scores.topk(k,dim=1).indices
            values=self.value(past).gather(1,indices[:,:,None].expand(-1,-1,state.shape[-1]))
            weights=F.softmax(scores.gather(1,indices),dim=-1)
            recalled=(values*weights[:,:,None]).sum(1)
            ans=ans+self.attn_gain*self.attn_out(recalled)
        if not disable_ffn:
            ans=ans+self.ffn_gain*self.ffn(state)
        return ans

def run_window(model,bridge,adapter,h,targets,tags,training=False,
               disable_attention=False,disable_ffn=False,inspect_prefix=0):
    """Exact R11 recurrence; optional causal residual before its SAME decoder."""
    B,T=targets.shape
    previous=torch.full((B,),model.memory.start_piece,dtype=torch.long)
    state=h[:,0]; last=tags[:,0]; retain=bridge.retain_logit.sigmoid()
    history=[];total=state.new_zeros(());nonspace_loss=0.;total_n=0;nonspace_n=0;space_n=0
    other_hits=0;space_hits=0;logits_prefix=[]
    for t in range(T):
        valid=targets[:,t]!=-100
        ix=tags[:,t].clamp(0,h.shape[1]-1)
        next_h=h.gather(1,ix[:,None,None].expand(-1,1,h.shape[-1]))[:,0]
        changed=(ix!=last)&valid
        state=torch.where(changed[:,None],retain*state+(1-retain)*bridge.context_gain*next_h,state)
        state=model.memory.subword_cell(model.memory.piece_emb(previous),state)
        out=adapter(state,history,disable_attention,disable_ffn) if adapter is not None else state
        logits=model.memory.subword_out(out)
        if inspect_prefix and t<inspect_prefix: logits_prefix.append(logits.detach().clone())
        if valid.any():
            ce=F.cross_entropy(logits[valid],targets[:,t][valid],reduction='none')
            total=total+ce.sum()
            n=int(valid.sum());total_n+=n
            if not training:
                sp=valid & (targets[:,t]==32);ot=valid & (targets[:,t]!=32)
                nonspace_n+=int(ot.sum());space_n+=int(sp.sum())
                if ot.any():nonspace_loss+=float(F.cross_entropy(logits[ot],targets[:,t][ot],reduction='sum'))
                pred=logits.argmax(-1)
                other_hits+=int(((pred==targets[:,t])&ot).sum())
                space_hits+=int(((pred==32)&sp).sum())
        previous=torch.where(valid,targets[:,t],torch.zeros_like(previous))
        last=torch.where(valid,ix,last)
        if adapter is not None:history.append(state)
    if training:return total/max(total_n,1),total_n
    return {'total_loss':float(total), 'num_tokens':total_n,'other_loss':nonspace_loss,
            'nonspace_tokens':nonspace_n,'space_tokens':space_n,'other_hits':other_hits,
            'space_hits':space_hits,'prefix_logits':logits_prefix}

def evaluate(model,bridge,adapter,docs,choices,span,batchsize=2,**kwargs):
    model.eval();bridge.eval()
    if adapter is not None: adapter.eval()
    loss=0.;n=0;bytes_=0;other_loss=0.;otn=0;spn=0;oth=0;sph=0
    with torch.no_grad():
        for start in range(0,len(choices),batchsize):
            picks=choices[start:start+batchsize]
            ex,lc,target,tag,bcount=batch(docs,picks,span)
            B,W=ex.shape[:2]
            h=model.core.hidden(ex.reshape(-1,7),lc.reshape(-1,72)).reshape(B,W,-1)
            val=run_window(model,bridge,adapter,h,target,tag,**kwargs)
            loss+=val['total_loss'];n+=val['num_tokens'];bytes_+=bcount
            other_loss+=val['other_loss'];otn+=val['nonspace_tokens']
            spn+=val['space_tokens'];oth+=val['other_hits'];sph+=val['space_hits']
    return {'bits_per_byte':loss/bytes_/math.log(2),'other_nll':other_loss/otn,
            'other_top1':oth/otn,'space_top1':sph/spn,'raw_bytes':bytes_,
            'tokens':n,'nonspace_tokens':otn,'windows':len(choices)}

def load_for_arm(root,checkpoint,adapter_enabled,seed=20261111,capacity=64,topk=8):
    saved,r6,comp,model=load_model(root,checkpoint)
    bridge=StreamingBridge();bridge.load_state_dict(saved['stream_bridge'])
    # Independent new-param initialisation, not perturbing inherited saved RNG.
    resume_r12=saved['format']=='PAL12_CAUSAL_RESIDUAL_V1'
    if resume_r12:
        # The stored adapter configuration takes precedence over CLI defaults.
        prev_config=saved.get('r12_config',{})
        capacity=int(prev_config.get('capacity',capacity))
        topk=int(prev_config.get('topk',topk))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        adapter=CausalRecall(capacity=capacity,topk=topk) if (adapter_enabled or resume_r12) else None
    if resume_r12:
        assert adapter is not None
        adapter.load_state_dict(saved['r12_residual'],strict=True)
    opt=torch.optim.AdamW([
        {'params':[model.core.E,model.core.H,model.core.P,model.core.M,model.core.Q,model.core.NOW,model.core.L],'lr':2e-8},
        {'params':[model.memory.piece_emb.weight,model.memory.subword_cell.weight_ih,model.memory.subword_cell.weight_hh,model.memory.subword_cell.bias_ih,model.memory.subword_cell.bias_hh,model.memory.subword_out.weight,model.memory.subword_out.bias],'lr':5e-5},
        {'params':list(bridge.parameters()),'lr':4e-4}
    ],weight_decay=.00001)
    if adapter is not None:
        opt.add_param_group({'params':list(adapter.parameters()),'lr':2e-4,'weight_decay':.00001})
    if resume_r12:
        assert len(saved['stream_optimizer']['param_groups'])==4
        opt.load_state_dict(saved['stream_optimizer'])
    else:
        # Adding a zero-initialised extension must not wipe the 3-group R11 AdamW moments.
        old_saved=saved['stream_optimizer']
        if adapter is None:
            opt.load_state_dict(old_saved)
        else:
            ext=opt.param_groups.pop()
            opt.load_state_dict(old_saved)
            opt.add_param_group(ext)

    torch.set_rng_state(saved['torch_rng']);np.random.set_state(saved['numpy_rng']);random.setstate(saved['python_rng'])
    rg=np.random.default_rng();rg.bit_generator.state=saved['stream_rng']
    return saved,r6,comp,model,bridge,adapter,opt,rg

def old_anchors(model,root,positions=64,future_positions=48):
    cor=Corpus(root,'dev')
    ix=np.arange(positions,dtype=np.int64)*(len(cor.seq)-1)//positions
    model.eval();result=0.
    with torch.no_grad():
        for block in np.array_split(ix,max(1,(len(ix)+31)//32)):
            ee=torch.from_numpy(np.array(cor.data[block],copy=True).astype(np.int64))
            lc=torch.from_numpy(np.array(cor.long[block],copy=True).astype(np.float32))
            hidden=model.core.hidden(ee,lc)
            result+=float(model.core.nll(hidden,ee[:,3]).sum())
    return result/positions,eval_temporal(model,cor,future_positions,2)

def train_arm(root,checkpoint,docs,picks,args,enabled):
    saved,r6,comp,model,bridge,adapter,opt,rng=load_for_arm(root,checkpoint,enabled,
         seed=args.seed,capacity=args.capacity,topk=args.topk)
    t0=time.perf_counter();seen=0;train_losses=[];initial_gain=[]
    if adapter is not None and not enabled:
        # Matched R12 control: same starting predictions and old/new optimizer
        # moments, but the new residual capacity receives no parameter updates.
        adapter.requires_grad_(False)

    for step in range(args.steps):
        model.train();bridge.train()
        if adapter is not None:adapter.train()
        chosen=[picks['train'][int(j)] for j in rng.integers(0,len(picks['train']),size=args.batch)]
        e,lc,t,tag,_=batch(docs['train'],chosen,args.span)
        B,W=e.shape[:2]
        h=model.core.hidden(e.reshape(-1,7),lc.reshape(-1,72)).reshape(B,W,-1)
        loss,n=run_window(model,bridge,adapter,h,t,tag,training=True)
        if not torch.isfinite(loss):raise ValueError('nonfinite loss')
        opt.zero_grad(set_to_none=True);loss.backward()
        if adapter is not None and enabled and step==0:
            initial_gain=[float(adapter.attn_gain.grad.item()),float(adapter.ffn_gain.grad.item())]
        torch.nn.utils.clip_grad_norm_(list(model.parameters())+list(bridge.parameters())+
                        ([] if adapter is None else list(adapter.parameters())),1.)
        opt.step();seen+=n;train_losses.append(float(loss.detach()))
    sec=time.perf_counter()-t0
    after=evaluate(model,bridge,adapter,docs['dev'],picks['dev'],args.span,args.eval_batch)
    old,temporal=old_anchors(model,root,args.word_eval,args.future_eval)
    extra={'train_seconds':sec,'train_tokens':seen,'tokens_per_second':seen/sec,
           'train_loss_mean':float(np.mean(train_losses)),'old_word_nll':old,'future':temporal,
           'extra_parameters':sum(p.numel() for p in adapter.parameters()) if adapter is not None else 0,
           'attention_gain':float(adapter.attn_gain) if adapter is not None else 0.,
           'ffn_gain':float(adapter.ffn_gain) if adapter is not None else 0.,
           'initial_gate_gradients':initial_gain}
    if adapter is not None and not getattr(args,'skip_ablations',False):
        extra['remove_attn']=evaluate(model,bridge,adapter,docs['dev'],picks['dev'],args.span,args.eval_batch,disable_attention=True)
        extra['remove_ffn']=evaluate(model,bridge,adapter,docs['dev'],picks['dev'],args.span,args.eval_batch,disable_ffn=True)
        extra['remove_both']=evaluate(model,bridge,adapter,docs['dev'],picks['dev'],args.span,args.eval_batch,disable_attention=True,disable_ffn=True)
    return after,extra,(saved,model,bridge,adapter,opt,rng,torch.get_rng_state().clone())

def main(args):
    torch.set_num_threads(args.threads)
    root=pathlib.Path(args.root);ckpt=pathlib.Path(args.checkpoint)
    original=sha(ckpt)
    print(json.dumps({'phase':'SOURCE','sha256':original}),flush=True)
    saved,r6,comp,model,bridge,adapter,opt,rng=load_for_arm(root,ckpt,True,args.seed,args.capacity,args.topk)
    docs,manifest=read_books(args.books,comp,r6.vocab,args.max_books)
    picks=windows(docs,args.span,args.eval,args.max_tokens)
    if saved['format']=='PAL12_CAUSAL_RESIDUAL_V1':
        # R12 -> R12 continuity: recompute the already-promoted learned adapter.
        baseline=evaluate(model,bridge,adapter,docs['dev'],picks['dev'],args.span,args.eval_batch)
    else:
        baseline=eval_raw(model,bridge,docs['dev'],picks['dev'],args.span,args.eval_batch)
    exact=evaluate(model,bridge,adapter,docs['dev'],picks['dev'],args.span,args.eval_batch)
    maxdelta=max(abs(exact[k]-baseline[k]) for k in ('bits_per_byte','other_nll','other_top1','space_top1'))
    assert maxdelta<=1e-6,('nonidentity migration',maxdelta)
    # Direct logit identity, not merely a close aggregate CE estimate.
    test_pick=picks['dev'][:1]
    ee,lc,tt,tag,_=batch(docs['dev'],test_pick,args.span)
    with torch.no_grad():
        hh=model.core.hidden(ee.reshape(-1,7),lc.reshape(-1,72)).reshape(1,ee.shape[1],-1)
        a=run_window(model,bridge,None,hh,tt,tag,inspect_prefix=12)
        b=run_window(model,bridge,adapter,hh,tt,tag,inspect_prefix=12)
        logit_abs=max(float((u-v).abs().max()) for u,v in zip(a['prefix_logits'],b['prefix_logits']))
    if saved['format']!='PAL12_CAUSAL_RESIDUAL_V1':
        assert logit_abs==0.,('logit mismatch during zero-gate expansion',logit_abs)
    old_before,future_before=old_anchors(model,root,args.word_eval,args.future_eval)
    print(json.dumps({'phase':'INITIAL_IDENTITY','max_eval_metric_diff':maxdelta,'max_initial_logit_diff':logit_abs,
                      'baseline':baseline,'old_word_nll':old_before,
                      'old_optimizer_states':len(saved['stream_optimizer']['state'])}),flush=True)
    if args.steps==0:return
    arms={}
    winners={}
    for enabled in (False,True):
        arm=('frozen_residual_control' if saved['format']=='PAL12_CAUSAL_RESIDUAL_V1' else 'baseline_continuation') if not enabled else 'causal_recall_plus_ffn'
        ev,extra,state=train_arm(root,ckpt,docs,picks,args,enabled)
        arms[arm]={'dev':ev,**extra}
        if enabled:
            winners[arm]=state
        else:
            # Do not retain the 100+MB control model + AdamW moment tensors.
            # If the frozen residual wins, decline promotion; the active model stays untouched.
            del state
            import gc
            gc.collect()
        print(json.dumps({'phase':'ARM','arm':arm,'dev':ev,'throughput':extra['tokens_per_second']}),flush=True)
    guards={}
    for label,data in arms.items():
        guard=(data['old_word_nll']<=old_before+args.word_margin and
               all(data['future'][str(h)]['nll']<=future_before[str(h)]['nll']+args.future_margin for h in (1,2,3)))
        gains=(data['dev']['bits_per_byte']+args.min_gain<baseline['bits_per_byte'] and
               data['dev']['other_nll']+args.min_gain<baseline['other_nll'])
        guards[label]={'legacy_ok':guard,'raw_improved':gains,'acceptable':guard and gains}
    acceptable=[k for k in guards if guards[k]['acceptable']]
    ctrl_name='frozen_residual_control' if saved['format']=='PAL12_CAUSAL_RESIDUAL_V1' else 'baseline_continuation'
    learned=arms['causal_recall_plus_ffn']['dev']
    control=arms[ctrl_name]['dev']
    # Strong evidence gate: a capacity upgrade must improve BOTH lexical and
    # total raw loss relative to an equal-compute native continuation.
    residual_wins=(learned['bits_per_byte']+args.min_gain<control['bits_per_byte'] and
                   learned['other_nll']+args.min_gain<control['other_nll'])
    if residual_wins and 'causal_recall_plus_ffn' in acceptable:
        best='causal_recall_plus_ffn'
    elif ctrl_name in acceptable:
        best=ctrl_name
    else:best=None
    report={'source':'PAL R11 single inherited checkpoint','source_sha256':original,
            'r11_accepted_updates':saved['r11_updates'],'comparison':'equal updates; same RNG, windows and initial optimizer',
            'r12_parameters_added':arms['causal_recall_plus_ffn']['extra_parameters'],
            'residual_beats_matched_continuation_on_both_metrics':residual_wins,
            'context_access_tokens':args.capacity,'attention_topk':args.topk,
            'baseline_before':baseline,'migration_abs_metric_diff':maxdelta,'migration_max_initial_logit_diff':logit_abs,
            'anchors_before':{'old_word_nll':old_before,'future':future_before},
            'arms':arms,'guards':guards,'selected':best,'promoted':False,
            'notes':['Architecture-arm comparisons are exploratory same-dev model selection.',
                     'No independent test was used for selection.',
                     'Memory-optimized runner does not promote control arms; control is scored only.',
                     'Promoting R11-only continuation means no architecture upgrade; R12 evidence is not a scaling law.']}
    if args.promote and best is not None and best in winners:
        checkpoint=ckpt if not args.output else pathlib.Path(args.output)
        # Only the winner is retained as active; all old state and RNG survive.
        saved_w,mod,br,ad,optimizer,rg,trng=winners[best]
        result=dict(saved_w)
        result.update({'model':mod.state_dict(),'stream_bridge':br.state_dict(),
              'stream_optimizer':optimizer.state_dict(),'stream_rng':rg.bit_generator.state,
              'torch_rng':trng,
              'parent_sha256':original,'r12_selected_arm':best,'r12_config':vars(args)})
        if ad is not None:
            result['format']='PAL12_CAUSAL_RESIDUAL_V1'
            result['r12_residual']=ad.state_dict()
            old_r11=saved_w['r11_updates']
            if saved_w['format']=='PAL12_CAUSAL_RESIDUAL_V1' and 'r12_updates' not in saved_w:
                # Repair legacy pilot metadata without altering inherited weights.
                old_r11-=int(saved_w.get('r12_config',{}).get('steps',0))
            result['r11_updates']=old_r11
            result['r12_updates']=saved_w.get('r12_updates',
                saved_w.get('r12_config',{}).get('steps',0) if saved_w['format']=='PAL12_CAUSAL_RESIDUAL_V1' else 0)+args.steps
        else:
            result['format']='PAL11_RAW_SINGLE_V1'
            result['r11_updates']=saved_w['r11_updates']+args.steps
        temp=checkpoint.with_suffix(checkpoint.suffix+'.tmp')
        torch.save(result,temp);os.replace(temp,checkpoint)
        report['promoted']=True;report['active_checkpoint_sha256']=sha(checkpoint)
        report['active_file']=str(checkpoint)
    pathlib.Path(args.report).parent.mkdir(parents=True,exist_ok=True)
    pathlib.Path(args.report).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'phase':'FINAL','selected':best,'promoted':report['promoted'],
                      'report':str(args.report)}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',default='.');p.add_argument('--books',default='nanrouter')
    p.add_argument('--checkpoint',default='checkpoint/r11_active.pal11')
    p.add_argument('--output',default='')
    p.add_argument('--report',default='reports/r12_experiment.json')
    p.add_argument('--steps',type=int,default=12);p.add_argument('--batch',type=int,default=2)
    p.add_argument('--span',type=int,default=12);p.add_argument('--eval',type=int,default=32)
    p.add_argument('--max-tokens',type=int,default=128);p.add_argument('--max-books',type=int,default=4)
    p.add_argument('--eval-batch',type=int,default=2);p.add_argument('--capacity',type=int,default=64)
    p.add_argument('--topk',type=int,default=8);p.add_argument('--threads',type=int,default=2)
    p.add_argument('--word-eval',type=int,default=64);p.add_argument('--future-eval',type=int,default=48)
    p.add_argument('--word-margin',type=float,default=.0008)
    p.add_argument('--future-margin',type=float,default=.00045)
    p.add_argument('--min-gain',type=float,default=.0001)
    p.add_argument('--seed',type=int,default=20261111)
    p.add_argument('--promote',action='store_true')
    main(p.parse_args())
