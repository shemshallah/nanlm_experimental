#!/usr/bin/env python3
"""PAL R18 joint fine-tuning from ONE accepted checkpoint, with every Adam moment.

The inherited four optimizer groups and R18 memory train jointly. Existing
R18 Q/K+gain optimizer moments are migrated into a fifth group. On an
unaccepted candidate the sole active checkpoint remains byte-identical.
Every train/eval target is strictly autoregressive, and R18 slots only hold
strictly older labels j<=t-65. This is CPU research, not frontier-scale SOTA.
"""
import argparse,copy,hashlib,json,math,os,pathlib,random,sys,time
import numpy as np,torch
from torch.nn import functional as F
sys.path.insert(0,str(pathlib.Path(__file__).parent))
import r14_slim_base as pal
from r13_cpu import run_window_fast
from r11_raw import batch,windows
from r13_corpus_cache import read_books_cached
from r18_long_memory import R18LongMemory,build_copy_queries,apply_copy_boost

def integrated_logits(model,bridge,adapter,net,ex,lc,tags,targets,detach_state=False):
    B,W=ex.shape[:2]
    if detach_state:
        with torch.no_grad():hidden=model.core.hidden(ex.reshape(-1,7),lc.reshape(-1,72)).reshape(B,W,-1)
    else:hidden=model.core.hidden(ex.reshape(-1,7),lc.reshape(-1,72)).reshape(B,W,-1)
    logits=run_window_fast(model,bridge,adapter,hidden,targets,tags,training=True,return_logits=True)
    out=[]
    for i in range(B):
        n=int((targets[i]!=-100).sum())
        q=build_copy_queries(targets[i,:n],mem=net.max_memory,gap=net.min_gap)
        local=apply_copy_boost(net,logits[i,:n],q)
        if n<targets.shape[1]:local=torch.cat((local,logits[i,n:]),0)
        out.append(local)
    return torch.stack(out)

def evaluate_joint(model,bridge,adapter,net,docs,choices,span):
    model.eval();bridge.eval();adapter.eval();net.eval()
    total=other=0.;raw=other_n=space_n=other_correct=space_correct=eligible=0
    with torch.no_grad():
        for pick in choices:
            ex,lc,target,tags,bytecount=batch(docs,[pick],span)
            logits=integrated_logits(model,bridge,adapter,net,ex,lc,tags,target)
            t=target.flatten();out=logits.reshape(-1,logits.shape[-1])
            loss=F.cross_entropy(out,t,reduction='none');other_mask=t!=32
            total+=float(loss.sum());other+=float(loss[other_mask].sum());raw+=bytecount
            other_n+=int(other_mask.sum());space_n+=int((~other_mask).sum())
            pred=out.argmax(-1)
            other_correct+=int(((pred==t)&other_mask).sum())
            space_correct+=int(((pred==t)&(~other_mask)).sum())
            q=build_copy_queries(t,mem=net.max_memory,gap=net.min_gap)
            if q is not None:eligible+=len(q['positions'])
    return {'bits_per_byte':total/raw/math.log(2),'other_nll':other/max(1,other_n),
            'other_top1':other_correct/max(1,other_n),'space_top1':space_correct/max(1,space_n),
            'eligible':eligible,'raw_bytes':raw,'windows':len(choices)}

def build_joint_optimizer(base_optimizer,net,saved,lr):
    base_optimizer.add_param_group({'params':list(net.parameters()),'lr':lr,'weight_decay':1e-5})
    if 'r18_joint_optimizer' in saved:
        base_optimizer.load_state_dict(saved['r18_joint_optimizer'])
        return base_optimizer,{'mode':'resumed_fifth_group'}
    assert len(base_optimizer.param_groups)==5
    retrieval=saved['r18_retrieval_optimizer'];gain=saved['r18_gain_optimizer']
    # Recover R18 first-stage AdamW moments (Q/K+temp) AND gain calibration
    # moments, rather than initializing a fresh optimizer on those parameters.
    transfer=[];retrieval_index=0
    for name,param in net.named_parameters():
        if name in ('logit_gain','recency'):
            ids=gain['param_groups'][0]['params'];idx=0 if name=='logit_gain' else 1
            original=gain['state'].get(ids[idx],{})
        else:
            ids=retrieval['param_groups'][0]['params'];ix=retrieval_index;retrieval_index+=1
            original=retrieval['state'].get(ids[ix],{})
        if original:
            base_optimizer.state[param]={k:copy.deepcopy(v) for k,v in original.items()}
        transfer.append(name)
    return base_optimizer,{'mode':'migrated_retrieval_and_gain_adam','param_names':transfer,
        'restored_moments':sum(bool(base_optimizer.state[p]) for p in net.parameters())}

def main(args):
    torch.set_num_threads(args.threads);torch.manual_seed(args.seed);random.seed(args.seed);np.random.seed(args.seed)
    root=pathlib.Path(args.root).resolve();path=root/'checkpoint/r12_active.pal12';parent=pal.sha(path)
    saved,r6,comp,model,bridge,adapter,opt,rng=pal.load_for_arm(root,path,True)
    assert 'r18_memory' in saved and len(opt.param_groups)==4
    net=R18LongMemory(model.memory.piece_emb.weight)
    net.load_state_dict(saved['r18_memory'])
    opt,history=build_joint_optimizer(opt,net,saved,args.lr_memory)
    if not (0 < args.lr_scale <= 1):
        raise ValueError("--lr-scale must be in (0,1]")
    for group in opt.param_groups:
        group["lr"] *= args.lr_scale
    history["lr_scale"] = args.lr_scale
    history["effective_lrs"] = [float(group["lr"]) for group in opt.param_groups]
    assert len(opt.param_groups)==5
    assert history.get('restored_moments',len(list(net.parameters())))==len(list(net.parameters()))
    docs,_=read_books_cached(root/'nanrouter',comp,r6.vocab,0)
    short_windows=windows(docs,24,8000,180)
    long_windows=windows(docs,args.long_span,max(args.train_windows,args.long_windows),args.max_tokens)
    dev_short=short_windows['dev'][:args.short_windows]
    dev_long=long_windows['dev'][:args.long_windows]
    old_anchor=pal.old_anchors(model,root,positions=32,future_positions=32)
    baseline={'short':evaluate_joint(model,bridge,adapter,net,docs['dev'],dev_short,24),
              'long':evaluate_joint(model,bridge,adapter,net,docs['dev'],dev_long,args.long_span)}
    print(json.dumps({'phase':'BASE','baseline':baseline,'optimizer':history}),flush=True)
    start=time.monotonic();updates=saved.get('r18_joint_updates',0)
    trace=[];processed=0;first=0;promoted=False
    for step in range(1,args.steps+1):
        model.train();bridge.train();adapter.train();net.train()
        choices=[long_windows['train'][int(v)] for v in rng.integers(0,len(long_windows['train']),size=args.batch)]
        ex,lc,t,tag,bytes_=batch(docs['train'],choices,args.span)
        out=integrated_logits(model,bridge,adapter,net,ex,lc,tag,t)
        logits=out.reshape(-1,out.shape[-1]);loss=F.cross_entropy(logits,t.reshape(-1),ignore_index=-100)
        assert torch.isfinite(loss)
        opt.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters())+list(bridge.parameters())+list(adapter.parameters())+list(net.parameters()),1.)
        opt.step()
        # The memory shares the inherited subword embedding basis. Its
        # non-persistent read-only copy must follow each unified update so a
        # checkpoint resumed from the latest embedding is logit-identical.
        with torch.no_grad():
            net.inherited_piece_embeddings.copy_(model.memory.piece_emb.weight.detach())
        processed+=int((t!=-100).sum());first+=1
        if step%5==0:print(json.dumps({'phase':'TRAIN','step':step,'loss':float(loss),'tokens':processed}),flush=True)
    after={'short':evaluate_joint(model,bridge,adapter,net,docs['dev'],dev_short,24),
           'long':evaluate_joint(model,bridge,adapter,net,docs['dev'],dev_long,args.long_span)}
    old_after=pal.old_anchors(model,root,positions=32,future_positions=32)
    improves=all(after[scope][metric]<baseline[scope][metric]-args.min_gain for scope in ('short','long') for metric in ('bits_per_byte','other_nll'))
    anchors_ok=(old_after[0]<=old_anchor[0]+args.legacy_margin and
       all(old_after[1][str(h)]['nll']<=old_anchor[1][str(h)]['nll']+args.legacy_margin for h in (1,2,3)))
    promoted=bool(args.promote and improves and anchors_ok)
    report={'round':'PAL R18 JOINT','source_checkpoint':parent,'old_updates':saved['r12_updates'],
        'old_r18_joint_updates':updates,'new_joint_updates_attempted':args.steps,'train_tokens':processed,
        'optimizer':history,'before':baseline,'after':after,'anchors_before':{'word':old_anchor[0],'future':old_anchor[1]},
        'anchors_after':{'word':old_after[0],'future':old_after[1]},'dev_all_gains':improves,
        'legacy_anchors_ok':anchors_ok,'promoted':promoted,'elapsed':time.monotonic()-start}
    if promoted:
        snap=opt.state_dict()
        oldids={int(i) for group in snap['param_groups'][:4] for i in group['params']}
        original_four={'state':{idx:v for idx,v in snap['state'].items() if idx in oldids},
                       'param_groups':snap['param_groups'][:4]}
        accepted=dict(saved)
        accepted.update({'model':model.state_dict(),'stream_bridge':bridge.state_dict(),
         'r12_residual':adapter.state_dict(),'r18_memory':net.state_dict(),
         'stream_optimizer':original_four,'r18_joint_optimizer':snap,
         'r12_updates':saved['r12_updates']+args.steps,
         'r18_joint_updates':updates+args.steps,'stream_rng':copy.deepcopy(rng.bit_generator.state),
         'torch_rng':torch.get_rng_state().clone(),
         'r18_joint_parent_sha256':parent})
        tmp=path.with_suffix('.pal12.tmp');torch.save(accepted,tmp);os.replace(tmp,path)
        report['promoted_sha256']=pal.sha(path)
    else:assert pal.sha(path)==parent
    (root/'reports/r18_joint_continuation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'phase':'FINAL','promoted':promoted,'short':after['short'],'long':after['long'],'history':history}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='.')
    p.add_argument('--threads',type=int,default=2);p.add_argument('--seed',type=int,default=18020)
    p.add_argument('--steps',type=int,default=20);p.add_argument('--batch',type=int,default=1)
    p.add_argument('--span',type=int,default=96);p.add_argument('--long-span',type=int,default=400)
    p.add_argument('--train-windows',type=int,default=500);p.add_argument('--short-windows',type=int,default=32)
    p.add_argument('--long-windows',type=int,default=8);p.add_argument('--max-tokens',type=int,default=2000)
    p.add_argument('--lr-memory',type=float,default=0.00005);p.add_argument('--min-gain',type=float,default=.0001)
    p.add_argument('--lr-scale',type=float,default=1.0,help='Scale all restored optimizer learning rates, preserving Adam moments')
    p.add_argument('--legacy-margin',type=float,default=.01)
    p.add_argument('--promote',action='store_true')
    main(p.parse_args())
