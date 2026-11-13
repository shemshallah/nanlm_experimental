#!/usr/bin/env python3
"""FA1 R7.21: train the SAME R7 unified model (near+far jointly) on CPU.

No new optimizer is created; fifth inherited optimizer with original near/far
Adam moments is continued. Core and R7.12 calibration remain frozen for this
first pilot. Development-only selection, causal hard-negative controls, never
promote to accepted checkpoint. This is NOT independent book validation.
"""
from __future__ import annotations
import argparse, copy, json, pathlib, random, time
import numpy as np
import torch
import torch.nn.functional as F
from r7_core import Corpus
from train_r7 import batch_data
from r74_stream import OUTER_CONTEXT
from r712_streaming import DirectedStreaming
from r710_checkpoint_integrity import sha256,deep_equal
from fa1_r716_router import extract_paths,RECENT,PARENT
from fa1_r718_far_residual import (far_only,make_feature,pick_positions,
                                    reroute_far,HORIZON)
from fa1_r721_unified import load_candidate,trainable_memory_only,write_candidate


@torch.no_grad()
def gather_joint(root,model,base,comp,split,n,seed,batch=8,corpus_dir=None):
    corpus=Corpus(root,split,corpus_dir=corpus_dir);positions=np.sort(pick_positions(corpus,n,seed))
    stream=DirectedStreaming(model.core)
    allb=[];nf=[];npb=[];nc=[];ff=[];fp=[];fc=[]
    for offset in range(0,n,batch):
        inds=positions[offset:offset+batch]
        _,ctx,pieces,lens,y,_,_=batch_data(base,corpus,comp,inds)
        older=torch.from_numpy(corpus.get(inds,OUTER_CONTEXT)[0][:,:OUTER_CONTEXT-RECENT].copy()).long()
        state=stream.initial(older)
        ex=torch.from_numpy(corpus.data[inds].copy()).long()
        lg=torch.from_numpy(corpus.long[inds].copy()).long()
        now,loss,_=stream.forward(ex,lg,ctx,pieces,lens,state,older,y)
        allb.append(loss.detach().cpu());hist=corpus.get(inds,HORIZON)[0]
        routes=[];modal=[]
        for j in range(len(inds)):
            q=ctx[j].numpy();target=int(y[j]);h=hist[j]
            near=extract_paths(h,q,target,8192)+extract_paths(h,q,target,32768)
            far=far_only(h,q,target)
            routes.append((near,far,q))
            modal.extend(r['modal'] for r in near+far)
        p=torch.exp(-model.core.trunk.nll(now.cpu().repeat_interleave(12,0),
                                torch.tensor(modal,dtype=torch.long)).reshape(len(inds),12)).numpy()
        for j,(near,far,q) in enumerate(routes):
            nf.append(make_feature(near,p[j,:8],q));npb.append([r['target_probability'] for r in near]);nc.append([r['n'] for r in near])
            ff.append(make_feature(far,p[j,8:],q,far=True));fp.append([r['target_probability'] for r in far]);fc.append([r['n'] for r in far])
    return {'base':torch.cat(allb).cpu(),'near_features':torch.tensor(np.asarray(nf),dtype=torch.float32),
            'near_prob':torch.tensor(np.asarray(npb),dtype=torch.float32),
            'near_count':torch.tensor(np.asarray(nc),dtype=torch.float32),
            'far_features':torch.tensor(np.asarray(ff),dtype=torch.float32),
            'far_prob':torch.tensor(np.asarray(fp),dtype=torch.float32),
            'far_count':torch.tensor(np.asarray(fc),dtype=torch.float32),
            'positions':positions.tolist(),'n':n}


def wrong_far(root,model,base,comp,split,data,seed,mode='shuffle'):
    # Use the R7.18 causal control implementation, but recompute near-path
    # likelihood from the current single unified near module at evaluation.
    dummy={'base':data['base'],'positions':data['positions'],'n':data['n']}
    counter=reroute_far(root,model.core,base,comp,split,dummy,seed,mode=mode)
    d=dict(data)
    for key in ('far_features','far_prob','far_count'):d[key]=counter[key]
    return d


def take(data,ids):
    return {k:(v[ids] if torch.is_tensor(v) else v) for k,v in data.items()}


@torch.no_grad()
def evaluate(model,correct,shuffle,donor):
    c,n=model.joint_nll(correct)
    s,_=model.joint_nll(shuffle)
    d,_=model.joint_nll(donor)
    return {'parent_nll':float(correct['base'].mean()),
            'near_nll':float(n.mean()),'correct_nll':float(c.mean()),
            'gain_vs_parent':float((correct['base']-c).mean()),
            'shuffled_advantage':float((s-c).mean()),
            'independent_train_split_donor_advantage':float((d-c).mean()),
            'samples':int(correct['n'])}


def run(a):
    torch.set_num_threads(a.threads);torch.manual_seed(a.seed);np.random.seed(a.seed);random.seed(a.seed)
    start=time.monotonic();parent=pathlib.Path(a.root)/'checkpoint/pal_active.pt'
    digest=sha256(parent);assert digest==PARENT
    model,opts,base,comp,blob=load_candidate(a.checkpoint,a.root,threads=a.threads)
    assert blob['candidate_joint_adapter_updates']==0,'R7.21 training needs original migration seed'
    originals=copy.deepcopy([o.state_dict() for o in opts])
    core_initial=copy.deepcopy(model.core.state_dict())
    trainable_memory_only(model)
    # Continue the saved Adam moments, with gentler learning rates for the
    # first unified update sequence; all existing legacy groups untouched.
    opts[4].param_groups[-2]['lr']=a.near_lr
    opts[4].param_groups[-1]['lr']=a.far_lr
    tr=gather_joint(a.root,model,base,comp,'train',a.train,a.seed+10,a.batch)
    dv=gather_joint(a.root,model,base,comp,'dev',a.dev,a.seed+20,a.batch)
    t_shuffle=wrong_far(a.root,model,base,comp,'train',tr,a.seed+10)
    d_shuffle=wrong_far(a.root,model,base,comp,'dev',dv,a.seed+20)
    d_donor=wrong_far(a.root,model,base,comp,'dev',dv,a.seed+30,'train_donor')
    initial=evaluate(model,dv,d_shuffle,d_donor)
    baseline=copy.deepcopy(initial)
    history=[];candidate=None;best=None
    rng=np.random.default_rng(a.seed+100)
    for step in range(1,a.steps+1):
        ids=torch.tensor(rng.choice(a.train,min(a.batch_train,a.train),replace=False),dtype=torch.long)
        good=take(tr,ids);wrong=take(t_shuffle,ids)
        c,_=model.joint_nll(good)
        bad,_=model.joint_nll(wrong)
        adv=(bad-c).mean()
        objective=c.mean()+a.contrast*F.softplus((a.margin-adv)*12)/12
        opts[4].zero_grad(set_to_none=True)
        objective.backward()
        torch.nn.utils.clip_grad_norm_(list(model.near.parameters())+list(model.far.parameters()),1.0)
        opts[4].step()
        if step==1 or step%a.every==0 or step==a.steps:
            ev=evaluate(model,dv,d_shuffle,d_donor)
            history.append({'step':step,'dev':ev})
            # Correct memory must beat two strictly causal controls, and
            # held-out dev must beat frozen R7.18 from the same old corpus.
            acceptable=(ev['correct_nll'] < baseline['correct_nll']-a.min_gain and
                        ev['shuffled_advantage']>a.min_adv and
                        ev['independent_train_split_donor_advantage']>a.min_adv)
            if acceptable and (best is None or ev['correct_nll']<best['correct_nll']):
                best=dict(ev,step=step)
                candidate={'model':copy.deepcopy(model.state_dict()),
                           'optimizers':[copy.deepcopy(opt.state_dict()) for opt in opts]}
    if candidate is not None:
        model.load_state_dict(candidate['model'])
        for o,s in zip(opts,candidate['optimizers']):o.load_state_dict(s)
    else:
        model.load_state_dict(blob['model'])
        for o,s in zip(opts,originals):o.load_state_dict(s)
    assert all(torch.equal(v,model.core.state_dict()[k]) for k,v in core_initial.items())
    assert sha256(parent)==digest
    accepted=best is not None
    if accepted:
        report_path=pathlib.Path(a.output)
        metrics={'status':'R7.21 unified candidate selected on reused historical dev; not promoted',
                 'historical_dev':best,'independent_books':False}
        save=write_candidate(a.output_checkpoint,model,opts,a.root,a.near,a.far,
                             candidate_updates=best['step'],selection=metrics)
        loaded,restored,_,_,meta=load_candidate(a.output_checkpoint,a.root)
        assert all(torch.equal(x,loaded.state_dict()[k]) for k,x in model.state_dict().items())
        assert all(deep_equal(x.state_dict(),y.state_dict()) for x,y in zip(opts,restored))
    else:save=None
    for i in range(4):
        assert deep_equal(originals[i],opts[i].state_dict()),f'inherited optimizer {i} modified'
    # Fifth original parameter groups are not altered by adapter-only steps.
    now=opts[4].state_dict();old=originals[4]
    assert all(deep_equal(old['state'][k],now['state'][k]) for k in old['state'] if k<25)
    result={'round':'FA1 R7.21 single unified model','accepted_parent_step':14416,
            'joint_adapter_updates_executed':a.steps,
            'selected_joint_adapter_updates':best['step'] if best is not None else 0,
            'historical_dev_initial':initial,'historical_dev_selected':best,
            'history':history,'training_objective':'joint near+far CE + shuffled-history contrastive',
            'joint_optimizer_count':len(opts),'all_original_trunk_parameters_unchanged':True,
            'all_original_adam_moments_unchanged':True,'candidate_saved':accepted,
            'saved_candidate':save,'checkpoint_digest_unchanged':sha256(parent)==digest,
            'new_book_validation':False,'promoted':False,'elapsed_s':time.monotonic()-start}
    pathlib.Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    pathlib.Path(a.output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='history'},indent=2),flush=True)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='/mnt/data/fa1_full')
    p.add_argument('--near',default='/mnt/data/fa1r720/adaptation/fa1_r716_225.pt')
    p.add_argument('--far',default='/mnt/data/fa1r720/adaptation/fa1_r718_full_candidate.pt')
    p.add_argument('--checkpoint',default='/mnt/data/fa1r721/checkpoint/r7_unified_candidate.pt')
    p.add_argument('--output-checkpoint',default='/mnt/data/fa1r721/checkpoint/r7_unified_trained_candidate.pt')
    p.add_argument('--output',default='/mnt/data/fa1r721/reports/R721_JOINT_TRAIN.json')
    p.add_argument('--train',type=int,default=256);p.add_argument('--dev',type=int,default=128)
    p.add_argument('--batch',type=int,default=8);p.add_argument('--batch-train',type=int,default=64)
    p.add_argument('--steps',type=int,default=96);p.add_argument('--every',type=int,default=16)
    p.add_argument('--seed',type=int,default=72121);p.add_argument('--threads',type=int,default=2)
    p.add_argument('--near-lr',type=float,default=0.0001);p.add_argument('--far-lr',type=float,default=0.0001)
    p.add_argument('--contrast',type=float,default=.15);p.add_argument('--margin',type=float,default=.02)
    p.add_argument('--min-gain',type=float,default=.0005);p.add_argument('--min-adv',type=float,default=.0001)
    run(p.parse_args())