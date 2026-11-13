#!/usr/bin/env python3
"""R7.24 geometry-only NaNROUTER CPU pilot; no checkpoint promotion.

This resumes a single unpromoted R7.23 Diamond with all five optimizer
histories, adds rank2/rank3 signed geometry and sparse rank4 tetra correction,
freezes every inherited tensor, and compares on whole-book-disjoint DEV.
Never uses the held-out TEST to tune weights or thresholds.
"""
from __future__ import annotations
import argparse,copy,json,math,pathlib,random,time
import numpy as np,torch
import torch.nn.functional as F
from r710_checkpoint_integrity import sha256,deep_equal
from fa1_r724_unified import migrate,save_geometric,load_geometric
from fa1_r723_train import prepare,forward,take
from fa1_r725_controls import unrelated_book_context

GEOMETRY_GATES=('edge_gain','face_gain','tetra_gain','meta_gate')

def scalar(x):return float(x.detach())

def boot_books(delta,bookids,seed):
    ids=bookids.cpu().numpy();unique=np.unique(ids);values=delta.detach().cpu().numpy()
    book_means=np.asarray([float(values[ids==k].mean()) for k in unique])
    rng=np.random.default_rng(seed)
    boot=rng.choice(book_means,size=(1024,len(unique)),replace=True).mean(axis=-1)
    return {'books':len(unique),'book_average':float(book_means.mean()),
      'book_bootstrap_ci95':list(map(float,np.percentile(boot,[2.5,97.5])))}

@torch.no_grad()
def compare(m,data,seed):
    m.eval()
    # Reference: SAME R7.23 weights / old five optimizers; only all three
    # new geometry gates switched off, including the tetra-conditioned apex
    # specialist. Leaving meta_gate live invalidates the geometry baseline.
    g=m.diamond.geometry
    gates=[getattr(g,k) for k in GEOMETRY_GATES]
    current=[x.detach().clone() for x in gates]
    for p in gates:p.zero_()
    old=forward(m,data)[0].detach()
    for p,x in zip(gates,current):p.copy_(x)
    candidate,ex=forward(m,data)
    foreign,donors=unrelated_book_context(data,seed)
    unrelated,_=forward(m,foreign)
    no_pair,_=forward(m,data,'no_pair')
    gain=old-candidate
    adv=unrelated-candidate
    ret={'old_r723_nll':scalar(old.mean()),'geom_nll':scalar(candidate.mean()),
         'gain':scalar(gain.mean()),'correct_vs_unrelated':scalar(adv.mean()),
         'pair_down_advantage':scalar((no_pair-candidate).mean()),
         'geometry_gains':{k:scalar(getattr(g,k)) for k in GEOMETRY_GATES},
         'foreign_book_control_count':int(donors.numel()),
         'geometry_metrics':{k:scalar(v) for k,v in ex['trace']['geometry'].items()},
         'active_apex_specialists':int(ex['trace']['selected_sectors'][:,0].unique().numel())}
    if 'book_ids' in data:
        ret['book_gain']=boot_books(gain,data['book_ids'],seed)
        ret['book_history']=boot_books(adv,data['book_ids'],seed+1)
    return ret


def run(a):
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed);np.random.seed(a.seed);random.seed(a.seed)
    if a.resume_geometric and pathlib.Path(a.resume_geometric).resolve()==pathlib.Path(a.checkpoint).resolve():
        raise ValueError('continuation output must differ from the protected source checkpoint')
    begin=time.monotonic()
    source=a.resume_geometric or a.parent
    old_sha=sha256(source)
    if a.resume_geometric:
        m,opts,base,comp,meta=load_geometric(source,a.root,threads=a.threads)
        prior_updates=int(meta['new_geometry_updates'])
    else:
        m,opts,base,comp,meta=migrate(source,a.root,from_r723=True,threads=a.threads)
        prior_updates=0
    frozen={k:v.detach().clone() for k,v in m.state_dict().items() if not k.startswith('diamond.geometry.')}
    prior_opts=[copy.deepcopy(o.state_dict()) for o in opts]
    for p in m.parameters():p.requires_grad_(False)
    for p in m.diamond.geometry.parameters():p.requires_grad_(True)
    print('PREP real books; accepted parent untouched',flush=True)
    tr=prepare(a,m,base,comp,'train',a.train,a.seed+10)
    dv=prepare(a,m,base,comp,'dev',a.dev,a.seed+20)
    if 'book_ids' not in tr or 'book_ids' not in dv:
        raise ValueError('R7.25 requires book-disjoint NaNROUTER corpus labels')
    if tr['book_ids'].unique().numel()<2 or dv['book_ids'].unique().numel()<2:
        raise ValueError('R7.25 foreign-book controls require at least two books per split')
    old=compare(m,dv,a.seed+30)
    print('BASELINE',json.dumps(old),flush=True)
    if not a.resume_geometric:
        with torch.no_grad():
            for k in GEOMETRY_GATES:getattr(m.diamond.geometry,k).fill_(0.025)
    opt=opts[4]
    opt.param_groups[-1]['lr']=a.lr
    opt.param_groups[-1]['weight_decay']=a.weight_decay
    best=None;best_weights=None;best_opts=None;log=[]
    rng=np.random.default_rng(a.seed+40)
    for step in range(1,a.steps+1):
        ids=torch.tensor(rng.choice(a.train,size=min(a.train,a.minibatch),replace=False))
        sample=take(tr,ids)
        if sample['book_ids'].unique().numel()<2:
            original=sample['book_ids'][0]
            alternatives=torch.where(tr['book_ids']!=original)[0]
            ids[0]=alternatives[int(rng.integers(len(alternatives)))]
            sample=take(tr,ids)
        m.train()
        correct,extras=forward(m,sample)
        foreign,_=unrelated_book_context(sample,a.seed+step)
        wrong,_=forward(m,foreign)
        advantage=(wrong-correct).mean()
        reg=F.softplus((a.margin-advantage)*8)/8
        loss=correct.mean()+a.contrast*reg
        opt.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(m.diamond.geometry.parameters(),1.)
        opt.step()
        if step==1 or step%a.every==0 or step==a.steps:
            ev=compare(m,dv,a.seed+100+step)
            ev.update({'step':step,'train_loss':scalar(loss),
                       'train_history_advantage':scalar(advantage)})
            # DEV choice is predeclared. TEST is never used for selection.
            gates=ev['gain']>a.min_gain and ev['correct_vs_unrelated']>0 and ev['pair_down_advantage']>0
            if a.resume_geometric:
                gates=gates and ev['geom_nll']<old['geom_nll']-a.min_continue_gain
            if a.require_book_ci and 'book_gain' in ev:
                gates=gates and ev['book_gain']['book_bootstrap_ci95'][0]>0
            ev['passes_dev_gates']=bool(gates)
            log.append(ev)
            print('STEP',step,json.dumps(ev),flush=True)
            if gates and (best is None or ev['geom_nll']<best['geom_nll']):
                best=ev;best_weights=copy.deepcopy(m.diamond.geometry.state_dict());best_opts=copy.deepcopy(opt.state_dict())
            if step>=a.min_steps and not any(x['passes_dev_gates'] for x in log) and a.stop_if_no_gate:
                print('STOP: no development improvements satisfying causal controls',flush=True)
                break
    for k,v in frozen.items():assert torch.equal(m.state_dict()[k],v),('inherited drift',k)
    for j in range(4):assert deep_equal(opts[j].state_dict(),prior_opts[j]),('prior optimizer changed',j)
    # Only geometry has new/updated states. Preserve every pre-Diamond Adam
    # moment; existing geometric moments must CONTINUE rather than be reset.
    old5=prior_opts[4]['state'];new5=opts[4].state_dict()['state']
    inherited_ids={i for group in prior_opts[4]['param_groups'][:-1] for i in group['params']}
    for i in inherited_ids:
        if i in old5:assert deep_equal(old5[i],new5[i]),('inherited fifth moments drift',i)
    report={'round':'FA1 R7.24 oriented geometric diamond CPU pilot',
       'source_checkpoint_sha256':old_sha,'parent_candidate_unpromoted':True,
       'resumed_geometry_from_unpromoted':bool(a.resume_geometric),
       'prior_geometry_updates':prior_updates,
       'inherited_optimizers':5,'retained_prior_optimizer_slots':True,
       'control':'different-book donors, target-free, all Diamond context swapped',
       'dataset':'NaNROUTER whole-book-disjoint corpus' if a.corpus_dir else 'historical corpus',
       'train_targets':a.train,'dev_targets':a.dev,'steps_attempted':step,
       'old_dev':old,'history':log,'best':best,
       'accepted_model_promoted':False,'test_books_consulted':False,
       'byte_bits_per_byte_measured':False,'runtime_seconds':time.monotonic()-begin}
    if best is not None:
        m.diamond.geometry.load_state_dict(best_weights)
        opt.load_state_dict(best_opts)
        report['selected_unpromoted_checkpoint']=save_geometric(a.checkpoint,m,opts,old_sha,update=prior_updates+best['step'],notes={'development_gate':best})
        restored,old_opts,_,_,_=load_geometric(a.checkpoint,a.root)
        assert all(torch.equal(x,restored.state_dict()[k]) for k,x in m.state_dict().items())
        assert all(deep_equal(o.state_dict(),v.state_dict()) for o,v in zip(opts,old_opts))
        report['roundtrip_verified']=True
    else:report['selected_unpromoted_checkpoint']=None
    pathlib.Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    pathlib.Path(a.output).write_text(json.dumps(report,indent=2)+'\n')
    print('FINAL',json.dumps({k:v for k,v in report.items() if k!='history'},indent=2),flush=True)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',default='/mnt/data/diamond_r723/r7_ancestor')
    p.add_argument('--parent',default='/mnt/data/diamond_r723/R723_DIAMOND_STRICT_DOWNPATH_UNPROMOTED.pt')
    p.add_argument('--resume-geometric',default='',help='resume a saved unpromoted R7.24 checkpoint with all five Adam histories')
    p.add_argument('--corpus-dir',default='/mnt/data/diamond_r723/nanrouter_verified/processed')
    p.add_argument('--output',default='/mnt/data/diamond_r724/cpu_pilot.json')
    p.add_argument('--checkpoint',default='/mnt/data/diamond_r724/FA1_R724_GEOMETRY_CANDIDATE_UNPROMOTED.pt')
    p.add_argument('--train',type=int,default=256)
    p.add_argument('--dev',type=int,default=96)
    p.add_argument('--steps',type=int,default=128)
    p.add_argument('--minibatch',type=int,default=24)
    p.add_argument('--batch',type=int,default=8)
    p.add_argument('--lr',type=float,default=0.0003)
    p.add_argument('--weight-decay',type=float,default=0.)
    p.add_argument('--contrast',type=float,default=0.4)
    p.add_argument('--margin',type=float,default=0.001)
    p.add_argument('--min-gain',type=float,default=0.0001)
    p.add_argument('--min-continue-gain',type=float,default=0.)
    p.add_argument('--min-steps',type=int,default=64)
    p.add_argument('--every',type=int,default=32)
    p.add_argument('--stop-if-no-gate',action='store_true')
    p.add_argument('--require-book-ci',action='store_true')
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--seed',type=int,default=724)
    run(p.parse_args())
