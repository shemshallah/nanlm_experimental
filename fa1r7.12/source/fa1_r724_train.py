#!/usr/bin/env python3
"""R7.24 local CPU continuation: geometric S+/S-, triangle/tetra curvature.

Inherited 14416-step R7.22 and R7.23 optimizer histories preserved. Stage
only unpromoted candidates. Sealed R7.23 test was used already; NEVER use it
for model selection here. New independent books required for capability claim.
"""
from __future__ import annotations
import argparse,copy,json,pathlib,time,random,math
import numpy as np,torch
from torch.nn import functional as F
from r7_core import Corpus
from r710_checkpoint_integrity import sha256,deep_equal
from fa1_r723_train import prepare,take,forward
from fa1_r724_unified import extend_r723,save_r724,load_r724

@torch.no_grad()
def evaluate(m,data,seed=724):
    m.eval()
    old=m.joint_nll(data,diamond_enabled=False)[0]
    full,extra=forward(m,data)
    swap,_=forward(m,data,'unrelated')
    apex,_=forward(m,data,'no_apex')
    pair,_=forward(m,data,'no_pair')
    no_geo,_,_=m.joint_nll(data,return_extras=True,without_geometry=True)
    g=extra['trace'].get('geometry')
    res={'base_r722_nll':float(old.mean()),'diamond_r724_nll':float(full.mean()),
         'improve_over_r722':float((old-full).mean()),
         'improve_over_r723':float((no_geo-full).mean()),
         'correct_vs_unrelated':float((swap-full).mean()),
         'apex_ablation':float((apex-full).mean()),
         'pair_ablation':float((pair-full).mean()),
         'curvature_closure':float(g['closure_loss']) if g else None,
         'geometry_gate':float(g['geo_gate']) if g else None,
         'sector_diversity':int(extra['trace']['selected_sectors'][:,0].unique().numel())}
    if 'book_ids' in data:
        ids=data['book_ids'].detach().numpy()
        unique=np.unique(ids)
        res['dev_books']=int(len(unique))
        for label,diff in [('improve_over_r723',no_geo-full),('correct_vs_unrelated',swap-full),
                           ('improve_over_r722',old-full),('apex_ablation',apex-full),('pair_ablation',pair-full)]:
            x=diff.detach().numpy()
            means=np.array([x[ids==doc].mean() for doc in unique],dtype=np.float64)
            rng=np.random.default_rng(seed+len(unique))
            boot=means[rng.integers(0,len(means),size=(1500,len(means)))].mean(1)
            res[label+'_cluster_CI95']=[float(z) for z in np.percentile(boot,[2.5,97.5])]
            res[label+'_book_wins']=int((means>0).sum())
    return res


def run(a):
    t=time.monotonic()
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed);np.random.seed(a.seed);random.seed(a.seed)
    base_sha=sha256(a.parent)
    m,opts,base,comp,meta=extend_r723(a.parent,a.root,threads=a.threads)
    inherited={k:v.clone() for k,v in m.state_dict().items() if not k.startswith('diamond.geometry.')}
    old_state=[copy.deepcopy(o.state_dict()) for o in opts[:4]]
    fifth_state=copy.deepcopy(opts[4].state_dict())
    for p in m.parameters():p.requires_grad_(False)
    if a.joint_diamond:
        for p in m.diamond.parameters():p.requires_grad_(True)
    else:
        for p in m.diamond.geometry.parameters():p.requires_grad_(True)
    m.diamond.train()
    print('PREPARE causal R7.24 data, inherited candidate',base_sha,flush=True)
    tr=prepare(a,m,base,comp,'train',a.train,a.seed+21)
    dv=prepare(a,m,base,comp,'dev',a.dev,a.seed+22)
    with torch.no_grad():
        f,_=forward(m,dv)
        old,_,_=m.joint_nll(dv,return_extras=True,without_geometry=True)
        assert torch.equal(f,old),'R7.24 initialization not exactly R7.23'
    initial=evaluate(m,dv,a.seed)
    print('INIT',json.dumps(initial),flush=True)
    if initial['dev_books']<4:raise ValueError('Not enough independent dev books')
    opt=opts[4]
    opt.param_groups[-1]['lr']=a.lr
    if a.joint_diamond:opt.param_groups[-2]['lr']=a.lr_diamond
    rng=np.random.default_rng(a.seed+23)
    best=None;best_state=None;history=[]
    for step in range(1,a.steps+1):
        m.diamond.train()
        ix=torch.tensor(rng.choice(a.train,min(a.minibatch,a.train),replace=False),dtype=torch.long)
        batch=take(tr,ix)
        nll,full_aux=forward(m,batch)
        bad,_=forward(m,batch,'unrelated')
        no_geo,_,_=m.joint_nll(batch,return_extras=True,without_geometry=True)
        g=full_aux['trace']['geometry']
        assert g is not None
        adv=(bad-nll).mean()
        local_gain=(no_geo-nll).mean()
        branch=full_aux['branch_nll'].mean()
        if a.branch_weight>0:
            _,bad_aux=forward(m,batch,'unrelated')
            branch_adv=(bad_aux['branch_nll']-full_aux['branch_nll']).mean()
        else:branch_adv=branch.new_zeros(())
        if a.apex_contrast>0:
            _,no_apex=forward(m,batch,'no_apex')
            apex_adv=(no_apex['branch_nll']-full_aux['branch_nll']).mean()
        else:apex_adv=branch.new_zeros(())
        if a.pair_contrast>0:
            _,no_pair=forward(m,batch,'no_pair')
            pair_adv=(no_pair['branch_nll']-full_aux['branch_nll']).mean()
        else:pair_adv=branch.new_zeros(())
        loss=(nll.mean()+a.context_contrast*F.softplus((a.history_margin-adv)*12)/12
             +a.geometry_contrast*F.softplus((a.geometry_margin-local_gain)*12)/12
             +a.branch_weight*(branch+.7*F.softplus((a.history_margin-branch_adv)*12)/12)
             +a.apex_contrast*F.softplus((a.apex_margin-apex_adv)*12)/12
             +a.pair_contrast*F.softplus((a.apex_margin-pair_adv)*12)/12
             +a.closure_weight*g['closure_loss'])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.diamond.parameters() if a.joint_diamond
                                       else m.diamond.geometry.parameters(),1.)
        opt.step()
        if step==1 or step%a.eval_every==0 or step==a.steps:
            metrics=evaluate(m,dv,a.seed+3)
            gates=(metrics['improve_over_r722']>a.min_gain and
                   metrics['improve_over_r723']>a.min_gain and
                   metrics['correct_vs_unrelated']>a.min_adv and
                   metrics['apex_ablation']>0 and metrics['pair_ablation']>0 and
                   metrics['sector_diversity']>=10 and
                   (not a.require_cluster_ci or (
                     metrics['improve_over_r723_cluster_CI95'][0]>0 and
                     metrics['correct_vs_unrelated_cluster_CI95'][0]>0)))
            metrics.update({'update':step,'train_loss':float(loss.detach()),
                           'train_history_adv':float(adv.detach()),
                           'train_geometry_gain':float(local_gain.detach()),
                           'passes_gates':bool(gates)})
            history.append(metrics)
            print('UPDATE',step,json.dumps(metrics),flush=True)
            if gates and (best is None or metrics['diamond_r724_nll']<best['diamond_r724_nll']):
                best=dict(metrics)
                best_state=(copy.deepcopy((m.diamond if a.joint_diamond else m.diamond.geometry).state_dict()),
                            copy.deepcopy(opts[4].state_dict()))
            if step>=a.min_steps and a.stop_on_plateau and not any(r['passes_gates'] for r in history):
                print('EARLY STOP no development-qualified geometry path',flush=True)
                break
    if best_state is not None:
        (m.diamond if a.joint_diamond else m.diamond.geometry).load_state_dict(best_state[0])
        opts[4].load_state_dict(best_state[1])
    assert sha256(a.parent)==base_sha
    assert all(k in m.state_dict() and torch.equal(m.state_dict()[k],v) for k,v in inherited.items() if not a.joint_diamond or not k.startswith('diamond.'))
    assert all(deep_equal(o.state_dict(),s) for o,s in zip(opts[:4],old_state))
    current=opts[4].state_dict()
    assert current['param_groups'][:-1]==fifth_state['param_groups'][:-1]
    if not a.joint_diamond:
        assert all(deep_equal(current['state'][k],v) for k,v in fifth_state['state'].items())
    else:
        assert set(fifth_state['state']).issubset(current['state'])
        for k,entry in fifth_state['state'].items():
            assert int(current['state'][k]['step'])>=int(entry['step'])
    report={'round':'FA1 R7.24 trig mirror curvature + tetrahedral consistency',
            'source':'SHA-pinned NaNROUTER whole-book train/dev; old R7.23 sealed test untouched',
            'seed_sha256':base_sha,'inherited_optimizers':5,'original_optimizer_history_bitwise':True,
            'checkpoint_accepted_step':14416,'r723_candidate_updates':meta['r723_updates'],
            'joint_diamond_training':bool(a.joint_diamond),
            'steps_attempted':step,'best_unpromoted':best,'initial':initial,
            'history':history,'elapsed_seconds':time.monotonic()-t,
            'new_sealed_test_untouched':True,'auto_promoted':False,
            'bits_per_byte':None}
    if best is not None:
        path=pathlib.Path(a.candidate)
        report['unpromoted_candidate']=save_r724(path,m,opts,meta,best['update'],best)
        if a.roundtrip:
            restored,ropts,_,_,_=load_r724(path,a.root)
            assert all(torch.equal(restored.state_dict()[k],v) for k,v in m.state_dict().items())
            assert all(deep_equal(x.state_dict(),y.state_dict()) for x,y in zip(opts,ropts))
            report['checkpoint_roundtrip']=True
    else:report['unpromoted_candidate']=None
    pathlib.Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    pathlib.Path(a.output).write_text(json.dumps(report,indent=2)+'\n')
    print('FINISHED',json.dumps({k:v for k,v in report.items() if k!='history'},indent=2),flush=True)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',default='/mnt/data/diamond_r723/r7_ancestor')
    p.add_argument('--parent',default='/mnt/data/diamond_r723/R723_DIAMOND_STRICT_DOWNPATH_UNPROMOTED.pt')
    p.add_argument('--corpus-dir',default='/mnt/data/diamond_r723/nanrouter_verified/processed')
    p.add_argument('--output',default='/mnt/data/diamond_r723/r724_cpu_pilot.json')
    p.add_argument('--candidate',default='/mnt/data/diamond_r723/R724_CURVED_DIAMOND_UNPROMOTED.pt')
    p.add_argument('--train',type=int,default=640)
    p.add_argument('--dev',type=int,default=256)
    p.add_argument('--minibatch',type=int,default=16)
    p.add_argument('--batch',type=int,default=8)
    p.add_argument('--steps',type=int,default=128)
    p.add_argument('--eval-every',type=int,default=32)
    p.add_argument('--min-steps',type=int,default=96)
    p.add_argument('--lr',type=float,default=.0006)
    p.add_argument('--closure-weight',type=float,default=.01)
    p.add_argument('--joint-diamond',action='store_true')
    p.add_argument('--lr-diamond',type=float,default=.00015)
    p.add_argument('--branch-weight',type=float,default=0.)
    p.add_argument('--apex-contrast',type=float,default=0.)
    p.add_argument('--pair-contrast',type=float,default=0.)
    p.add_argument('--apex-margin',type=float,default=.002)
    p.add_argument('--context-contrast',type=float,default=.7)
    p.add_argument('--geometry-contrast',type=float,default=.4)
    p.add_argument('--history-margin',type=float,default=.002)
    p.add_argument('--geometry-margin',type=float,default=.001)
    p.add_argument('--min-gain',type=float,default=.0001)
    p.add_argument('--min-adv',type=float,default=.0001)
    p.add_argument('--require-cluster-ci',action='store_true')
    p.add_argument('--stop-on-plateau',action='store_true')
    p.add_argument('--roundtrip',action='store_true')
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--seed',type=int,default=72424)
    run(p.parse_args())