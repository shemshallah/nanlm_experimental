#!/usr/bin/env python3
"""R7.15 controlled long-range copy diagnostic with disjoint new query IDs.

Synthetic is a *mechanism check only*, not evidence of natural-language gains.
The comparison is to a 32,000-way uniform probability baseline on random keys;
FA1's trained natural-language baseline is NOT evaluated on synthetic IDs.
"""
import argparse, pathlib, json, copy, time
import numpy as np, torch
from torch import nn
from torch.nn import functional as F
from r712_validate import restore as restore_r712
from fa1_r715_mirror import QueryMirrorLM,wrong_history,slice_d,judge
from r74_stream import OLDER
from r710_checkpoint_integrity import sha256

def dataset(n,seed,lo,hi,E,k=96,batch=8):
    rng=np.random.default_rng(seed)
    H=rng.integers(2000,9000,size=(n,OLDER),dtype=np.int32)
    qs=rng.integers(lo,hi,size=n,dtype=np.int64)
    pv=rng.integers(9000,20000,size=n,dtype=np.int64)
    targets=rng.integers(24000,32000,size=n,dtype=np.int64)
    for i in range(n):
        # Exactly one true key/value lies within the 6144-token older history.
        pos=int(rng.integers(256,OLDER-256))
        H[i,pos-1]=pv[i];H[i,pos]=qs[i];H[i,pos+1]=targets[i]
        # Hard negatives: nearest keys among 256 random nonmatching IDs, with
        # unrelated successors. The sought value is NOT used in selection.
        pool=rng.integers(9000,24000,size=256,dtype=np.int64)
        pool=pool[pool!=qs[i]]
        es=F.normalize(E[torch.from_numpy(pool)],dim=-1)
        sim=(es*F.normalize(E[qs[i]],dim=-1)).sum(-1)
        distractors=pool[sim.topk(10).indices.numpy()]
        for d in distractors:
            j=int(rng.integers(16,OLDER-16))
            if abs(j-pos)>4:
                H[i,j]=int(d);H[i,j+1]=int(rng.integers(24000,32000))
    D={'history':torch.from_numpy(H.astype('int64')),'query':torch.from_numpy(qs),
       'prev_query':torch.from_numpy(pv),'prev_prev':torch.from_numpy(pv),
       'target':torch.from_numpy(targets),
       'base':torch.full((n,),float(np.log(32000))), 'n':n}
    Eunit=F.normalize(E.detach(),dim=-1)
    cols={key:[] for key in ['key','val','prev_key','age','valid','exact','exact2']}
    with torch.no_grad():
        for ids in torch.arange(n).split(batch):
            h=D['history'][ids];q=D['query'][ids];p=D['prev_query'][ids]
            keys=h[:,1:-1];vals=h[:,2:];pr=h[:,:-2]
            good=(keys>1)&(vals>1)
            sims=torch.bmm(Eunit[keys],Eunit[q].unsqueeze(-1)).squeeze(-1)
            sp=torch.bmm(Eunit[pr],Eunit[p].unsqueeze(-1)).squeeze(-1)
            exact=(keys==q[:,None])&good
            exact2=exact&(pr==p[:,None])
            recency=torch.linspace(-1.,0.,keys.shape[1])[None,:]
            sc=.8*sims+.2*sp+1.1*exact.float()+1.1*exact2.float()+.03*recency
            sc=sc.masked_fill(~good,-1e8)
            ix=sc.topk(k,dim=1).indices
            for nm,arr in (('key',keys),('val',vals),('prev_key',pr)):
                cols[nm].append(arr.gather(1,ix))
            cols['age'].append(ix.float()/keys.shape[1]-1)
            cols['valid'].append(good.gather(1,ix))
            cols['exact'].append(exact.gather(1,ix).float())
            cols['exact2'].append(exact2.gather(1,ix).float())
    D.update({key:torch.cat(val,0) for key,val in cols.items()})
    return D

def run(a):
    start=time.monotonic();torch.set_num_threads(a.threads);torch.manual_seed(716)
    root=pathlib.Path(a.root);parent=sha256(root/'checkpoint/pal_active.pt')
    model,opts,_,_,_,_=restore_r712(root,root/'adaptation/r712_gate_delta.pt')
    model.eval()
    for v in model.parameters():v.requires_grad_(False)
    E=model.trunk.E.detach()
    train=dataset(a.train,1,9000,14000,E,a.topk,a.batch)
    dev=dataset(a.dev,2,14000,19000,E,a.topk,a.batch)
    test=dataset(a.test,3,19000,24000,E,a.topk,a.batch)
    train_wrong=wrong_history(train,max(1,a.train//3),E,a.topk,a.batch)
    dev_wrong=wrong_history(dev,max(1,a.dev//3),E,a.topk,a.batch)
    test_wrong=wrong_history(test,max(1,a.test//3),E,a.topk,a.batch)
    head=QueryMirrorLM(E,model.r74.outer.dir_key,model.r74.outer.dir_value)
    opts[4].add_param_group({'params':list(head.parameters()),'lr':a.lr,'weight_decay':0.0})
    opt=opts[4]
    first=judge(head,dev,dev_wrong)
    best={'step':0,'nll':first['base_nll'],'state':None}
    hist=[];rng=np.random.default_rng(716)
    for step in range(1,a.updates+1):
        idx=torch.tensor(rng.choice(a.train,min(a.train,a.train_batch),replace=False))
        lc=head(slice_d(train,idx)).mean();lw=head(slice_d(train_wrong,idx)).mean()
        loss=lc+.20*F.softplus(lc-lw+.05)
        opt.zero_grad(set_to_none=True);loss.backward()
        nn.utils.clip_grad_norm_(head.parameters(),1.);opt.step()
        if step%a.eval_every==0 or step==1 or step==a.updates:
            d=judge(head,dev,dev_wrong)
            hist.append({'step':step,'dev':d})
            print(json.dumps({'phase':'SYNTHETIC','step':step,'dev_correct_nll':round(d['correct_nll'],5),
               'gain':round(d['correct_gain'],5),'correct_vs_wrong':round(d['correct_vs_wrong_advantage'],5)}),flush=True)
            if d['correct_gain']>.01 and d['correct_vs_wrong_advantage']>.01 and d['correct_nll']<best['nll']:
                best={'step':step,'nll':d['correct_nll'],'state':copy.deepcopy(head.state_dict())}
    if best['state']:head.load_state_dict(best['state'])
    selected=judge(head,dev,dev_wrong)
    heldout=judge(head,test,test_wrong)
    with torch.no_grad():
        oldscore=head.scores.detach().clone()
        head.scores[4:6]=0 # ablate symmetric and antisymmetric mirror readrails
    no_rails=judge(head,test,test_wrong)
    with torch.no_grad():
        head.scores.copy_(oldscore)
        head.scores[2]=0 # ablate explicit 2-token exact-context contribution
    no_context2=judge(head,test,test_wrong)
    with torch.no_grad():head.scores.copy_(oldscore)
    # the learned mechanism must outperform its own rail ablation to attribute
    # any gains specifically to S+/S- rather than exact key matching.
    report={'experiment':'FA1 R7.15 synthetic long-range hidden key/value copy',
       'source':'synthetic random token IDs; independent train/dev/test key ID bands',
       'status':'diagnostic only, not full LM checkpoint promotion',
       'accepted_parent':parent,'accepted_step':14416,'five_optimizer_histories_preserved':True,
       'uniform_baseline_not_original_LM':True,
       'history_distance_words':'older 2049..8192 tokens; one planted key-value pair',
       'hard_negatives':'10 embedding-nearest incorrect keys from random 256-key pool',
       'count':{'train':a.train,'dev':a.dev,'test':a.test},
       'first_dev':first,'best_step':best['step'],
       'selected_dev':selected,'held_out_test':heldout,'test_ablate_mirror_rails':no_rails,'test_ablate_explicit_two_token_bonus':no_context2,
       'training_log':hist,'seconds':round(time.monotonic()-start,2)}
    (root/'reports/FA1_R7p15_synthetic.json').write_text(json.dumps(report,indent=2)+'\n')
    torch.save({'format':'FA1_R715_SYNTH_DIAGNOSTIC_ONLY','parent':parent,
      'selected_step':best['step'],'parameters':head.state_dict()},
      root/'adaptation/fa1_r715_synthetic_diagnostic.pt')
    if parent!=sha256(root/'checkpoint/pal_active.pt'):raise RuntimeError('accepted changed')
    print(json.dumps({'phase':'SYNTHETIC_DONE','step':best['step'],'dev':report['selected_dev'],
                      'test':report['held_out_test'],'seconds':report['seconds']}),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',default='.');p.add_argument('--train',type=int,default=256)
 p.add_argument('--dev',type=int,default=128);p.add_argument('--test',type=int,default=128)
 p.add_argument('--updates',type=int,default=128);p.add_argument('--eval-every',type=int,default=16)
 p.add_argument('--topk',type=int,default=96);p.add_argument('--batch',type=int,default=8)
 p.add_argument('--train-batch',type=int,default=32);p.add_argument('--threads',type=int,default=2)
 p.add_argument('--lr',type=float,default=.008);run(p.parse_args())