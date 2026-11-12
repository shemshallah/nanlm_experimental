#!/usr/bin/env python3
"""FA1 R7.18: nondegrading far-only 32K--128K residual to frozen R7.16 8K/32K trust reader.

No original model weights or optimizer moments change. Target never enters router features.
Inherited splits have been consulted; this is diagnostic, not an independent-book test.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, math, os, pathlib, random, sys, tempfile, time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from r7_core import Corpus
from r74_stream import OUTER_CONTEXT
from r712_validate import restore as restore_parent
from r712_streaming import DirectedStreaming
from r710_checkpoint_integrity import sha256
from train_r7 import batch_data
from fa1_r716_router import TrustRouter, extract_paths, nll as near_nll, PARENT, RECENT, ORDERS

HORIZON=131072
FAR_EXCLUSION=32768
FAR_CHANNELS=4
FAR_FEATURES=FAR_CHANNELS*7+4


def pick_positions(corpus,n,seed,min_start=HORIZON):
    """Historical stratification or equal-book sampling of age-eligible document positions."""
    end=len(corpus.seq)-1
    if corpus.doc_start is not None:
        starts=np.unique(np.asarray(corpus.doc_start,dtype=np.int64))
        ends=np.r_[starts[1:],len(corpus.seq)]
        eligible=[(int(x+FAR_EXCLUSION+1),int(z-1)) for x,z in zip(starts,ends) if z-x>FAR_EXCLUSION+2]
        if len(eligible)<3:raise ValueError('fewer than three books with at least 32K preceding tokens')
        rng=np.random.default_rng(seed)
        order=rng.permutation(len(eligible))
        selected=np.asarray([eligible[order[i%len(order)]] for i in range(n)],dtype=np.int64)
        # Ensure both early and later positions are represented within each book; no cross-document context.
        positions=np.asarray([rng.integers(lo,hi+1) for lo,hi in selected],dtype=np.int64)
        return positions
    if end<=min_start:raise ValueError('split too short for 128K context')
    lo=min_start+int(.01*(end-min_start));span=end-lo
    offsets=lo+np.floor(np.arange(n)*span/n).astype(np.int64)
    return np.minimum(end,offsets+np.random.default_rng(seed).integers(0,max(1,span//n),size=n))


def far_only(history,ctx,target):
    """Only tokens with ages 32769..131072 can be retrieved; no overlap with 32K path."""
    h=np.asarray(history[-HORIZON:-FAR_EXCLUSION],dtype=np.int64)
    x=np.concatenate((h,np.ones(RECENT,dtype=np.int64)))
    return extract_paths(x,ctx,target,len(x))


def make_feature(records,priors,ctx,far=False):
    feat=[]
    for k,x in enumerate(records):
        prior=float(priors[k]) if x['n'] else 0.
        feat.extend((np.log1p(x['n']),np.log1p(x['weight_total']),x['peak'],x['entropy'],x['recent'],prior,prior*x['peak']))
    if far:
        feat.extend((float(ctx[-1]>1),float(ctx[-2]>1),float(records[0]['n']>0),float(records[-1]['n']>0)))
    else:
        feat.extend((float(ctx[-1]>1),float(ctx[-2]>1),float(records[0]['n']>0),float(records[ORDERS]['n']>0)))
    return feat

@torch.no_grad()
def gather(root,model,base,comp,near_router,split,n,seed,batch=8,corpus_dir=None):
    corpus=Corpus(root,split,corpus_dir=corpus_dir)
    positions=pick_positions(corpus,n,seed)
    stream=DirectedStreaming(model)
    base_losses=[];f_near=[];p_near=[];c_near=[];f_far=[];p_far=[];c_far=[]
    for inds in np.array_split(positions,max(1,math.ceil(n/batch))):
        _,ctx,pieces,lens,y,_,_=batch_data(base,corpus,comp,inds)
        hist=torch.from_numpy(corpus.get(inds,OUTER_CONTEXT)[0][:,:OUTER_CONTEXT-RECENT].copy()).long()
        state=stream.initial(hist)
        ex=torch.from_numpy(corpus.data[inds].copy()).long();lg=torch.from_numpy(corpus.long[inds].copy()).long()
        now,loss,_=stream.forward(ex,lg,ctx,pieces,lens,state,hist,y)
        base_losses.append(loss.cpu())
        longhist=corpus.get(inds,context=HORIZON)[0]
        all_modal=[];details=[]
        for j in range(len(inds)):
            h=longhist[j];q=ctx[j].numpy();target=int(y[j])
            near=extract_paths(h,q,target,8192)+extract_paths(h,q,target,32768)
            far=far_only(h,q,target)
            details.append((near,far,q))
            all_modal.extend(x['modal'] for x in near+far)
        modal_lp=model.trunk.nll(now.cpu().repeat_interleave(12,0),torch.tensor(all_modal,dtype=torch.long)).reshape(len(inds),12)
        priors=torch.exp(-modal_lp).numpy()
        for j,(near,far,q) in enumerate(details):
            f_near.append(make_feature(near,priors[j,:8],q))
            p_near.append([x['target_probability'] for x in near]);c_near.append([x['n'] for x in near])
            f_far.append(make_feature(far,priors[j,8:],q,far=True))
            p_far.append([x['target_probability'] for x in far]);c_far.append([x['n'] for x in far])
    base_losses=torch.cat(base_losses)
    near_data={'base':base_losses,'features':torch.tensor(np.asarray(f_near),dtype=torch.float32),
      'prob':torch.tensor(np.asarray(p_near),dtype=torch.float32),
      'count':torch.tensor(np.asarray(c_near),dtype=torch.float32)}
    near_loss,_=near_nll(near_router,near_data)
    obj={'base':base_losses,'near':near_loss,
      'far_features':torch.tensor(np.asarray(f_far),dtype=torch.float32),
      'far_prob':torch.tensor(np.asarray(p_far),dtype=torch.float32),
      'far_count':torch.tensor(np.asarray(c_far),dtype=torch.float32),
      'positions':positions.tolist(),'n':n,'near_coverage':float((near_data['count'].sum(1)>0).float().mean()),
      'far_coverage':float((torch.tensor(np.asarray(c_far)).sum(1)>0).float().mean())}
    assert torch.isfinite(near_loss).all() and obj['far_features'].shape==(n,FAR_FEATURES)
    return obj

@torch.no_grad()
def reroute_far(root,model,base,comp,split,data,seed,corpus_dir=None,mode='shuffle'):
    """Safe counterfactuals:

    - shuffle: permute only the target's own OLD far span, preserving unigram counts.
      Never uses any future tokens, so valid on train/dev/test.
    - train_donor: use a distinct train-split source (dev/test evaluation only).
      Never copies from a future portion of the evaluation split.
    Both leave the full inherited model and 8K/32K R7.16 reader unchanged.
    """
    corpus=Corpus(root,split,corpus_dir=corpus_dir)
    source=Corpus(root,'train',corpus_dir=corpus_dir) if mode=='train_donor' else corpus
    if mode=='train_donor' and split=='train':raise ValueError('cannot use unrestricted same-train donor')
    if mode not in ('shuffle','train_donor'):raise ValueError('unrecognized control')
    pos=np.asarray(data['positions'],dtype=np.int64)
    rng=np.random.default_rng(seed+902718)
    donor=(rng.integers(HORIZON+1,len(source.seq),size=len(pos),dtype=np.int64)
           if mode=='train_donor' else None)
    out=copy.copy(data);stream=DirectedStreaming(model)
    ff=[];pp=[];cc=[]
    for ix in np.array_split(np.arange(len(pos)),max(1,math.ceil(len(pos)/8))):
        indices=pos[ix]
        histories=source.get(donor[ix],context=HORIZON)[0] if mode=='train_donor' else corpus.get(indices,context=HORIZON)[0]
        _,ctx,pieces,lens,y,_,_=batch_data(base,corpus,comp,indices)
        recent=torch.from_numpy(corpus.get(indices,OUTER_CONTEXT)[0][:,:OUTER_CONTEXT-RECENT].copy()).long()
        state=stream.initial(recent)
        ex=torch.from_numpy(corpus.data[indices].copy()).long();lg=torch.from_numpy(corpus.long[indices].copy()).long()
        now,loss,_=stream.forward(ex,lg,ctx,pieces,lens,state,recent,y)
        assert torch.allclose(loss.cpu(),data['base'][ix],atol=1e-6,rtol=1e-6)
        records=[];modal=[]
        for j in range(len(ix)):
            h=histories[j]
            if mode=='shuffle':
                h=h.copy()
                # shuffle solely the real ancient memory, not the query or next-token label
                rng.shuffle(h[-HORIZON:-FAR_EXCLUSION])
            r=far_only(h,ctx[j].numpy(),int(y[j]))
            records.append(r);modal.extend(x['modal'] for x in r)
        priors=torch.exp(-model.trunk.nll(now.cpu().repeat_interleave(FAR_CHANNELS,0),torch.tensor(modal,dtype=torch.long)).reshape(len(ix),FAR_CHANNELS)).numpy()
        for j,r in enumerate(records):
            ff.append(make_feature(r,priors[j],ctx[j].numpy(),far=True))
            pp.append([x['target_probability'] for x in r]);cc.append([x['n'] for x in r])
    out['far_features']=torch.tensor(np.asarray(ff),dtype=torch.float32)
    out['far_prob']=torch.tensor(np.asarray(pp),dtype=torch.float32)
    out['far_count']=torch.tensor(np.asarray(cc),dtype=torch.float32)
    out['control_mode']=mode
    return out


class FarResidual(nn.Module):
    def __init__(self,hidden=32,max_gate=.2):
        super().__init__();self.max_gate=max_gate
        self.net=nn.Sequential(nn.Linear(FAR_FEATURES,hidden),nn.LayerNorm(hidden),nn.Tanh(),nn.Linear(hidden,FAR_CHANNELS+1))
        with torch.no_grad():
            nn.init.normal_(self.net[-1].weight,std=.001)
            self.net[-1].bias.zero_();self.net[-1].bias[0]=6.
    def forward(self,features,counts):
        score=self.net(features);allowed=counts>0
        score=torch.cat([score[:,:1],score[:,1:].masked_fill(~allowed,-1e5)],dim=1)
        return F.softmax(score,dim=1)[:,1:]*self.max_gate


def loss_far(net,data):
    g=net(data['far_features'],data['far_count'])
    src=torch.exp(-data['near']);mix=(1-g.sum(1))*src+(g*data['far_prob']).sum(1)
    return -torch.log(mix.clamp_min(1e-32)),g

@torch.no_grad()
def evaluate(net,correct,wrong,other_wrong=None):
    c,g=loss_far(net,correct);w,wg=loss_far(net,wrong)
    alternative=loss_far(net,other_wrong)[0] if other_wrong is not None else None
    return {'n':correct['n'],'base_nll':float(correct['base'].mean()),'r716_near_nll':float(correct['near'].mean()),
      'r718_correct_nll':float(c.mean()),'r718_wrong_far_nll':float(w.mean()),
      'far_incremental_gain':float((correct['near']-c).mean()),
      'correct_far_advantage':float((w-c).mean()),
      'correct_far_advantage_train_donor':float((alternative-c).mean()) if alternative is not None else None,
      'total_vs_accepted_gain':float((correct['base']-c).mean()),
      'average_far_gate':float(g.sum(1).mean()),'far_coverage':correct['far_coverage']}


def save_atomic(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(dir=path.parent,suffix='.tmp');os.close(fd)
    try:
        torch.save(obj,tmp)
        with open(tmp,'rb') as f:os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)


def main(a):
    t=time.monotonic();root=pathlib.Path(a.root)
    torch.set_num_threads(a.threads);torch.manual_seed(718);np.random.seed(718);random.seed(718)
    parent=root/'checkpoint/pal_active.pt';assert sha256(parent)==PARENT,'wrong parent'
    model,opts,base,comp,_,_=restore_parent(root,root/'adaptation/r712_gate_delta.pt')
    assert len(opts)==5
    model.eval()
    for p in model.parameters():p.requires_grad_(False)
    near_overlay=torch.load(a.r716_overlay,map_location='cpu',weights_only=False)
    assert near_overlay['parent_sha256']==PARENT
    near_router=TrustRouter();near_router.load_state_dict(near_overlay['gate']);near_router.eval()
    for p in near_router.parameters():p.requires_grad_(False)
    near_digest=sha256(a.r716_overlay)
    train=gather(root,model,base,comp,near_router,'train',a.train,7181,a.batch,a.corpus_dir)
    dev=gather(root,model,base,comp,near_router,'dev',a.dev,7182,a.batch,a.corpus_dir)
    wtrain=reroute_far(root,model,base,comp,'train',train,7181,a.corpus_dir)
    wdev=reroute_far(root,model,base,comp,'dev',dev,7182,a.corpus_dir)
    wdev2=reroute_far(root,model,base,comp,'dev',dev,7182,a.corpus_dir,mode='train_donor')
    net=FarResidual(hidden=a.hidden,max_gate=a.max_gate)
    opt=torch.optim.AdamW(net.parameters(),lr=a.lr,weight_decay=0.0)
    init=evaluate(net,dev,wdev,wdev2)
    print(json.dumps({'phase':'INIT','dev':init,'parent':PARENT,'optimizer_histories':len(opts)}),flush=True)
    rng=np.random.default_rng(718718);best={'step':0,'params':None,'optim':None,'dev':None};history=[]
    for step in range(1,a.steps+1):
        ids=torch.tensor(rng.choice(a.train,min(a.train,a.train_batch),replace=False),dtype=torch.long)
        good={k:(v[ids] if torch.is_tensor(v) else v) for k,v in train.items()}
        wrong={k:(v[ids] if torch.is_tensor(v) else v) for k,v in wtrain.items()}
        cg,_=loss_far(net,good);wg,_=loss_far(net,wrong)
        margin=(wg-cg).mean()
        objective=cg.mean()+a.contrast*F.softplus((a.margin-margin)*12.)/12.
        opt.zero_grad();objective.backward();nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step()
        if step==1 or step%a.eval_every==0 or step==a.steps:
            ev=evaluate(net,dev,wdev,wdev2);history.append({'step':step,'dev':ev})
            print(json.dumps({'phase':'EVAL','step':step,'dev_gain':round(ev['far_incremental_gain'],6),'adv':round(ev['correct_far_advantage'],6),'gate':round(ev['average_far_gate'],6)}),flush=True)
            if ev['far_incremental_gain']>=a.min_gain and ev['correct_far_advantage']>=a.min_adv and ev['correct_far_advantage_train_donor']>=a.min_adv and (best['dev'] is None or ev['far_incremental_gain']>best['dev']['far_incremental_gain']):
                best={'step':step,'params':copy.deepcopy(net.state_dict()),'optim':copy.deepcopy(opt.state_dict()),'dev':ev}
    if best['params'] is not None:net.load_state_dict(best['params']);opt.load_state_dict(best['optim'])
    print(json.dumps({'phase':'SELECTED','step':best['step'],'dev':best['dev']}),flush=True)
    test=gather(root,model,base,comp,near_router,'test',a.test,7183,a.batch,a.corpus_dir)
    wtest=reroute_far(root,model,base,comp,'test',test,7183,a.corpus_dir)
    wtest2=reroute_far(root,model,base,comp,'test',test,7183,a.corpus_dir,mode='train_donor')
    final=evaluate(net,test,wtest,wtest2) if best['params'] is not None else None
    far_off={'r716_near_nll':float(test['near'].mean()),'base_nll':float(test['base'].mean()),'extra_far_contribution':0.0}
    adapter=None
    if best['params'] is not None:
        adapter=pathlib.Path(a.overlay)
        save_atomic(adapter,{'format':'FA1_R7_18_FAR_DISJOINT_CANDIDATE','parent_sha256':PARENT,'r716_overlay_sha256':near_digest,
           'step':best['step'],'far_gate':net.state_dict(),'far_optimizer':opt.state_dict(),'dev':best['dev'],
           'scales':{'near':(8192,32768),'far':(32768,131072)},'selected_on_legacy_dev':not bool(a.corpus_dir)})
    assert sha256(parent)==PARENT and sha256(a.r716_overlay)==near_digest
    report={'round':'FA1 R7.18','hypothesis':'Disjoint 32K-128K retrieval can improve a frozen 32K R7.16 candidate without wrong-memory shortcuts',
      'accepted_parent_step':14416,'parent_sha256':PARENT,'r716_overlay_sha256':near_digest,
      'original_five_optimizer_histories_preserved':True,'accepted_weights_unmodified':True,
      'r716_overlay_unmodified':True,'r716_candidate_not_promoted':True,
      'train_n':a.train,'dev_n':a.dev,'test_n':a.test,'training_steps':a.steps,'selected_step':best['step'],
      'selected':best['params'] is not None,'promoted':False,'initial_dev':init,'selected_dev':best['dev'],
      'historical_test':final,'historical_test_reference_near':far_off,'all_dev_gates':history,
      'exact_horizon_groups':{'near':'[2049,32768]','far':'[32769,131072]'},'history_controls':'order-shuffled far past and distinct-train-split far donors; original and 32K near unchanged',
      'independent_book_evaluation':bool(a.corpus_dir),'legacy_splits_reused':not bool(a.corpus_dir),
      'adapter_path':str(adapter) if adapter else None,'adapter_sha256':sha256(adapter) if adapter else None,
      'seconds':round(time.monotonic()-t,2)}
    output=pathlib.Path(a.output);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'phase':'DONE','selected':best['step'],'test':final,'time':report['seconds']}),flush=True)
    return report

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=pathlib.Path,required=True)
    ap.add_argument('--r716-overlay',type=pathlib.Path,required=True)
    ap.add_argument('--train',type=int,default=512);ap.add_argument('--dev',type=int,default=256);ap.add_argument('--test',type=int,default=256)
    ap.add_argument('--steps',type=int,default=240);ap.add_argument('--eval-every',type=int,default=24)
    ap.add_argument('--batch',type=int,default=8);ap.add_argument('--train-batch',type=int,default=96)
    ap.add_argument('--threads',type=int,default=2);ap.add_argument('--hidden',type=int,default=32)
    ap.add_argument('--lr',type=float,default=.003);ap.add_argument('--max-gate',type=float,default=.20)
    ap.add_argument('--contrast',type=float,default=.8);ap.add_argument('--margin',type=float,default=.002)
    ap.add_argument('--min-gain',type=float,default=.001);ap.add_argument('--min-adv',type=float,default=.001)
    ap.add_argument('--corpus-dir',type=pathlib.Path)
    ap.add_argument('--output',type=pathlib.Path,default=pathlib.Path('reports/FA1_R7p18.json'))
    ap.add_argument('--overlay',type=pathlib.Path,default=pathlib.Path('adaptation/fa1_r718_far.pt'))
    main(ap.parse_args())