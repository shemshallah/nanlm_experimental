#!/usr/bin/env python3
"""FA1 R7.17: validated 32K/128K n-gram memory, CPU multi-scale trust router.

This is a causal experimental readout, NOT a new LM or a replacement checkpoint.
Parent 14,416-step PAL is immutable, including five optimizer histories.
Each history is limited to positions <= prediction_position-2049.
Do not claim independent-book evaluation when using the inherited legacy split.
"""
from __future__ import annotations
import argparse,copy,hashlib,json,math,os,pathlib,random,tempfile,time
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

PARENT='54e835b00642013d4f1bf486a8150eb1ff06c547565771bd21933e143d050f2d'
SCALES=(32768,131072)  # Includes a 2048-token unscored recent window.
RECENT=2048
ORDERS=4
CHANNELS=len(SCALES)*ORDERS
PER_CHANNEL=7  # [count, total, peak, entropy, recent, modal_base, agreement]
FEATURE_DIM=CHANNELS*PER_CHANNEL+4

def pick_positions(corpus,n,seed,min_start=131072):
    end=len(corpus.seq)-1
    if end<=min_start:raise ValueError('corpus too short for selected 32K horizon')
    # Deterministic stratified sample; avoid reusing exact positions for seeds.
    lo=min_start+int(.01*(end-min_start));span=end-lo
    starts=lo+np.floor(np.arange(n,dtype=np.float64)*span/n).astype(np.int64)
    step=max(1,span//n)
    return np.minimum(end, starts+np.random.default_rng(seed).integers(0,step,size=n))

def extract_paths(history,ctx,target,scale):
    """History full window includes RECENT last tokens, but its last RECENT never enter candidate indices.

    Returns only target-free features plus *training-label* likelihoods used for CE.
    """
    h=np.asarray(history[-scale:-RECENT],dtype=np.int64)
    q=np.asarray(ctx[-ORDERS:],dtype=np.int64)
    L=len(h)
    if L<ORDERS+2:raise ValueError('short history')
    ages=np.arange(L-1,dtype=np.float32)/(L-1)
    # Fixed decaying weights, stronger for newer *older* transitions.
    timew=np.exp(2.0*(ages-1.0)).astype(np.float32)
    valid=(h[:-1]>1)&(h[1:]>1)
    masks=valid&(h[:-1]==q[-1])&(q[-1]>1)
    records=[]
    for order in range(1,ORDERS+1):
        if order>1:
            masks[:order-1]=False
            masks[order-1:] &= (h[:-(order)]==q[-order]) & (q[-order]>1)
        idx=np.flatnonzero(masks)
        count=len(idx)
        if count:
            vals=h[idx+1]
            w=timew[idx]
            uniq,inverse=np.unique(vals,return_inverse=True)
            ww=np.bincount(inverse,weights=w,minlength=len(uniq))
            frac=(ww/max(1e-30,float(ww.sum()))).astype(np.float64)
            imax=int(ww.argmax());top=int(uniq[imax]);peak=float(frac[imax]);ent=float(-(frac*np.log(frac+1e-30)).sum()/np.log1p(len(uniq)))
            ptarget=float(w[vals==target].sum()/w.sum())
            rec=float(idx[-1]/L)
        else: top=1;peak=ent=ptarget=rec=0.
        records.append({'n':count,'weight_total':float(timew[idx].sum()) if count else 0.,'peak':peak,
          'entropy':ent,'recent':rec,'modal':top,'target_probability':ptarget})
    return records

@torch.no_grad()
def gather(root,model,base,comp,split,n,seed,batch=8,corpus_dir=None):
    corpus=Corpus(root,split,corpus_dir=corpus_dir)
    pos=pick_positions(corpus,n,seed)
    stream=DirectedStreaming(model)
    E=model.trunk.E.detach().cpu()
    baseline=[];features=[];probabilities=[];counts=[];positions=[]
    for inds in np.array_split(pos,max(1,math.ceil(n/batch))):
        _,ctx,pieces,lens,y,_,_=batch_data(base,corpus,comp,inds)
        past=torch.from_numpy(corpus.get(inds,OUTER_CONTEXT)[0][:,:OUTER_CONTEXT-RECENT].copy()).long()
        # Input model legacy needs exactly 6,144 older tokens (8,192 total minus 2,048 recent).
        state=stream.initial(past)
        ex=torch.from_numpy(corpus.data[inds].copy()).long()
        lg=torch.from_numpy(corpus.long[inds].copy()).long()
        now,loss,_=stream.forward(ex,lg,ctx,pieces,lens,state,past,y)
        baseline.append(loss.detach().cpu())
        # Grab 32K past + 2048 most recent. Source document boundaries are honored by Corpus.get.
        long_hist=corpus.get(inds,context=max(SCALES))[0]
        # Important: local model context is held constant across both experimental memory scales.
        modals=[];meta=[];p=[]
        for j in range(len(inds)):
            item=[]
            for scale in SCALES:
                item.extend(extract_paths(long_hist[j],ctx[j].cpu().numpy(),int(y[j]),scale))
            meta.append(item)
            modals.extend(x['modal'] for x in item)
            p.append([x['target_probability'] for x in item])
        modal_t=torch.tensor(modals,dtype=torch.long)
        repeat_now=now.detach().cpu().repeat_interleave(CHANNELS,0)
        modal_lp=model.trunk.nll(repeat_now,modal_t).reshape(len(inds),CHANNELS)
        # modal probabilities from inherited trunk, *not* from current target, thus safe at inference.
        modal_prior=torch.exp(-modal_lp).numpy()
        for j,item in enumerate(meta):
            ft=[]
            for k,x in enumerate(item):
                ft.extend((np.log1p(x['n']),np.log1p(x['weight_total']),x['peak'],x['entropy'],x['recent'],
                    float(modal_prior[j,k]) if x['n'] else 0.,
                    float(modal_prior[j,k])*x['peak'] if x['n'] else 0.))
            # Query-only global features; never include the next-token label.
            query=ctx[j,-1].item();prev=ctx[j,-2].item()
            ft.extend((float(query>1),float(prev>1),float(item[0]['n']>0),float(item[ORDERS]['n']>0)))
            features.append(ft)
            counts.append([x['n'] for x in item])
        probabilities.extend(p)
        positions.extend(int(x) for x in inds)
    arr={'base':torch.cat(baseline), 'features':torch.tensor(np.asarray(features),dtype=torch.float32),
        'prob':torch.tensor(np.asarray(probabilities),dtype=torch.float32),
        'count':torch.tensor(np.asarray(counts),dtype=torch.float32),
        'positions':positions,'n':len(positions)}
    assert arr['features'].shape==(n,FEATURE_DIM) and arr['prob'].shape==(n,CHANNELS)
    assert torch.isfinite(arr['features']).all() and torch.isfinite(arr['base']).all()
    return arr

@torch.no_grad()
def reroute_wrong(root,model,base,comp,split,data,seed,corpus_dir=None):
    """Entire histories are shifted to other positions; reselect all 8 n-gram routes using original queries.

    Baseline NLL is matched byte-for-byte with correct history. Never shuffle prepared probabilities.
    """
    corpus=Corpus(root,split,corpus_dir=corpus_dir)
    n=data['n'];positions=np.asarray(data['positions'],dtype=np.int64)
    shift=max(1,n//3);donor=np.roll(positions,shift)
    wrong=copy.deepcopy(data)
    E=model.trunk.E.detach()
    stream=DirectedStreaming(model)
    ff=[];pp=[];cc=[]
    for ix in np.array_split(np.arange(n),max(1,math.ceil(n/8))):
        inds=positions[ix];wrong_h=corpus.get(donor[ix],context=max(SCALES))[0]
        _,ctx,pieces,lens,y,_,_=batch_data(base,corpus,comp,inds)
        # Re-evaluate model with its ORIGINAL history; only donor memory varies.
        history=torch.from_numpy(corpus.get(inds,OUTER_CONTEXT)[0][:,:OUTER_CONTEXT-RECENT].copy()).long()
        state=stream.initial(history)
        ex=torch.from_numpy(corpus.data[inds].copy()).long();lg=torch.from_numpy(corpus.long[inds].copy()).long()
        now,loss,_=stream.forward(ex,lg,ctx,pieces,lens,state,history,y)
        assert torch.allclose(loss.detach(),data['base'][ix],atol=1e-6,rtol=1e-6)
        modal=[];records=[]
        for j in range(len(ix)):
            channels=[]
            for scale in SCALES:
                channels.extend(extract_paths(wrong_h[j],ctx[j].cpu().numpy(),int(y[j]),scale))
            records.append(channels);modal.extend(x['modal'] for x in channels)
        modal_nll=model.trunk.nll(now.detach().repeat_interleave(CHANNELS,0),torch.tensor(modal)).reshape(len(ix),CHANNELS)
        modal_prior=torch.exp(-modal_nll).numpy()
        for j,chs in enumerate(records):
            fs=[]
            for k,x in enumerate(chs):
                fs.extend((np.log1p(x['n']),np.log1p(x['weight_total']),x['peak'],x['entropy'],x['recent'],
                    float(modal_prior[j,k]) if x['n'] else 0.,float(modal_prior[j,k])*x['peak'] if x['n'] else 0.))
            fs.extend((float(ctx[j,-1]>1),float(ctx[j,-2]>1),float(chs[0]['n']>0),float(chs[ORDERS]['n']>0)))
            ff.append(fs);pp.append([x['target_probability'] for x in chs]);cc.append([x['n'] for x in chs])
    wrong['features']=torch.tensor(np.asarray(ff),dtype=torch.float32)
    wrong['prob']=torch.tensor(np.asarray(pp),dtype=torch.float32)
    wrong['count']=torch.tensor(np.asarray(cc),dtype=torch.float32)
    return wrong

class TrustRouter(nn.Module):
    def __init__(self,hidden=48,max_gate=.35):
        super().__init__()
        self.max_gate=max_gate
        self.net=nn.Sequential(nn.Linear(FEATURE_DIM,hidden),nn.LayerNorm(hidden),nn.Tanh(),nn.Linear(hidden,CHANNELS+1))
        with torch.no_grad():
            nn.init.normal_(self.net[-1].weight,std=.001)
            self.net[-1].bias.zero_();self.net[-1].bias[0]=9.0
    def forward(self,features,count,allowed_scales=(True,True)):
        scores=self.net(features)
        # Counts are query/historical-only: no target-dependent gating.
        eligible=count>0
        if not allowed_scales[0]:eligible[:,:ORDERS]=False
        if not allowed_scales[1]:eligible[:,ORDERS:]=False
        scores=torch.cat([scores[:,:1],scores[:,1:].masked_fill(~eligible,-1e5)],dim=1)
        weights=F.softmax(scores,dim=1)[:,1:]*self.max_gate
        return weights

def nll(router,data,scales=(True,True)):
    g=router(data['features'],data['count'],scales)
    t=g.sum(1).clamp(max=.999)
    base=torch.exp(-data['base'])
    p=(1-t)*base+(g*data['prob']).sum(1)
    return -torch.log(p.clamp_min(1e-32)),g

@torch.no_grad()
def eval_pair(router,data,wrong,scales=(True,True)):
    c,g=nll(router,data,scales);w,gw=nll(router,wrong,scales)
    return {'n':data['n'],'base_nll':float(data['base'].mean()),'correct_nll':float(c.mean()),
            'wrong_nll':float(w.mean()),'gain':float((data['base']-c).mean()),
            'advantage':float((w-c).mean()),'mean_gate':float(g.sum(1).mean()),
            'nonzero_routes':float((g>1e-5).float().sum(1).mean()),
            'coverage_32k':float((data['count'][:,:ORDERS].sum(1)>0).float().mean()),
            'coverage_128k':float((data['count'][:,ORDERS:].sum(1)>0).float().mean())}

def secure_save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(dir=path.parent,suffix='.tmp');os.close(fd)
    try:
        torch.save(data,tmp)
        with open(tmp,'rb') as fp:os.fsync(fp.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)


def main(a):
    t0=time.monotonic();root=pathlib.Path(a.root);parent=root/'checkpoint/pal_active.pt'
    if sha256(parent)!=PARENT:raise RuntimeError('refuse to train wrong parent checkpoint')
    torch.set_num_threads(a.threads);random.seed(716);torch.manual_seed(716);np.random.seed(716)
    model,opts,base,comp,_,_=restore_parent(root,root/'adaptation/r712_gate_delta.pt')
    assert len(opts)==5
    model.eval()
    for p in model.parameters():p.requires_grad_(False)
    original_state=copy.deepcopy(opts[4].state_dict());n_original=len(opts[4].param_groups)
    print(json.dumps({'phase':'RESTORED','parent':PARENT,'optimizers':len(opts),'accepted_step':14416}),flush=True)
    train=gather(root,model,base,comp,'train',a.train,7161,a.batch,a.corpus_dir)
    dev=gather(root,model,base,comp,'dev',a.dev,7162,a.batch,a.corpus_dir)
    wrongtrain=reroute_wrong(root,model,base,comp,'train',train,7161,a.corpus_dir)
    wrongdev=reroute_wrong(root,model,base,comp,'dev',dev,7162,a.corpus_dir)
    router=TrustRouter(hidden=a.hidden,max_gate=a.max_gate)
    opts[4].add_param_group({'params':list(router.parameters()),'lr':a.lr,'weight_decay':0.0})
    optimizer=opts[4];base_dev=eval_pair(router,dev,wrongdev)
    print(json.dumps({'phase':'INITIAL','dev':base_dev,'seconds':round(time.monotonic()-t0,1)}),flush=True)
    best={'step':0,'gain':0.,'params':None,'optim':None};runs=[]
    rng=np.random.default_rng(716716)
    for step in range(1,a.steps+1):
        ids=torch.as_tensor(rng.choice(a.train,min(a.train,a.train_batch),replace=False),dtype=torch.long)
        good={k:(v[ids] if torch.is_tensor(v) else v) for k,v in train.items()}
        wrong={k:(v[ids] if torch.is_tensor(v) else v) for k,v in wrongtrain.items()}
        lc,_=nll(router,good);lw,_=nll(router,wrong)
        # Train next-token likelihood, with a mild selective-history contrast, never labels in router inputs.
        loss=lc.mean()+a.contrast*F.softplus((lc-lw).mean()+.005)
        optimizer.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(router.parameters(),1.)
        optimizer.step()
        if step==1 or step%a.eval_every==0 or step==a.steps:
            tr=eval_pair(router,train,wrongtrain);dv=eval_pair(router,dev,wrongdev)
            runs.append({'step':step,'train':tr,'dev':dv})
            print(json.dumps({'phase':'GATE','step':step,'train_gain':round(tr['gain'],7),
                'dev_gain':round(dv['gain'],7),'advantage':round(dv['advantage'],7),'gate':round(dv['mean_gate'],5)}),flush=True)
            if dv['gain']>max(a.min_gain,best['gain']) and dv['advantage']>a.min_adv:
                best={'step':step,'gain':dv['gain'],'params':copy.deepcopy(router.state_dict()),
                    'optim':copy.deepcopy(optimizer.state_dict())}
    if best['params'] is not None:
        router.load_state_dict(best['params']);optimizer.load_state_dict(best['optim'])
    selected=eval_pair(router,dev,wrongdev) if best['params'] else None
    ablations={}
    if best['params']:
        for scale,mask in [('32k_only',(True,False)),('128k_only',(False,True)),('all',(True,True))]:
            ablations[scale]=eval_pair(router,dev,wrongdev,mask)
    print(json.dumps({'phase':'SELECTED','step':best['step'],'dev':selected,'elapsed':round(time.monotonic()-t0,2)}),flush=True)
    # Test split consulted once only after dev-only selection.
    print(json.dumps({'phase':'TEST_GATHER_BEGIN','n':a.test}),flush=True)
    test=gather(root,model,base,comp,'test',a.test,7163,a.batch,a.corpus_dir)
    print(json.dumps({'phase':'TEST_GATHER_COMPLETE','seconds':round(time.monotonic()-t0,2)}),flush=True)
    wrongtest=reroute_wrong(root,model,base,comp,'test',test,7163,a.corpus_dir)
    print(json.dumps({'phase':'TEST_REROUTE_COMPLETE','seconds':round(time.monotonic()-t0,2)}),flush=True)
    historical=eval_pair(router,test,wrongtest) if best['params'] else {'n':a.test,'base_nll':float(test['base'].mean()),
            'correct_nll':float(test['base'].mean()),'wrong_nll':float(test['base'].mean()),'gain':0.,'advantage':0.,'mean_gate':0.}
    assert sha256(parent)==PARENT
    now=optimizer.state_dict()
    assert now['param_groups'][:n_original]==original_state['param_groups']
    for key,value in original_state['state'].items():
        assert key in now['state']
        for sub,x in value.items():
            y=now['state'][key][sub]
            if torch.is_tensor(x):assert torch.equal(x,y),f'inherited optimizer state mutated {key}.{sub}'
            else:assert x==y
    # Always store diagnostic adapter + optimizer *only* if dev-selected, never a full replacement.
    overlay=None
    if best['params']:
        overlay=pathlib.Path(a.overlay)
        state=optimizer.state_dict();ids=state['param_groups'][-1]['params']
        secure_save(overlay,{'format':'FA1_R7_17_128K_TRUST_CANDIDATE_V1','parent_sha256':PARENT,
             'parent_joint_steps':14416,'gate':router.state_dict(),'optimizer_group':state['param_groups'][-1],
             'optimizer_new_state':{k:v for k,v in state['state'].items() if k in ids},
             'scale':SCALES,'channels':CHANNELS,'book_independent':bool(a.corpus_dir),
             'dev_metrics':selected})
    report={'title':'FA1 R7.17 32K/128K causal ngram trust router','accepted_step':14416,'parent_sha256':PARENT,
        'parent_verified_unmodified':True,'five_optimizer_histories_unchanged':True,'original_groups':n_original,
        'additional_optimizer_group_training_only':True,'candidate_selected':best['step']>0,
        'candidate_promoted':False,'best_step':best['step'],'train_samples':a.train,'dev_samples':a.dev,
        'historical_test_samples':a.test,'dev':selected,'dev_init':base_dev,'ablations':ablations,
        'historical_test':historical,'logs':runs,'heldout_books':bool(a.corpus_dir),
        'book_disjointness_not_automatically_verified':True,
        'train_dev_test_repeatedly_consulted':not bool(a.corpus_dir),
        'accepted_binary_unchanged':True,'adapter_path':str(overlay) if overlay else None,
        'adapter_sha256':sha256(overlay) if overlay else None,
        'elapsed_s':round(time.monotonic()-t0,2)}
    path=pathlib.Path(a.output);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'phase':'DONE','best':best['step'],'gain':selected['gain'] if selected else 0.,
        'historical_test_gain':historical['gain'],'accepted_unchanged':True,'seconds':report['elapsed_s']}),flush=True)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=pathlib.Path,default=pathlib.Path('.'))
    p.add_argument('--corpus-dir',type=pathlib.Path)
    p.add_argument('--train',type=int,default=768);p.add_argument('--dev',type=int,default=384)
    p.add_argument('--test',type=int,default=384);p.add_argument('--steps',type=int,default=512)
    p.add_argument('--eval-every',type=int,default=32);p.add_argument('--batch',type=int,default=8)
    p.add_argument('--train-batch',type=int,default=96);p.add_argument('--threads',type=int,default=2)
    p.add_argument('--hidden',type=int,default=48);p.add_argument('--max-gate',type=float,default=.35)
    p.add_argument('--lr',type=float,default=.002);p.add_argument('--contrast',type=float,default=.04)
    p.add_argument('--min-gain',type=float,default=.0002);p.add_argument('--min-adv',type=float,default=.0001)
    p.add_argument('--output',type=pathlib.Path,default=pathlib.Path('reports/FA1_R7p16_CPU.json'))
    p.add_argument('--overlay',type=pathlib.Path,default=pathlib.Path('adaptation/fa1_r716_candidate.pt'))
    main(p.parse_args())