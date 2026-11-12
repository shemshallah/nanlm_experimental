#!/usr/bin/env python3
"""PAL R7.12: causal sparse directed-memory LM bridge and dev-gated calibration.

Readout scans ONLY the 768 tokens older than the exact 2048-token window.
No present target ever affects retrieval scores, candidate selection or gate
features. This is an experimental probability-mixture head, not a separate LM.
Correct/unrelated-memory paired ablation is supported with frozen features.
"""
from __future__ import annotations
import argparse,json,pathlib,time,copy,hashlib,os,tempfile
import numpy as np,torch
from torch import nn
from torch.nn import functional as F
from r711_delta import restore_overlay
from r712_streaming import DirectedStreaming
from r74_stream import OUTER_CONTEXT,OLDER
from r7_core import Corpus
from train_r7 import batch_data
from r710_checkpoint_integrity import sha256

class CalibrationGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(4,16),nn.Tanh(),nn.Linear(16,1))
        with torch.no_grad():
            self.net[0].weight.zero_();self.net[0].bias.zero_()
            self.net[2].weight.zero_();self.net[2].bias.fill_(-4.5)
    def forward(self,features):
        return .3*torch.sigmoid(self.net(features).squeeze(-1))

@torch.no_grad()
def memory_probability(outer,E,ring,query,target):
    # Every (key,value) transition was observed before the current target.
    old_key=ring[:,:-1];old_val=ring[:,1:]
    k=F.normalize(outer.dir_key(F.embedding(old_key,E)),dim=-1)
    q=F.normalize(outer.dir_key(F.embedding(query,E)),dim=-1)
    sims=(k*q[:,None,:]).sum(-1)
    # Reuse the experimentally trained directed mirror to score a candidate
    # value; this is NOT a target-conditioned candidate-mining step.
    M,km,vm=outer.pair_state(ring,E)
    kq=q-km
    response=F.normalize(torch.bmm(M,kq.unsqueeze(-1)).squeeze(-1),dim=-1)
    val_emb=F.normalize(outer.dir_value(F.embedding(old_val,E))-vm[:,None,:],dim=-1)
    directed=(val_emb*response[:,None,:]).sum(-1)
    time_bias=torch.linspace(-.25,0.,old_key.shape[1],device=ring.device)
    scores=12.*sims+2.*directed+time_bias
    good=(old_key>1)&(old_val>1)&(query[:,None]>1)
    scores=scores.masked_fill(~good,-1.e4)
    attention=torch.softmax(scores,dim=1)
    attention=torch.where(good,attention,torch.zeros_like(attention))
    attention=attention/attention.sum(-1,keepdim=True).clamp_min(1e-20)
    prob=(attention*(old_val==target[:,None])).sum(-1)
    # Query-dependent features computed without the target, for proper
    # deployment at inference time.
    maxsim=sims.masked_fill(~good,-1).max(-1).values
    entropy=-(attention.clamp_min(1e-20).log()*attention).sum(-1)/np.log(old_key.shape[1])
    confidence=attention.max(-1).values
    valid=good.any(-1).float()
    features=torch.stack((maxsim,entropy,confidence,valid),dim=-1)
    return prob,features

@torch.no_grad()
def collect(root,model,base,comp,split,count,seed=0,batch=8,wrong=False):
    corpus=Corpus(root,split);end=len(corpus.seq)-1
    # Fixed positions for a separately chosen split; avoid split starts.
    positions=8192+np.arange(count,dtype=np.int64)*(end-8192)//count
    if seed:
        # A deterministic offset smaller than the sampling stride, preventing
        # repeated evaluation on identical positions with a different seed.
        stride=max(1,(end-8192)//count)
        positions=np.minimum(end,positions+np.random.default_rng(seed).integers(0,stride,size=count))
    bs=[];ps=[];fs=[]
    E=model.trunk.E.detach();outer=model.r74.outer
    model.eval();stream=DirectedStreaming(model)
    for ix in np.array_split(positions,int(np.ceil(count/batch))):
        _,ctx,pieces,lens,y,_,_=batch_data(base,corpus,comp,ix)
        hist=torch.tensor(corpus.get(ix,OUTER_CONTEXT)[0][:,:OLDER].copy()).long()
        s=stream.initial(hist)
        ex=torch.tensor(corpus.data[ix].copy()).long()
        lg=torch.tensor(corpus.long[ix].copy()).long()
        _,nll,_=stream.forward(ex,lg,ctx,pieces,lens,s,hist,y)
        ring=s.ring.roll(1,0) if wrong else s.ring
        prob,feat=memory_probability(outer,E,ring,ctx[:,-1],y)
        bs.append(nll.detach().cpu());ps.append(prob.detach().cpu());fs.append(feat.detach().cpu())
    return {'baseline':torch.cat(bs),'prob':torch.cat(ps),'features':torch.cat(fs),'count':count}

def score(gate,data,ablate=False):
    g=gate(data['features'])
    if ablate:g=g*0
    # Exact probability-mixture NLL with inherited full model, no approximation
    logp=torch.logaddexp(torch.log1p(-g.clamp(max=1.-1e-7))-data['baseline'],
                         torch.log(g.clamp_min(1.e-30))+torch.log(data['prob'].clamp_min(1e-30)))
    return -logp

def summary(gate,data):
    with torch.no_grad():
        losses=score(gate,data)
        return {'nll':float(losses.mean()),'base_nll':float(data['baseline'].mean()),
          'gain':float(data['baseline'].mean()-losses.mean()),
          'avg_gate':float(gate(data['features']).mean()),
          'memory_target_hit_rate':float((data['prob']>0).float().mean()),
          'avg_mem_prob_when_hit':float(data['prob'][data['prob']>0].mean()) if bool((data['prob']>0).any()) else 0.,
          'n':data['count']}

def run(args):
    torch.set_num_threads(args.threads);root=args.root
    parent=root/'checkpoint/pal_active.pt';startsha=sha256(parent)
    model,opts,base,comp,patch=restore_overlay(root,root/'adaptation/r711_experimental_delta.pt')
    torch.manual_seed(712);np.random.seed(712)
    gate=CalibrationGate()
    t0=time.monotonic()
    data={s:collect(root,model,base,comp,s,n,seed,args.batch) for s,n,seed in (
      ('train',args.train,0),('dev',args.dev,7122))}
    before={k:summary(gate,v) for k,v in data.items()}
    print(json.dumps({'phase':'FROZEN_EVIDENCE','before':before,'seconds':round(time.monotonic()-t0,2)}),flush=True)
    # Reuse the same fifth optimizer: a single extra parameter group; no new
    # optimizer, no change to any accepted state. Its moments are stored in
    # the experimental overlay, never in the accepted full checkpoint.
    opts[4].add_param_group({'params':list(gate.parameters()),'lr':args.lr,'weight_decay':0.0})
    optimizer=opts[4]
    best={'nll':before['dev']['base_nll'],'update':0,'parameters':None,'optimizer':None}
    train_history=[];n=len(data['train']['baseline']);rng=np.random.default_rng(712712)
    for step in range(1,args.updates+1):
        idx=rng.choice(n,min(n,args.train_batch),replace=False)
        b={k:(v[idx] if isinstance(v,torch.Tensor) else len(idx)) for k,v in data['train'].items()}
        loss=score(gate,b).mean()
        optimizer.zero_grad(set_to_none=True);loss.backward()
        nn.utils.clip_grad_norm_(gate.parameters(),1.0)
        optimizer.step()
        if step%args.eval_every==0 or step==1 or step==args.updates:
            dev=summary(gate,data['dev']);train=summary(gate,data['train'])
            train_history.append({'step':step,'train':train,'dev':dev})
            print(json.dumps({'phase':'GATE','step':step,'train_gain':train['gain'],
                   'dev_gain':dev['gain'],'gate':dev['avg_gate']}),flush=True)
            if dev['nll']<best['nll']-1e-5:
                best={'nll':dev['nll'],'update':step,
                      'parameters':copy.deepcopy(gate.state_dict()),
                      'optimizer':copy.deepcopy(optimizer.state_dict())}
    if best['parameters'] is not None:
        gate.load_state_dict(best['parameters']);optimizer.load_state_dict(best['optimizer'])
    else:
        # Explicit no-promotion fallback, but we still retain the calibrated
        # engineering experiment with its before state.
        gate=CalibrationGate()
    # Test is consulted once, after selection. Still historical, unsealed.
    test=collect(root,model,base,comp,'test',args.test,7123,args.batch)
    result={'r712':'Causal sparse directed-memory LM bridge',
      'parent_sha256':startsha,'accepted_joint_steps':14416,'r711_selected_aux_updates':80,
      'r712_calibration_updates_attempted':args.updates,'dev_selected_step':best['update'],
      'candidate_promoted':False,'dev_selected_positive':best['update']>0,
      'before':before,'selected':{k:summary(gate,v) for k,v in data.items()},
      'historical_test':summary(gate,test),'logs':train_history,
      'full_model_checkpoint_unchanged':sha256(parent)==startsha,
      'independent_books':False,'elapsed_s':round(time.monotonic()-t0,2),
      'notes':['The sparse readout uses older directed events; it is not a 32k-vocabulary dense transformer head.',
               'No independent book splits, no frontier comparison.',
               'Existing model weights and all optimizer moments are preserved in the parent.',
               'Calibration optimizes only the new gate; no full-model joint update has occurred.']}
    # Compact signed overlay, model+retrieval and fifth optimizer group only.
    out=args.output;out.parent.mkdir(parents=True,exist_ok=True)
    payload={'format':'PAL_R7_12_CAUSAL_LM_GATE_OVERLAY_V1','parent_sha256':startsha,
       'r711_delta_sha256':sha256(root/'adaptation/r711_experimental_delta.pt'),
       'gate_state':gate.state_dict(),
       'new_fifth_group':optimizer.state_dict()['param_groups'][-1],
       'new_fifth_states':{k:v for k,v in optimizer.state_dict()['state'].items()
                           if k in optimizer.state_dict()['param_groups'][-1]['params']},
       'selection_metrics':result}
    fd,tmp=tempfile.mkstemp(prefix='.r712_gate_',suffix='.pending',dir=out.parent);os.close(fd)
    try:
        torch.save(payload,tmp)
        with open(tmp,'rb') as f:os.fsync(f.fileno())
        os.replace(tmp,out)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)
    out.with_suffix(out.suffix+'.sha256').write_text(sha256(out)+'  '+out.name+'\n')
    result['r712_overlay_sha256']=sha256(out)
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'phase':'DONE','selected_step':best['update'],'dev_gain':result['selected']['dev']['gain'],
          'test_gain':result['historical_test']['gain'],'accepted_unchanged':result['full_model_checkpoint_unchanged'],
          'elapsed_s':result['elapsed_s']}),flush=True)
    return result

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--root',type=pathlib.Path,default=pathlib.Path('.'))
    a.add_argument('--train',type=int,default=256);a.add_argument('--dev',type=int,default=96)
    a.add_argument('--test',type=int,default=96);a.add_argument('--batch',type=int,default=8)
    a.add_argument('--train-batch',type=int,default=64);a.add_argument('--updates',type=int,default=128)
    a.add_argument('--eval-every',type=int,default=16);a.add_argument('--lr',type=float,default=.008)
    a.add_argument('--threads',type=int,default=2)
    a.add_argument('--output',type=pathlib.Path,default=pathlib.Path('adaptation/r712_gate_delta.pt'))
    a.add_argument('--report',type=pathlib.Path,default=pathlib.Path('reports/R7p12_lm_bridge.json'))
    run(a.parse_args())
