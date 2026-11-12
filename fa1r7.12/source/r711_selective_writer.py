#!/usr/bin/env python3
"""PAL R7.11 CPU research: train a causal directed associative read in ONE rank64 matrix.

Zero-gated migration from R7.7 preserves ALL inherited logits and five optimizer
histories. No independent recurrent state or checkpoint lineage. Experimental
auxiliary training uses only historic adjacent tokens and an aligned matrix
readout (never next-word labels or post-history context). Current stream advance
is only exact when gate=0; a nonzero-gate candidate is not production-ready.
"""
from __future__ import annotations
import argparse, copy, json, pathlib, time, random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from r74_stream import CrossWindowMirror, OLDER, OUTER_CONTEXT
from r7_core import D,R,Corpus
from r77_semantic import restore as restore77, build_model
from r710_hard_negative import pick
from r710_checkpoint_integrity import durable_save,verify,deep_equal,sha256
from r75_retrieval import OPT_NAMES

FMT='PAL_R7_11_SELECTIVE_DIRECTED_CANDIDATE_V1'
PARENT='PAL_R7_7_SEMANTIC_ASSOCIATIVE_V1'
HORIZON=768

class SelectivePairMirror(CrossWindowMirror):
    def add_pair_parameters(self):
        self.dir_key=nn.Linear(D,R,bias=False)
        self.dir_value=nn.Linear(D,R,bias=False)
        self.dir_select=nn.Linear(2*R,1,bias=True)
        self.dir_gate=nn.Parameter(torch.zeros(()))
        # Reproducible nonrandom migration, so original checkpoint RNG unchanged.
        with torch.no_grad():
            self.dir_key.weight.copy_(self.key.weight)
            self.dir_value.weight.copy_(self.value.weight)
            self.dir_select.weight.zero_()
            self.dir_select.bias.zero_()

    def pair_state(self, history, embeddings):
        """Causal ordered bigrams, late-old 768 tokens, centered key/value covariance."""
        h=history[:,-HORIZON:].long(); a=h[:,:-1]; b=h[:,1:]
        mask=((a>1)&(b>1)).float()
        ek=F.embedding(a,embeddings);ev=F.embedding(b,embeddings)
        k=F.normalize(self.dir_key(ek),dim=-1)
        v=F.normalize(self.dir_value(ev),dim=-1)
        w=mask*torch.linspace(-1.0,0.,a.shape[1],device=h.device).exp()[None,:]
        salience=2*torch.sigmoid(self.dir_select(torch.cat((k,v),-1)).squeeze(-1))
        w=w*salience
        total=w.sum(1).clamp_min(1.0)
        kmean=(k*w.unsqueeze(-1)).sum(1)/total[:,None]
        vmean=(v*w.unsqueeze(-1)).sum(1)/total[:,None]
        kc=k-kmean[:,None,:]; vc=v-vmean[:,None,:]
        M=torch.bmm((vc*w.unsqueeze(-1)).transpose(1,2),kc)/total[:,None,None]
        return M,kmean,vmean

    def initial(self,history,embeddings):
        old=super().initial(history,embeddings)
        gate=torch.tanh(self.dir_gate)
        # At gain zero, arithmetic remains bitwise equal to inherited state.
        # IMPORTANT: do not short-circuit zero, or the gate cannot get
        # nonzero gradients during eventual joint LM training.
        pair,_,_=self.pair_state(history,embeddings)
        return old+gate*(pair-old)

    def advance(self,state,leaving,embeddings):
        # The legacy update has only the leaving word, not its preceding key;
        # it cannot update a directed pair sketch faithfully. Fail closed.
        if float(torch.abs(torch.tanh(self.dir_gate.detach())))>1e-12:
            raise RuntimeError('Nonzero directed memory requires exact history reinitialization; streaming pair advance not implemented')
        return super().advance(state,leaving,embeddings)

    def pair_score(self,M,keys,answers,embeddings,kmean,vmean,temp=.12):
        k=F.normalize(self.dir_key(F.embedding(keys.long(),embeddings)),dim=-1)-kmean
        v=F.normalize(self.dir_value(F.embedding(answers.long(),embeddings)),dim=-1)-vmean[:,None,:]
        response=torch.bmm(M,k.unsqueeze(-1)).squeeze(-1)
        # Centering and 2x normalization undo high-frequency unigram bias.
        logits=torch.einsum('bd,bkd->bk',F.normalize(response,dim=-1),F.normalize(v,dim=-1))/temp
        return logits


def upgrade(model,optimizers):
    old=model.r74.outer
    if type(old) is not CrossWindowMirror:raise AssertionError('unexpected source outer writer')
    old.__class__=SelectivePairMirror
    old.add_pair_parameters()
    params=[*old.dir_key.parameters(),*old.dir_value.parameters(),*old.dir_select.parameters(),old.dir_gate]
    optimizers[3].add_param_group({'params':params,'lr':.0015,'weight_decay':.0001})
    return params


def restore(root,source):
    root=pathlib.Path(root);source=pathlib.Path(source)
    data=torch.load(source,map_location='cpu',weights_only=False)
    if data['format']==PARENT:
        model,optimizers,base,comp,parent=restore77(root,source,torch.device('cpu'))
        upgrade(model,optimizers)
        torch.set_rng_state(parent['torch_rng']);np.random.set_state(parent['numpy_rng']);random.setstate(parent['python_rng'])
        return model,optimizers,base,comp,parent
    if data['format']!=FMT:raise ValueError('unsupported checkpoint')
    model,optimizers,base,comp=build_model(root,torch.device('cpu'),(2e-6,6e-5,1e-4,8e-5,.002))
    upgrade(model,optimizers)
    model.load_state_dict(data['model'],strict=True)
    for name,opt in zip(OPT_NAMES,optimizers):opt.load_state_dict(data['optimizers'][name])
    torch.set_rng_state(data['torch_rng']);np.random.set_state(data['numpy_rng']);random.setstate(data['python_rng'])
    return model,optimizers,base,comp,data


def build_examples(root, split, count, seed, E, kind='hard'):
    corpus=Corpus(root,split)
    positions=8192+np.arange(count,dtype=np.int64)*(len(corpus.seq)-8193)//count
    data=[]
    for pos in positions:
        history=corpus.get(np.asarray([pos]),OUTER_CONTEXT)[0][0,:OLDER].copy()
        entry=pick(history,E,seed+int(pos)*11,kind=kind)
        if entry is not None:
            data.append((int(pos),history[-HORIZON:].astype(np.int64),entry['query'],entry['candidates']))
    if not data:raise ValueError('no valid recall events')
    return {'split':split,'positions':np.asarray([v[0] for v in data]),
      'history':torch.tensor(np.stack([v[1] for v in data]),dtype=torch.long),
      'query':torch.tensor([v[2] for v in data],dtype=torch.long),
      'answers':torch.tensor([v[3] for v in data],dtype=torch.long)}


def score_batches(outer,E,data,batch=16,permuted=False,reversed=False,zero=False,grad=False):
    preds=[];refs=[];margins=[];losses=[];n=len(data['positions'])
    for start in range(0,n,batch):
        z=slice(start,min(start+batch,n)); h=data['history'][z];q=data['query'][z];ans=data['answers'][z]
        if reversed:
            # Reverse original 64-word blocks within the pair horizon; preserve same multiset.
            h=h.reshape(len(h),HORIZON//64,64).flip(-1).reshape(len(h),HORIZON)
        M,kmean,vmean=outer.pair_state(h,E)
        if permuted:
            # Wrong-source state: deliberately pair queries with a different history.
            if len(h)>1:
                M=M.roll(1,0);kmean=kmean.roll(1,0);vmean=vmean.roll(1,0)
            else:continue
        if zero:M=M*0
        logits=outer.pair_score(M,q,ans,E,kmean,vmean)
        loss=F.cross_entropy(logits,torch.zeros(len(h),dtype=torch.long),reduction='none')
        losses.append(loss)
        with torch.no_grad():
            is_max=(logits>=logits.max(-1,keepdim=True).values-1e-7)
            preds.append(is_max[:,0].float()/is_max.sum(-1))
            margins.append(logits[:,0]-logits[:,1:].max(-1).values)
    if not losses:raise RuntimeError('insufficient batch size')
    return torch.cat(losses),torch.cat(preds),torch.cat(margins)

@torch.no_grad()
def evaluate(outer,E,data,batch=16):
    conditions={}
    for name,kwargs in [('correct',{}),('unrelated',{'permuted':True}),
                        ('reversed',{'reversed':True}),('zero',{'zero':True})]:
        losses,acc,marg=score_batches(outer,E,data,batch=batch,**kwargs)
        conditions[name]={'ce':float(losses.mean()),'top1':float(acc.mean()),'margin':float(marg.mean())}
    conditions['memory_ce_advantage']=conditions['unrelated']['ce']-conditions['correct']['ce']
    conditions['order_ce_advantage']=conditions['reversed']['ce']-conditions['correct']['ce']
    conditions['examples']=len(data['positions'])
    return conditions


def save_candidate(root,out,model,opts,prior,steps,sha_parent,report):
    prior={**prior,'format':FMT,'model':model.state_dict(),
        'optimizers':{name:opt.state_dict() for name,opt in zip(OPT_NAMES,opts)},
        'joint_steps':int(prior['joint_steps']), 'directed_aux_updates':int(steps),
        'torch_rng':torch.get_rng_state(),'numpy_rng':np.random.get_state(),
        'python_rng':random.getstate(),'batch_rng_state':prior['batch_rng_state'],
        'accepted_parent_sha256':sha_parent,
        'architecture':'PAL R7.7 + optional zero-gated causal directed ordered 768-token rank64 writer (experimental)',
        'metrics':report}
    return durable_save(prior,out,root,sha_parent)


def run(args):
    torch.set_num_threads(args.threads);torch.manual_seed(711);np.random.seed(711)
    t0=time.monotonic();root=args.root;accepted=root/'checkpoint/pal_active.pt';source_sha=sha256(accepted)
    parent,summary=verify(accepted,root,expected_step=14416)
    model,opts,base,comp,prior=restore(root,accepted)
    E=model.trunk.E.detach()
    assert all(torch.equal(v,parent['model'][name]) for name,v in model.state_dict().items() if name in parent['model'])
    assert model.r74.outer.dir_gate.item()==0
    oldstates=copy.deepcopy({k:o.state_dict() for k,o in zip(OPT_NAMES,opts)})
    for n in OPT_NAMES:
        initial=parent['optimizers'][n]
        new=oldstates[n]
        if n=='outer_adamw':
            assert deep_equal(initial['state'],new['state'])
            assert deep_equal(initial['param_groups'],new['param_groups'][:-1])
        else:assert deep_equal(initial,new)
    print(json.dumps({'stage':'RESTORE_OK','source':source_sha,'old_joint_steps':prior['joint_steps']}),flush=True)
    train=build_examples(root,'train',args.train_examples,711,E)
    dev=build_examples(root,'dev',args.dev_examples,712,E)
    test=build_examples(root,'test',args.test_examples,713,E)
    print(json.dumps({'stage':'EXAMPLES','n':{k:len(v['positions']) for k,v in (('train',train),('dev',dev),('test',test))}}),flush=True)
    outer=model.r74.outer
    before={k:evaluate(outer,E,v,batch=args.batch) for k,v in (('train',train),('dev',dev))}
    print(json.dumps({'stage':'BEFORE','train':before['train'],'dev':before['dev']}),flush=True)
    p=[*outer.dir_key.parameters(),*outer.dir_value.parameters(),*outer.dir_select.parameters()]
    # One additional group in the existing FOURTH optimizer; no sixth optimizer.
    # Gradient to all legacy weights is blocked during this targeted auxiliary stage.
    best=None;best_step=0;best_dev=before['dev']['correct']['ce'];best_params=copy.deepcopy(outer.state_dict());best_optimizer=copy.deepcopy(opts[3].state_dict());best_rng=None
    rng=np.random.default_rng(71131)
    history=[]
    for step in range(1,args.updates+1):
        model.train();inds=rng.choice(len(train['positions']),min(args.batch,len(train['positions'])),replace=False)
        h=train['history'][inds];q=train['query'][inds];ans=train['answers'][inds]
        M,km,vm=outer.pair_state(h,E);logits=outer.pair_score(M,q,ans,E,km,vm)
        ce=F.cross_entropy(logits,torch.zeros(len(h),dtype=torch.long))
        wrong=outer.pair_score(M.roll(1,0),q,ans,E,km.roll(1,0),vm.roll(1,0))
        # Minimize correct CE and impose a small gap to wrong-memory CE.
        contrast=F.softplus(2*(.08+ce-F.cross_entropy(wrong,torch.zeros(len(h),dtype=torch.long))))/2
        loss=ce+.25*contrast
        opts[3].zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(p,1.0);opts[3].step()
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite directed memory loss')
        if step==1 or step%args.eval_every==0 or step==args.updates:
            model.eval()
            with torch.no_grad():dv=evaluate(outer,E,dev,batch=args.batch)
            history.append({'update':step,'train_ce':float(ce.detach()),'train_loss':float(loss.detach()),
                     'dev':dv,'elapsed_s':round(time.monotonic()-t0,2)})
            print(json.dumps({'stage':'GATE','step':step,'train_ce':float(ce.detach()),'dev_ce':dv['correct']['ce'],
                'dev_mem_advantage':dv['memory_ce_advantage'],'dev_accuracy':dv['correct']['top1'],'elapsed':round(time.monotonic()-t0,1)}),flush=True)
            if dv['correct']['ce']<best_dev-1e-4 and dv['memory_ce_advantage']>0:
                best_dev=dv['correct']['ce'];best_step=step;best_params=copy.deepcopy(outer.state_dict());best_optimizer=copy.deepcopy(opts[3].state_dict());best_rng=copy.deepcopy(rng.bit_generator.state)
    outer.load_state_dict(best_params);opts[3].load_state_dict(best_optimizer)
    after={k:evaluate(outer,E,v,batch=args.batch) for k,v in (('train',train),('dev',dev),('test',test))}
    assert sha256(accepted)==source_sha
    # Training is auxiliary-only and zero-gated. All existing model and optimizer
    # state tensors MUST be bit-identical to the accepted parent.
    for name,v in parent['model'].items():assert torch.equal(model.state_dict()[name],v),name
    for n in OPT_NAMES:
        now=opts[OPT_NAMES.index(n)].state_dict()
        if n=='outer_adamw':
            assert all(k in now['state'] and deep_equal(now['state'][k],v) for k,v in parent['optimizers'][n]['state'].items())
        else:assert deep_equal(now,parent['optimizers'][n])
    result={'parent_step':14416,'parent_sha256':source_sha,'auxiliary_updates_attempted':args.updates,
      'best_auxiliary_update':best_step,'accepted_joint_steps_unchanged':14416,'selected_directed_aux_updates':best_step,
      'candidate_promoted':False,'gate_zero_preserves_inherited_predictions':True,
      'five_legacy_optimizer_histories_bitwise_intact':True,'historical_heldouts_not_sealed':True,
      'directed_memory_horizon':HORIZON,'conditions_before':before,'conditions_after':after,
      'auxiliary_rng_at_selected_step':best_rng,'train_log':history,'elapsed_seconds':round(time.monotonic()-t0,2),
      'limitations':['memory reader is auxiliary; not yet integrated into streaming NOW prediction',
         'nonzero blend gate not tested for incremental advance equivalence',
         'test corpus repeatedly consulted historically; no independent book-level claim']}
    if args.output and best_step>0:
        # Restore BOTH selected parameters and optimizer moments before save.
        result['candidate_save']=save_candidate(root,args.output,model,opts,prior,best_step,source_sha,result)
    if args.report:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'stage':'DONE','best_step':best_step,'after_dev':after['dev'],
                       'after_test':after['test'],'elapsed':result['elapsed_seconds']}),flush=True)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=pathlib.Path,default=pathlib.Path('.'))
    p.add_argument('--updates',type=int,default=128)
    p.add_argument('--train-examples',type=int,default=384)
    p.add_argument('--dev-examples',type=int,default=128)
    p.add_argument('--test-examples',type=int,default=128)
    p.add_argument('--batch',type=int,default=16)
    p.add_argument('--eval-every',type=int,default=32)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--output',type=pathlib.Path)
    p.add_argument('--report',type=pathlib.Path)
    run(p.parse_args())
