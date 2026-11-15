"""Function-preserving, confidence-conditioned trust gate for one v102 model.

The inherited step-480 contextual correction and its optimizer states are
embedded in each checkpoint unchanged. Only ten new Adam parameters learn.
At initialization the trust multiplier is exactly one, reproducing the
candidate's original predictions. The optional target-free top-k guard makes
the final ranking identical to the protected step-288 parent.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random

import numpy as np

from .contextual import ContextualResidual, previous_byte_classes
from .residual import ResidualPolicy, load_scored


def normalize(logits):
    z=logits-logits.max(axis=1,keepdims=True)
    e=np.exp(z)
    return e/e.sum(axis=1,keepdims=True)


def features(probs,classes):
    """Ten causal gate features: bias, donor entropy/margin, seven prior classes."""
    if not (probs.ndim==2 and probs.shape[1]==256 and len(probs)==len(classes)):
        raise ValueError('Mismatched donor distributions and prior classes')
    p=np.maximum(probs,1e-20)
    entropy=-(p*np.log2(p)).sum(axis=1)/8.
    largest=np.partition(probs,-2,axis=1)[:,-2:]
    margin=np.max(largest,axis=1)-np.min(largest,axis=1)
    f=np.zeros((len(probs),10),np.float64)
    f[:,0]=1
    f[:,1]=entropy
    f[:,2]=margin
    f[np.arange(len(probs)),3+classes]=1
    return f


class TrustGate:
    def __init__(self,candidate:ContextualResidual):
        self.candidate=candidate
        self.g=np.zeros(10,np.float64)
        self.m=np.zeros(10,np.float64)
        self.v=np.zeros(10,np.float64)
        self.step=0

    def distributions(self,probs,classes):
        base=np.log(np.maximum(probs,1e-20))+self.candidate.reference_weights
        contextual=np.log(np.maximum(probs,1e-20))+self.candidate.weights+self.candidate.context[classes]
        delta=contextual-base
        # 1+tanh is in (0,2), and is exactly one at zero initialization.
        strength=1.+np.tanh(features(probs,classes)@self.g)
        fitted=normalize(base+strength[:,None]*delta)
        parent=normalize(base)
        return parent,fitted

    def score(self,probs,targets,per,classes,*,guard=True):
        base,fitted=self.distributions(probs,classes)
        if guard:
            b5=np.sort(np.argpartition(base,-5,axis=1)[:,-5:],axis=1)
            f5=np.sort(np.argpartition(fitted,-5,axis=1)[:,-5:],axis=1)
            use=np.all(b5==f5,axis=1)&(base.argmax(axis=1)==fitted.argmax(axis=1))
            final=np.where(use[:,None],fitted,base)
            if not np.array_equal(np.sort(np.argpartition(final,-5,axis=1)[:,-5:],axis=1),b5):
                raise ValueError('Top-five safety invariant violated')
            if not np.array_equal(final.argmax(axis=1),base.argmax(axis=1)):
                raise ValueError('Top-one safety invariant violated')
        else:
            final=fitted;use=np.ones(len(targets),bool)
        nll=-np.log(np.maximum(final[np.arange(len(targets)),targets],1e-30))
        top5=np.argpartition(final,-5,axis=1)[:,-5:]
        return {'bpc':float(nll.mean()/math.log(2)),
                'top1':float(np.mean(final.argmax(axis=1)==targets)),
                'top5':float(np.mean(np.any(top5==targets[:,None],axis=1))),
                'fraction_context_applied':float(np.mean(use)),
                'per_book_bpc':(nll.reshape(-1,per).mean(axis=1)/math.log(2)).tolist()}

    def train(self,probs,targets,classes,*,steps,lr,seed=2300):
        """New Adam group alone; original residual/conditional moments remain."""
        rng=np.random.default_rng(seed+self.step)
        ff=features(probs,classes)
        delta=self.candidate.weights-self.candidate.reference_weights+self.candidate.context[classes]
        base=np.log(np.maximum(probs,1e-20))+self.candidate.reference_weights
        for _ in range(steps):
            ix=rng.integers(0,len(targets),size=min(512,len(targets)))
            fx=ff[ix];dl=delta[ix];y=targets[ix]
            tanh=np.tanh(fx@self.g)
            q=normalize(base[ix]+(1+tanh)[:,None]*dl)
            dloss=(q*dl).sum(axis=1)-dl[np.arange(len(ix)),y]
            grad=(fx*((1-tanh*tanh)*dloss)[:,None]).mean(axis=0)+.005*self.g
            self.step+=1
            self.m=.9*self.m+.1*grad
            self.v=.999*self.v+.001*grad*grad
            self.g-=lr*(self.m/(1-.9**self.step))/(np.sqrt(self.v/(1-.999**self.step))+1e-8)

    def checkpoint(self,path:Path):
        c=self.candidate
        np.savez_compressed(path,weights=c.weights,adam_m=c.m,adam_v=c.v,
                            critic=c.critic,adam_step=np.array(c.t),
                            reference_weights=c.reference_weights,
                            context=c.context,context_m=c.cm,context_v=c.cv,
                            context_step=np.array(c.ct),trust_g=self.g,
                            trust_m=self.m,trust_v=self.v,trust_step=np.array(self.step))

    @classmethod
    def from_checkpoint(cls,path:Path):
        with np.load(path,allow_pickle=False) as d:
            required={'weights','adam_m','adam_v','critic','adam_step',
                      'reference_weights','context','context_m','context_v',
                      'context_step','trust_g','trust_m','trust_v','trust_step'}
            if set(d.files)!=required:raise ValueError('Incomplete unified checkpoint')
            parent=ResidualPolicy()
            for key,field,shape in (('weights','weights',(256,)),
                                    ('adam_m','m',(256,)),('adam_v','v',(256,)),
                                    ('critic','critic',(2,))):
                if d[key].shape!=shape or not np.isfinite(d[key]).all():
                    raise ValueError('Invalid inherited optimizer field '+key)
                setattr(parent,field,d[key].astype(np.float64))
            parent.t=int(d['adam_step'])
            candidate=ContextualResidual(parent)
            for key,field,shape in (('reference_weights','reference_weights',(256,)),
                                    ('context','context',(7,256)),
                                    ('context_m','cm',(7,256)),
                                    ('context_v','cv',(7,256))):
                if d[key].shape!=shape or not np.isfinite(d[key]).all():
                    raise ValueError('Invalid context optimizer field '+key)
                setattr(candidate,field,d[key].astype(np.float64))
            candidate.ct=int(d['context_step'])
            out=cls(candidate)
            for key,field in (('trust_g','g'),('trust_m','m'),('trust_v','v')):
                if d[key].shape!=(10,) or not np.isfinite(d[key]).all():
                    raise ValueError('Invalid gate optimizer field '+key)
                setattr(out,field,d[key].astype(np.float64))
            out.step=int(d['trust_step'])
            return out


def train_stages(work:Path):
    """Search only existing CALIBRATION/DEV3; never train on TEST or DEV4."""
    prior=json.loads((work/'context_search.json').read_text())
    candidate_path=work/prior['best_checkpoint_path']
    inherited=ContextualResidual.from_checkpoint(candidate_path)
    cal_p,cal_y,cal_per=load_scored(work,'calibration')
    dev_p,dev_y,dev_per=load_scored(work,'dev3')
    cal_classes=previous_byte_classes(work,'calibration')
    dev_classes=previous_byte_classes(work,'dev3')
    model=TrustGate(inherited)
    previous_context=inherited.score_guarded(dev_p,dev_y,dev_per,dev_classes)
    initial=model.score(dev_p,dev_y,dev_per,dev_classes)
    if abs(previous_context['bpc']-initial['bpc'])>1e-12:
        raise ValueError('Trust gate failed function-preserving init')
    stages=[];best=initial['bpc'];best_step=0
    for count,lr in ((128,.01),(128,.005),(256,.002)):
        model.train(cal_p,cal_y,cal_classes,steps=count,lr=lr)
        dev=model.score(dev_p,dev_y,dev_per,dev_classes)
        stages.append({'gate_adam_step':model.step,'learning_rate':lr,
                       'dev3':{k:v for k,v in dev.items() if k!='per_book_bpc'}})
        if dev['bpc']<best-0.0002:
            best=dev['bpc'];best_step=model.step
            model.checkpoint(work/'trust_candidate.npz')
    report={'schema':'terminal-rsi-v102-confidence-trust-adam-v1',
            'parent_context_checkpoint_sha256':hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
            'initial_gate_prediction_exact':True,
            'inherited_residual_adam_step':inherited.t,
            'inherited_context_adam_step':inherited.ct,
            'gate_training_book_count':cal_per and len(cal_y)//cal_per,
            'gate_dev_book_count':dev_per and len(dev_y)//dev_per,
            'initial_dev3':{k:v for k,v in initial.items() if k!='per_book_bpc'},
            'stages':stages,'selected_gate_step':best_step,
            'selected_gate_dev3_bpc':best,'candidate_promoted':False,
            'caveat':'DEV3 was previously used to select step-480; no independent test of this new gate yet'}
    (work/'trust_search.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    return report


def evaluate_saved(work:Path,split:str):
    """Read-only verification on an existing, explicitly labelled corpus split."""
    if split not in ('test3','dev4','holdout4'):
        raise ValueError('Unknown evaluation split')
    selected=work/'trust_candidate.npz'
    if not selected.exists():raise ValueError('No DEV-selected confidence gate')
    gate=TrustGate.from_checkpoint(selected)
    inherited=ResidualPolicy.load_checkpoint(work/'residual_checkpoint.npz')
    probs,targets,per=load_scored(work,split)
    cls=previous_byte_classes(work,split)
    parent=inherited.score(probs,targets,per)
    contextual=gate.candidate.score_guarded(probs,targets,per,cls)
    trust_raw=gate.score(probs,targets,per,cls,guard=False)
    trust_safe=gate.score(probs,targets,per,cls,guard=True)
    if ((trust_safe['top1'],trust_safe['top5'])!=(parent['top1'],parent['top5'])):
        raise ValueError('Trust gate failed exact top-1/top-5 preservation')
    nbooks=len(targets)//per
    diff=np.array(contextual['per_book_bpc'])-np.array(trust_safe['per_book_bpc'])
    rng=random.Random(4000+len(targets))
    boot=sorted(float(np.mean(rng.choices(diff.tolist(),k=nbooks))) for _ in range(2000))
    caveats={
        'test3':'TEST3 was used to inspect the earlier contextual architecture; result is exploratory/post hoc',
        'dev4':'Eight fresh DEV books, not a full independent corpus',
        'holdout4':'64 never-trained TRAIN-remainder books; still auxiliary to sealed TEST'}
    report={'schema':'terminal-rsi-v102-confidence-trust-evaluation-v1',
            'split':split,'books':nbooks,'scored_bytes':len(targets),
            'training_updates_during_evaluation':0,
            'trust_checkpoint_sha256':hashlib.sha256(selected.read_bytes()).hexdigest(),
            'parent':parent,'inherited_context_guard':contextual,
            'trust_raw':trust_raw,'trust_guarded':trust_safe,
            'trust_vs_context_guard_bpc_gain_ci95':[boot[50],boot[1949]],
            'books_improved_over_context_guard':int(np.sum(diff>0)),
            'candidate_promoted':False,'caveat':caveats[split]}
    (work/f'trust_{split}_search.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('train','evaluate'))
    p.add_argument('--work',type=Path,required=True)
    p.add_argument('--split',choices=('test3','dev4','holdout4'))
    a=p.parse_args()
    if a.command=='train':report=train_stages(a.work)
    else:
        if not a.split:p.error('--split required for evaluate')
        report=evaluate_saved(a.work,a.split)
    print(json.dumps(report,indent=2,sort_keys=True))


if __name__=='__main__':main()
