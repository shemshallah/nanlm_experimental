"""Causal previous-byte-class expansion of the SAME Terminal-RSI correction.

Function-preserving migration: inherit the accepted residual weights AND Adam
moments at step 288; initialize seven new context rows to zero. Original donor
v102 remains frozen for all compared evaluations. No FA-model imports.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import struct

import numpy as np

from . import nancm_v102 as donor
from .controller import (assign_books, digest, select_windows, write_windows)
from .residual import ResidualPolicy, load_scored


def prepare_new_books(*, corpus:Path, work:Path):
    """Never reuse original DEV1, TEST1, DEV2, or TEST2 source books."""
    parts=assign_books(corpus)
    orig=json.loads((work/'split_manifest.json').read_text())
    bpe=donor.BPE.load(work/'bpe.txt')
    width,skip=orig['width'],orig['skip']
    def subset(books,n,seed):
        return select_windows(books,bpe,count=n,width=width,skip=skip,seed=seed)
    def remaining(group,old):
        names={n for n,_ in old}
        return [(n,b) for n,b in group if n not in names]
    dev_unused=remaining(parts['dev'],subset(parts['dev'],48,1001))
    dev_unused=remaining(dev_unused,subset(dev_unused,24,4004))
    test_unused=remaining(parts['test'],subset(parts['test'],48,2002))
    test_unused=remaining(test_unused,subset(test_unused,24,5005))
    sets={'dev3':subset(dev_unused,24,6006),
          'test3':subset(test_unused,8,7007)}
    result={'schema':'terminal-rsi-contextual-fresh-book-splits-v1',
            'parent_residual_checkpoint_sha256':digest((work/'residual_checkpoint.npz').read_bytes()),
            'splits':{key:write_windows(work,key,items,bpe,skip=skip)
                      for key,items in sets.items()}}
    (work/'context_manifest.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    return result


def prepare_guard_books(*,corpus:Path,work:Path):
    """Last untouched DEV source books, after prior context selection."""
    parts=assign_books(corpus)
    original=json.loads((work/'split_manifest.json').read_text())
    bpe=donor.BPE.load(work/'bpe.txt')
    width,skip=original['width'],original['skip']
    books=parts['dev']
    for n,seed in ((48,1001),(24,4004),(24,6006)):
        already=select_windows(books,bpe,count=n,width=width,skip=skip,seed=seed)
        names={name for name,_ in already}
        books=[(name,content) for name,content in books if name not in names]
    selected=select_windows(books,bpe,count=8,width=width,skip=skip,seed=8008)
    result={'schema':'terminal-rsi-guard-independent-dev4-v1',
            'prior_dev_book_counts':[48,24,24],
            'splits':{'dev4':write_windows(work,'dev4',selected,bpe,skip=skip)}}
    (work/'guard_manifest.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    return result


def previous_byte_classes(work:Path,split:str):
    """Features are *strictly prior* observed bytes, never scored targets."""
    payload=(work/f'win_{split}.bin').read_bytes()
    count,skip=struct.unpack_from('<II',payload)
    pos=8;values=[]
    for _ in range(count):
        width=struct.unpack_from('<I',payload,pos)[0];pos+=4
        window=payload[pos:pos+width];pos+=width
        if skip<1:raise ValueError('Need previous observed byte before scoring')
        values.extend(window[skip-1:width-1])
    if pos!=len(payload):raise ValueError('Corrupt heldout window container')
    prev=np.frombuffer(bytes(values),dtype=np.uint8)
    category=np.full(len(prev),6,dtype=np.int64)
    category[(prev>=97)&(prev<=122)]=0
    category[(prev>=65)&(prev<=90)]=1
    category[(prev>=48)&(prev<=57)]=2
    category[prev==32]=3
    category[prev==10]=4
    category[prev>=128]=5
    return category


class ContextualResidual(ResidualPolicy):
    """Zero-init seven-context extension with inherited parent optimizer."""
    def __init__(self,parent:ResidualPolicy):
        super().__init__()
        self.weights=parent.weights.copy();self.m=parent.m.copy()
        self.v=parent.v.copy();self.t=parent.t;self.critic=parent.critic.copy()
        self.reference_weights=parent.weights.copy()
        self.context=np.zeros((7,256),np.float64)
        self.cm=np.zeros_like(self.context)
        self.cv=np.zeros_like(self.context)
        self.ct=0

    def score(self,probs,targets,per,classes):
        if len(targets)!=len(classes):raise ValueError('Class label/target mismatch')
        logits=np.log(np.maximum(probs,1e-20))+self.weights+self.context[classes]
        logits-=logits.max(axis=1,keepdims=True)
        p=np.exp(logits);p/=p.sum(axis=1,keepdims=True)
        losses=-np.log(np.maximum(p[np.arange(len(targets)),targets],1e-30))
        top5=np.argpartition(p,-5,axis=1)[:,-5:]
        return {'bpc':float(losses.mean()/math.log(2)),
                'top1':float(np.mean(p.argmax(axis=1)==targets)),
                'top5':float(np.mean(np.any(top5==targets[:,None],axis=1))),
                'per_book_bpc':(losses.reshape(-1,per).mean(axis=1)/math.log(2)).tolist()}

    def score_guarded(self,probs,targets,per,classes):
        """Causal confidence gate: preserve inherited top-1 and top-5 exactly.

        No ground-truth target participates in the route decision. A change
        to either predicted top-1 or the membership of the predicted top-5
        disables the conditional correction at that position.
        """
        if len(targets)!=len(classes):raise ValueError('Class/target mismatch')
        def normalized(logits):
            logits=logits-logits.max(axis=1,keepdims=True)
            x=np.exp(logits)
            return x/x.sum(axis=1,keepdims=True)
        original_logits=np.log(np.maximum(probs,1e-20))
        inherited=normalized(original_logits+self.reference_weights)
        corrected=normalized(original_logits+self.weights+self.context[classes])
        old_top=np.sort(np.argpartition(inherited,-5,axis=1)[:,-5:],axis=1)
        new_top=np.sort(np.argpartition(corrected,-5,axis=1)[:,-5:],axis=1)
        use=(np.all(old_top==new_top,axis=1) &
             (inherited.argmax(axis=1)==corrected.argmax(axis=1)))
        final=np.where(use[:,None],corrected,inherited)
        losses=-np.log(np.maximum(final[np.arange(len(targets)),targets],1e-30))
        # Invariant is checked independently of true targets.
        if (not np.array_equal(final.argmax(axis=1),inherited.argmax(axis=1)) or
                not np.array_equal(np.sort(np.argpartition(final,-5,axis=1)[:,-5:],axis=1),old_top)):
            raise ValueError('Causal top-k guard invariant violated')
        return {'bpc':float(losses.mean()/math.log(2)),
                'top1':float(np.mean(final.argmax(axis=1)==targets)),
                'top5':float(np.mean(np.any(old_top==targets[:,None],axis=1))),
                'positions_using_context':int(np.sum(use)),
                'fraction_using_context':float(np.mean(use)),
                'per_book_bpc':(losses.reshape(-1,per).mean(axis=1)/math.log(2)).tolist()}

    def train(self,probs,targets,classes,*,steps,lr,phase='RSR',seed=2030):
        if phase not in ('RSR','RPR'):raise ValueError('Unknown learning phase')
        rng=np.random.default_rng(seed+self.t)
        for _ in range(steps):
            indices=rng.integers(0,len(targets),size=min(len(targets),512))
            p=probs[indices];y=targets[indices];c=classes[indices]
            logits=np.log(np.maximum(p,1e-20))+self.weights+self.context[c]
            logits-=logits.max(axis=1,keepdims=True)
            q=np.exp(logits);q/=q.sum(axis=1,keepdims=True)
            losses=-np.log(np.maximum(q[np.arange(len(y)),y],1e-30))
            groups=(y!=32).astype(np.int64)
            if phase=='RPR':
                # Inherited critic reallocates a bounded share of gradient
                # weight to whichever verified byte family remains harder.
                weights=np.clip(1+.15*(self.critic[groups]-self.critic.mean()),.9,1.1)
            else:weights=np.ones(len(y),dtype=np.float64)
            grad=q;grad[np.arange(len(y)),y]-=1.
            grad*=weights[:,None]
            gg=grad.mean(axis=0)
            cg=np.zeros((7,256),np.float64)
            np.add.at(cg,c,grad)
            cg=cg/len(y)+.01*self.context
            # Original Adam group resumes *exactly* from inherited moments.
            self.t+=1
            self.m=.9*self.m+.1*gg;self.v=.999*self.v+.001*gg*gg
            self.weights-=lr*(self.m/(1-.9**self.t))/(np.sqrt(self.v/(1-.999**self.t))+1e-8)
            # New group starts fresh while belonging to the same policy.
            self.ct+=1
            self.cm=.9*self.cm+.1*cg;self.cv=.999*self.cv+.001*cg*cg
            self.context-=lr*(self.cm/(1-.9**self.ct))/(np.sqrt(self.cv/(1-.999**self.ct))+1e-8)
            if phase=='RPR':
                for j in range(2):
                    if np.any(groups==j):
                        self.critic[j]=.95*self.critic[j]+.05*float(losses[groups==j].mean())

    def checkpoint(self,path:Path):
        np.savez_compressed(path,weights=self.weights,m=self.m,v=self.v,
                            critic=self.critic,adam_step=np.array(self.t),
                            reference_weights=self.reference_weights,
                            context=self.context,cm=self.cm,cv=self.cv,
                            context_step=np.array(self.ct))

    @classmethod
    def from_checkpoint(cls,path:Path):
        with np.load(path,allow_pickle=False) as data:
            needed={'weights','m','v','critic','adam_step','reference_weights',
                    'context','cm','cv','context_step'}
            if set(data.files)!=needed:raise ValueError('Incomplete contextual checkpoint')
            parent=ResidualPolicy()
            for name in ('weights','m','v','critic'):
                expected=(2,) if name=='critic' else (256,)
                if data[name].shape!=expected or not np.isfinite(data[name]).all():
                    raise ValueError('Corrupt inherited state')
                setattr(parent,name,data[name].astype(np.float64))
            parent.t=int(data['adam_step'])
            out=cls(parent)
            if data['reference_weights'].shape!=(256,) or not np.isfinite(data['reference_weights']).all():
                raise ValueError('Corrupt inherited reference')
            out.reference_weights=data['reference_weights'].astype(np.float64)
            for name in ('context','cm','cv'):
                if data[name].shape!=(7,256) or not np.isfinite(data[name]).all():
                    raise ValueError('Corrupt context state')
                setattr(out,name,data[name].astype(np.float64))
            out.ct=int(data['context_step'])
            return out


def run(work:Path,*,steps=64):
    parent=ResidualPolicy.load_checkpoint(work/'residual_checkpoint.npz')
    train_p,train_y,train_per=load_scored(work,'calibration')
    train_c=previous_byte_classes(work,'calibration')
    dev_p,dev_y,dev_per=load_scored(work,'dev3')
    dev_c=previous_byte_classes(work,'dev3')
    candidate=ContextualResidual(parent)
    original=parent.score(dev_p,dev_y,dev_per)
    migrated=candidate.score(dev_p,dev_y,dev_per,dev_c)
    if original!=migrated:
        raise ValueError('Function-preserving migration failed')
    history=[{'stage':'INHERITED','adam_step':parent.t,'context_step':0,
              'dev3':original}]
    best=original['bpc'];best_checkpoint=None
    for stage,rate in [('CONTEXT-RSR1',.006),('CONTEXT-RPR1',.0015),
                       ('CONTEXT-RSR2',.001),('CONTEXT-RPR2',.0005)]:
        candidate.train(train_p,train_y,train_c,steps=steps,lr=rate,
                        phase='RPR' if 'RPR' in stage else 'RSR')
        measured=candidate.score(dev_p,dev_y,dev_per,dev_c)
        # Require joint BPC improvement and at least non-regression on both
        # exact prediction accuracies, measured only on DEV3.
        eligible=(measured['bpc']<best-.0002 and
                  measured['top1']+1e-12>=original['top1'] and
                  measured['top5']+1e-12>=original['top5'])
        if eligible:
            best=measured['bpc'];best_checkpoint=work/f'context_candidate_step{candidate.t}.npz'
            candidate.checkpoint(best_checkpoint)
        history.append({'stage':stage,'adam_step':candidate.t,'context_step':candidate.ct,
                        'dev3':measured,'gate_pass':eligible})
    if best_checkpoint:
        tested=ContextualResidual.from_checkpoint(best_checkpoint)
    else:
        tested=ContextualResidual(parent)
    # TEST3 never influences the candidate selection or phase learning.
    test_p,test_y,test_per=load_scored(work,'test3')
    test_c=previous_byte_classes(work,'test3')
    base_test=parent.score(test_p,test_y,test_per)
    test=tested.score(test_p,test_y,test_per,test_c)
    guarded=tested.score_guarded(test_p,test_y,test_per,test_c)
    guarded_dev=tested.score_guarded(dev_p,dev_y,dev_per,dev_c)
    improvements=np.array(base_test['per_book_bpc'])-np.array(test['per_book_bpc'])
    rnd=random.Random(126)
    ci=sorted(float(np.mean(rnd.choices(improvements.tolist(),k=len(improvements))))
              for _ in range(2000))
    report={'schema':'terminal-rsi-v102-context-expansion-v1',
            'parent_checkpoint_step':parent.t,'parent_checkpoint_sha256':digest((work/'residual_checkpoint.npz').read_bytes()),
            'source_split':json.loads((work/'context_manifest.json').read_text()),
            'migration_exact':True,'stages':history,
            'dev3_accepted':bool(best_checkpoint),
            'best_checkpoint_path':best_checkpoint.name if best_checkpoint else None,
            'test3_parent':base_test,'test3_contextual':test,
            'dev3_guarded_posthoc':guarded_dev,
            'test3_guarded_posthoc':guarded,
            'test3_bpc_gain_ci95':[ci[50],ci[1949]],
            'promoted':False,
            'caveat':'Only 8 fresh test books remain in this release; exploratory, not promotion-grade'}
    (work/'context_search.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    return report


def evaluate_guard(work:Path):
    """Confirm the post-hoc causal top-k guard on genuinely new DEV4 books."""
    previous=json.loads((work/'context_search.json').read_text())
    name=previous['best_checkpoint_path']
    if not name:raise ValueError('No selected DEV3 contextual candidate')
    candidate=ContextualResidual.from_checkpoint(work/name)
    parent=ResidualPolicy.load_checkpoint(work/'residual_checkpoint.npz')
    p,y,per=load_scored(work,'dev4')
    classes=previous_byte_classes(work,'dev4')
    base=parent.score(p,y,per)
    guarded=candidate.score_guarded(p,y,per,classes)
    raw=candidate.score(p,y,per,classes)
    if (base['top1']!=guarded['top1'] or base['top5']!=guarded['top5']):
        raise ValueError('Exact top1/top5 preservation failed')
    d=np.array(base['per_book_bpc'])-np.array(guarded['per_book_bpc'])
    rng=random.Random(145)
    ci=sorted(float(np.mean(rng.choices(d.tolist(),k=len(d)))) for _ in range(2000))
    result={'schema':'terminal-rsi-context-topk-guard-independent-check-v1',
            'books':len(d),'scored_bytes':len(y),'candidate_checkpoint':name,
            'guard': 'Use candidate only where argmax and unordered top-5 agree with inherited model',
            'decision_reads_target':False,'top1_top5_proven_identical':True,
            'inherited':base,'raw_context':raw,'guarded_context':guarded,
            'guarded_bpc_gain_ci95':[ci[50],ci[1949]],
            'candidate_promoted':False,
            'caveat':'Eight independent DEV books; preserve model and revalidate on larger fresh TEST corpora before promotion'}
    (work/'guard_search.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','train','prepare-guard','eval-guard'))
    parser.add_argument('--work',type=Path,required=True)
    parser.add_argument('--corpus',type=Path)
    args=parser.parse_args()
    if args.command=='prepare':
        if not args.corpus:parser.error('--corpus required')
        result=prepare_new_books(corpus=args.corpus,work=args.work)
    elif args.command=='prepare-guard':
        if not args.corpus:parser.error('--corpus required')
        result=prepare_guard_books(corpus=args.corpus,work=args.work)
    elif args.command=='eval-guard':result=evaluate_guard(args.work)
    else:result=run(args.work)
    if args.command=='train':
        print(json.dumps({'parent_step':result['parent_checkpoint_step'],
                          'stages':[(x['stage'],x['dev3']['bpc'],x.get('gate_pass')) for x in result['stages']],
                          'dev3_accepted':result['dev3_accepted'],
                          'test3_parent':{k:v for k,v in result['test3_parent'].items() if k!='per_book_bpc'},
                          'test3_context':{k:v for k,v in result['test3_contextual'].items() if k!='per_book_bpc'},
                          'test3_guarded_posthoc':{k:v for k,v in result['test3_guarded_posthoc'].items() if k!='per_book_bpc'},
                          'test3_gain_ci95':result['test3_bpc_gain_ci95']},indent=2))
    elif args.command=='eval-guard':
        print(json.dumps({'books':result['books'],
                          'inherited':{k:v for k,v in result['inherited'].items() if k!='per_book_bpc'},
                          'raw':{k:v for k,v in result['raw_context'].items() if k!='per_book_bpc'},
                          'guarded':{k:v for k,v in result['guarded_context'].items() if k!='per_book_bpc'},
                          'bpc_gain_ci95':result['guarded_bpc_gain_ci95']},indent=2))
    else:print(json.dumps(result,indent=2))


if __name__=='__main__':main()
