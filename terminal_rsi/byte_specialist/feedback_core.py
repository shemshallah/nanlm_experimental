"""RSI-centered, prequential feedback controller over one v102 byte model.

Every prediction passes through this controller. Its recurring critic observes
the *previous* target only after that prediction is made, tracks counterfactual
regret of the trust and causal-recall pathways, and controls their next logit
contributions. The v102 donor, residual, context, trust and echo weights and
their full Adam histories remain inherited and immutable in this stage.

No held-out target is used to fit the controller. The candidate is never
automatically promoted. This is a byte-prediction RSI analogue, not a claim
of terminal-task RL or general recursive self-improvement.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import random

import numpy as np

from .causal_echo import CausalEcho, _data, _same_top5
from .trustgate import normalize


class FeedbackState:
    """Four slowly changing, causal critic registers, reset per source book.

    Registers mean: trust-vs-residual log advantage, echo-vs-trust log
    advantage, base model surprise and base model top-1 error. The last two
    are scaled to roughly [0,1]. They are based on frozen experts, so all
    candidate controllers see *identical* evidence and training is stable.
    """
    def __init__(self, decay=.95):
        if not 0 <= decay < 1:
            raise ValueError('Invalid RSI critic decay')
        self.decay = decay
        self.values = np.zeros(4, np.float64)
        self.observations = 0

    def features(self, recall_order: int):
        if not 0 <= int(recall_order) <= 3:
            raise ValueError('Invalid causal recall order')
        return np.array([1., *self.values, int(recall_order)/3.], np.float64)

    def observe(self, residual, trust, echo, target):
        """Call only after the target byte has been scored/predicted."""
        if not 0 <= int(target) < 256:
            raise ValueError('Bad target byte')
        y = int(target)
        p0 = max(float(residual[y]), 1e-30)
        p1 = max(float(trust[y]), 1e-30)
        p2 = max(float(echo[y]), 1e-30)
        measurements = np.array([
            np.clip(math.log(p1/p0), -4., 4.),
            np.clip(math.log(p2/p1), -4., 4.),
            min(-math.log(p0)/8., 4.),
            float(np.argmax(residual) != y),
        ], np.float64)
        self.values = self.decay*self.values + (1.-self.decay)*measurements
        self.observations += 1


class FeedbackRSI:
    """Single-model controller with its own Adam group and recurrent critic."""
    def __init__(self, echo: CausalEcho, *, decay=.95):
        self.echo = echo
        self.decay = decay
        self.weights = np.zeros((2,6), np.float64)
        self.m = np.zeros((2,6), np.float64)
        self.v = np.zeros((2,6), np.float64)
        self.step = 0

    def _experts(self, probs, classes, counts, orders):
        residual, trust, echo = self.echo.distributions(
            probs, classes, counts, orders, guard=True)
        return residual, trust, echo

    def replay_features(self, experts, targets, orders, per):
        """Trace pre-prediction state; independently reset at each book."""
        residual, trust, echo = experts
        n = len(targets)
        if n == 0 or per <= 0 or n%per or len(orders) != n:
            raise ValueError('Malformed book-aligned RSI sequence')
        inputs = np.zeros((n,6), np.float64)
        critic = FeedbackState(self.decay)
        for t in range(n):
            if t%per == 0:
                critic = FeedbackState(self.decay)
            inputs[t] = critic.features(int(orders[t]))
            # This state affects only times >= t+1. No target lookahead.
            critic.observe(residual[t], trust[t], echo[t], targets[t])
        return inputs

    def _mix(self, experts, feature_rows, *, guard=True):
        residual, trust, echo = experts
        if not (len(residual)==len(feature_rows)==len(trust)==len(echo)):
            raise ValueError('Mismatched RSI expert rows')
        if not np.any(self.weights):
            # Preserve the exact inherited output at controller migration.
            return echo.copy(), np.ones((len(echo),2),np.float64)
        z = feature_rows @ self.weights.T
        factors = 1. + .5*np.tanh(z)
        l0 = np.log(np.maximum(residual, 1e-20))
        l1 = np.log(np.maximum(trust, 1e-20))
        l2 = np.log(np.maximum(echo, 1e-20))
        proposed = normalize(l0 + factors[:,0,None]*(l1-l0) +
                             factors[:,1,None]*(l2-l1))
        if guard:
            accepted = _same_top5(residual, proposed)
            proposed = np.where(accepted[:,None], proposed, residual)
            if not np.all(_same_top5(residual,proposed)):
                raise AssertionError('RSI rank-protection invariant broken')
        return proposed, factors

    def distributions(self, probs, targets, per, classes, counts, orders, *, guard=True):
        experts = self._experts(probs, classes, counts, orders)
        features = self.replay_features(experts, targets, orders, per)
        return self._mix(experts, features, guard=guard)

    def score(self, probs, targets, per, classes, counts, orders, *, guard=True):
        distribution, factors = self.distributions(probs,targets,per,classes,counts,orders,
                                                   guard=guard)
        losses = -np.log(np.maximum(distribution[np.arange(len(targets)),targets],1e-30))
        top5 = np.argpartition(distribution,-5,axis=1)[:,-5:]
        return {'bpc':float(np.mean(losses)/math.log(2)),
                'top1':float(np.mean(distribution.argmax(axis=1)==targets)),
                'top5':float(np.mean(np.any(top5==targets[:,None],axis=1))),
                'per_book_bpc':(losses.reshape(-1,per).mean(axis=1)/math.log(2)).tolist(),
                'mean_trust_scale':float(np.mean(factors[:,0])),
                'mean_echo_scale':float(np.mean(factors[:,1])),
                'rsi_adam_step':self.step}

    def train(self, probs, targets, per, classes, counts, orders, *, steps, lr,
              seed=8800, guarded_objective=True, freeze_bias=True):
        """RSR-like supervised proper-score updates of only the RSI controls.

        This is *not PPO*: the critic is a causal, fixed counterfactual state,
        while the model learns calibrated pathway scales by Adam. A future
        true RPR experiment must be separately reported and verified.
        """
        experts=self._experts(probs,classes,counts,orders)
        features=self.replay_features(experts,targets,orders,per)
        l0=np.log(np.maximum(experts[0],1e-20))
        d1=np.log(np.maximum(experts[1],1e-20))-l0
        d2=np.log(np.maximum(experts[2],1e-20))-np.log(np.maximum(experts[1],1e-20))
        rng=np.random.default_rng(seed+self.step)
        for _ in range(steps):
            ix=rng.integers(0,len(targets),size=min(512,len(targets)))
            f=features[ix];y=targets[ix]
            z=f @ self.weights.T
            tanh=np.tanh(z)
            factors=1.+.5*tanh
            q=normalize(l0[ix]+factors[:,0,None]*d1[ix]+factors[:,1,None]*d2[ix])
            derivatives=np.column_stack((
                (q*d1[ix]).sum(axis=1)-d1[ix,y],
                (q*d2[ix]).sum(axis=1)-d2[ix,y]))
            dz=derivatives*.5*(1.-tanh*tanh)
            if guarded_objective:
                # Rejected examples revert to their protected parent, so
                # their true deployed gradient is zero. Match that objective.
                accepted=_same_top5(experts[0][ix],q)
                dz[~accepted]=0.
            grad=dz.T @ f/len(ix) + .004*self.weights
            if freeze_bias:
                # RSI must specialize from *previous observations*, not
                # merely learn another global rescaling of expert weights.
                grad[:,0]=0.
            self.step+=1
            self.m=.9*self.m+.1*grad
            self.v=.999*self.v+.001*grad*grad
            self.weights-=lr*(self.m/(1.-.9**self.step))/(
                np.sqrt(self.v/(1.-.999**self.step))+1e-8)
            if freeze_bias:
                self.weights[:,0]=0.

    def checkpoint(self,path:Path):
        buf=io.BytesIO()
        self.echo.checkpoint(buf)
        buf.seek(0)
        with np.load(buf,allow_pickle=False) as d:
            fields={key:d[key].copy() for key in d.files}
        fields.update(rsi_weights=self.weights,rsi_m=self.m,rsi_v=self.v,
                      rsi_step=np.array(self.step),rsi_decay=np.array(self.decay))
        np.savez_compressed(path,**fields)

    @classmethod
    def from_checkpoint(cls,path:Path):
        fields={'rsi_weights','rsi_m','rsi_v','rsi_step','rsi_decay'}
        with np.load(path,allow_pickle=False) as d:
            if not fields.issubset(d.files):
                raise ValueError('Incomplete RSI optimizer state')
            prior={key:d[key].copy() for key in d.files if key not in fields}
            rsi={key:d[key].copy() for key in fields}
        buf=io.BytesIO()
        np.savez_compressed(buf,**prior)
        buf.seek(0)
        out=cls(CausalEcho.from_checkpoint(buf),decay=float(rsi['rsi_decay']))
        for key,field in (('rsi_weights','weights'),('rsi_m','m'),('rsi_v','v')):
            if rsi[key].shape!=(2,6) or not np.isfinite(rsi[key]).all():
                raise ValueError('Malformed RSI field '+key)
            setattr(out,field,rsi[key].astype(np.float64))
        out.step=int(rsi['rsi_step'])
        if out.step<0:raise ValueError('Bad RSI step')
        return out


def train_stages(work:Path):
    inherited_path=work/'echo_candidate.npz'
    inherited=FeedbackRSI(CausalEcho.from_checkpoint(inherited_path))
    cal=_data(work,'calibration')
    dev=_data(work,'dev3')
    baseline=inherited.score(*dev)
    inherited.checkpoint(work/'rsi_candidate.npz')
    best=baseline['bpc'];step=0;stages=[]
    for updates,lr in ((128,.005),(128,.002),(128,.001)):
        inherited.train(cal[0],cal[1],cal[2],cal[3],cal[4],cal[5],steps=updates,lr=lr)
        result=inherited.score(*dev)
        stages.append({'step':inherited.step,'lr':lr,
                       'dev3':{k:v for k,v in result.items() if k!='per_book_bpc'}})
        if result['bpc']<best-.0002:
            best=result['bpc'];step=inherited.step
            inherited.checkpoint(work/'rsi_candidate.npz')
    report={'schema':'terminal-rsi-v102-recurrent-feedback-core-v1',
            'inherited_echo_sha256':hashlib.sha256(inherited_path.read_bytes()).hexdigest(),
            'migration_prediction_preserving':True,
            'initial_dev3':{k:v for k,v in baseline.items() if k!='per_book_bpc'},
            'stages':stages,'selected_step':step,'selected_dev3_bpc':best,
            'candidate_promoted':False,
            'caveat':'Prequential recurrence tested only on exploratory DEV3; fresh held-out books required'}
    (work/'rsi_search.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    return report


def evaluate_saved(work:Path,split:str):
    if split not in ('dev4','holdout4'):
        raise ValueError('Reserved RSI evaluation split required')
    path=work/'rsi_candidate.npz'
    model=FeedbackRSI.from_checkpoint(path)
    data=_data(work,split)
    baseline=FeedbackRSI(model.echo,decay=model.decay).score(*data)
    candidate=model.score(*data)
    per=data[2];n=len(data[1])//per
    bookdiff=np.array(baseline['per_book_bpc'])-np.array(candidate['per_book_bpc'])
    rng=random.Random(10100+len(data[1]))
    ci=sorted(float(np.mean(rng.choices(bookdiff.tolist(),k=n))) for _ in range(2000))
    if (baseline['top1'],baseline['top5']) != (candidate['top1'],candidate['top5']):
        raise ValueError('RSI feedback changed inherited top-k rankings')
    report={'schema':'terminal-rsi-v102-recurrent-feedback-frozen-eval-v1',
            'split':split,'books':n,'positions':len(data[1]),
            'training_updates_during_evaluation':0,
            'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'inherited_echo':baseline,'rsi_feedback':candidate,
            'bpc_gain_ci95':[ci[50],ci[1949]],
            'books_improved':int(np.sum(bookdiff>0)),
            'candidate_promoted':False}
    (work/f'rsi_{split}_search.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('train','evaluate'))
    p.add_argument('--work',type=Path,required=True)
    p.add_argument('--split',choices=('dev4','holdout4'))
    args=p.parse_args()
    if args.command=='train':out=train_stages(args.work)
    else:
        if not args.split:p.error('--split required')
        out=evaluate_saved(args.work,args.split)
    print(json.dumps(out,sort_keys=True,indent=2))


if __name__=='__main__':main()
