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
from collections import Counter, defaultdict
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
        # A rank-preserving, recurrent temperature head. Zero initialization
        # is the identity; it has an independent Adam group in this *same*
        # model and can train while older parameter groups stay frozen.
        self.temp_w = np.zeros(6, np.float64)
        self.temp_m = np.zeros(6, np.float64)
        self.temp_v = np.zeros(6, np.float64)
        self.temp_step = 0

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

    def _calibrate(self, distribution, feature_rows):
        if not np.any(self.temp_w):
            return distribution.copy()
        log_t=np.clip(feature_rows@self.temp_w,-.4,.4)
        result=normalize(np.log(np.maximum(distribution,1e-30))*np.exp(log_t)[:,None])
        if not np.all(_same_top5(distribution,result)):
            raise AssertionError('RSI calibration changed the top-1/top-5 ranking')
        return result

    def distributions(self, probs, targets, per, classes, counts, orders, *, guard=True):
        experts = self._experts(probs, classes, counts, orders)
        features = self.replay_features(experts, targets, orders, per)
        proposed,factors=self._mix(experts, features, guard=guard)
        return self._calibrate(proposed,features),factors

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
                'rsi_adam_step':self.step,'temperature_adam_step':self.temp_step}

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

    def train_temperature(self, probs, targets, per, classes, counts, orders, *,
                          steps, lr, seed=9900, bias_only=False):
        """Optimize only the rank-exact calibration head from causal evidence."""
        experts=self._experts(probs,classes,counts,orders)
        features=self.replay_features(experts,targets,orders,per)
        base,_=self._mix(experts,features,guard=True)
        logp=np.log(np.maximum(base,1e-30))
        rng=np.random.default_rng(seed+self.temp_step)
        for _ in range(steps):
            ix=rng.integers(0,len(targets),size=min(512,len(targets)))
            f=features[ix];y=targets[ix];lp=logp[ix]
            unclipped=f@self.temp_w
            log_t=np.clip(unclipped,-.4,.4)
            temp=np.exp(log_t)
            pred=normalize(lp*temp[:,None])
            derivative=((pred*lp).sum(axis=1)-lp[np.arange(len(ix)),y])*temp
            derivative[np.abs(unclipped)>=.4]=0.
            grad=np.mean(f*derivative[:,None],axis=0)+.001*self.temp_w
            if bias_only:
                grad[1:]=0.
            self.temp_step+=1
            self.temp_m=.9*self.temp_m+.1*grad
            self.temp_v=.999*self.temp_v+.001*grad*grad
            self.temp_w-=lr*(self.temp_m/(1.-.9**self.temp_step))/(
                np.sqrt(self.temp_v/(1.-.999**self.temp_step))+1e-8)
            if bias_only:self.temp_w[1:]=0.

    def checkpoint(self,path:Path):
        buf=io.BytesIO()
        self.echo.checkpoint(buf)
        buf.seek(0)
        with np.load(buf,allow_pickle=False) as d:
            fields={key:d[key].copy() for key in d.files}
        fields.update(rsi_weights=self.weights,rsi_m=self.m,rsi_v=self.v,
                      rsi_step=np.array(self.step),rsi_decay=np.array(self.decay),
                      rsi_temp_w=self.temp_w,rsi_temp_m=self.temp_m,
                      rsi_temp_v=self.temp_v,rsi_temp_step=np.array(self.temp_step))
        np.savez_compressed(path,**fields)

    @classmethod
    def from_checkpoint(cls,path:Path):
        fields={'rsi_weights','rsi_m','rsi_v','rsi_step','rsi_decay'}
        temp_fields={'rsi_temp_w','rsi_temp_m','rsi_temp_v','rsi_temp_step'}
        with np.load(path,allow_pickle=False) as d:
            if not fields.issubset(d.files):
                raise ValueError('Incomplete RSI optimizer state')
            if len(temp_fields.intersection(d.files)) not in (0,len(temp_fields)):
                raise ValueError('Partial RSI calibration optimizer state')
            prior={key:d[key].copy() for key in d.files if key not in fields|temp_fields}
            rsi={key:d[key].copy() for key in fields}
            temp={key:d[key].copy() for key in temp_fields} if temp_fields.issubset(d.files) else {}
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
        for key,field in (('rsi_temp_w','temp_w'),('rsi_temp_m','temp_m'),('rsi_temp_v','temp_v')):
            if key in temp:
                if temp[key].shape!=(6,) or not np.isfinite(temp[key]).all():
                    raise ValueError('Malformed RSI calibration field '+key)
                setattr(out,field,temp[key].astype(np.float64))
        if temp:
            out.temp_step=int(temp['rsi_temp_step'])
            if out.temp_step<0:raise ValueError('Bad RSI calibration step')
        return out


class StreamingRSICore:
    """Production-shaped predict -> observe interface, with zero future access.

    Owns the within-book suffix memories, previous-byte class, RSI critic,
    and a pending prediction receipt. The caller supplies only a donor
    distribution when it requests a prediction; no target is accepted until
    after that distribution has been returned.
    """
    def __init__(self, model: FeedbackRSI):
        self.model = model
        buf=io.BytesIO()
        model.checkpoint(buf)
        self.model_sha256=hashlib.sha256(buf.getvalue()).hexdigest()
        self.reset_book()

    def reset_book(self, book_id='unassigned'):
        if getattr(self, '_pending', None) is not None:
            raise ValueError('Observe pending prediction before book reset')
        if not isinstance(book_id,str) or not book_id:
            raise ValueError('A nonempty book ID is required')
        self.book_id=book_id
        self.history = bytearray()
        self.memories = [None] + [defaultdict(Counter) for _ in range(3)]
        self.critic = FeedbackState(self.model.decay)
        self._pending = None
        self._pending_digest = None
        self.stream_digest=hashlib.sha256(
            b'terminal-rsi-v102-stream-v1\x00'+self.model_sha256.encode()+
            b'\x00'+book_id.encode('utf-8')).hexdigest()

    def _record(self, byte):
        if not isinstance(byte, (int, np.integer)) or not 0 <= int(byte) < 256:
            raise ValueError('Expected an observed byte in [0,255]')
        byte = int(byte)
        for k in (1,2,3):
            if len(self.history) >= k:
                self.memories[k][bytes(self.history[-k:])][byte] += 1
        self.history.append(byte)

    def warmup(self, byte):
        """Consume an already-observed prefix byte, before any prediction."""
        if self._pending is not None:
            raise ValueError('Cannot warm up before observing pending target')
        self._record(byte)

    @staticmethod
    def _prior_class(byte):
        if 97 <= byte <= 122:return 0
        if 65 <= byte <= 90:return 1
        if 48 <= byte <= 57:return 2
        if byte == 32:return 3
        if byte == 10:return 4
        if byte >= 128:return 5
        return 6

    def _causal_recall(self):
        row = np.zeros((1,256), np.uint16)
        order = 0
        for k in (3,2,1):
            if len(self.history) >= k:
                old = self.memories[k].get(bytes(self.history[-k:]))
                if old:
                    for byte, count in old.items():
                        if count > np.iinfo(np.uint16).max:
                            raise ValueError('Recall count overflow')
                        row[0,byte] = count
                    order = k
                    break
        return row, np.array([order], np.uint8)

    def predict(self, donor_distribution):
        if self._pending is not None:
            raise ValueError('RSI prediction already awaiting observation')
        if not self.history:
            raise ValueError('At least one observed prefix byte required')
        p = np.asarray(donor_distribution, np.float64)
        if (p.shape!=(256,) or not np.isfinite(p).all() or
                np.any(p<0) or abs(float(p.sum())-1.)>.003):
            raise ValueError('Malformed frozen donor distribution')
        counts,orders = self._causal_recall()
        classes=np.array([self._prior_class(self.history[-1])],np.int64)
        expert=self.model._experts(p.reshape(1,256), classes, counts, orders)
        features=self.critic.features(int(orders[0])).reshape(1,6)
        proposed,_=self.model._mix(expert,features,guard=True)
        proposed=self.model._calibrate(proposed,features)
        self._pending=tuple(channel[0].copy() for channel in expert)
        self._pending_digest=hashlib.sha256(
            bytes.fromhex(self.stream_digest)+
            np.asarray(p,dtype='<f8').tobytes()+
            np.asarray(proposed[0],dtype='<f8').tobytes()+
            len(self.history).to_bytes(8,'little')).digest()
        return proposed[0].copy()

    def observe(self, target_byte):
        if self._pending is None:
            raise ValueError('RSI observation without prior prediction')
        if not isinstance(target_byte,(int,np.integer)) or not 0 <= int(target_byte) < 256:
            raise ValueError('Invalid scored byte')
        self.critic.observe(*self._pending,int(target_byte))
        self._record(int(target_byte))
        self.stream_digest=hashlib.sha256(
            self._pending_digest+bytes([int(target_byte)])+
            len(self.history).to_bytes(8,'little')).hexdigest()
        self._pending=None
        self._pending_digest=None

    def snapshot(self):
        """Canonical, self-checking checkpoint at a completed prediction boundary.

        This has accidental-corruption integrity, not authenticity: SHA-256
        here is unkeyed. Serialized history rebuilds exact suffix tables;
        the critic registers are stored because donor probabilities cannot be
        reconstructed from observed bytes alone.
        """
        if self._pending is not None:
            raise ValueError('Cannot checkpoint an unobserved prediction')
        state={'schema':'terminal-rsi-v102-stream-state-v1',
               'model_sha256':self.model_sha256,
               'book_id':self.book_id,
               'observed_history_hex':self.history.hex(),
               'critic_values':self.critic.values.tolist(),
               'critic_observations':self.critic.observations,
               'stream_digest':self.stream_digest}
        data=json.dumps(state,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
        state['state_sha256']=hashlib.sha256(data).hexdigest()
        return state

    @classmethod
    def from_snapshot(cls,model:FeedbackRSI,state):
        if not isinstance(state,dict) or set(state)!={
                'schema','model_sha256','book_id','observed_history_hex',
                'critic_values','critic_observations','stream_digest','state_sha256'}:
            raise ValueError('Malformed RSI stream snapshot')
        payload={k:v for k,v in state.items() if k!='state_sha256'}
        try:
            digest=hashlib.sha256(json.dumps(
                payload,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
            history=bytes.fromhex(state['observed_history_hex'])
            values=np.asarray(state['critic_values'],dtype=np.float64)
        except (ValueError,TypeError,OverflowError) as e:
            raise ValueError('Corrupt RSI stream snapshot') from e
        if digest!=state['state_sha256'] or state['schema']!='terminal-rsi-v102-stream-state-v1':
            raise ValueError('RSI snapshot integrity check failed')
        out=cls(model)
        if out.model_sha256!=state['model_sha256']:
            raise ValueError('RSI snapshot targets different model/optimizer state')
        if (not isinstance(state['stream_digest'],str) or
                len(state['stream_digest'])!=64 or
                any(c not in '0123456789abcdef' for c in state['stream_digest'])):
            raise ValueError('Malformed RSI stream digest')
        n=state['critic_observations']
        if type(n) is not int or not 0<=n<=len(history):
            raise ValueError('Impossible RSI observation count')
        if (values.shape!=(4,) or not np.isfinite(values).all() or
                np.max(np.abs(values))>4.000000001):
            raise ValueError('Invalid RSI critic history')
        out.reset_book(state['book_id'])
        for b in history:
            out._record(b)
        out.critic.values=values
        out.critic.observations=n
        out.stream_digest=state['stream_digest']
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


def train_temperature_stages(work:Path):
    """Continue the accepted RSI migration with proper-score calibration.

    A matched-update bias-only control measures whether prior-outcome feedback
    adds value beyond a global rank-preserving temperature adjustment.
    """
    parent_path=work/'rsi_candidate.npz'
    model=FeedbackRSI.from_checkpoint(parent_path)
    control=FeedbackRSI.from_checkpoint(parent_path)
    if model.temp_step or control.temp_step:
        raise ValueError('Expected unchanged temperature migration')
    calibration=_data(work,'calibration')
    dev=_data(work,'dev3')
    initial=model.score(*dev)
    selected_step=0;best=initial['bpc']
    model.checkpoint(work/'rsi_temperature_candidate.npz')
    stages=[]
    for updates,lr in ((64,.002),(64,.001),(128,.0005)):
        model.train_temperature(*calibration,steps=updates,lr=lr)
        control.train_temperature(*calibration,steps=updates,lr=lr,bias_only=True)
        tested=model.score(*dev)
        bias_only=control.score(*dev)
        stages.append({'temperature_step':model.temp_step,'lr':lr,
                       'full_feedback_dev3':{k:v for k,v in tested.items() if k!='per_book_bpc'},
                       'bias_only_dev3_bpc':bias_only['bpc']})
        if tested['bpc']<best-.0001:
            best=tested['bpc'];selected_step=model.temp_step
            model.checkpoint(work/'rsi_temperature_candidate.npz')
    selected=FeedbackRSI.from_checkpoint(work/'rsi_temperature_candidate.npz')
    selected_dyn=selected.score(*dev)
    selected_w=selected.temp_w.copy()
    selected.temp_w[1:]=0.
    selected_bias_ablation=selected.score(*dev)
    gain=np.array(initial['per_book_bpc'])-np.array(selected_dyn['per_book_bpc'])
    feedback_gain=(np.array(selected_bias_ablation['per_book_bpc'])-
                   np.array(selected_dyn['per_book_bpc']))
    nbooks=len(gain)
    rng=random.Random(9713)
    boot=sorted(float(np.mean(rng.choices(gain.tolist(),k=nbooks))) for _ in range(2000))
    boot_feedback=sorted(float(np.mean(rng.choices(feedback_gain.tolist(),k=nbooks)))
                         for _ in range(2000))
    report={'schema':'terminal-rsi-v102-causal-feedback-temperature-v1',
            'parent_checkpoint_sha256':hashlib.sha256(parent_path.read_bytes()).hexdigest(),
            'zero_initialization_prediction_preserving':True,
            'baseline_dev3':{k:v for k,v in initial.items() if k!='per_book_bpc'},
            'stages':stages,'selected_temperature_step':selected_step,
            'selected_dev3_bpc':best,
            'selected_feedback_coefficients':selected_w.tolist(),
            'selected_bias_ablation_dev3_bpc':selected_bias_ablation['bpc'],
            'selected_full_feedback_dev3_bpc':selected_dyn['bpc'],
            'full_vs_inherited_dev3_gain_ci95':[boot[50],boot[1949]],
            'dynamic_vs_bias_ablation_dev3_gain_ci95':[boot_feedback[50],boot_feedback[1949]],
            'books_improved_over_inherited':int(np.sum(gain>0)),
            'books_improved_over_bias_ablation':int(np.sum(feedback_gain>0)),
            'candidate_promoted':False,
            'caveat':'DEV3 is reused. Validation of the recurrent effect requires untouched books'}
    (work/'rsi_temperature_search.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    return report


def evaluate_temperature_saved(work:Path,split:str):
    if split not in ('dev4','holdout4'):
        raise ValueError('Reserved independent evaluation split required')
    parent=FeedbackRSI.from_checkpoint(work/'rsi_candidate.npz')
    path=work/'rsi_temperature_candidate.npz'
    selected=FeedbackRSI.from_checkpoint(path)
    data=_data(work,split)
    original=parent.score(*data)
    corrected=selected.score(*data)
    saved=selected.temp_w.copy()
    selected.temp_w[1:]=0.
    bias_only=selected.score(*data)
    selected.temp_w[:]=saved
    if (original['top1'],original['top5'])!=(corrected['top1'],corrected['top5']):
        raise AssertionError('Temperature changed top-1/top-5')
    n=len(data[1])//data[2]
    difference=np.array(original['per_book_bpc'])-np.array(corrected['per_book_bpc'])
    rng=random.Random(12800+len(data[1]))
    ci=sorted(float(np.mean(rng.choices(difference.tolist(),k=n))) for _ in range(2000))
    report={'schema':'terminal-rsi-v102-causal-temperature-independent-v1',
            'split':split,'books':n,'positions':len(data[1]),
            'training_updates_during_evaluation':0,
            'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'inherited_rsi':original,'calibrated_rsi':corrected,
            'bias_only_ablation':bias_only,
            'bpc_gain_ci95':[ci[50],ci[1949]],
            'books_improved':int(np.sum(difference>0)),
            'candidate_promoted':False}
    (work/f'rsi_temperature_{split}_search.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    return report


def diagnose_legacy_temperature(work:Path):
    """Read-only stress test on previously inspected books; NEVER a promotion gate."""
    baseline=FeedbackRSI.from_checkpoint(work/'rsi_candidate.npz')
    selected=FeedbackRSI.from_checkpoint(work/'rsi_temperature_candidate.npz')
    report={'schema':'terminal-rsi-v102-feedback-temperature-legacy-diagnostic-v1',
            'status':'POST_HOC_INSPECTED_BOOKS_NOT_INDEPENDENT',
            'training_updates_during_diagnostic':0,
            'checkpoint_sha256':hashlib.sha256(
                (work/'rsi_temperature_candidate.npz').read_bytes()).hexdigest(),
            'splits':{},'candidate_promoted':False}
    for split in ('dev2','test2','test3'):
        data=_data(work,split)
        inherited=baseline.score(*data)
        feedback=selected.score(*data)
        saved=selected.temp_w.copy()
        selected.temp_w[1:]=0.
        global_only=selected.score(*data)
        selected.temp_w[:]=saved
        if ((inherited['top1'],inherited['top5']) !=
                (feedback['top1'],feedback['top5'])):
            raise AssertionError('Legacy feedback altered protected rankings')
        n=len(data[1])//data[2]
        diff=(np.array(inherited['per_book_bpc'])-
              np.array(feedback['per_book_bpc']))
        rng=random.Random(15000+n+len(data[1]))
        boot=sorted(float(np.mean(rng.choices(diff.tolist(),k=n))) for _ in range(2000))
        report['splits'][split]={
            'books':n,'scored_bytes':len(data[1]),
            'inherited_bpc':inherited['bpc'],
            'recurrent_feedback_bpc':feedback['bpc'],
            'global_temperature_ablation_bpc':global_only['bpc'],
            'top1':feedback['top1'],'top5':feedback['top5'],
            'feedback_bpc_gain_ci95':[boot[50],boot[1949]],
            'books_improved':int(np.sum(diff>0)),
            'reason_not_independent':'This book set was inspected in prior architecture rounds'}
    (work/'rsi_temperature_legacy_diagnostic.json').write_text(
        json.dumps(report,sort_keys=True,indent=2)+'\n')
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
    p.add_argument('command',choices=('train','evaluate','temperature-train',
                                      'temperature-evaluate','temperature-legacy-diagnostic'))
    p.add_argument('--work',type=Path,required=True)
    p.add_argument('--split',choices=('dev4','holdout4'))
    args=p.parse_args()
    if args.command=='train':out=train_stages(args.work)
    elif args.command=='temperature-train':out=train_temperature_stages(args.work)
    elif args.command=='temperature-legacy-diagnostic':out=diagnose_legacy_temperature(args.work)
    else:
        if not args.split:p.error('--split required')
        out=(evaluate_saved(args.work,args.split) if args.command=='evaluate'
             else evaluate_temperature_saved(args.work,args.split))
    print(json.dumps(out,sort_keys=True,indent=2))


if __name__=='__main__':main()
