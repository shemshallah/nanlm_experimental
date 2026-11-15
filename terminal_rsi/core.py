"""Standalone Terminal-RSI verified rewrite and turn-aware credit contracts.

No Poincare/FA dependencies, command execution, uncontrolled self-modification,
or checkpoint promotion. Historical compatibility lives in poincare_mirror.
Ref: Yang et al., Terminal-RSI, arXiv:2610.22192 (2026).
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Callable, Mapping, Sequence


def finite(x):
    if isinstance(x, bool) or not math.isfinite(float(x)):
        raise ValueError('Nonfinite value')
    return float(x)


@dataclass(frozen=True)
class Outcome:
    family: str
    success: bool
    passed: int = 0
    total: int = 0
    valid: bool = False

    def __post_init__(self):
        if not self.family or type(self.success) is not bool or type(self.valid) is not bool:
            raise ValueError('Invalid task/result fields')
        if type(self.passed) is not int or type(self.total) is not int or not 0 <= self.passed <= self.total:
            raise ValueError('Invalid semantic checks')


@dataclass(frozen=True)
class Calibration:
    divisors: Mapping[str, float]
    fallback: float = 82.

    def __post_init__(self):
        if finite(self.fallback) <= 0 or any(not f or finite(v) <= 0 for f,v in self.divisors.items()):
            raise ValueError('Divisors must be positive')
        object.__setattr__(self, 'divisors', tuple(sorted((f, float(v)) for f,v in self.divisors.items())))

    @classmethod
    def from_history(cls, samples: Mapping[str, Sequence[int]]):
        result = {}
        for family, counts in samples.items():
            if any(type(n) is not int or n < 0 for n in counts):
                raise ValueError('Bad historical count')
            if counts:
                ordered = sorted(counts)
                n = ordered[math.ceil(.9 * len(ordered))-1]
                if n: result[family] = n
        return cls(result)

    def divisor(self, family):
        return dict(self.divisors).get(family, self.fallback)

    def adapted(self, high_failure=(), long_low=()):
        high, low = set(high_failure), set(long_low)
        result = dict(self.divisors)
        for f in low-high: result[f] = self.divisor(f)/4
        for f in high: result[f] = 2*self.divisor(f)
        return Calibration(result, self.fallback)


def fdr_reward(o: Outcome, calibration: Calibration):
    return o.passed/calibration.divisor(o.family) if o.valid else float(o.success)


def spr_reward(o: Outcome, calibration: Calibration):
    return float(o.success) + o.passed/(o.passed+calibration.divisor(o.family)) if o.valid else float(o.success)


@dataclass(frozen=True)
class Action:
    turn_id: int
    value: float
    reward: float = 0.
    terminated: bool = False


@dataclass(frozen=True)
class Continuation:
    value: float
    advantage: float
    turn_id: int


@dataclass(frozen=True)
class Credit:
    advantages: tuple[float, ...]
    targets: tuple[float, ...]
    deltas: tuple[float, ...]


def tb_gae(actions: Sequence[Action], gamma_turn=.99, lambda_turn=.95,
           continuation: Continuation | None = None) -> Credit:
    """Same assistant turn: gamma=lambda=1; between turns: discounted.

    Observations do not appear in `actions`. Artificial truncations require
    successor value, advantage and turn_id; actual termination never bootstraps.
    """
    if not 0 <= finite(gamma_turn) <= 1 or not 0 <= finite(lambda_turn) <= 1:
        raise ValueError('Discounts outside [0,1]')
    if not actions: return Credit((),(),())
    if any(type(a.turn_id) is not int or a.turn_id < 0 for a in actions):
        raise ValueError('Bad turn ID')
    if any(b.turn_id < a.turn_id for a,b in zip(actions,actions[1:])):
        raise ValueError('Backwards turn')
    if any(a.terminated for a in actions[:-1]):
        raise ValueError('Actions after termination')
    for a in actions: finite(a.value); finite(a.reward)
    if not actions[-1].terminated and continuation is None:
        raise ValueError('Artificial cutoff requires complete continuation')
    if continuation:
        finite(continuation.value); finite(continuation.advantage)
        if continuation.turn_id < actions[-1].turn_id: raise ValueError('Backwards continuation')
    adv = [0.]*len(actions); targets = adv.copy(); deltas = adv.copy()
    next_adv = continuation.advantage if continuation else 0.
    for i in range(len(actions)-1,-1,-1):
        a=actions[i]
        if a.terminated:
            gamma, lam, next_value = 1., 1., 0.
        else:
            next_id,next_value = ((actions[i+1].turn_id, actions[i+1].value)
                                  if i+1 < len(actions) else (continuation.turn_id, continuation.value))
            gamma,lam = (1.,1.) if a.turn_id==next_id else (gamma_turn,lambda_turn)
        delta=a.reward+(0. if a.terminated else gamma*next_value)-a.value
        advantage=delta+(0. if a.terminated else gamma*lam*next_adv)
        deltas[i]=delta; adv[i]=advantage; targets[i]=advantage+a.value
        next_adv=advantage
    return Credit(tuple(adv),tuple(targets),tuple(deltas))


@dataclass(frozen=True)
class Event:
    role: str
    text: str
    turn_id: int
    visibility: str = 'public'


@dataclass(frozen=True)
class Trajectory:
    task_id: str
    family: str
    harness_id: str
    run_id: str
    events: tuple[Event,...]
    success: bool

    def digest(self):
        record=dict(task_id=self.task_id,family=self.family,harness_id=self.harness_id,
                    run_id=self.run_id,events=[vars(x) for x in self.events],success=self.success)
        return hashlib.sha256(json.dumps(record,sort_keys=True,separators=(',',':')).encode()).hexdigest()


@dataclass(frozen=True)
class Receipt:
    task_id: str
    run_id: str
    harness_id: str
    digest: str
    success: bool


@dataclass(frozen=True)
class SFTExample:
    donor_digest: str
    replay_digest: str
    messages: tuple[tuple[str,str],...]
    loss_mask: tuple[bool,...]


def screen(t: Trajectory):
    if not (t.task_id and t.harness_id and t.run_id and t.events and
            any(e.role=='assistant' for e in t.events)):
        raise ValueError('Incomplete public trajectory')
    if any(e.visibility!='public' or e.role not in ('user','assistant','tool') or
           type(e.turn_id) is not int or e.turn_id<0 for e in t.events):
        raise ValueError('Private material is forbidden')
    if any(b.turn_id<a.turn_id for a,b in zip(t.events,t.events[1:])):
        raise ValueError('Nonmonotone turns')


def adopt_rewrite(donor: Trajectory, replay: Callable, verifier: Callable) -> SFTExample:
    """Replay public donor actions in a fresh target harness, then independently verify.

    The supplied verifier must itself be trusted. A matching digest does not
    authenticate a malicious verifier, and this module never runs commands.
    """
    screen(donor)
    if not donor.success: raise ValueError('Failed donor')
    actions=tuple(e.text for e in donor.events if e.role=='assistant')
    target=replay(donor.task_id, actions)
    screen(target)
    if (target.task_id!=donor.task_id or target.run_id==donor.run_id or
        target.harness_id==donor.harness_id or
        tuple(e.text for e in target.events if e.role=='assistant')!=actions):
        raise ValueError('Replay not fresh, mismatched or changed')
    result=verifier(target)
    if not (target.success and result.success and result.task_id==target.task_id and
            result.run_id==target.run_id and result.harness_id==target.harness_id and
            result.digest==target.digest()):
        raise ValueError('Unverified target replay')
    return SFTExample(donor.digest(),target.digest(),
                      tuple((e.role,e.text) for e in target.events),
                      tuple(e.role=='assistant' for e in target.events))


@dataclass(frozen=True)
class TaskGate:
    """Independent terminal-task progress; no language-model measurements."""
    strict_success_rate: float
    process_reward: float
    verified_tasks: int
    covered_families: int


def preliminary_task_gate(base: TaskGate, candidate: TaskGate,
                          tolerance=1e-6) -> tuple[bool,tuple[str,...]]:
    """Preliminary candidate screen, *never* sufficient to auto-promote.

    Real promotion additionally needs authenticated external receipts, an
    untouched test split, a nonrecursive control, and uncertainty estimates.
    """
    if finite(tolerance) < 0:
        raise ValueError('Bad tolerance')
    for result in (base, candidate):
        if (type(result.verified_tasks) is not int or result.verified_tasks <= 0 or
                type(result.covered_families) is not int or result.covered_families <= 0 or
                result.covered_families > result.verified_tasks):
            raise ValueError('Invalid external verification coverage')
        for score in (result.strict_success_rate, result.process_reward):
            if not 0 <= finite(score) <= 1:
                raise ValueError('Task scores must lie in [0,1]')
    failures = []
    if candidate.verified_tasks < base.verified_tasks:
        failures.append('verified_task_coverage')
    if candidate.covered_families < base.covered_families:
        failures.append('family_coverage')
    if candidate.strict_success_rate <= base.strict_success_rate + tolerance:
        failures.append('no_strict_success_improvement')
    if candidate.process_reward + tolerance < base.process_reward:
        failures.append('process_reward_regression')
    return not failures, tuple(failures)


STAGES=('BASE','RSR-1','RPR-1','RSR-2','RPR-2')


@dataclass(frozen=True)
class Stage:
    name: str
    actor_sha: str
    critic_sha: str
    optimizer_sha: str
    verified_sft: int = 0
    critic_warmup: int = 0


def check_transition(previous: Stage, candidate: Stage):
    if (previous.name not in STAGES or candidate.name not in STAGES or
        STAGES.index(candidate.name)!=STAGES.index(previous.name)+1):
        raise ValueError('Unexpected phase change')
    for record in (previous,candidate):
        for digest in (record.actor_sha,record.critic_sha,record.optimizer_sha):
            if len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):
                raise ValueError('Invalid SHA-256 stage identity')
    if candidate.name.startswith('RSR'):
        if candidate.critic_sha!=previous.critic_sha or candidate.verified_sft<=0:
            raise ValueError('RSR changes critic or lacks verified SFT')
    elif candidate.critic_warmup<4:
        raise ValueError('RPR requires >=4 critic-only warmup updates')
