"""From-scratch, FA-independent finite-action Terminal-RSI research pilot.

This is an intentionally small *synthetic* verifier-backed task harness, not
Terminal-Bench and not a pretrained terminal agent. A single actor and its
Adam moments persist across BASE -> RSR-1 -> RPR-1 -> RSR-2 -> RPR-2.
No shell commands are executed and there is no checkpoint auto-promotion.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import random
import re
import time

import numpy as np

from .core import (Action, Calibration, Event, Outcome, Receipt, Stage,
                   Trajectory, adopt_rewrite, check_transition, spr_reward, tb_gae)


FAMILIES = (
    'COPY_A', 'COPY_B', 'NEXT_A', 'NEXT_B', 'PREV_A', 'PREV_B',
    'DOUBLE_A', 'DOUBLE_B', 'NINE_A', 'NINE_B',
)
_PROMPT = re.compile(r'^([A-Z_]+) ([0-9]) ([0-9])$')


def verified_answer(prompt: str) -> str:
    """External, deterministic specification; does not trust the actor label."""
    match = _PROMPT.fullmatch(prompt)
    if not match or match[1] not in FAMILIES:
        raise ValueError('Unrecognized finite-action task')
    op, left, right = match[1], int(match[2]), int(match[3])
    value = left if op.endswith('_A') else right
    if op.startswith('NEXT_'):
        value = (value + 1) % 10
    elif op.startswith('PREV_'):
        value = (value - 1) % 10
    elif op.startswith('DOUBLE_'):
        value = (2 * value) % 10
    elif op.startswith('NINE_'):
        value = 9 - value
    return str(value)


@dataclass(frozen=True)
class Task:
    family: str
    a: int
    b: int

    @property
    def prompt(self):
        return f'{self.family} {self.a} {self.b}'

    @property
    def task_id(self):
        return self.prompt


def task_split(seed: int):
    """80/20 nonoverlapping task IDs within each seen family, deterministic."""
    train, dev = [], []
    for family in FAMILIES:
        tasks = [Task(family, a, b) for a in range(10) for b in range(10)]
        random.Random(f'{seed}:{family}').shuffle(tasks)
        train.extend(tasks[:80])
        dev.extend(tasks[80:])
    assert len(train) == 800 and len(dev) == 200
    assert not {t.task_id for t in train}.intersection(t.task_id for t in dev)
    return tuple(train), tuple(dev)


def verified_rewrite(task: Task, serial: int):
    """Check successful public donor actions against a fresh replay and receipt."""
    answer = verified_answer(task.prompt)
    donor = Trajectory(task.task_id, task.family, 'donor', f'donor-{serial}',
                       (Event('user', task.prompt, 0), Event('assistant', answer, 1)), True)

    def replay(task_id, actions):
        good = len(actions) == 1 and actions[0] == verified_answer(task_id)
        return Trajectory(task_id, task.family, 'fresh-verifier', f'replay-{serial}',
                          (Event('user', task.prompt, 0),
                           Event('assistant', actions[0], 1)), good)

    def receipt(t):
        actions = [e.text for e in t.events if e.role == 'assistant']
        passed = len(actions) == 1 and actions[0] == verified_answer(t.task_id)
        return Receipt(t.task_id, t.run_id, t.harness_id, t.digest(), passed)

    result = adopt_rewrite(donor, replay, receipt)
    if result.loss_mask != (False, True):
        raise RuntimeError('Unverified assistant loss mask')
    return answer


class Actor:
    """Feature-based 10-action categorical policy with persistent Adam state."""

    def __init__(self, seed: int):
        rng = np.random.default_rng(seed)
        self.weights = rng.normal(0, .002, (len(FAMILIES) * 21, 10))
        self.m = np.zeros_like(self.weights)
        self.v = np.zeros_like(self.weights)
        self.t = 0

    @staticmethod
    def indices(task: Task):
        family = FAMILIES.index(task.family)
        base = family * 21
        return np.array((base, base + 1 + task.a, base + 11 + task.b))

    def probabilities(self, task: Task):
        logits = self.weights[self.indices(task)].sum(axis=0)
        odds = np.exp(logits - logits.max())
        return odds / odds.sum()

    def update(self, task: Task, action: int, multiplier: float, lr: float):
        probs = self.probabilities(task)
        gradient = probs.copy()
        gradient[action] -= 1.
        gradient *= multiplier
        self.t += 1
        # All three active feature rows have exactly this derivative.
        for row in self.indices(task):
            self.m[row] = .9 * self.m[row] + .1 * gradient
            self.v[row] = .999 * self.v[row] + .001 * gradient**2
            mhat = self.m[row] / (1 - .9**self.t)
            vhat = self.v[row] / (1 - .999**self.t)
            self.weights[row] -= lr * mhat / (np.sqrt(vhat) + 1e-8)

    def digest(self):
        return hashlib.sha256(self.weights.tobytes()).hexdigest()

    def optimizer_digest(self):
        return hashlib.sha256(self.m.tobytes() + self.v.tobytes() +
                              str(self.t).encode()).hexdigest()

    def state(self):
        return dict(weights=self.weights.tolist(), adam_m=self.m.tolist(),
                    adam_v=self.v.tolist(), adam_step=self.t)


def measure(actor: Actor, tasks):
    start = time.perf_counter()
    ce = 0.
    top1 = top5 = 0
    for task in tasks:
        correct = int(verified_answer(task.prompt))
        probs = actor.probabilities(task)
        ranks = np.argsort(-probs, kind='stable')
        ce -= math.log(max(float(probs[correct]), 1e-300))
        top1 += int(int(ranks[0]) == correct)
        top5 += int(correct in ranks[:5])
    n = len(tasks)
    return dict(tasks=n, strict_successes=top1,
                strict_success_rate=round(top1 / n, 6),
                top1=round(top1 / n, 6), top5=round(top5 / n, 6),
                nats_per_action=round(ce / n, 6),
                bits_per_action=round(ce / (n * math.log(2)), 6),
                tasks_per_second=round(n / max(time.perf_counter()-start, 1e-9), 2))


def critic_digest(values):
    return hashlib.sha256(values.tobytes()).hexdigest()


def run(*, seed=2026, sft_updates=200, rpr_updates=100, lr=.03,
        actor_rpr_scale=.1, include_checkpoint=False):
    if not (1 <= sft_updates <= 10000 and 1 <= rpr_updates <= 10000):
        raise ValueError('Update counts must be between 1 and 10000')
    if not (0 < lr <= 1 and 0 < actor_rpr_scale <= 1):
        raise ValueError('Bad actor learning rate')
    train, dev = task_split(seed)
    actor = Actor(seed)
    critic = np.zeros(len(FAMILIES), dtype=np.float64)
    rng = random.Random(seed + 1)
    stage = Stage('BASE', actor.digest(), critic_digest(critic), actor.optimizer_digest())
    stages = [dict(stage='BASE', heldout=measure(actor, dev), train=measure(actor, train))]
    verified_replays = 0
    total_attempts = total_successes = 0
    historical_training_state = actor.optimizer_digest()
    for phase in (1, 2):
        # RSR: only externally replayed successful actions can become targets.
        old_critic = critic_digest(critic)
        for i in range(sft_updates):
            task = train[rng.randrange(len(train))]
            answer = verified_rewrite(task, serial=phase * 10000 + i)
            actor.update(task, int(answer), 1., lr)
            verified_replays += 1
        next_stage = Stage(f'RSR-{phase}', actor.digest(), critic_digest(critic),
                           actor.optimizer_digest(), verified_sft=sft_updates)
        check_transition(stage, next_stage)
        assert old_critic == critic_digest(critic)
        stage = next_stage
        stages.append(dict(stage=stage.name, heldout=measure(actor, dev),
                           train=measure(actor, train)))

        # Critic-only warmup; retain actor weights and Adam moments exactly.
        frozen_actor = actor.digest()
        frozen_optimizer = actor.optimizer_digest()
        for _ in range(4):
            for family_id in range(len(FAMILIES)):
                critic[family_id] += .1 * (1. - critic[family_id])
        assert actor.digest() == frozen_actor
        assert actor.optimizer_digest() == frozen_optimizer
        successes = 0
        calibration = Calibration({f: 82 for f in FAMILIES})
        for _ in range(rpr_updates):
            task = train[rng.randrange(len(train))]
            old_probs = actor.probabilities(task)
            chosen = rng.choices(range(10), weights=old_probs, k=1)[0]
            strict = str(chosen) == verified_answer(task.prompt)
            successes += int(strict)
            reward = spr_reward(Outcome(task.family, strict, int(strict), 1, True), calibration)
            family_id = FAMILIES.index(task.family)
            advantage = tb_gae((Action(0, float(critic[family_id]), reward, True),)).advantages[0]
            # One on-policy PPO update at ratio=1; clipped repeated updates
            # are intentionally omitted in this bounded pilot.
            actor.update(task, chosen, -advantage, lr * actor_rpr_scale)
            critic[family_id] += .1 * (reward - critic[family_id])
        total_attempts += rpr_updates
        total_successes += successes
        next_stage = Stage(f'RPR-{phase}', actor.digest(), critic_digest(critic),
                           actor.optimizer_digest(), critic_warmup=4)
        check_transition(stage, next_stage)
        stage = next_stage
        stages.append(dict(stage=stage.name, heldout=measure(actor, dev),
                           train=measure(actor, train),
                           sampled_train_successes=successes,
                           sampled_train_attempts=rpr_updates))
    report = dict(schema='terminal-rsi-standalone-v1',
                  isolation='No Poincare/FA actor, model, checkpoint, or corpus',
                  benchmark_eligible=False, terminal_bench_success=None,
                  benchmark_ineligibility='Synthetic finite-action tasks; all families seen in training',
                  seed=seed, train_tasks=len(train), heldout_tasks=len(dev),
                  train_task_ids_sha256=hashlib.sha256('\n'.join(sorted(t.task_id for t in train)).encode()).hexdigest(),
                  dev_task_ids_sha256=hashlib.sha256('\n'.join(sorted(t.task_id for t in dev)).encode()).hexdigest(),
                  verified_replays=verified_replays,
                  sampled_train_attempts=total_attempts,
                  sampled_train_strict_successes=total_successes,
                  adam_updates=actor.t, stages=stages,
                  candidate_promoted=False,
                  note='Top-1 is exact one-character task success; bits/action is not bits/byte or BPE.')
    if include_checkpoint:
        report['candidate_checkpoint'] = dict(actor=actor.state(), critic=critic.tolist(),
                                              stage=stage.name, parent_optimizer_sha256=historical_training_state,
                                              promoted=False)
    return report
