"""Causal, within-window n-gram recall for the inherited v102 byte specialist.

This is a candidate *component* of the existing Terminal-RSI model, not a
replacement model. The three new Adam parameters start at zero influence.
All earlier weights and Adam moments are carried in one flat checkpoint.
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
import struct

import numpy as np

from .contextual import previous_byte_classes
from .residual import load_scored
from .trustgate import TrustGate, normalize


def causal_counts(work: Path, split: str):
    """Use only transitions with positions strictly before each predicted byte.

    History is reset for each window/book. We use the longest populated suffix
    from the preceding 1, 2 or 3 bytes, with a zero expert if none exists.
    This code never reads a target before producing that position's features.
    """
    payload = (work / f'win_{split}.bin').read_bytes()
    nwin, skip = struct.unpack_from('<II', payload)
    offset = 8
    all_counts = []
    all_orders = []
    for _ in range(nwin):
        width = struct.unpack_from('<I', payload, offset)[0]
        offset += 4
        window = payload[offset:offset + width]
        offset += width
        if width <= skip or len(window) != width:
            raise ValueError('Malformed causal window')
        memories = [None] + [defaultdict(Counter) for _ in range(3)]
        for t, observed in enumerate(window):
            if t >= skip:
                selected = Counter()
                order = 0
                for k in (3, 2, 1):
                    if t >= k:
                        key = window[t - k:t]
                        prior = memories[k].get(key)
                        if prior:
                            selected = prior.copy()
                            order = k
                            break
                row = np.zeros(256, np.uint16)
                for byte, occurrences in selected.items():
                    if occurrences > np.iinfo(np.uint16).max:
                        raise ValueError('Window count overflow')
                    row[byte] = occurrences
                all_counts.append(row)
                all_orders.append(order)
            # The current target first becomes historical data *after* scoring.
            for k in (1, 2, 3):
                if t >= k:
                    memories[k][window[t - k:t]][observed] += 1
    if offset != len(payload):
        raise ValueError('Trailing causal window data')
    return np.asarray(all_counts, np.uint16).reshape(-1, 256), np.asarray(all_orders, np.uint8)


def _same_top5(a, b):
    return ((a.argmax(axis=1) == b.argmax(axis=1)) &
            np.all(np.sort(np.argpartition(a, -5, axis=1)[:, -5:], axis=1) ==
                   np.sort(np.argpartition(b, -5, axis=1)[:, -5:], axis=1), axis=1))


class CausalEcho:
    def __init__(self, parent: TrustGate):
        self.parent = parent
        self.g = np.zeros(3, np.float64)
        self.m = np.zeros(3, np.float64)
        self.v = np.zeros(3, np.float64)
        self.step = 0

    def _base(self, probs, classes):
        residual, trust = self.parent.distributions(probs, classes)
        accept = _same_top5(residual, trust)
        inherited = np.where(accept[:, None], trust, residual)
        return residual, inherited

    @staticmethod
    def _signal(inherited, counts):
        if inherited.shape != counts.shape or inherited.shape[1] != 256:
            raise ValueError('Echo counts must match 256-way probabilities')
        # The log Bayes factor is zero without observations and strictly
        # target-free. Strength is learned; pseudocount eight limits spikes.
        return np.log1p(counts / (8. * np.maximum(inherited, 1e-20)))

    def distributions(self, probs, classes, counts, orders, *, guard=True):
        residual, inherited = self._base(probs, classes)
        if len(orders) != len(probs) or np.any(orders > 3):
            raise ValueError('Invalid preceding-context orders')
        signal = self._signal(inherited, counts)
        strengths = np.zeros(len(orders), np.float64)
        active = orders > 0
        strengths[active] = np.tanh(self.g[orders[active] - 1])
        proposed = normalize(np.log(np.maximum(inherited, 1e-20)) +
                             strengths[:, None] * signal)
        if not guard:
            return residual, inherited, proposed
        accepted = _same_top5(residual, proposed)
        result = np.where(accepted[:, None], proposed, residual)
        if not np.all(_same_top5(residual, result)):
            raise AssertionError('Parent top-k preservation failed')
        return residual, inherited, result

    def score(self, probs, targets, per, classes, counts, orders, *, guard=True):
        residual, inherited, final = self.distributions(
            probs, classes, counts, orders, guard=guard)
        loss = -np.log(np.maximum(final[np.arange(len(targets)), targets], 1e-30))
        top5 = np.argpartition(final, -5, axis=1)[:, -5:]
        return {'bpc': float(np.mean(loss) / math.log(2)),
                'top1': float(np.mean(final.argmax(axis=1) == targets)),
                'top5': float(np.mean(np.any(top5 == targets[:, None], axis=1))),
                'per_book_bpc': (loss.reshape(-1, per).mean(axis=1) / math.log(2)).tolist(),
                'fraction_echo_recalled': float(np.mean(orders > 0)),
                'echo_adam_step': self.step}

    def train(self, probs, targets, classes, counts, orders, *, steps, lr, seed=4150):
        """Train only the new three-parameter Adam group; parent is immutable."""
        rng = np.random.default_rng(seed + self.step)
        _, inherited = self._base(probs, classes)
        signal = self._signal(inherited, counts)
        logits = np.log(np.maximum(inherited, 1e-20))
        for _ in range(steps):
            ix = rng.integers(0, len(targets), size=min(512, len(targets)))
            grades = orders[ix]
            strength = np.zeros(len(ix), np.float64)
            active = grades > 0
            strength[active] = np.tanh(self.g[grades[active] - 1])
            predicted = normalize(logits[ix] + strength[:, None] * signal[ix])
            d = (predicted * signal[ix]).sum(axis=1) - signal[ix, targets[ix]]
            grad = .004 * self.g.copy()
            for k in (1, 2, 3):
                mask = grades == k
                grad[k - 1] += np.sum(d[mask]) * (1 - np.tanh(self.g[k - 1]) ** 2) / len(ix)
            self.step += 1
            self.m = .9 * self.m + .1 * grad
            self.v = .999 * self.v + .001 * grad * grad
            self.g -= lr * (self.m / (1 - .9 ** self.step)) / (
                np.sqrt(self.v / (1 - .999 ** self.step)) + 1e-8)

    def checkpoint(self, path: Path):
        # Construct a *flat* npz: all inherited model/optimizer arrays, then
        # the additive echo parameter group. No model or Adam reset occurs.
        temporary = io.BytesIO()
        self.parent.checkpoint(temporary)
        temporary.seek(0)
        with np.load(temporary, allow_pickle=False) as data:
            fields = {name: data[name].copy() for name in data.files}
        fields.update(echo_g=self.g, echo_m=self.m, echo_v=self.v,
                      echo_step=np.array(self.step))
        np.savez_compressed(path, **fields)

    @classmethod
    def from_checkpoint(cls, path: Path):
        with np.load(path, allow_pickle=False) as data:
            required = {'echo_g', 'echo_m', 'echo_v', 'echo_step'}
            if not required.issubset(data.files):
                raise ValueError('Echo optimizer state missing')
            prior = {name: data[name].copy() for name in data.files if name not in required}
            extra = {name: data[name].copy() for name in required}
        buf = io.BytesIO()
        np.savez_compressed(buf, **prior)
        buf.seek(0)
        out = cls(TrustGate.from_checkpoint(buf))
        for name, field in (('echo_g', 'g'), ('echo_m', 'm'), ('echo_v', 'v')):
            if extra[name].shape != (3,) or not np.isfinite(extra[name]).all():
                raise ValueError('Invalid echo optimizer field ' + name)
            setattr(out, field, extra[name].astype(np.float64))
        out.step = int(extra['echo_step'])
        if out.step < 0:
            raise ValueError('Invalid echo step')
        return out


def _data(work, split):
    probs, targets, per = load_scored(work, split)
    classes = previous_byte_classes(work, split)
    counts, orders = causal_counts(work, split)
    if len(orders) != len(targets):
        raise ValueError('Causal recall does not align with evaluated positions')
    return probs, targets, per, classes, counts, orders


def train_stages(work: Path):
    path = work / 'trust_candidate.npz'
    parent = TrustGate.from_checkpoint(path)
    cal = _data(work, 'calibration')
    dev = _data(work, 'dev3')
    echo = CausalEcho(parent)
    baseline = echo.score(*dev)
    echo.checkpoint(work / 'echo_candidate.npz')
    stages = []
    best = baseline['bpc']
    selected_step = 0
    for updates, lr in ((128, .01), (128, .004), (128, .002)):
        echo.train(cal[0], cal[1], cal[3], cal[4], cal[5], steps=updates, lr=lr)
        scored = echo.score(*dev)
        stages.append({'step': echo.step, 'lr': lr,
                       'dev3': {k: v for k, v in scored.items() if k != 'per_book_bpc'}})
        if scored['bpc'] < best - .0002:
            best = scored['bpc']
            selected_step = echo.step
            echo.checkpoint(work / 'echo_candidate.npz')
    report = {'schema': 'terminal-rsi-v102-causal-echo-trial-v1',
              'parent_checkpoint_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'parent_residual_adam_step': parent.candidate.t,
              'parent_context_adam_step': parent.candidate.ct,
              'parent_trust_adam_step': parent.step,
              'baseline_dev3': {k: v for k, v in baseline.items() if k != 'per_book_bpc'},
              'stages': stages, 'selected_step': selected_step,
              'selected_dev3_bpc': best, 'candidate_promoted': False,
              'caveat': 'DEV3 already used for earlier selection; independent book gate pending'}
    (work / 'echo_search.json').write_text(json.dumps(report, sort_keys=True, indent=2) + '\n')
    return report


def evaluate_saved(work: Path, split: str):
    if split not in ('dev4', 'holdout4'):
        raise ValueError('Independent evaluation restricted to reserved splits')
    path = work / 'echo_candidate.npz'
    echo = CausalEcho.from_checkpoint(path)
    data = _data(work, split)
    initial = CausalEcho(echo.parent).score(*data)
    proposed = echo.score(*data)
    nbooks = len(data[1]) // data[2]
    book_diff = np.asarray(initial['per_book_bpc']) - np.asarray(proposed['per_book_bpc'])
    rng = random.Random(7000 + len(data[1]))
    ci = sorted(np.mean(rng.choices(book_diff.tolist(), k=nbooks)) for _ in range(2000))
    if (initial['top1'], initial['top5']) != (proposed['top1'], proposed['top5']):
        raise AssertionError('Top-k safety invariant violated')
    report = {'schema': 'terminal-rsi-v102-causal-echo-frozen-evaluation-v1',
              'split': split, 'books': nbooks, 'positions': len(data[1]),
              'training_updates_during_evaluation': 0,
              'checkpoint_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'inherited_trust': initial, 'causal_echo': proposed,
              'bpc_gain_ci95': [float(ci[50]), float(ci[1949])],
              'books_improved': int(np.sum(book_diff > 0)),
              'candidate_promoted': False}
    (work / f'echo_{split}_search.json').write_text(json.dumps(report, sort_keys=True, indent=2) + '\n')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('train', 'evaluate'))
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--split', choices=('dev4', 'holdout4'))
    a = p.parse_args()
    if a.command == 'train':
        result = train_stages(a.work)
    else:
        if a.split is None:
            p.error('--split required for evaluation')
        result = evaluate_saved(a.work, a.split)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
