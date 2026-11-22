"""R38 Adam continuation: archive all inherited groups without altering one bit.

Exact R37 checkpoint availability is a strict prerequisite for R37 continuation.
A separate 32-group R32 staging path is permissible for smoke tests, but such
runs must never be described as R37 descendants.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
from .episodic import GROUPS

R37_ACCEPTED_SHA256 = 'cd7a6fa9a4ba4368b8a423ac860038dd30062eba40bcbbf2b451c8ccbd520cc4'


class ContinuationAdam:
    def __init__(self, checkpoint: Path, expected_sha256: str,
                 expected_groups: int = 42):
        checkpoint = Path(checkpoint)
        actual = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if actual != expected_sha256:
            raise ValueError(f'checkpoint SHA mismatch: {actual}')
        with np.load(checkpoint, allow_pickle=False) as z:
            self.parent = {key: z[key].copy() for key in z.files}
        for field in ('theta', 'adam_m', 'adam_v', 'adam_steps'):
            if field not in self.parent or self.parent[field].shape != (expected_groups,):
                raise ValueError(f'incomplete inherited {field}')
        self.parent_sha = actual
        self.groups = expected_groups
        self.theta = np.zeros(GROUPS, dtype=np.float64)
        self.m = np.zeros(GROUPS, dtype=np.float64)
        self.v = np.zeros(GROUPS, dtype=np.float64)
        self.steps = 0

    def update(self, gradient, lr=0.003, weight_decay=1e-5):
        grad = np.asarray(gradient, dtype=np.float64)
        if grad.shape != (GROUPS,) or not np.isfinite(grad).all():
            raise ValueError('invalid gradient')
        if not 0 < lr < 0.1:
            raise ValueError('invalid learning rate')
        g = np.clip(grad + weight_decay*self.theta, -3., 3.)
        self.steps += 1
        self.m = 0.9*self.m + 0.1*g
        self.v = 0.999*self.v + 0.001*g*g
        self.theta -= lr * (self.m/(1.-0.9**self.steps))/(np.sqrt(self.v/(1.-0.999**self.steps))+1e-8)
        np.clip(self.theta, -2.0, 2.0, out=self.theta)

    def save(self, path, *, fit_indices, gate_indices, dev_indices):
        data = dict(self.parent)
        for key, extra in (('theta', self.theta), ('adam_m', self.m), ('adam_v', self.v)):
            data[key] = np.concatenate((self.parent[key], extra))
        data['adam_steps'] = np.concatenate((
            self.parent['adam_steps'], np.full(GROUPS, self.steps, dtype=np.int64)))
        data['r38_parent_sha256'] = np.array(self.parent_sha)
        data['r38_parent_groups'] = np.array(self.groups)
        for key, val in (('fit_indices', fit_indices), ('gate_indices', gate_indices),
                         ('dev_indices', dev_indices)):
            data['r38_'+key] = np.asarray(val, dtype=np.int64)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **data)
        with np.load(path, allow_pickle=False) as saved:
            for key in ('theta', 'adam_m', 'adam_v', 'adam_steps'):
                if not np.array_equal(saved[key][:self.groups], self.parent[key]):
                    raise AssertionError('inherited optimizer mutation detected')
            if not np.array_equal(saved['theta'][self.groups:], self.theta):
                raise AssertionError('R38 continuation lost')
        return path