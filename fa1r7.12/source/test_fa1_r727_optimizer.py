#!/usr/bin/env python3
"""R7.27: lossless optimizer consolidation and one-step parity regression."""
import copy
import unittest

import torch
from torch import nn

from r73_joint import LegacyAccumulator
from fa1_r727_optimizer import UnifiedR7Optimizer, _deep_equal


class ToyR7(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = nn.Linear(4, 3)
        self.adapter = nn.Linear(3, 4)
        self.mirror = nn.Linear(4, 4)
        self.outer = nn.Linear(4, 2)
        self.near = nn.Linear(2, 5)
        self.apex = nn.Linear(5, 2)


def legacy_optimizers(model):
    legacy = LegacyAccumulator(model.trunk, None, lr=1.5e-3, init_from_legacy=False)
    for p in model.trunk.parameters():
        legacy.state[p]['sum'] = torch.full_like(p, 0.125)
    adapter = torch.optim.AdamW(model.adapter.parameters(), lr=7e-4)
    mirror = torch.optim.AdamW(model.mirror.parameters(), lr=8e-4, weight_decay=1e-3)
    outer = torch.optim.AdamW(model.outer.parameters(), lr=9e-4)
    last = torch.optim.AdamW([
        {'params': list(model.near.parameters()), 'lr': 1e-3},
        {'params': list(model.apex.parameters()), 'lr': 2e-3, 'weight_decay': 0}], lr=4e-3)
    return [legacy, adapter, mirror, outer, last]


def gradients(model, step):
    for k, p in enumerate(model.parameters()):
        p.grad = torch.full_like(p, 0.0125 + k * 0.002 + step * 0.0001)


def check_equivalent(left, old, right, unified):
    for (a, p), (b, q) in zip(left.named_parameters(), right.named_parameters()):
        assert a == b and torch.equal(p, q), ('parameter mismatch', a)
    for p, q in zip(left.trunk.parameters(), right.trunk.parameters()):
        assert _deep_equal(old[0].state[p], unified._legacy_state[q])
    for opt in old[1:]:
        for a, p in left.named_parameters():
            if p in opt.state:
                q = dict(right.named_parameters())[a]
                assert _deep_equal(opt.state[p], unified._adamw.state[q]), ('moment mismatch', a)


class TestR727(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_cold_and_warm_moments_step_exact(self):
        torch.manual_seed(727)
        old_model = ToyR7()
        new_model = copy.deepcopy(old_model)
        old = legacy_optimizers(old_model)
        past = legacy_optimizers(new_model)
        for step in range(3):
            gradients(old_model, step)
            gradients(new_model, step)
            for o in old: o.step()
            for o in past: o.step()
        merged = UnifiedR7Optimizer(new_model, past)
        check_equivalent(old_model, old, new_model, merged)
        for step in range(3, 8):
            gradients(old_model, step)
            gradients(new_model, step)
            for o in old: o.step()
            merged.step()
            check_equivalent(old_model, old, new_model, merged)

        # Restoring a complete unified state retains the same update trajectory.
        data = copy.deepcopy(merged.state_dict())
        recovered_model = copy.deepcopy(new_model)
        recovered = UnifiedR7Optimizer(recovered_model, legacy_optimizers(recovered_model))
        recovered.load_state_dict(data)
        gradients(new_model, 8)
        gradients(recovered_model, 8)
        merged.step()
        recovered.step()
        for a, b in zip(new_model.parameters(), recovered_model.parameters()):
            self.assertTrue(torch.equal(a, b))
        self.assertTrue(_deep_equal(merged.state_dict(), recovered.state_dict()))

    def test_no_duplicate_or_unknown_parameters(self):
        model = ToyR7()
        inherited = legacy_optimizers(model)
        inherited[4].param_groups[-1]['params'].append(next(model.near.parameters()))
        with self.assertRaises(ValueError):
            UnifiedR7Optimizer(model, inherited)


if __name__ == '__main__':
    unittest.main()
