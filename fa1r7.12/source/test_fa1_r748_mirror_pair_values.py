"""Exact C(256,2), causal probability, sign, controls and gradient regressions."""
import unittest

import torch
from torch import nn

from fa1_r747_head_address_copy import HeadAddressWordBackoff
from fa1_r748_mirror_pair_values import (PAIR_COUNT, VALUE_DIM,
    MirrorPairValueBackoff, pair_indices, pair_value_control)


class MirrorPairValueTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(748)
        self.heads = torch.tensor([[0, 1], [2, 5], [5, 2], [3, 7],
                                   [6, 9], [1, 4], [0, 2], [2, 8],
                                   [0, 0], [3, 5], [5, 3], [7, 6]])
        self.active = torch.tensor([[0, 2, 5, 1, 3, 4, 6, 7],
                                    [0, 5, 2, 1, 3, 4, 6, 7]])
        self.embeddings = torch.randn(12, 96)
        self.data = {'word_ids': torch.tensor([[1, 2, 3, 5], [2, 3, 4, 5]]),
                     'diamond_backoff_recent_mask': torch.ones(2, 4, dtype=torch.bool),
                     'vertex_ids': self.active}
        self.trace = {'triangle_attention': torch.full((2, 56), 1 / 56),
                      'trust_logit': torch.zeros(2), 'apex_std': torch.tensor(0.),
                      'descended_point_summary': torch.randn(2, 32)}

    def module(self, opened=True):
        mod = MirrorPairValueBackoff()
        mod.geometric_query = nn.Linear(32, 96, bias=False)
        mod.geometric_gain = nn.Parameter(torch.tensor(.5))
        mod.address_coefficients = nn.Parameter(torch.zeros(5))
        mod.address_gain = nn.Parameter(torch.tensor(.4))
        mod.address_copy_gain = nn.Parameter(torch.tensor(.3))
        mod.address_route_bias = nn.Parameter(torch.tensor(-1.5))
        mod.address_route_weight = nn.Parameter(torch.zeros(()))
        mod.register_buffer('word_head_lookup', self.heads)
        mod.mirror_pair_values = nn.Embedding(PAIR_COUNT, VALUE_DIM)
        mod.mirror_head_values = nn.Embedding(256, VALUE_DIM)
        mod.mirror_pair_query = nn.Linear(32, VALUE_DIM, bias=False)
        mod.mirror_signed_weight = nn.Parameter(torch.tensor(1.))
        mod.mirror_pair_gain = nn.Parameter(torch.tensor(.65 if opened else 0.))
        mod.mirror_pair_copy_gain = nn.Parameter(torch.tensor(.5 if opened else 0.))
        return mod

    def test_all_32640_pair_addresses_unique_and_valid(self):
        p = torch.triu_indices(256, 256, 1)
        ids, valid = pair_indices(p[0], p[1])
        self.assertTrue(bool(valid.all()))
        self.assertTrue(torch.equal(ids.sort().values, torch.arange(PAIR_COUNT)))
        self.assertEqual(pair_indices(torch.tensor([5]), torch.tensor([5]))[1].item(), False)

    def test_closed_pair_gates_reproduce_r747_bitwise(self):
        mod = self.module(opened=False)
        target = torch.tensor([2, 3])
        legacy = HeadAddressWordBackoff.forward(mod, self.data, self.trace, target,
                                                self.embeddings)
        candidate = mod(self.data, self.trace, target, self.embeddings)
        self.assertTrue(torch.equal(legacy[0], candidate[0]))
        self.assertTrue(torch.equal(legacy[1], candidate[1]))

    def test_target_independence_and_normalized_copy_distribution(self):
        mod = self.module()
        alpha_values, q_values, attn = [], [], []
        for y in range(12):
            alpha, q, extras = mod(self.data, self.trace, torch.full((2,), y),
                                   self.embeddings)
            alpha_values.append(alpha)
            q_values.append(q)
            attn.append(extras['r748_pair_attention'])
        for a, b in zip(alpha_values, alpha_values[1:]):
            self.assertTrue(torch.equal(a, b))
        for a, b in zip(attn, attn[1:]):
            self.assertTrue(torch.equal(a, b))
        self.assertTrue(torch.allclose(torch.stack(q_values, -1).sum(-1),
                                       torch.ones(2), atol=1.e-6))
        self.assertTrue(bool((alpha_values[0] >= 0).all()))
        self.assertTrue(bool((alpha_values[0] < 1).all()))

    def test_address_only_reordering_and_signed_controls(self):
        mod = self.module()
        target = torch.tensor([1, 1])
        _, q, _ = mod(self.data, self.trace, target, self.embeddings)
        changed = dict(self.data, vertex_ids=self.data['vertex_ids'].roll(1, dims=1))
        _, reord, _ = mod(changed, self.trace, target, self.embeddings)
        unrelated = dict(self.data, vertex_ids=self.data['vertex_ids'].flip(0))
        _, foreign, _ = mod(unrelated, self.trace, target, self.embeddings)
        with pair_value_control(type('M', (), {'word_backoff': mod})(),
                                'without_signed_orientation'):
            _, neutral, _ = mod(self.data, self.trace, target, self.embeddings)
        with pair_value_control(type('M', (), {'word_backoff': mod})(),
                                'scramble_pair_values'):
            _, permuted, _ = mod(self.data, self.trace, target, self.embeddings)
        for other in (reord, foreign, neutral, permuted):
            self.assertGreater(float((q - other).abs().sum()), 1.e-7)

    def test_pair_memory_is_trainable_with_no_inherited_gradient(self):
        mod = self.module()
        alpha, q, _ = mod(self.data, self.trace, torch.tensor([2, 3]), self.embeddings)
        loss = -(alpha * q + (1 - alpha) * .01).log().mean()
        loss.backward()
        for p in (mod.mirror_pair_values.weight, mod.mirror_head_values.weight,
                  mod.mirror_pair_query.weight):
            self.assertIsNotNone(p.grad)
            self.assertTrue(bool(torch.isfinite(p.grad).all()))
            self.assertGreater(float(p.grad.abs().sum()), 0.)

    def test_control_restores_original_parameters(self):
        mod = self.module()
        former = (mod.mirror_pair_gain.detach().clone(),
                  mod.mirror_pair_copy_gain.detach().clone())
        with pair_value_control(type('M', (), {'word_backoff': mod})()):
            self.assertEqual(mod.mirror_pair_gain.item(), 0.)
            self.assertEqual(mod.mirror_pair_copy_gain.item(), 0.)
        self.assertTrue(torch.equal(mod.mirror_pair_gain, former[0]))
        self.assertTrue(torch.equal(mod.mirror_pair_copy_gain, former[1]))


if __name__ == '__main__':
    unittest.main()
