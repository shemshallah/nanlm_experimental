"""R7.49 exact migration, address causality, oriented controls and gradients."""
import unittest

import torch
from torch import nn

from fa1_r748_mirror_pair_values import MirrorPairValueBackoff
from fa1_r749_address_reliability import (ReliableAddressedMirrorBackoff,
                                          address_reliability_control)
from test_fa1_r748_mirror_pair_values import MirrorPairValueTests


class AddressReliabilityTests(unittest.TestCase):
    def setUp(self):
        self.fixture = MirrorPairValueTests()
        self.fixture.setUp()
        self.mod = self.fixture.module()
        self.data = self.fixture.data
        self.trace = self.fixture.trace
        self.ids = torch.tensor([[1, 2, 3, 5], [2, 3, 4, 5]])

    def append(self):
        self.mod.__class__ = ReliableAddressedMirrorBackoff
        self.mod.mirror_signed_suppression = nn.Parameter(torch.zeros(()))
        self.mod.mirror_reliability_bias = nn.Parameter(torch.zeros(()))
        self.mod.mirror_reliability_overlap = nn.Parameter(torch.zeros(()))
        self.mod.mirror_exact_address_gain = nn.Parameter(torch.zeros(()))
        self.mod.mirror_partial_address_gain = nn.Parameter(torch.zeros(()))
        self.mod.mirror_orientation_address_gain = nn.Parameter(torch.zeros(()))

    def test_exact_zero_gate_bitwise_migration(self):
        before = self.mod.pair_logits(self.data, self.trace, self.ids)[0]
        target = torch.tensor([1, 2])
        prior = self.mod(self.data, self.trace, target, self.fixture.embeddings)
        self.append()
        after = self.mod.pair_logits(self.data, self.trace, self.ids)[0]
        new = self.mod(self.data, self.trace, target, self.fixture.embeddings)
        self.assertTrue(torch.equal(before, after))
        self.assertTrue(torch.equal(prior[0], new[0]))
        self.assertTrue(torch.equal(prior[1], new[1]))

    def test_exact_pair_address_changes_only_its_candidates(self):
        self.append()
        with torch.no_grad():
            self.mod.mirror_exact_address_gain.fill_(1.25)
            self.mod.mirror_partial_address_gain.fill_(.5)
        baseline, info = self.mod.pair_logits(self.data, self.trace, self.ids)
        with address_reliability_control(type('M', (), {'word_backoff': self.mod})(),
                                         'without_new_address'):
            neutral, _ = self.mod.pair_logits(self.data, self.trace, self.ids)
        self.assertTrue(bool((baseline != neutral).any()))
        self.assertGreater(float(info['r749_exact_active_pair_fraction'].max()), 0.)
        self.assertTrue(bool(torch.isfinite(baseline).all()))

    def test_signed_suppression_reliably_reduces_inherited_signed_rail(self):
        self.append()
        with torch.no_grad():
            self.mod.mirror_signed_suppression.fill_(3.)
            self.mod.mirror_reliability_bias.fill_(10.)
        gated = self.mod.pair_logits(self.data, self.trace, self.ids)[0]
        with address_reliability_control(type('M', (), {'word_backoff': self.mod})(),
                                         'without_new_signed_gate'):
            original = self.mod.pair_logits(self.data, self.trace, self.ids)[0]
        self.mod._r748_control = 'without_signed_orientation'
        try:
            neutral = self.mod.pair_logits(self.data, self.trace, self.ids)[0]
        finally:
            self.mod._r748_control = 'full'
        self.assertLess(float((gated - neutral).abs().mean()),
                        float((original - neutral).abs().mean()))

    def test_target_independent_attention_and_counterfactual_restoration(self):
        self.append()
        with torch.no_grad():
            self.mod.mirror_partial_address_gain.fill_(.75)
            self.mod.mirror_signed_suppression.fill_(.2)
        outputs = [self.mod(self.data, self.trace, torch.full((2,), t),
                            self.fixture.embeddings)[2]['r748_pair_attention']
                   for t in range(12)]
        self.assertTrue(all(torch.equal(outputs[0], x) for x in outputs[1:]))
        before = self.mod.pair_logits(self.data, self.trace, self.ids)[0]
        with address_reliability_control(type('M', (), {'word_backoff': self.mod})()):
            self.assertEqual(float(self.mod.mirror_partial_address_gain), 0.)
            self.assertEqual(float(self.mod.mirror_signed_suppression), 0.)
        after = self.mod.pair_logits(self.data, self.trace, self.ids)[0]
        self.assertTrue(torch.equal(before, after))

    def test_direct_address_gain_receives_gradient(self):
        self.append()
        self.mod.mirror_exact_address_gain.data.fill_(.05)
        loss = self.mod.pair_logits(self.data, self.trace, self.ids)[0].square().mean()
        loss.backward()
        self.assertIsNotNone(self.mod.mirror_exact_address_gain.grad)
        self.assertGreater(float(self.mod.mirror_exact_address_gain.grad.abs()), 0.)


if __name__ == '__main__':
    unittest.main()
