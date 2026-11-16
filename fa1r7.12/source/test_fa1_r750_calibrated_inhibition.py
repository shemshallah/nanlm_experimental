"""R7.50 zero-migration, exact address, stored values and causal controls."""
import unittest

import torch
from torch import nn

from fa1_r750_calibrated_inhibition import (CalibratedPairInhibitionBackoff,
                                             inhibition_control)
from test_fa1_r749_address_reliability import AddressReliabilityTests


class InhibitionTests(unittest.TestCase):
    def setUp(self):
        self.fixture = AddressReliabilityTests()
        self.fixture.setUp()
        self.fixture.append()
        self.mod = self.fixture.mod
        self.data = self.fixture.data
        self.trace = self.fixture.trace
        self.ids = self.fixture.ids

    def append(self):
        m = self.mod
        m.__class__ = CalibratedPairInhibitionBackoff
        m.mirror_inhibition_table = nn.Embedding(32640, 1)
        nn.init.zeros_(m.mirror_inhibition_table.weight)
        m.mirror_inhibition_context = nn.Linear(32, 1)
        nn.init.zeros_(m.mirror_inhibition_context.weight)
        nn.init.zeros_(m.mirror_inhibition_context.bias)
        m.mirror_inhibition_bias = nn.Parameter(torch.tensor(-1.))
        m.mirror_inhibition_gain = nn.Parameter(torch.zeros(()))

    def test_zero_gain_preserves_all_r749_scores_bitwise(self):
        prior = self.mod.pair_logits(self.data, self.trace, self.ids)[0]
        ys = torch.tensor([1, 2])
        old = self.mod(self.data, self.trace, ys, self.fixture.fixture.embeddings)
        self.append()
        after = self.mod.pair_logits(self.data, self.trace, self.ids)[0]
        now = self.mod(self.data, self.trace, ys, self.fixture.fixture.embeddings)
        self.assertTrue(torch.equal(prior, after))
        self.assertTrue(torch.equal(old[0], now[0]))
        self.assertTrue(torch.equal(old[1], now[1]))

    def test_positive_penalties_only_on_exact_valid_pairs(self):
        self.append()
        with torch.no_grad():
            self.mod.mirror_inhibition_gain.fill_(1.)
        full, telemetry = self.mod.pair_logits(self.data, self.trace, self.ids)
        model = type('M', (), {'word_backoff':self.mod})()
        with inhibition_control(model):
            baseline, _ = self.mod.pair_logits(self.data, self.trace, self.ids)
        difference = baseline - full
        self.assertTrue(bool((difference >= 0).all()))
        self.assertGreater(float(difference.max()), 0.)
        self.assertGreater(float(telemetry['r750_exact_hit_fraction'].max()), 0.)

    def test_learned_values_remap_and_restore_without_param_mutation(self):
        self.append()
        with torch.no_grad():
            self.mod.mirror_inhibition_gain.fill_(1.)
            self.mod.mirror_inhibition_table.weight.normal_(std=.2)
        original = self.mod.mirror_inhibition_table.weight.detach().clone()
        full = self.mod.pair_logits(self.data, self.trace, self.ids)[0]
        model = type('M', (), {'word_backoff':self.mod})()
        with inhibition_control(model, 'permute_inhibitory_values'):
            different = self.mod.pair_logits(self.data, self.trace, self.ids)[0]
        self.assertGreater(float((full - different).abs().sum()), 0.)
        self.assertTrue(torch.equal(original, self.mod.mirror_inhibition_table.weight))

    def test_target_independent_attention_and_causal_context(self):
        self.append()
        with torch.no_grad():
            self.mod.mirror_inhibition_gain.fill_(.8)
        attentions = [self.mod(self.data, self.trace, torch.full((2,), t),
                               self.fixture.fixture.embeddings)[2]['r748_pair_attention']
                      for t in range(12)]
        self.assertTrue(all(torch.equal(attentions[0], attn) for attn in attentions[1:]))
        with torch.no_grad():
            self.mod.mirror_inhibition_context.weight.fill_(.03)
        shifted = dict(self.trace, descended_point_summary=self.trace['descended_point_summary']*3.)
        orig = self.mod.pair_logits(self.data, self.trace, self.ids)[0]
        other = self.mod.pair_logits(self.data, shifted, self.ids)[0]
        self.assertGreater(float((orig-other).abs().sum()), 0.)

    def test_trainable_inhibition_table_gets_gradient(self):
        self.append()
        with torch.no_grad():
            self.mod.mirror_inhibition_gain.fill_(.8)
        logits = self.mod.pair_logits(self.data, self.trace, self.ids)[0]
        logits.square().sum().backward()
        self.assertGreater(float(self.mod.mirror_inhibition_table.weight.grad.abs().sum()), 0.)


if __name__=='__main__':
    unittest.main()
