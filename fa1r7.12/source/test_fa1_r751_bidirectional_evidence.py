"""FA1 R7.51: exact zero migration, signed evidence, confidence, controls."""
import unittest

import torch

from fa1_r748_mirror_pair_values import pair_indices
from fa1_r751_bidirectional_evidence import (
    BidirectionalPairEvidenceBackoff,
    bidirectional_control,
    causal_confidence_features,
)
from test_fa1_r749_address_reliability import AddressReliabilityTests


class BidirectionalEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.fixture = AddressReliabilityTests()
        self.fixture.setUp()
        self.fixture.append()
        self.module = self.fixture.mod
        self.data = self.fixture.data
        self.trace = self.fixture.trace
        self.ids = self.fixture.ids
        self.target = torch.tensor([1, 2])

    def append(self):
        from torch import nn
        m = self.module
        m.__class__ = BidirectionalPairEvidenceBackoff
        m.mirror_r751_positive = nn.Embedding(32640, 1)
        m.mirror_r751_negative = nn.Embedding(32640, 1)
        m.mirror_r751_context = nn.Linear(35, 1)
        m.mirror_r751_temperature = nn.Parameter(torch.zeros(()))
        m.mirror_r751_global_pair_bias = nn.Parameter(torch.zeros(()))
        m.mirror_r751_address_gain = nn.Parameter(torch.zeros(()))
        nn.init.zeros_(m.mirror_r751_positive.weight)
        nn.init.zeros_(m.mirror_r751_negative.weight)
        nn.init.zeros_(m.mirror_r751_context.weight)
        nn.init.zeros_(m.mirror_r751_context.bias)
        return m

    def shell(self):
        return type("BackoffHolder", (), {"word_backoff": self.module})()

    def test_zero_migration_is_bitwise_identical_to_r749(self):
        prior_logits = self.module.pair_logits(self.data, self.trace, self.ids)[0]
        old = self.module(self.data, self.trace, self.target,
                          self.fixture.fixture.embeddings)
        self.append()
        new_logits = self.module.pair_logits(self.data, self.trace, self.ids)[0]
        new = self.module(self.data, self.trace, self.target,
                          self.fixture.fixture.embeddings)
        self.assertTrue(torch.equal(prior_logits, new_logits))
        self.assertTrue(torch.equal(old[0], new[0]))
        self.assertTrue(torch.equal(old[1], new[1]))

    def test_signed_bank_can_reward_and_penalize_exact_pairs(self):
        m = self.append()
        with torch.no_grad():
            m.mirror_r751_address_gain.fill_(1.)
            m.mirror_r751_positive.weight.fill_(2.)
        reward, stats = m.pair_logits(self.data, self.trace, self.ids)
        with bidirectional_control(self.shell(), "calibration_only"):
            parent, _ = m.pair_logits(self.data, self.trace, self.ids)
        increase = reward - parent
        self.assertGreater(float(increase.max()), 0.)
        self.assertTrue(bool((increase >= 0).all()))
        self.assertGreater(float(stats["r751_exact_hit_fraction"].max()), 0.)
        with torch.no_grad():
            m.mirror_r751_positive.weight.zero_()
            m.mirror_r751_negative.weight.fill_(2.)
        penalize, _ = m.pair_logits(self.data, self.trace, self.ids)
        self.assertLess(float((penalize - parent).min()), 0.)
        self.assertTrue(bool((penalize <= parent).all()))

    def test_temperature_only_control_is_not_address_evidence(self):
        m = self.append()
        with torch.no_grad():
            m.mirror_r751_temperature.fill_(.7)
        full, _ = m.pair_logits(self.data, self.trace, self.ids)
        with bidirectional_control(self.shell(), "calibration_only"):
            calibrated, _ = m.pair_logits(self.data, self.trace, self.ids)
        with bidirectional_control(self.shell(), "disabled"):
            parent, _ = m.pair_logits(self.data, self.trace, self.ids)
        self.assertTrue(torch.equal(full, calibrated))
        self.assertGreater(float((calibrated - parent).abs().sum()), 0.)

    def test_row_permutation_changes_values_not_address_membership(self):
        m = self.append()
        with torch.no_grad():
            m.mirror_r751_address_gain.fill_(1.)
            m.mirror_r751_positive.weight.normal_(std=.2)
        original = m.mirror_r751_positive.weight.detach().clone()
        full, info = m.pair_logits(self.data, self.trace, self.ids)
        with bidirectional_control(self.shell(), "permute_pair_values"):
            permuted, shifted = m.pair_logits(self.data, self.trace, self.ids)
        self.assertGreater(float((full - permuted).abs().sum()), 0.)
        self.assertTrue(torch.equal(info["r751_exact_hit_fraction"],
                                    shifted["r751_exact_hit_fraction"]))
        self.assertTrue(torch.equal(original, m.mirror_r751_positive.weight))

    def test_causal_features_and_copy_attention_ignore_target_labels(self):
        m = self.append()
        with torch.no_grad():
            m.mirror_r751_address_gain.fill_(1.)
            m.mirror_r751_positive.weight.fill_(.1)
            m.mirror_r751_context.weight.fill_(.03)
        features = causal_confidence_features(self.data, self.trace, self.ids)
        self.assertEqual(tuple(features.shape), (len(self.ids), 35))
        attentions = [m(self.data, self.trace, torch.full_like(self.target, i),
                        self.fixture.fixture.embeddings)[2]["r748_pair_attention"]
                      for i in range(12)]
        self.assertTrue(all(torch.equal(attentions[0], a) for a in attentions[1:]))
        changed = dict(self.trace,
                       descended_point_summary=self.trace[
                           "descended_point_summary"] * 3.)
        original = m.pair_logits(self.data, self.trace, self.ids)[0]
        altered = m.pair_logits(self.data, changed, self.ids)[0]
        self.assertGreater(float((altered - original).abs().sum()), 0.)

    def test_trainable_signed_address_banks_get_gradients(self):
        m = self.append()
        with torch.no_grad():
            m.mirror_r751_address_gain.fill_(1.)
        logits = m.pair_logits(self.data, self.trace, self.ids)[0]
        logits.square().sum().backward()
        self.assertGreater(float(m.mirror_r751_positive.weight.grad.abs().sum()), 0.)
        self.assertGreater(float(m.mirror_r751_negative.weight.grad.abs().sum()), 0.)
        self.assertIsNotNone(m.mirror_r751_temperature.grad)

    def test_invalid_controls_rejected(self):
        self.append()
        with self.assertRaises(ValueError):
            with bidirectional_control(self.shell(), "fake"):
                pass


if __name__ == "__main__":
    unittest.main()
