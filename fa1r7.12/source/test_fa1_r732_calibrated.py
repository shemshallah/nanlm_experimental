"""R7.32: loss-gated checkpoint selection, exact migration, causal trust."""
import copy
import math
import unittest

import torch
from torch import nn

from fa1_r730_factorized_geometry import FactorizedContextualGeometry
from fa1_r731_layered_decoder import EOW, LayeredDiamondDecoder, attach_layered
from fa1_r732_geometry_calibration import GeometricTrust, attach_calibrated
from fa1_r728_geometric_memory import (candidate_priority, passes_gate,
    layered_byte_metrics, geometry_disabled, layered_language_context_disabled)


class CalibratedDiamondTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(732)
        torch.set_num_threads(2)

    @staticmethod
    def data():
        return [torch.randn(3, 32) for _ in range(3)]

    def test_attach_is_exactly_function_preserving_and_still_causal(self):
        before = LayeredDiamondDecoder()
        before.eval()
        with torch.no_grad():
            before.word_gain.fill_(.25)
            before.byte_geometry_gain.fill_(.4)
        point, apex, query = self.data()
        labels = torch.tensor([[97, 98, EOW, -100], [99, 100, EOW, -100],
                               [120, 121, EOW, -100]])
        original_word, original_condition = before.descend(point, apex, query)
        original_logits = before.autoregressive(labels, original_condition)
        after = copy.deepcopy(before)
        after.word_geometry_router = GeometricTrust(after.width)
        after.byte_geometry_router = GeometricTrust(after.width)
        word, condition = after.descend(point, apex, query)
        logits = after.autoregressive(labels, condition)
        self.assertTrue(torch.equal(original_word, word))
        self.assertTrue(torch.equal(original_condition, condition))
        self.assertTrue(torch.equal(original_logits, logits))
        altered = labels.clone()
        altered[:, 2] = 200
        self.assertTrue(torch.equal(logits[:, :3], after.autoregressive(altered, condition)[:, :3]))

    def test_router_learns_abstention_without_bypassing_apex_or_pair(self):
        decoder = LayeredDiamondDecoder()
        decoder.word_geometry_router = GeometricTrust(decoder.width)
        decoder.byte_geometry_router = GeometricTrust(decoder.width)
        with torch.no_grad():
            decoder.word_gain.fill_(.4)
            decoder.byte_geometry_gain.fill_(.4)
        point, apex, query = self.data()
        word, condition = decoder.descend(point, apex, query)
        baseline = decoder.descend(torch.zeros_like(point), apex, query)[1]
        with torch.no_grad():
            decoder.byte_geometry_router.net[-1].bias.fill_(-9.)
            decoder.word_geometry_router.net[-1].bias.fill_(-9.)
        suppressed_word, suppressed = decoder.descend(point, apex, query)
        self.assertLess((suppressed-baseline).norm(), (condition-baseline).norm()*.001)
        self.assertLess(suppressed_word.norm(), word.norm()*.001)
        self.assertTrue(torch.equal(decoder.descend(point, torch.zeros_like(apex), query)[1],baseline))
        with torch.no_grad():
            decoder.byte_geometry_router.net[-1].bias.zero_()
            decoder.word_geometry_router.net[-1].bias.zero_()
        word, condition = decoder.descend(point, apex, query)
        loss = word.square().mean()+condition.square().mean()
        loss.backward()
        for r in (decoder.word_geometry_router,decoder.byte_geometry_router):
            g = r.net[-1].weight.grad
            self.assertIsNotNone(g)
            self.assertGreater(float(g.abs().sum()),0.)

    def test_new_group_preserves_original_optimizer_moments(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.diamond = nn.Module()
                self.diamond.d = 32
                self.diamond.geometry = FactorizedContextualGeometry()
        model = Model()
        previous = list(model.parameters())
        opt = type('Optimizer', (), {})()
        opt._name_by_id = {id(p): n for n, p in model.named_parameters()}
        opt._group_names = [{'algorithm': 'adamw', 'names': list(opt._name_by_id.values())}]
        opt._adamw = torch.optim.AdamW(previous, lr=.001)
        opt._legacy_groups = []
        opt._source_group_counts = [1]
        opt.param_groups = opt._adamw.param_groups
        previous[0].grad = torch.ones_like(previous[0])
        opt._adamw.step()
        attach_layered(model, opt)
        old_state = {id(p): copy.deepcopy(opt._adamw.state[p]) for p in previous}
        existing_param_ids = {id(p) for p in model.parameters()}
        attach_calibrated(model, opt)
        self.assertEqual(len(opt.param_groups),3)
        self.assertEqual(opt._source_group_counts,[3])
        self.assertEqual(len({id(p) for group in opt.param_groups for p in group['params']}),
                         len(list(model.parameters())))
        self.assertTrue(existing_param_ids.issubset({id(p) for p in model.parameters()}))
        for p in previous:
            for key, value in old_state[id(p)].items():
                self.assertTrue(torch.equal(value,opt._adamw.state[p][key]))
        with self.assertRaises(ValueError):
            attach_calibrated(model, opt)

    def test_geometry_ablation_keeps_language_context_active(self):
        m = nn.Module()
        m.diamond = nn.Module()
        m.diamond.geometry = FactorizedContextualGeometry()
        m.diamond.hierarchy = LayeredDiamondDecoder()
        h = m.diamond.hierarchy
        with torch.no_grad():
            h.context_gain.fill_(.27)
            h.point_gain.fill_(.2)
            h.byte_geometry_gain.fill_(.3)
        with geometry_disabled(m):
            self.assertEqual(float(h.context_gain),float(torch.tensor(.27)))
            self.assertEqual(float(h.point_gain),0.)
            self.assertEqual(float(h.byte_geometry_gain),0.)
        self.assertEqual(float(h.point_gain),float(torch.tensor(.2)))
        self.assertEqual(float(h.byte_geometry_gain),float(torch.tensor(.3)))
        with layered_language_context_disabled(m):
            self.assertEqual(float(h.context_gain),0.)
            self.assertEqual(float(h.point_gain),float(torch.tensor(.2)))
        self.assertEqual(float(h.context_gain),float(torch.tensor(.27)))

    @staticmethod
    def candidate(**overrides):
        record = dict(full_word_nll=5.3, geometry_word_gain=.002,
                      branch_geometry_gain=.003,
                      correct_book_advantage_nats_per_word=.001,
                      apex_word_gain=.001, pair_word_gain=.001,
                      book_geometry_gain={'book_bootstrap_ci95':[.0001,.004]},
                      book_correct_history={'book_bootstrap_ci95':[.0001,.003]},
                      conditional_byte_geometry_gain_nats_per_byte=.002,
                      conditional_byte_apex_gain_nats_per_byte=.003,
                      conditional_byte_pair_gain_nats_per_byte=.003,
                      conditional_byte_geometry_gain_book_bootstrap={
                          'book_bootstrap_ci95':[.0001,.005]},
                      conditional_spelling_bits_per_byte=6.7)
        record.update(overrides)
        return record

    def test_selection_never_overrides_strict_gate_with_lower_word_loss(self):
        baseline = self.candidate(full_word_nll=5.4,
                                  conditional_spelling_bits_per_byte=7.2)
        admissible = self.candidate(full_word_nll=5.3)
        negative_geometry = self.candidate(full_word_nll=5.29,
                                           geometry_word_gain=-.00001)
        low_confidence = self.candidate(full_word_nll=5.28,
            book_correct_history={'book_bootstrap_ci95':[-.001,.002]})
        self.assertTrue(passes_gate(baseline, admissible,.0001))
        self.assertFalse(passes_gate(baseline, low_confidence,.0001))
        self.assertGreater(candidate_priority(baseline, admissible,.0001),
                           candidate_priority(baseline, low_confidence,.0001))
        self.assertGreater(candidate_priority(baseline, low_confidence,.0001),
                           candidate_priority(baseline, negative_geometry,.0001))

    def test_conditional_bpb_weights_real_bytes_not_word_count(self):
        labels = torch.tensor([[97,EOW,-100,-100,-100], [98,99,100,101,EOW]])
        logits = torch.zeros(2,5,257)
        logits[0,0,97] = -10.
        logits[1,:4,98:102] = 3.
        m = layered_byte_metrics({'byte_ar_logits':logits},
                                 {'byte_target_sequence':labels})
        ce = nn.functional.cross_entropy(logits.transpose(1,2),labels,
                                         reduction='none',ignore_index=-100)
        expected = (ce[0,0]+ce[1,:4].sum())/5/math.log(2)
        self.assertAlmostEqual(m['conditional_spelling_bits_per_byte'],float(expected),places=5)
        self.assertEqual(m['conditional_spelling_bytes_per_word'],2.5)


if __name__ == '__main__':
    unittest.main()
