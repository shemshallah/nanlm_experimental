"""FA1 in-place apex feedback: prediction, gradient, and optimizer parity."""
import copy
import unittest
from types import SimpleNamespace

import torch
from torch import nn

from fa1_r730_factorized_geometry import FactorizedContextualGeometry
from fa1_r731_layered_decoder import (CausalApexFeedback, EOW,
                                      LayeredDiamondDecoder)
from fa1_r732_geometry_calibration import (GeometricTrust,
                                             attach_apex_feedback)
from fa1_r728_geometric_memory import (apex_feedback_disabled,
                                        geometry_disabled, passes_gate)
from fa1_r724_geometry import GeometricDiamond


class FA1ApexFeedbackTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(735)
        torch.set_num_threads(2)

    @staticmethod
    def calibrated_decoder():
        h = LayeredDiamondDecoder()
        h.word_geometry_router = GeometricTrust(h.width)
        h.byte_geometry_router = GeometricTrust(h.width)
        with torch.no_grad():
            h.word_gain.fill_(.3)
            h.byte_geometry_gain.fill_(.4)
        return h

    def test_causal_feedback_is_exactly_zero_at_migration(self):
        before = self.calibrated_decoder().eval()
        point, apex, query = [torch.randn(3, 32) for _ in range(3)]
        labels = torch.tensor([[97, 98, EOW], [99, 100, EOW], [120, 121, EOW]])
        old_word, old_byte = before.descend(point, apex, query)
        old_logits = before.autoregressive(labels, old_byte)
        after = copy.deepcopy(before)
        after.apex_feedback = CausalApexFeedback(after.width)
        after.apex_feedback_gain = nn.Parameter(torch.zeros(()))
        word, byte = after.descend(point, apex, query)
        self.assertTrue(torch.equal(old_word, word))
        self.assertTrue(torch.equal(old_byte, byte))
        self.assertTrue(torch.equal(old_logits, after.autoregressive(labels, byte)))
        later_label = labels.clone()
        later_label[:, 2] = 111
        self.assertTrue(torch.equal(old_logits[:, :3],
                                    after.autoregressive(later_label, byte)[:, :3]))
        # Crucially, zero contribution does not mean a dead learning path.
        (word.square().mean() + byte.square().mean()).backward()
        self.assertIsNotNone(after.apex_feedback_gain.grad)
        self.assertGreater(float(after.apex_feedback_gain.grad.abs()), 0.)

    def test_active_feedback_needs_both_apex_and_descended_point(self):
        h = self.calibrated_decoder()
        old = copy.deepcopy(h)
        h.apex_feedback = CausalApexFeedback(h.width)
        h.apex_feedback_gain = nn.Parameter(torch.tensor(.5))
        point, apex, query = [torch.randn(3, 32) for _ in range(3)]
        word, byte = h.descend(point, apex, query)
        orig_word, orig_byte = old.descend(point, apex, query)
        self.assertGreater(float((word-orig_word).abs().sum()), 0.)
        self.assertGreater(float((byte-orig_byte).abs().sum()), 0.)
        (word.square().mean()+byte.square().mean()).backward()
        attn_grad = h.apex_feedback.cross.in_proj_weight.grad
        self.assertIsNotNone(attn_grad)
        self.assertGreater(float(attn_grad.abs().sum()), 0.)
        for p, a in ((torch.zeros_like(point), apex),
                     (point, torch.zeros_like(apex))):
            w, b = h.descend(p, a, query)
            old_w, old_b = old.descend(p, a, query)
            self.assertTrue(torch.equal(w, old_w))
            self.assertTrue(torch.equal(b, old_b))

    def test_attaches_inside_original_diamond_forward_without_a_bypass(self):
        model = GeometricDiamond()
        model.hierarchy = self.calibrated_decoder()
        trunk = SimpleNamespace(E=nn.Parameter(torch.randn(256,96)),
                                H=nn.Parameter(torch.randn(256,96)),
                                P=nn.Parameter(torch.randn(32640,64)),
                                M=nn.Parameter(torch.randn(32640,64)))
        b = 2
        args = (torch.randn(b,96), torch.randint(0,256,(b,32)),
                torch.randint(0,256,(b,128)),torch.randint(0,256,(b,8)),
                torch.stack([torch.randperm(256)[:8] for _ in range(b)]), trunk)
        with torch.no_grad():
            old_hidden,_,_,old_trace = model(*args,return_trace=True)
        model.hierarchy.apex_feedback = CausalApexFeedback(32)
        model.hierarchy.apex_feedback_gain = nn.Parameter(torch.zeros(()))
        with torch.no_grad():
            inherited,_,_,trace = model(*args,return_trace=True)
            inherited_no_apex,_,_=model(*args,without_apex=True)
        self.assertTrue(torch.equal(inherited, old_hidden))
        self.assertTrue(torch.equal(trace['layered_byte_condition'],
                                    old_trace['layered_byte_condition']))
        with torch.no_grad():
            model.hierarchy.apex_feedback_gain.fill_(.5)
            modified,_,_,new_trace = model(*args,return_trace=True)
            no_pair,_,_=model(*args,without_pair=True)
            no_apex,_,_=model(*args,without_apex=True)
        self.assertGreater(float((modified-old_hidden).abs().sum()), 0.)
        self.assertGreater(float((new_trace['layered_byte_condition']-
                                  trace['layered_byte_condition']).abs().sum()),0.)
        self.assertTrue(torch.equal(no_pair,args[0]))
        # Other inherited geometry can remain active without the apex; the
        # *new* feedback must not bypass its missing apex signal.
        self.assertTrue(torch.equal(no_apex,inherited_no_apex))

    def test_attach_preserves_all_existing_parameters_and_adam_moments(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.diamond = nn.Module()
                self.diamond.geometry = FactorizedContextualGeometry()
                self.diamond.hierarchy = FA1ApexFeedbackTests.calibrated_decoder()

        model = Model()
        old = list(model.parameters())
        optimizer = type('UnifiedOptimizerShell', (), {})()
        optimizer._name_by_id = {id(p): n for n, p in model.named_parameters()}
        optimizer._group_names = [{'algorithm':'adamw',
                                   'names':list(optimizer._name_by_id.values())}]
        optimizer._adamw = torch.optim.AdamW(old, lr=.001)
        optimizer._legacy_groups = []
        optimizer._source_group_counts = [1]
        optimizer.param_groups = optimizer._adamw.param_groups
        old[0].grad = torch.ones_like(old[0])
        optimizer._adamw.step()
        prior = copy.deepcopy(optimizer._adamw.state[old[0]])
        prior_ids = {id(p) for p in old}
        attach_apex_feedback(model, optimizer)
        self.assertEqual(len(optimizer.param_groups), 2)
        self.assertEqual(optimizer._source_group_counts, [2])
        self.assertTrue(prior_ids.issubset({id(p) for p in model.parameters()}))
        self.assertEqual({id(p) for g in optimizer.param_groups for p in g['params']},
                         {id(p) for p in model.parameters()})
        for k, v in prior.items():
            self.assertTrue(torch.equal(v, optimizer._adamw.state[old[0]][k]))
        with self.assertRaisesRegex(ValueError, 'already attached'):
            attach_apex_feedback(model, optimizer)

        with torch.no_grad():
            model.diamond.hierarchy.apex_feedback_gain.fill_(.5)
        with apex_feedback_disabled(model):
            self.assertEqual(float(model.diamond.hierarchy.apex_feedback_gain),0.)
        self.assertAlmostEqual(float(model.diamond.hierarchy.apex_feedback_gain),.5)
        with geometry_disabled(model):
            self.assertEqual(float(model.diamond.hierarchy.apex_feedback_gain),0.)

    def test_negative_feedback_cannot_pass_dev_gate(self):
        baseline = dict(full_word_nll=5.4)
        candidate = dict(full_word_nll=5.3, geometry_word_gain=.01,
                         branch_geometry_gain=.01,
                         correct_book_advantage_nats_per_word=.01,
                         apex_word_gain=.01, pair_word_gain=.01,
                         book_geometry_gain={'book_bootstrap_ci95':[.001,.02]},
                         book_correct_history={'book_bootstrap_ci95':[.001,.02]},
                         apex_feedback_word_gain=-.01,
                         apex_feedback_branch_gain=.01,
                         apex_feedback_word_bootstrap={'book_bootstrap_ci95':[-.02,.01]},
                         apex_feedback_byte_gain_nats_per_byte=.01,
                         apex_feedback_byte_bootstrap={'book_bootstrap_ci95':[.001,.02]})
        self.assertFalse(passes_gate(baseline, candidate, .0001))
        candidate['apex_feedback_word_gain'] = .01
        candidate['apex_feedback_word_bootstrap'] = {'book_bootstrap_ci95':[.001,.02]}
        self.assertTrue(passes_gate(baseline, candidate, .0001))
        candidate['apex_feedback_byte_bootstrap'] = {'book_bootstrap_ci95':[-.001,.02]}
        self.assertFalse(passes_gate(baseline, candidate, .0001))


if __name__ == '__main__':
    unittest.main()
