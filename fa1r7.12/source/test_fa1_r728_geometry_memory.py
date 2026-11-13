"""Fast R7.28 checks; the real-weight book tests run separately."""
import unittest

import torch
from torch import nn

from fa1_r728_geometric_memory import (
    GEOMETRY_GATES, geometry_disabled, passes_gate, soft_margin,
)


class DummyGeometry(nn.Module):
    def __init__(self):
        super().__init__()
        for i, key in enumerate(GEOMETRY_GATES):
            setattr(self, key, nn.Parameter(torch.tensor(float(i+1))))


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.diamond = nn.Module()
        self.diamond.geometry = DummyGeometry()


class GeometricMemoryTests(unittest.TestCase):
    def test_geometry_off_restores_exact_gates(self):
        model = DummyModel()
        saved = {k: getattr(model.diamond.geometry,k).detach().clone() for k in GEOMETRY_GATES}
        with geometry_disabled(model):
            for key in GEOMETRY_GATES:
                self.assertEqual(float(getattr(model.diamond.geometry,key).detach()),0.)
        for key in GEOMETRY_GATES:
            self.assertTrue(torch.equal(saved[key],getattr(model.diamond.geometry,key)))

    def test_exception_restores_gates(self):
        model = DummyModel()
        with self.assertRaisesRegex(ValueError,'expected'):
            with geometry_disabled(model):
                raise ValueError('expected')
        self.assertEqual(float(model.diamond.geometry.meta_gate.detach()),4.)

    def test_margin_rewards_causal_advantage(self):
        true = torch.tensor([2.,2.],requires_grad=True)
        wrong = torch.tensor([3.,3.])
        good = soft_margin(true,wrong,0.01)
        bad = soft_margin(wrong,true,0.01)
        self.assertLess(float(good),float(bad))
        good.backward()
        self.assertTrue(torch.isfinite(true.grad).all())

    def test_promotion_policy_requires_all_real_gates(self):
        reference={'full_word_nll':5.}
        proposal={
            'full_word_nll':4.99,'geometry_word_gain':0.002,
            'branch_geometry_gain':0.01,
            'correct_book_advantage_nats_per_word':0.01,
            'apex_word_gain':0.002,'pair_word_gain':0.002,
            'book_geometry_gain':{'book_bootstrap_ci95':[0.001,0.01]},
            'book_correct_history':{'book_bootstrap_ci95':[0.001,0.01]},
        }
        self.assertTrue(passes_gate(reference,proposal,0.001))
        for field in ('geometry_word_gain','branch_geometry_gain','correct_book_advantage_nats_per_word',
                      'apex_word_gain','pair_word_gain'):
            failed=dict(proposal,**{field:-0.001})
            self.assertFalse(passes_gate(reference,failed,0.001),field)
        for field in ('book_geometry_gain','book_correct_history'):
            failed=dict(proposal,**{field:{'book_bootstrap_ci95':[-0.001,0.01]}})
            self.assertFalse(passes_gate(reference,failed,0.001),field)
        self.assertFalse(passes_gate(reference,dict(proposal,full_word_nll=4.999),0.001))


if __name__=='__main__':
    unittest.main()
