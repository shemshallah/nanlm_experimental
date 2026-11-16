"""R51 margin objectives and joint exact/shuffled autograd contracts."""
import math
import sys
from pathlib import Path
import unittest

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from poincare_frontier import geometric_address_margin_loss
from geometry_address_controls import address_ablation, fixed_permutations
from test_r50_geometry_addresses import FakeModel


class ContrastiveAddressLearning(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(51)

    def test_better_exact_predictions_reduce_margin_penalty(self):
        targets = torch.tensor([[2, 1]])
        exact = torch.zeros(1,2,4)
        shuffled = torch.zeros(1,2,4)
        before, gap0 = geometric_address_margin_loss(exact, shuffled, targets)
        exact[:,:,1:3] = 2.
        after, gap1 = geometric_address_margin_loss(exact, shuffled, targets)
        self.assertLess(after.item(), before.item())
        self.assertGreater(gap1.item(), gap0.item())

    def test_both_counterfactual_paths_receive_real_gradients(self):
        target = torch.tensor([[0, 2, 1]])
        exact = torch.randn(1,3,4, requires_grad=True)
        scrambled = torch.randn(1,3,4, requires_grad=True)
        loss, gap = geometric_address_margin_loss(exact, scrambled, target)
        loss.backward()
        self.assertTrue(torch.isfinite(gap))
        self.assertGreater(exact.grad.abs().sum().item(), 0.)
        self.assertGreater(scrambled.grad.abs().sum().item(), 0.)
        self.assertLess(exact.grad[0,0,0].item(), 0.)
        self.assertGreater(scrambled.grad[0,0,0].item(), 0.)

    def test_same_logits_have_known_finite_loss(self):
        logits = torch.randn(2,3,8)
        target = torch.randint(0,8,(2,3))
        value,gap = geometric_address_margin_loss(logits, logits, target,.005,.01)
        self.assertEqual(gap.item(), 0.)
        self.assertAlmostEqual(value.item(), .01*math.log1p(math.exp(.5)),places=6)

    def test_invalid_shapes_or_hyperparameters_rejected(self):
        x = torch.zeros(1,3,5)
        y = torch.zeros(1,3,dtype=torch.long)
        for m,t in ((-.1,.01),(.1,0.),(float('nan'),.01),(.1,float('inf'))):
            with self.assertRaises(ValueError):
                geometric_address_margin_loss(x,x,y,m,t)
        with self.assertRaises(ValueError):
            geometric_address_margin_loss(x,x[:,:-1],y)

    def test_new_training_step_gets_different_permutations(self):
        a = fixed_permutations(5001)
        b = fixed_permutations(5002)
        self.assertFalse(torch.equal(a['pair_addresses'],b['pair_addresses']))
        self.assertFalse(torch.equal(a['faces'],b['faces']))

    def test_exact_and_scrambled_backward_coexist_without_buffer_mutation(self):
        m = FakeModel()
        s = m.geometry.simplex
        original_d1, original_d2, original_d3 = s.d1, s.d2, s.d3
        src = s.pair_shared(torch.arange(28))[None]
        exact = torch.einsum('ep,ner->npr',s.d1,src)
        with address_ablation(m,'all_scrambled',fixed_permutations(2026)):
            changed = torch.einsum('ep,ner->npr',s.d1,src)
            self.assertIsNot(s.d1,original_d1)
        self.assertIs(s.d1,original_d1)
        self.assertIs(s.d2,original_d2)
        self.assertIs(s.d3,original_d3)
        (exact.square().mean()+changed.square().mean()).backward()
        self.assertTrue(torch.isfinite(s.pair_shared.weight.grad).all())


if __name__ == '__main__':
    unittest.main()
