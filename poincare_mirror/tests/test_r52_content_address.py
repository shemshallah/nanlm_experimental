"""R52: non-space-content geometric specificity, with R51 compatibility."""
import sys
from pathlib import Path
import unittest

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from poincare_frontier import geometric_address_margin_loss


class ContentOnlyAddressContrastTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(52)

    def test_default_objective_stays_bitwise_equivalent_to_r51(self):
        exact = torch.randn(2, 5, 257)
        shuffled = torch.randn(2, 5, 257)
        targets = torch.randint(0, 257, (2, 5))
        expected_exact = F.cross_entropy(exact.flatten(0, 1), targets.flatten())
        expected_shuffled = F.cross_entropy(shuffled.flatten(0, 1), targets.flatten())
        gap = expected_shuffled - expected_exact
        expected_loss = .01 * F.softplus((.005-gap)/.01)
        loss, actual_gap = geometric_address_margin_loss(exact, shuffled, targets)
        self.assertTrue(torch.equal(loss, expected_loss))
        self.assertTrue(torch.equal(actual_gap, gap.detach()))

    def test_only_content_bytes_contribute(self):
        exact = torch.randn(1, 5, 257, requires_grad=True)
        shuffled = torch.randn(1, 5, 257, requires_grad=True)
        targets = torch.tensor([[32, 65, 256, 66, 32]])
        loss, gap = geometric_address_margin_loss(
            exact, shuffled, targets, content_only=True)
        content_exact = F.cross_entropy(exact[:, [1, 3], :].reshape(-1, 257),
                                        targets[:, [1, 3]].reshape(-1))
        content_shuffled = F.cross_entropy(shuffled[:, [1, 3], :].reshape(-1, 257),
                                           targets[:, [1, 3]].reshape(-1))
        expected_gap = content_shuffled-content_exact
        self.assertTrue(torch.allclose(gap, expected_gap.detach()))
        self.assertTrue(torch.allclose(loss, .01*F.softplus((.005-expected_gap)/.01)))
        loss.backward()
        self.assertEqual(exact.grad[0, [0, 2, 4]].abs().sum().item(), 0.)
        self.assertEqual(shuffled.grad[0, [0, 2, 4]].abs().sum().item(), 0.)
        self.assertGreater(exact.grad[0, [1, 3]].abs().sum().item(), 0.)
        self.assertGreater(shuffled.grad[0, [1, 3]].abs().sum().item(), 0.)

    def test_all_space_or_eos_batch_is_noop_with_backward(self):
        exact = torch.randn(2, 4, 257, requires_grad=True)
        shuffled = torch.randn(2, 4, 257, requires_grad=True)
        targets = torch.tensor([[32, 32, 256, 32], [256, 32, 32, 256]])
        loss, gap = geometric_address_margin_loss(exact, shuffled, targets,
                                                    content_only=True)
        self.assertEqual(loss.item(), 0.)
        self.assertEqual(gap.item(), 0.)
        loss.backward()
        self.assertEqual(exact.grad.abs().sum().item(), 0.)
        self.assertEqual(shuffled.grad.abs().sum().item(), 0.)

    def test_bad_shapes_rejected_before_content_mask(self):
        logits = torch.zeros(1, 3, 257)
        targets = torch.tensor([[32, 65, 32]])
        with self.assertRaises(ValueError):
            geometric_address_margin_loss(logits, logits[:, :2], targets,
                                          content_only=True)


if __name__ == '__main__':
    unittest.main()
