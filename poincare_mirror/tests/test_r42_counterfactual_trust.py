"""R42 opt-in, training-only counterfactual trust supervision tests."""
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from poincare_frontier import (
    Config, MirrorLM, trust_counterfactual_binary_loss,
)


class CounterfactualTrustTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_supervision_favors_reciprocal_when_it_helps(self):
        # The correct byte is index 0. The first position benefits from the
        # reciprocal bridge while the second is harmed by it.
        on = torch.tensor([[[4., 0.], [0., 4.]]], requires_grad=True)
        off = torch.tensor([[[0., 4.], [4., 0.]]], requires_grad=True)
        targets = torch.tensor([[0, 0]])
        trust_logits = torch.zeros(1, 2, 1, requires_grad=True)
        aux, better = trust_counterfactual_binary_loss(trust_logits, on, off, targets)
        self.assertAlmostEqual(better.item(), 0.5)
        aux.backward()
        self.assertLess(trust_logits.grad[0, 0, 0].item(), 0.)
        self.assertGreater(trust_logits.grad[0, 1, 0].item(), 0.)
        self.assertIsNone(on.grad)
        self.assertIsNone(off.grad)

    def test_rejects_misaligned_predictions(self):
        with self.assertRaises(ValueError):
            trust_counterfactual_binary_loss(torch.zeros(1, 2, 1),
                torch.zeros(1, 2, 5), torch.zeros(1, 1, 5), torch.zeros(1, 2, dtype=torch.long))

    def test_real_reciprocal_gate_receives_auxiliary_gradient(self):
        torch.manual_seed(42)
        cfg = Config(vocab_size=257, dim=32, layers=2, heads=4,
            pairs=4, topk_pairs=2, expert_hidden=16, context=8,
            moe_every=2, symbiotic_rank=8, hierarchy_rank=8,
            micro_meta=True, simplex_transfer=True, reciprocal_scales=True,
            reciprocal_content_trust=True)
        model = MirrorLM(cfg)
        model.reciprocal.record_trust_for_aux = True
        with torch.no_grad():
            model.reciprocal.gate.fill_(.06)
        batch = torch.tensor([[65, 66, 32, 67, 68, 69, 70, 71]])
        on = model.loss_with_future(batch, return_next_logits=True)
        self.assertEqual(model.reciprocal.latest_trust_logits.shape, (1, 5, 1))
        model.reciprocal.enabled = False
        try:
            with torch.no_grad():
                off, _, _ = model(batch[:, :-3])
        finally:
            model.reciprocal.enabled = True
        aux, fraction = trust_counterfactual_binary_loss(
            model.reciprocal.latest_trust_logits, on['next_logits'], off, batch[:, 1:-2])
        self.assertTrue(0. <= fraction.item() <= 1.)
        model.zero_grad(set_to_none=True)
        aux.backward()
        self.assertGreater(model.reciprocal.context_trust.weight.grad.norm().item(), 0.)
        # The auxiliary head has a detached feature input, therefore only
        # trust weights receive its gradients, not the whole inherited LM.
        self.assertIsNone(model.embed.weight.grad)
        self.assertIsNone(model.reciprocal.scale_qkv.weight.grad)


if __name__ == '__main__':
    unittest.main()
