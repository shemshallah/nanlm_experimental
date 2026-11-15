"""R44 causal reciprocal representation-supervision regression tests."""
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from poincare_frontier import Config, MirrorLM, reciprocal_residual_prediction_loss


class GeometricResidualObjectiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_tied_vocabulary_teacher_and_gradient_isolation(self):
        baseline = torch.zeros(1, 2, 2, requires_grad=True)
        tied = torch.tensor([[1., 0.], [-1., 0.]], requires_grad=True)
        residual = torch.tensor([[[1., 0.], [1., 0.]]], requires_grad=True)
        labels = torch.tensor([[0, 0]])
        loss = reciprocal_residual_prediction_loss(baseline, residual, tied, labels, 1.)
        self.assertLess(loss.item(), torch.log(torch.tensor(2.)).item())
        loss.backward()
        self.assertIsNone(baseline.grad)
        self.assertIsNone(tied.grad)
        self.assertIsNotNone(residual.grad)
        self.assertLess(residual.grad[0, 0, 0].item(), 0.)

    def test_rejects_invalid_scale_and_misaligned_shapes(self):
        baseline = torch.zeros(1, 3, 257)
        residual = torch.zeros(1, 3, 8)
        tied = torch.zeros(257, 8)
        labels = torch.zeros(1, 3, dtype=torch.long)
        for scale in (-1., 0., float('nan')):
            with self.assertRaises(ValueError):
                reciprocal_residual_prediction_loss(baseline,residual,tied,labels,scale)
        with self.assertRaises(ValueError):
            reciprocal_residual_prediction_loss(baseline,residual[:, :2],tied,labels,1.)

    def test_real_ungated_geometry_is_causal_and_receives_gradient(self):
        torch.manual_seed(44)
        cfg = Config(vocab_size=257, dim=32, layers=2, heads=4,
            pairs=4, topk_pairs=2, expert_hidden=16, context=8,
            moe_every=2, symbiotic_rank=8, hierarchy_rank=8,
            micro_meta=True, simplex_transfer=True, reciprocal_scales=True,
            reciprocal_content_trust=True)
        model = MirrorLM(cfg)
        model.reciprocal.record_residual_for_aux = True
        batch = torch.tensor([[65,66,32,67,68,69,70,71]])
        _ = model.loss_with_future(batch)
        geom = model.reciprocal.latest_residual_for_aux
        self.assertEqual(geom.shape, (1,5,32))
        model.reciprocal.enabled = False
        try:
            with torch.no_grad():
                off, _, _ = model(batch[:,:-3])
        finally:
            model.reciprocal.enabled = True
        aux = reciprocal_residual_prediction_loss(off, geom, model.head.weight,
                                                   batch[:,1:-2], 8.)
        model.zero_grad(set_to_none=True)
        aux.backward()
        self.assertGreater(model.reciprocal.to_hidden.weight.grad.norm().item(), 0.)
        self.assertIsNotNone(model.reciprocal.scale_qkv.weight.grad)
        self.assertIs(model.head.weight, model.embed.weight)


if __name__ == '__main__':
    unittest.main()
