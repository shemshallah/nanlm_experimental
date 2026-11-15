"""R39 unweighted causal diagnostics integrated into the same FA2 trainer."""
import math
import sys
import unittest
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from poincare_frontier import Config, MirrorLM, evaluate_byte_breakdown


class ArithmeticPredictor(nn.Module):
    """Exactly solves a synthetic ascending-byte task at +1/+2/+3."""
    @staticmethod
    def predict(ids, offset):
        logits = torch.full((*ids.shape, 257), -6.)
        logits.scatter_(-1, (ids+offset).unsqueeze(-1), 9.)
        return logits

    def forward(self, ids):
        return self.predict(ids,1), ids, {}

    def future_logits(self, hidden):
        return self.predict(hidden,2), self.predict(hidden,3)


class ByteBreakdownTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_exact_causal_horizons_and_space_strata(self):
        batch = torch.tensor([[29,30,31,32,33,34,35],
                              [62,63,64,65,66,67,68]])
        predictor = ArithmeticPredictor().train()
        before = predictor.training
        report = evaluate_byte_breakdown(predictor,[batch])
        self.assertEqual(predictor.training,before)
        for horizon in ('next','plus2','plus3'):
            self.assertEqual(report[horizon]['all']['positions'],8)
            self.assertEqual(report[horizon]['space']['positions'],1)
            self.assertEqual(report[horizon]['nonspace']['positions'],7)
            for group in ('all','nonspace','space'):
                row = report[horizon][group]
                self.assertEqual(row['top1_percent'],100.)
                self.assertEqual(row['top5_percent'],100.)
                self.assertAlmostEqual(row['bits_per_byte'],
                    row['ce_nats_per_byte']/math.log(2),places=5)

    def test_missing_subset_is_null_never_fabricated_accuracy(self):
        report=evaluate_byte_breakdown(ArithmeticPredictor(),
            [torch.tensor([[60,61,62,63,64,65,66]])])
        for horizon in ('next','plus2','plus3'):
            self.assertIsNone(report[horizon]['space'])

    def test_matches_inherited_unweighted_losses_without_mutation(self):
        torch.manual_seed(120)
        cfg=Config(vocab_size=257,dim=32,heads=4,layers=2,pairs=4,
                   topk_pairs=2,expert_hidden=16,context=8,moe_every=2,
                   future2_weight=.1,future3_weight=.05)
        model=MirrorLM(cfg).train()
        batch=torch.tensor([[65,32,66,32,67,32,68,32,69,32,70]])
        before={k:v.detach().clone() for k,v in model.state_dict().items()}
        with torch.no_grad():
            model.eval()
            inherited=model.loss_with_future(batch)
        model.train()
        breakdown=evaluate_byte_breakdown(model,[batch])
        self.assertTrue(model.training)
        for horizon,field in (('next','next_ce'),('plus2','future2_ce'),
                              ('plus3','future3_ce')):
            self.assertAlmostEqual(breakdown[horizon]['all']['ce_nats_per_byte'],
                                   inherited[field].item(),places=5)
            self.assertEqual(breakdown[horizon]['all']['positions'],8)
        for name,tensor in before.items():
            self.assertTrue(torch.equal(tensor,model.state_dict()[name]),name)
        self.assertFalse(any(p.grad is not None for p in model.parameters()))

    def test_diagnostics_reject_bad_length(self):
        with self.assertRaisesRegex(ValueError,'causal evaluation windows'):
            evaluate_byte_breakdown(ArithmeticPredictor(),[torch.ones(1,3,dtype=torch.long)])


if __name__=='__main__':
    unittest.main()
