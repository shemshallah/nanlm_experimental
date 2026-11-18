"""Opt-in R70: make causal future-memory correction top-1 rank preserving."""
from pathlib import Path
import sys
import unittest

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rank_safe_fusion import protect_parent_winner_rank
from poincare_frontier import Config, MirrorLM
from book_stream_eval import evaluate_causal_book


class R70FutureWinnerTests(unittest.TestCase):
    def test_counterfactual_memory_cannot_displace_parent_winner(self):
        parent = torch.full((4, 7, 257), -7.)
        winners = torch.arange(28).reshape(4, 7) + 30
        parent.scatter_(-1, winners.unsqueeze(-1), 3.)
        candidate = parent.clone()
        replacement = (winners + 40) % 257
        candidate.scatter_(-1, replacement.unsqueeze(-1), 9.)
        safe = protect_parent_winner_rank(parent, candidate)
        self.assertTrue(torch.equal(safe.argmax(-1), parent.argmax(-1)))
        self.assertTrue(torch.allclose(safe.exp().sum(-1),
                                       torch.ones_like(winners,dtype=torch.float32),atol=1e-5))
        self.assertTrue(torch.isfinite(safe).all())
        self.assertTrue((safe.exp()-F.softmax(parent,-1)).abs().sum()>0)

    def test_no_divergence_when_memory_does_not_flip_parent_rank(self):
        parent = torch.full((3, 257), -4.)
        parent[:,65] = 5.
        candidate = parent.clone()
        candidate[:,66] = 0.5
        safe = protect_parent_winner_rank(parent, candidate)
        self.assertTrue(torch.allclose(F.softmax(safe,-1),
                                       F.softmax(candidate,-1),atol=1e-6))

    def test_invalid_shapes_rejected(self):
        with self.assertRaises(ValueError):
            protect_parent_winner_rank(torch.randn(2, 257),torch.randn(1,257))

    def test_model_future_path_option_and_error(self):
        torch.manual_seed(70)
        cfg=Config(dim=16,layers=2,heads=2,pairs=4,topk_pairs=1,
                   expert_hidden=16,context=16,moe_every=2,symbiotic_rank=8,
                   micro_meta=True,simplex_transfer=True,dispatch_mode='gather',
                   balance_weight=0.,pair_successor_copy=True,
                   pair_successor_conditional=True,pair_successor_joint=True,
                   pair_successor_future=True)
        model=MirrorLM(cfg).eval()
        ids=torch.tensor([[65,66,67,65,66,67,65,66]],dtype=torch.long)
        with torch.no_grad():
            _,hidden,_=model(ids)
            raw=model.future_logits(hidden)
            hist=torch.zeros(1,8,257)
            hist[...,88]=1.
            coverage=torch.ones(1,8)
            hits=torch.ones(1,8)
            records=(hist,coverage,hits,hist,coverage,hits)
            model.pair_successor.future_weight.fill_(5.)
            guarded=model.future_logits(hidden,future_memory_override=records,
                                       future_winner_guard=True)
        for expected,actual in zip(raw,guarded):
            self.assertTrue(torch.equal(expected.argmax(-1),actual.argmax(-1)))
        with self.assertRaisesRegex(ValueError, 'requires causal'):
            model.future_logits(hidden,future_winner_guard=True)

    def test_evaluator_refuses_guard_without_future_memory(self):
        cfg=Config(dim=16,layers=2,heads=2,pairs=4,topk_pairs=1,
                   expert_hidden=16,context=16,moe_every=2,symbiotic_rank=8,
                   micro_meta=True,simplex_transfer=True,dispatch_mode='gather',
                   balance_weight=0.,pair_successor_copy=True,
                   pair_successor_conditional=True,pair_successor_joint=True)
        model=MirrorLM(cfg).eval()
        book=torch.tensor([65,66,67,32,68,69,70,32,71,72],dtype=torch.long)
        with self.assertRaisesRegex(ValueError,'requires an R69'):
            evaluate_causal_book(model,book,'R70-guard-toy',context=8,chunk=4,
                                 future_winner_guard=True)


if __name__ == '__main__':
    unittest.main()
