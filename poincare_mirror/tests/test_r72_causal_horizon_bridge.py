"""R72 fallback from observed +2 memory to sparse observed +3 memory."""
import sys
from pathlib import Path
import unittest

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pair_successor_memory import PairSuccessorTrust
from book_stream_eval import evaluate_causal_book
from poincare_frontier import Config,MirrorLM


class HorizonBridgeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(72)
        self.memory=PairSuccessorTrust(conditional_content=True,
                                      joint_content=True,future_content=True)
        with torch.no_grad(): self.memory.future_weight.fill_(.6)
        self.raw=tuple(torch.randn(1,5,257) for _ in range(2))
        self.h2=torch.zeros(1,5,257)
        self.h3=torch.zeros(1,5,257)
        self.h2[...,88]=1.
        self.h3[...,90]=1.
        self.c2=torch.full((1,5),.8)
        self.c3=torch.full((1,5),.2)
        self.n2=torch.full((1,5),3.)
        self.n3=torch.full((1,5),1.)

    def history(self):
        return (self.h2,self.c2,self.n2,self.h3,self.c3,self.n3)

    def test_zero_is_bitwise_identical_to_parent(self):
        old=self.memory.future_forward(self.raw,self.history())
        new=self.memory.future_forward(self.raw,self.history(),horizon_bridge=0.)
        for a,b in zip(old,new): self.assertTrue(torch.equal(a,b))

    def test_causal_bridge_only_modifies_plus_three(self):
        before=self.memory.future_forward(self.raw,self.history())
        after=self.memory.future_forward(self.raw,self.history(),horizon_bridge=.4)
        self.assertTrue(torch.equal(before[0],after[0]))
        self.assertGreater((before[1]-after[1]).abs().sum().item(),0.001)
        self.assertTrue(torch.equal(before[1][...,32],after[1][...,32]))
        self.assertTrue(torch.equal(before[1][...,256],after[1][...,256]))
        mask=torch.ones(257,dtype=torch.bool)
        mask[[32,256]]=False
        self.assertTrue(torch.allclose(torch.logsumexp(before[1][...,mask],-1),
                                       torch.logsumexp(after[1][...,mask],-1),atol=1e-6))

    def test_reliable_plus_three_suppresses_fallback(self):
        self.c3.fill_(1.)
        before=self.memory.future_forward(self.raw,self.history())
        after=self.memory.future_forward(self.raw,self.history(),horizon_bridge=1.)
        self.assertTrue(torch.allclose(before[1],after[1],atol=1e-7))

    def test_invalid_bridge_or_evidence_fails_closed(self):
        with self.assertRaisesRegex(ValueError,'between zero and one'):
            self.memory.future_forward(self.raw,self.history(),horizon_bridge=1.2)
        self.c2.fill_(-.1)
        with self.assertRaisesRegex(ValueError,'Invalid (R72|causal)'):
            self.memory.future_forward(self.raw,self.history(),horizon_bridge=.5)

    def test_evaluator_rejects_bridge_without_r69(self):
        cfg=Config(dim=16,layers=2,heads=2,pairs=4,topk_pairs=1,
                   expert_hidden=16,context=16,moe_every=2,symbiotic_rank=8,
                   micro_meta=True,simplex_transfer=True,dispatch_mode='gather',
                   balance_weight=0.,pair_successor_copy=True,
                   pair_successor_conditional=True,pair_successor_joint=True)
        model=MirrorLM(cfg).eval()
        tokens=torch.tensor([65,66,67,32,68,69,70,32,71,72])
        with self.assertRaisesRegex(ValueError,'requires an R69'):
            evaluate_causal_book(model,tokens,'R72-toy',context=8,chunk=4,
                                 future_horizon_bridge=.2)


if __name__=='__main__': unittest.main()
