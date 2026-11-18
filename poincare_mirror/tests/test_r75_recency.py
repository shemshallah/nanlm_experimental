"""R75 zero-parity recency, causal retrieval and gradient tests."""
import sys
from pathlib import Path
import unittest

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from book_successor_stream import CausalBookSuccessor
from pair_successor_memory import PairSuccessorTrust


class RecencyTests(unittest.TestCase):
    def test_legacy_default_bitwise_equal(self):
        data=[65,66,67,68,65,66,70,72,65]
        a=CausalBookSuccessor(key_mode='pair')
        b=CausalBookSuccessor(key_mode='pair',recency_half_life=0.)
        for s in (a,b):s.begin_book('same')
        for byte in data:
            x=a.step('same',byte,[10,20])
            y=b.step('same',byte,[10,20])
            self.assertTrue(torch.equal(x[0],y[0]))
            self.assertEqual(x[1:],y[1:])

    def test_recency_changes_values_not_hit_counts(self):
        old=CausalBookSuccessor(key_mode='pair')
        recent=CausalBookSuccessor(key_mode='pair',recency_half_life=1.)
        for s in (old,recent):s.begin_book('book')
        for byte in [60,61,62,63,64,65,66,67]:
            uniform=old.step('book',byte,[11])
            decayed=recent.step('book',byte,[11])
        self.assertGreater((uniform[0]-decayed[0]).abs().sum().item(),.3)
        self.assertEqual(uniform[1:],decayed[1:])
        self.assertAlmostEqual(float(decayed[0].sum()),1.,places=6)

    def test_stream_snapshots_and_future_causality(self):
        first=CausalBookSuccessor(key_mode='pair',recency_half_life=8.)
        second=CausalBookSuccessor(key_mode='pair',recency_half_life=8.)
        first.begin_book('one')
        for b in (65,66,65,67,65,68):first.step('one',b,[10])
        second.load_state_dict(first.state_dict())
        a=first.step('one',65,[10]);b=second.step('one',65,[10])
        self.assertTrue(torch.equal(a[0],b[0]))
        self.assertEqual(a[1:],b[1:])
        with self.assertRaisesRegex(ValueError,'recency_half_life'):
            CausalBookSuccessor(key_mode='pair').load_state_dict(first.state_dict())

    def test_new_zero_weight_preserves_r69_and_has_gradients(self):
        torch.manual_seed(75)
        m=PairSuccessorTrust(conditional_content=True,joint_content=True,
                             future_content=True)
        base=torch.randn(1,3,257)
        uniform=torch.zeros_like(base);uniform[...,65]=.5;uniform[...,66]=.5
        recent=torch.zeros_like(base);recent[...,65]=.9;recent[...,66]=.1
        cov=torch.full((1,3),.6);hits=torch.full((1,3),4.)
        a=(uniform,cov,hits);b=(recent,cov.clone(),hits.clone())
        old={k:v.detach().clone() for k,v in m.state_dict().items()}
        weight=m.enable_recency_content()
        self.assertTrue(torch.equal(base,m.recency_forward(base,a,b)))
        self.assertTrue(all(torch.equal(v,m.state_dict()[k]) for k,v in old.items()))
        with torch.no_grad():weight.fill_(.5)
        output=m.recency_forward(base,a,b)
        for x in (32,256):self.assertTrue(torch.equal(base[...,x],output[...,x]))
        mask=torch.ones(257,dtype=torch.bool);mask[[32,256]]=False
        self.assertTrue(torch.allclose(torch.logsumexp(base[...,mask],-1),
                                       torch.logsumexp(output[...,mask],-1),atol=1e-6))
        loss=torch.nn.functional.cross_entropy(output.flatten(0,1),
                                               torch.full((3,),65,dtype=torch.long))
        loss.backward()
        self.assertGreater(weight.grad.abs().sum().item(),0)


if __name__=='__main__':unittest.main()
