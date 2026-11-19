"""R76 confidence weighting retains exact R69 parity when disabled."""
import sys
from pathlib import Path
import unittest

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from book_successor_stream import CausalBookSuccessor
from calibrate_r76_head_weight import evaluate_book
from poincare_frontier import Config,MirrorLM


def pair(a,b):
    return a*(511-a)//2+b-a-1


class R76Tests(unittest.TestCase):
    def test_default_unweighted_matches_explicit_one(self):
        a=CausalBookSuccessor(key_mode='hierarchical')
        b=CausalBookSuccessor(key_mode='hierarchical',head_backoff_weight=1.)
        for s in (a,b):s.begin_book('p')
        for byte,edges in ((65,[pair(0,1),pair(0,3)]),
                           (66,[pair(6,7),pair(6,8)]),
                           (65,[pair(0,1),pair(6,7)]),
                           (67,[pair(6,7),pair(6,8)]),
                           (65,[pair(0,1),pair(0,2)])):
            x=a.step('p',byte,edges)
            y=b.step('p',byte,edges)
            self.assertTrue(torch.equal(x[0],y[0]))
            self.assertEqual(x[1:],y[1:])

    def test_favor_exact_over_shared_head_changes_hist_only(self):
        a=CausalBookSuccessor(key_mode='hierarchical')
        b=CausalBookSuccessor(key_mode='hierarchical',head_backoff_weight=.125)
        for s in (a,b):s.begin_book('p')
        episodes=((65,[pair(0,1),pair(0,3)]),
                  (66,[pair(6,7),pair(6,8)]),
                  (65,[pair(0,1),pair(6,7)]),
                  (67,[pair(6,7),pair(6,8)]),
                  (65,[pair(0,1),pair(0,2)]))
        for byte,edges in episodes:
            old=a.step('p',byte,edges)
            new=b.step('p',byte,edges)
        self.assertGreater((old[0]-new[0]).abs().sum().item(),.001)
        self.assertEqual(old[1:],new[1:])
        self.assertAlmostEqual(float(new[0].sum()),1.,places=6)
        b.load_state_dict(b.state_dict())
        with self.assertRaisesRegex(ValueError,'head_backoff_weight'):
            a.load_state_dict(b.state_dict())

    def test_three_horizon_calibration_toy_parity(self):
        cfg=Config(dim=16,layers=2,heads=2,pairs=4,topk_pairs=1,
                   expert_hidden=16,context=16,moe_every=2,symbiotic_rank=8,
                   micro_meta=True,simplex_transfer=True,dispatch_mode='gather',
                   balance_weight=0.,pair_successor_copy=True,
                   pair_successor_conditional=True,pair_successor_joint=True,
                   pair_successor_future=True)
        model=MirrorLM(cfg).eval()
        tokens=torch.tensor(([65,66,67,32,68,69,70,32]*4),dtype=torch.long)
        result=evaluate_book(model,tokens,'toy',warmup=8,positions=8,
                             context=16,chunk=4,weights=(.25,1.))
        self.assertEqual(result['positions'],8)
        self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in
                            result['metrics']['1.0'].values()))


if __name__=='__main__':unittest.main()
