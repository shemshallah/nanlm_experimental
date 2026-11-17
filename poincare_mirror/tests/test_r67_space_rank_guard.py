"""R67 rank guard is causal, probability-mass preserving and control-matched."""
from pathlib import Path
import sys
import unittest
import torch
from torch.nn import functional as F

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from book_stream_eval import protect_parent_space_rank
from poincare_frontier import Config,MirrorLM


class RankGuardTests(unittest.TestCase):
    def test_parent_space_winner_survives_memory_pressure(self):
        base=torch.full((3,257),-7.)
        base[:,32]=3.
        base[:,4]=2.
        proposed=base.clone()
        proposed[:,4]=6.
        calibrated=protect_parent_space_rank(base,proposed)
        self.assertTrue(calibrated.argmax(-1).eq(32).all())
        pb=F.softmax(base,-1)
        pc=F.softmax(proposed,-1)
        pg=calibrated.exp()
        self.assertTrue(torch.allclose(pg.sum(-1),torch.ones(3),atol=1e-6))
        self.assertTrue(torch.all(pg[:,4]>=pb[:,4]))
        self.assertTrue(torch.all(pg[:,4]<=pc[:,4]))

    def test_nonspace_parent_uses_full_memory(self):
        base=torch.full((5,257),-5.)
        base[:,4]=2.
        candidate=base.clone()
        candidate[:,7]=4.
        out=protect_parent_space_rank(base,candidate)
        self.assertTrue(torch.allclose(F.softmax(out,-1),
                                       F.softmax(candidate,-1),atol=1e-7))

    def test_eos_parent_is_protected_without_targets(self):
        base=torch.zeros(2,257)
        base[:,256]=5.
        candidate=base.clone()
        candidate[:,65]=6.
        guarded=protect_parent_space_rank(base,candidate)
        self.assertTrue(guarded.argmax(-1).eq(256).all())
        self.assertTrue(torch.isfinite(guarded).all())

    def test_rank_guard_is_wired_inside_single_inherited_model(self):
        torch.manual_seed(67)
        cfg=Config(dim=16,layers=2,heads=2,pairs=4,topk_pairs=1,
                   expert_hidden=16,context=16,moe_every=2,symbiotic_rank=8,
                   micro_meta=True,simplex_transfer=True,dispatch_mode='gather',
                   balance_weight=0.,pair_successor_copy=True,
                   pair_successor_conditional=True,pair_successor_joint=True,
                   pair_successor_space_rank_guard=True)
        model=MirrorLM(cfg).eval()
        self.assertTrue(model.pair_successor.space_rank_guard)
        ids=torch.tensor([[65,32,66,32,67,68,32,69]],dtype=torch.long)
        with torch.no_grad():
            _,hidden,_=model(ids)
            pairs=model._r54_geometry().r54_last_pairs.detach()
            logits=model.head(hidden)
            hist=torch.zeros_like(logits)
            hist[...,65]=1.
            cov=torch.ones_like(ids,dtype=torch.float32)
            hits=torch.ones_like(cov)
            model.pair_successor.joint_weight.copy_(torch.tensor([2.,1.,1.,-1.]))
            original,_=model.pair_successor(logits,pairs,ids,
                joint_memory_override=(torch.zeros_like(hist),
                    torch.zeros_like(cov),torch.zeros_like(hits)))
            guarded,_=model.pair_successor(logits,pairs,ids,
                                           joint_memory_override=(hist,cov,hits))
            top=original.argmax(-1)
            protected=top.eq(32)|top.eq(256)
            self.assertTrue(torch.equal(guarded.argmax(-1)[protected],top[protected]))
            self.assertTrue(torch.isfinite(guarded).all())


if __name__=="__main__":
    unittest.main()
