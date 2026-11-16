"""R61: parity between FA2 vectorized and causal indexed successor engines."""
from dataclasses import asdict
from pathlib import Path
import sys
import time
import unittest

import torch
from torch.nn import functional as F

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pair_successor_memory import PairSuccessorTrust,retrieve_pair_successors
from poincare_frontier import Config,MirrorLM


class IndexedEngineParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    @staticmethod
    def fixture(length=32,edges=3,batch=2):
        torch.manual_seed(611)
        # Use a small population of real canonical pair addresses for
        # nontrivial exact and head overlaps (not all-miss comparisons).
        candidates=torch.tensor([1,2,3,4,5,6,255,256,258,510,766,1020,
                                 2000,3000,4000,9000],dtype=torch.long)
        pairs=candidates[torch.randint(0,len(candidates),(batch,length,edges))]
        options=torch.tensor([32,65,66,67,68,69],dtype=torch.long)
        seen=options[torch.randint(0,len(options),(batch,length))]
        return seen,pairs

    def compare_engines(self,observed,pairs,horizon,mode,control,query_map=None,
                        query_byte_map=None):
        kwargs=dict(horizon=horizon,key_mode=mode,value_control=control,
                    query_map=query_map,query_byte_map=query_byte_map)
        oracle=retrieve_pair_successors(pairs,observed,engine="vectorized",**kwargs)
        indexed=retrieve_pair_successors(pairs,observed,engine="indexed",**kwargs)
        for a,b in zip(oracle,indexed):
            torch.testing.assert_close(a,b,rtol=0,atol=1e-6)
        return oracle

    def test_all_key_modes_horizons_and_value_controls(self):
        seen,pairs=self.fixture()
        for mode in ("pair","pair_byte","pair_bigram","head_byte","hierarchical"):
            for horizon in (1,2,7,32,64):
                for control in ("exact","rotated"):
                    with self.subTest(mode=mode,horizon=horizon,control=control):
                        self.compare_engines(seen,pairs,horizon,mode,control)

    def test_isolated_query_permutations_and_batch_independence(self):
        seen,pairs=self.fixture(length=24)
        for mode in ("pair_byte","pair_bigram","head_byte","hierarchical"):
            for pmap,bmap in (
                    (torch.arange(32640).roll(137),None),
                    (None,torch.arange(257).roll(1)),
                    (torch.arange(32640).roll(137),torch.arange(257).roll(1))):
                self.compare_engines(seen,pairs,8,mode,"exact",pmap,bmap)
        a=self.compare_engines(seen,pairs,9,"hierarchical","exact")
        for b in range(seen.shape[0]):
            single=self.compare_engines(seen[b:b+1],pairs[b:b+1],9,
                                        "hierarchical","exact")
            for both,one in zip(a,single):
                torch.testing.assert_close(both[b:b+1],one,rtol=0,atol=1e-6)

    def test_synthetic_repetition_retained_exactly_with_eviction(self):
        seen=torch.tensor([[65,66,65,67,65,66,65,67,65,66,65,67]])
        addr=lambda a,b: a*(511-a)//2+b-a-1
        pairs=torch.tensor([[[addr(1,2),addr(2,3)],
                             [addr(100,101),addr(99,100)],
                             [addr(1,3),addr(1,2)],
                             [addr(100,101),addr(99,100)],
                             [addr(1,4),addr(1,3)],
                             [addr(100,101),addr(99,100)],
                             [addr(1,2),addr(1,4)],
                             [addr(100,101),addr(99,100)],
                             [addr(1,5),addr(1,6)],
                             [addr(100,101),addr(99,100)],
                             [addr(1,7),addr(1,8)],
                             [addr(100,101),addr(99,100)]]])
        for h in (2,4,8,32):
            original=self.compare_engines(seen,pairs,h,"hierarchical","exact")
            self.assertTrue(torch.isfinite(original[0]).all())
        changed=seen.clone()
        changed[:,8:]=90
        head=self.compare_engines(changed,pairs,12,"hierarchical","rotated")
        original=self.compare_engines(seen,pairs,12,"hierarchical","rotated")
        for a,b in zip(head,original):
            self.assertTrue(torch.equal(a[:,:8],b[:,:8]))

    def test_nonzero_rails_prediction_and_gradient_parity(self):
        torch.manual_seed(612)
        seen,pairs=self.fixture(length=20,batch=1,edges=3)
        logits=torch.randn(1,20,257)*.1
        rail=PairSuccessorTrust(conditional_content=True,joint_content=True)
        rail.joint_key_mode="hierarchical"
        with torch.no_grad():
            rail.weight.copy_(torch.tensor([.5,.2,.1,0.]))
            rail.content_weight.copy_(torch.tensor([.4,.3,0.,.1]))
            rail.joint_weight.copy_(torch.tensor([.8,.2,.0,-.1]))
        rail.read_engine="vectorized"
        oracle,_=rail(logits,pairs,seen)
        rail.read_engine="indexed"
        candidate,_=rail(logits,pairs,seen)
        torch.testing.assert_close(oracle,candidate,rtol=0,atol=2e-5)
        target=seen[:,1:]
        loss=F.cross_entropy(candidate[:,:-1].reshape(-1,257),target.reshape(-1))
        loss.backward()
        self.assertGreater(rail.joint_weight.grad.abs().sum().item(),0)
        with self.assertRaisesRegex(ValueError,"CPU-only"):
            retrieve_pair_successors(pairs.to("meta"),seen.to("meta"),
                                      key_mode="hierarchical",engine="indexed")

    def test_indexed_engine_config_roundtrip(self):
        cfg=Config(dim=16,layers=2,heads=2,pairs=4,topk_pairs=1,
                   expert_hidden=16,context=16,moe_every=2,symbiotic_rank=8,
                   micro_meta=True,simplex_transfer=True,dispatch_mode="gather",
                   balance_weight=0.,pair_successor_copy=True,
                   pair_successor_conditional=True,pair_successor_joint=True,
                   pair_successor_key_mode="hierarchical",
                   pair_successor_engine="indexed")
        model=MirrorLM(cfg).eval()
        self.assertEqual(model.pair_successor.read_engine,"indexed")
        self.assertEqual(model.pair_successor.joint_key_mode,"hierarchical")
        recovered=MirrorLM(Config(**asdict(cfg))).eval()
        recovered.load_state_dict(model.state_dict(),strict=True)
        seen=torch.tensor([[65,66,65,66,65,66,65,66]])
        with torch.no_grad():
            self.assertTrue(torch.equal(model(seen)[0],recovered(seen)[0]))

    def test_long_window_parity_and_bounded_cpu_usage(self):
        seen,pairs=self.fixture(length=192,batch=1,edges=4)
        started=time.perf_counter()
        self.compare_engines(seen,pairs,96,"hierarchical","exact")
        elapsed=time.perf_counter()-started
        # A generous CPU-CI budget, not a measured speedup claim.
        self.assertLess(elapsed,30.0)


if __name__=="__main__":
    unittest.main()
