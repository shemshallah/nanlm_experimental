"""R60: exact-to-shared-head compound successor fallback, confidence and causal gates."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import sys
import unittest

import torch
from torch.nn import functional as F

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from pair_successor_memory import PairSuccessorTrust,retrieve_pair_successors
from poincare_frontier import Config,MirrorLM
from train_r55_pair_successor import memory_arm,train_only_gate,audit_inheritance


def pair(a,b):
    return a*(511-a)//2+b-a-1


class HierarchicalCompoundTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def fixture(self):
        seen=torch.tensor([[65,66,65,67,65,66,65,67,65,66,65,67]],
                          dtype=torch.long)
        ids=torch.full((1,seen.shape[1],1),pair(100,101),dtype=torch.long)
        for t,a,b in ((0,1,2),(2,1,3),(4,1,4),(6,1,2),(8,1,5),(10,1,6)):
            ids[:,t,0]=pair(a,b)
        return seen,ids

    def test_canonical_pair_decoding_round_trip(self):
        table=torch.triu_indices(256,256,1)
        idx=table[0]*(511-table[0])//2+table[1]-table[0]-1
        self.assertTrue(torch.equal(idx,torch.arange(32640)))
        self.assertEqual(pair(0,1),0)
        self.assertEqual(pair(254,255),32639)

    def test_hierarchical_backoff_preserves_exact_priority(self):
        seen,ids=self.fixture()
        exact,cov_exact,_=retrieve_pair_successors(
            ids,seen,horizon=12,key_mode="pair_byte")
        mixed,cov,hits=retrieve_pair_successors(
            ids,seen,horizon=12,key_mode="hierarchical")
        self.assertEqual(cov_exact[0,4].item(),0)
        self.assertEqual(cov[0,4].item(),0.25)
        self.assertEqual(hits[0,4].item(),2)
        self.assertAlmostEqual(mixed[0,4,66].item(),.5)
        self.assertAlmostEqual(mixed[0,4,67].item(),.5)
        self.assertEqual(cov[0,6].item(),1.0)
        self.assertAlmostEqual(mixed[0,6,66].item(),1.0)
        self.assertAlmostEqual(mixed[0,6,67].item(),0.0)
        self.assertGreater(cov.sum().item(),cov_exact.sum().item())

    def test_all_three_counterfactuals_keep_their_semantics(self):
        seen,ids=self.fixture()
        a,cov,hits=retrieve_pair_successors(ids,seen,horizon=12,
                                            key_mode="hierarchical")
        b,addr_cov,_=retrieve_pair_successors(
            ids,seen,horizon=12,key_mode="hierarchical",
            query_map=torch.full((32640,),pair(200,201),dtype=torch.long))
        c,byte_cov,_=retrieve_pair_successors(
            ids,seen,horizon=12,key_mode="hierarchical",
            query_byte_map=torch.arange(257).roll(1))
        d,value_cov,value_hits=retrieve_pair_successors(
            ids,seen,horizon=12,key_mode="hierarchical",
            value_control="rotated")
        self.assertEqual(addr_cov[0,4].item(),0.0)
        self.assertEqual(byte_cov[0,4].item(),0.0)
        self.assertTrue(torch.equal(cov,value_cov))
        self.assertTrue(torch.equal(hits,value_hits))
        self.assertFalse(torch.equal(a,b))
        self.assertFalse(torch.equal(a,c))
        self.assertFalse(torch.equal(a,d))

    def test_causal_suffix_and_finite_edge_cases(self):
        seen,ids=self.fixture()
        original=retrieve_pair_successors(ids,seen,horizon=12,
                                           key_mode="hierarchical")
        changed_seen=seen.clone()
        changed_seen[:,8:]=252
        changed_ids=ids.clone()
        changed_ids[:,8:]=pair(200,201)
        altered=retrieve_pair_successors(changed_ids,changed_seen,horizon=12,
                                          key_mode="hierarchical")
        for x,y in zip(original,altered):
            self.assertTrue(torch.equal(x[:,:8],y[:,:8]))
            self.assertTrue(torch.isfinite(x).all())
        for mode in ("head_byte","hierarchical"):
            zero=retrieve_pair_successors(ids[:,:1],seen[:,:1],key_mode=mode)
            self.assertEqual(zero[0].sum().item(),0)
        with self.assertRaises(ValueError):
            retrieve_pair_successors(ids,seen,key_mode="not_a_key")

    def test_joint_mass_conservation_and_mode_serialization(self):
        torch.manual_seed(60)
        seen,ids=self.fixture()
        logits=torch.randn(1,seen.shape[1],257)*.1
        trust=PairSuccessorTrust(conditional_content=True,joint_content=True)
        trust.joint_key_mode="hierarchical"
        before,_=trust(logits,ids,seen)
        with torch.no_grad():
            trust.joint_weight.copy_(torch.tensor([1.,.5,.0,.0]))
        after,stats=trust(logits,ids,seen)
        self.assertTrue(torch.equal(before[...,32],after[...,32]))
        self.assertTrue(torch.equal(before[...,256],after[...,256]))
        self.assertTrue(torch.allclose(before.softmax(-1)[...,32],
                                       after.softmax(-1)[...,32],atol=1e-7))
        mask=torch.ones(257,dtype=torch.bool)
        mask[[32,256]]=False
        self.assertTrue(torch.allclose(
            torch.logsumexp(before[...,mask],-1),
            torch.logsumexp(after[...,mask],-1),atol=1e-6))
        self.assertAlmostEqual(stats["joint_coverage"][0,4].item(),.25)

        cfg=Config(dim=16,layers=2,heads=2,pairs=4,topk_pairs=1,
                   expert_hidden=16,context=16,moe_every=2,symbiotic_rank=8,
                   micro_meta=True,simplex_transfer=True,dispatch_mode="gather",
                   balance_weight=0.)
        model=MirrorLM(cfg).eval()
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
        model.enable_pair_successor(optimizer)
        model.enable_conditional_pair_successor(optimizer)
        unchanged=model(seen)[0].detach().clone()
        model.enable_joint_pair_successor(optimizer,key_mode="hierarchical")
        self.assertTrue(torch.equal(unchanged,model(seen)[0]))
        self.assertEqual(model.cfg.pair_successor_key_mode,"hierarchical")
        restored=MirrorLM(Config(**asdict(model.cfg))).eval()
        restored.load_state_dict(model.state_dict(),strict=True)
        self.assertEqual(restored.pair_successor.joint_key_mode,"hierarchical")
        self.assertTrue(torch.equal(model(seen)[0],restored(seen)[0]))

    def test_synthetic_training_really_improves_a_known_copy_task(self):
        # Independent toy diagnostic, not DEV or a real model milestone.
        torch.manual_seed(602)
        seen=torch.tensor([[65,66]*8],dtype=torch.long)
        ids=torch.full((1,16,1),pair(100,101),dtype=torch.long)
        for t,(a,b) in enumerate(((1,2),(100,101),(1,3),(100,101),
                                  (1,4),(100,101),(1,5),(100,101),
                                  (1,6),(100,101),(1,7),(100,101),
                                  (1,8),(100,101),(1,9),(100,101))):
            ids[:,t,0]=pair(a,b)
        logits=torch.zeros(1,16,257)
        targets=seen.roll(-1,dims=1)
        specialist=PairSuccessorTrust(conditional_content=True,joint_content=True)
        specialist.joint_key_mode="hierarchical"
        opt=torch.optim.AdamW(specialist.parameters(),lr=.02)
        # R59/R60 training driver isolates the newest rail, as production does.
        for name,param in specialist.named_parameters():
            param.requires_grad_(name=="joint_weight")
        records=[[(logits,logits,logits,ids,seen,(targets,targets,targets))]]
        with torch.no_grad():
            initial=F.cross_entropy(specialist(logits,ids,seen)[0][:,2:-1].reshape(-1,257),
                                    targets[:,2:-1].reshape(-1)).item()
        train_only_gate(specialist,opt,records,16,.02,602)
        with torch.no_grad():
            learned=F.cross_entropy(specialist(logits,ids,seen)[0][:,2:-1].reshape(-1,257),
                                    targets[:,2:-1].reshape(-1)).item()
        self.assertLess(learned,initial)
        self.assertGreater(specialist.joint_weight.norm().item(),0)
        control=specialist(logits,ids,seen)[0]
        with memory_arm(specialist,"neutral",None):
            neutral=specialist(logits,ids,seen)[0]
        self.assertTrue(torch.equal(neutral,logits))
        self.assertTrue(torch.equal(control,specialist(logits,ids,seen)[0]))


if __name__=="__main__":
    unittest.main()
