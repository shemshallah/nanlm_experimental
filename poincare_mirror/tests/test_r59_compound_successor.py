"""R59: causal compound-key rail, negative controls and unified lineage."""
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
from train_r55_pair_successor import audit_inheritance,train_only_gate,memory_arm


class CompoundSuccessorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    @staticmethod
    def sample():
        seen = torch.tensor([[65,66,65,67,65,66,65,67,65,66,65,67]],
                            dtype=torch.long)
        pair = torch.full((1,seen.shape[1],2),7,dtype=torch.long)
        return seen,pair

    def test_pair_byte_retrieval_and_all_three_negative_controls(self):
        seen,pairs = self.sample()
        exact,cov,hits = retrieve_pair_successors(
            pairs,seen,horizon=12,key_mode="pair_byte")
        self.assertEqual(hits[0,4].item(),2)
        self.assertEqual(cov[0,4].item(),1)
        self.assertAlmostEqual(exact[0,4,66].item(),.5)
        self.assertAlmostEqual(exact[0,4,67].item(),.5)
        # Wrong pair: only the QUERY side is permuted.
        addr,cov_addr,_ = retrieve_pair_successors(
            pairs,seen,horizon=12,key_mode="pair_byte",
            query_map=torch.arange(32640).roll(137))
        self.assertEqual(cov_addr.sum().item(),0)
        # Wrong observed-byte identity: historical stored bytes unchanged.
        wrong,cov_wrong,_ = retrieve_pair_successors(
            pairs,seen,horizon=12,key_mode="pair_byte",
            query_byte_map=torch.arange(257).roll(1))
        self.assertEqual(cov_wrong[0,4].item(),0)
        self.assertFalse(torch.equal(exact,wrong))
        # Wrong successor: hit counts/coverage are IDENTICAL.
        values,cov_values,hits_values = retrieve_pair_successors(
            pairs,seen,horizon=12,key_mode="pair_byte",value_control="rotated")
        self.assertTrue(torch.equal(cov,cov_values))
        self.assertTrue(torch.equal(hits,hits_values))
        self.assertFalse(torch.equal(exact,values))
        self.assertFalse(torch.equal(exact,addr))

    def test_bigram_mode_restricts_keys_and_handles_book_boundary(self):
        seen,pairs = self.sample()
        one,_,_ = retrieve_pair_successors(pairs,seen,horizon=12,
                                            key_mode="pair_byte")
        two,cov,hits = retrieve_pair_successors(pairs,seen,horizon=12,
                                                key_mode="pair_bigram")
        self.assertEqual(hits[0,4].item(),0)
        self.assertEqual(cov[0,4].item(),0)
        self.assertGreater(hits[0,6].item(),0)
        self.assertEqual(two[0,0].sum().item(),0)
        self.assertLessEqual(two[0,6].sum().item(),1+1e-6)

    def test_suffix_causality_and_retrieval_guards(self):
        seen,pairs = self.sample()
        original = retrieve_pair_successors(pairs,seen,horizon=12,
                                             key_mode="pair_byte")
        altered = seen.clone()
        altered[:,8:] = 99
        pair_altered = pairs.clone()
        pair_altered[:,8:] = 99
        replay = retrieve_pair_successors(pair_altered,altered,horizon=12,
                                           key_mode="pair_byte")
        for a,b in zip(original,replay):
            self.assertTrue(torch.equal(a[:,:8],b[:,:8]))
        with self.assertRaisesRegex(ValueError,"control"):
            retrieve_pair_successors(pairs,seen,key_mode="future")
        with self.assertRaisesRegex(ValueError,"Query byte map"):
            retrieve_pair_successors(pairs,seen,key_mode="pair_byte",
                                      query_byte_map=torch.arange(256))
        # Degenerate single-position prefix has zero historical matches.
        one = retrieve_pair_successors(pairs[:,:1],seen[:,:1],key_mode="pair_byte")
        self.assertEqual(one[0].abs().sum().item(),0)

    def test_zero_init_exact_parity_and_probability_mass(self):
        torch.manual_seed(591)
        seen,pairs = self.sample()
        logits = torch.randn(1,seen.shape[1],257)*.2
        trust = PairSuccessorTrust(conditional_content=True)
        with torch.no_grad():
            trust.weight.copy_(torch.tensor([.5,.2,.0,-.1]))
            trust.content_weight.copy_(torch.tensor([.5,.3,.0,.0]))
            before,_ = trust(logits,pairs,seen)
            trust.enable_joint_content()
            unchanged,_ = trust(logits,pairs,seen)
            self.assertTrue(torch.equal(before,unchanged))
            trust.joint_weight.copy_(torch.tensor([1.3,.4,.1,0.]))
        after,stats = trust(logits,pairs,seen)
        self.assertGreater(stats["joint_coverage"].sum().item(),0)
        self.assertFalse(torch.equal(before,after))
        for protected in (32,256):
            self.assertTrue(torch.equal(before[...,protected],after[...,protected]))
            self.assertTrue(torch.allclose(before.softmax(-1)[...,protected],
                                           after.softmax(-1)[...,protected],atol=1e-7))
        mask = torch.ones(257,dtype=torch.bool)
        mask[[32,256]] = False
        self.assertTrue(torch.allclose(torch.logsumexp(before[...,mask],-1),
                                       torch.logsumexp(after[...,mask],-1),
                                       atol=1e-6))
        target = seen.roll(-1,dims=1)[:,:-1]
        loss = F.cross_entropy(after[:,:-1].reshape(-1,257),target.reshape(-1))
        loss.backward()
        self.assertGreater(trust.joint_weight.grad.abs().sum().item(),0)
        with self.assertRaisesRegex(ValueError,"already enabled"):
            trust.enable_joint_content()

    def test_arm_interventions_restore_exactly(self):
        torch.manual_seed(592)
        seen,pairs = self.sample()
        logits = torch.randn(1,seen.shape[1],257)*.2
        trust = PairSuccessorTrust(conditional_content=True,joint_content=True)
        with torch.no_grad():
            trust.weight.copy_(torch.tensor([.5,.2,0.,0.]))
            trust.content_weight.copy_(torch.tensor([.4,.1,0.,0.]))
            trust.joint_weight.copy_(torch.tensor([2.,.8,0.,0.]))
        before = trust(logits,pairs,seen)[0]
        variants = {}
        for arm in ("neutral","wrong_address","wrong_successor","wrong_byte"):
            with memory_arm(trust,arm,torch.arange(32640).roll(137)):
                variants[arm] = trust(logits,pairs,seen)[0]
            self.assertTrue(torch.equal(before,trust(logits,pairs,seen)[0]))
        self.assertFalse(torch.equal(before,variants["neutral"]))
        self.assertFalse(torch.equal(before,variants["wrong_address"]))
        self.assertFalse(torch.equal(before,variants["wrong_successor"]))
        self.assertFalse(torch.equal(before,variants["wrong_byte"]))
        self.assertIsNone(trust.joint_query_map)
        self.assertIsNone(trust.joint_query_byte_map)

    def test_migrate_train_reload_same_model_and_adam(self):
        torch.manual_seed(593)
        cfg = Config(dim=16,layers=2,heads=2,pairs=4,topk_pairs=1,
                     expert_hidden=16,context=16,moe_every=2,symbiotic_rank=8,
                     micro_meta=True,simplex_transfer=True,dispatch_mode="gather",
                     balance_weight=0.)
        model = MirrorLM(cfg)
        opt = torch.optim.AdamW(model.parameters(),lr=.001)
        seen,pairs = self.sample()
        _,_,out = model(seen,seen)
        out["loss"].backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        model.eval()
        model.enable_pair_successor(opt)
        model.enable_conditional_pair_successor(opt)
        with torch.no_grad():
            model.pair_successor.weight.copy_(torch.tensor([.5,.2,0.,0.]))
            model.pair_successor.content_weight.copy_(torch.tensor([.4,.1,0.,0.]))
            inherited_logits = model(seen)[0]
        parent = {"model":deepcopy(model.state_dict()),
                  "optimizer":deepcopy(opt.state_dict())}
        with self.assertRaisesRegex(ValueError,"requires"):
            PairSuccessorTrust(joint_content=True)
        model.enable_joint_pair_successor(opt)
        self.assertTrue(model.cfg.pair_successor_joint)
        with torch.no_grad():
            self.assertTrue(torch.equal(inherited_logits,model(seen)[0]))
        for name,param in model.named_parameters():
            param.requires_grad_(name=="pair_successor.joint_weight")
        raw = model.head(model(seen)[1]).detach()
        target = seen.roll(-1,dims=1)
        records = [[(raw,raw,raw,pairs,seen,(target,target,target))]]
        result = train_only_gate(model.pair_successor,opt,records,2,.02,593)
        self.assertEqual(result["updates"],2)
        self.assertTrue(audit_inheritance(parent,model,opt,False,False,True)["pass"])
        recovered = MirrorLM(Config(**asdict(model.cfg))).eval()
        recovered.load_state_dict(model.state_dict(),strict=True)
        restored_opt = torch.optim.AdamW(recovered.parameters())
        restored_opt.load_state_dict(opt.state_dict())
        self.assertEqual(len(restored_opt.param_groups),1)
        with torch.no_grad():
            self.assertTrue(torch.equal(model(seen)[0],recovered(seen)[0]))
        with self.assertRaisesRegex(ValueError,"already enabled"):
            model.enable_joint_pair_successor(opt)


if __name__=="__main__":
    unittest.main()
