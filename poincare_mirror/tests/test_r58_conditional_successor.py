"""R58 mass-conserving geometric content specialist and lineage regression."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import sys
import unittest

import torch
from torch.nn import functional as F

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from pair_successor_memory import PairSuccessorTrust
from poincare_frontier import Config, MirrorLM
from train_r55_pair_successor import memory_arm, audit_inheritance, train_only_gate


class R58MassPreservingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def sample(self):
        observed = torch.tensor([[65,32,66,65,32,66,65,32,66,65,32,66]])
        pairs = torch.where(observed[...,None] == 32,
                            torch.tensor([11,12]),torch.tensor([7,8]))
        logits = torch.randn(1,observed.shape[1],257)*.2
        logits[...,32] += .6
        return observed,pairs,logits

    def test_content_gate_conserves_space_eos_and_content_mass(self):
        torch.manual_seed(58)
        seen,pairs,logits = self.sample()
        trust = PairSuccessorTrust(conditional_content=True)
        with torch.no_grad():
            trust.weight[:] = torch.tensor([.2,.03,.0,-.1])
            neutral,_ = trust(logits,pairs,seen)
            trust.content_weight[:] = torch.tensor([1.0,.5,.0,.0])
            specialized,_ = trust(logits,pairs,seen)
        self.assertFalse(torch.equal(neutral,specialized))
        for protected in (32,256):
            self.assertTrue(torch.equal(neutral[...,protected],specialized[...,protected]))
            self.assertTrue(torch.allclose(neutral.softmax(-1)[...,protected],
                                           specialized.softmax(-1)[...,protected],atol=1e-7))
        mask = torch.ones(257,dtype=torch.bool)
        mask[[32,256]] = False
        self.assertTrue(torch.allclose(
            torch.logsumexp(neutral[...,mask],-1),
            torch.logsumexp(specialized[...,mask],-1),atol=1e-6))
        trust.content_weight.requires_grad_(True)
        corrected,_ = trust(logits,pairs,seen)
        target = seen.roll(-1,dims=1)[:,:-1]
        loss = F.cross_entropy(corrected[:,2:-1].reshape(-1,257),target[:,2:].reshape(-1))
        loss.backward()
        self.assertGreater(trust.content_weight.grad.abs().sum().item(),0)

    def test_causal_suffix_and_isolated_negative_controls(self):
        torch.manual_seed(581)
        seen,pairs,logits = self.sample()
        trust = PairSuccessorTrust(conditional_content=True)
        with torch.no_grad():
            trust.weight[:] = torch.tensor([.6,.2,.1,-.1])
            trust.content_weight[:] = torch.tensor([.8,.2,.0,.0])
        exact,_ = trust(logits,pairs,seen)
        changed = seen.clone()
        changed[:,8:] = 99
        suffix,_ = trust(logits,pairs,changed)
        self.assertTrue(torch.equal(exact[:,:8],suffix[:,:8]))
        baseline = trust(logits,pairs,seen)[0]
        with memory_arm(trust,'neutral',None):
            protected = trust(logits,pairs,seen)[0]
        with memory_arm(trust,'wrong_address',torch.arange(32640).roll(137)):
            shuffled = trust(logits,pairs,seen)[0]
        with memory_arm(trust,'wrong_successor',None):
            rotated = trust(logits,pairs,seen)[0]
        self.assertTrue(torch.equal(baseline,trust(logits,pairs,seen)[0]))
        self.assertFalse(torch.equal(baseline,protected))
        self.assertFalse(torch.equal(baseline,shuffled))
        self.assertFalse(torch.equal(baseline,rotated))
        self.assertIsNone(trust.query_map)  # inherited trust never scrambled
        self.assertIsNone(trust.conditional_query_map)

    def test_same_model_same_adam_migration(self):
        torch.manual_seed(59)
        cfg = Config(dim=16,layers=2,heads=2,pairs=4,topk_pairs=1,
                     expert_hidden=16,context=16,moe_every=2,symbiotic_rank=8,
                     micro_meta=True,simplex_transfer=True,dispatch_mode='gather',
                     balance_weight=0.)
        model = MirrorLM(cfg)
        optimizer = torch.optim.AdamW(model.parameters(),lr=.001)
        seq = torch.tensor([[65,32,66,67,65,32,66,67]])
        _,_,metrics = model(seq,seq)
        metrics['loss'].backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        model.eval()
        model.enable_pair_successor(optimizer)
        with torch.no_grad():
            model.pair_successor.weight[:] = torch.tensor([.2,.0,.0,-.04])
            before = model(seq)[0]
        parent = {'model':deepcopy(model.state_dict()),
                  'optimizer':deepcopy(optimizer.state_dict())}
        model.enable_conditional_pair_successor(optimizer)
        with torch.no_grad():
            self.assertTrue(torch.equal(before,model(seq)[0]))
        self.assertTrue(model.cfg.pair_successor_conditional)
        for name,param in model.named_parameters():
            param.requires_grad_(name == 'pair_successor.content_weight')
        logits,hidden,_ = model(seq)
        raw = model.head(hidden).detach()
        pair_ids = model._r54_geometry().r54_last_pairs.detach()
        train_cache = [[(raw,raw,raw,pair_ids,seq,(seq,seq,seq))]]
        train_log = train_only_gate(model.pair_successor,optimizer,train_cache,
                                    2,.02,59)
        self.assertEqual(train_log['updates'],2)
        self.assertTrue(audit_inheritance(parent,model,optimizer,False,True)['pass'])
        recovered = MirrorLM(Config(**asdict(model.cfg))).eval()
        recovered.load_state_dict(model.state_dict(),strict=True)
        recovery_optimizer = torch.optim.AdamW(recovered.parameters())
        recovery_optimizer.load_state_dict(optimizer.state_dict())
        with torch.no_grad():
            self.assertTrue(torch.equal(model(seq)[0],recovered(seq)[0]))
        with self.assertRaisesRegex(ValueError,'already enabled'):
            model.enable_conditional_pair_successor(optimizer)


if __name__=='__main__':
    unittest.main()
