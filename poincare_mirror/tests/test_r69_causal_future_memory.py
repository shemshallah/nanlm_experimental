"""R69 geometric future memory preserves causality and replay."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from book_successor_stream import CausalBookSuccessor,CausalLagBookSuccessor
import torch
from pair_successor_memory import PairSuccessorTrust
from poincare_frontier import Config,MirrorLM
from book_stream_eval import evaluate_causal_book


class FutureMemoryTests(unittest.TestCase):
    def test_periodic_future_retrieval(self):
        for lag,target in ((2,67),(3,65)):
            s=CausalLagBookSuccessor(lag=lag,key_mode='pair_byte',horizon=32)
            s.begin_book('A')
            rows=[s.step('A',x,[0]) for x in (65,66,67,65,66,67,65)]
            self.assertEqual(rows[3][0][target].item(),1.)
            self.assertEqual(rows[3][1],1.)

    def test_future_suffix_invariance(self):
        prefix=[65,66,67,65,66,67,65]
        for lag in (2,3):
            runs=[]
            for suffix in ((11,12,13),(98,99,100)):
                s=CausalLagBookSuccessor(lag=lag,key_mode='hierarchical',horizon=32)
                s.begin_book('book')
                runs.append([s.step('book',x,[0,1]) for x in prefix+list(suffix)])
            for a,b in zip(runs[0][:len(prefix)],runs[1][:len(prefix)]):
                self.assertTrue(a[0].equal(b[0]))
                self.assertEqual(a[1:],b[1:])

    def test_snapshot_book_boundary(self):
        a=CausalLagBookSuccessor(lag=3,horizon=32)
        a.begin_book('X')
        for x in (65,66,67,65,66,67): a.step('X',x,[4,5])
        b=CausalLagBookSuccessor(lag=3,horizon=32)
        b.load_state_dict(a.state_dict())
        for x in (65,66,67,65):
            left,right=a.step('X',x,[4,5]),b.step('X',x,[4,5])
            self.assertTrue(left[0].equal(right[0]))
            self.assertEqual(left[1:],right[1:])
        b.begin_book('Y')
        self.assertEqual(b.step('Y',65,[4,5])[1:],(0.,0.))

    def test_original_next_rail_parity(self):
        s=CausalBookSuccessor(horizon=8)
        s.begin_book('C')
        self.assertEqual(s.step('C',65,[0])[1:],(0.,0.))
        self.assertEqual(s.step('C',65,[0])[0][65].item(),1.)

    def test_future_zero_initialization_and_optimizer_append(self):
        torch.manual_seed(69)
        cfg=Config(dim=16,layers=2,heads=2,pairs=4,topk_pairs=1,
                   expert_hidden=16,context=16,moe_every=2,symbiotic_rank=8,
                   micro_meta=True,simplex_transfer=True,dispatch_mode='gather',
                   balance_weight=0.,pair_successor_copy=True,
                   pair_successor_conditional=True,pair_successor_joint=True)
        m=MirrorLM(cfg).eval()
        o=torch.optim.AdamW(m.parameters(),lr=.001)
        hidden=torch.randn(1,4,16)
        parent=m.future_logits(hidden)
        prior=len(o.param_groups[0]['params'])
        added=m.enable_future_pair_successor(o)
        self.assertEqual(tuple(added.shape),(2,4))
        self.assertEqual(len(o.param_groups[0]['params']),prior+1)
        hist=torch.zeros(1,4,257)
        hist[...,65]=1
        cov=torch.ones(1,4)
        hits=torch.full((1,4),8.)
        future=m.future_logits(hidden,future_memory_override=(hist,cov,hits)*2)
        self.assertTrue(torch.equal(parent[0],future[0]))
        self.assertTrue(torch.equal(parent[1],future[1]))

    def test_future_memory_updates_only_content_distribution(self):
        rail=PairSuccessorTrust(conditional_content=True,joint_content=True,
                                future_content=True)
        with torch.no_grad(): rail.future_weight[:,0]=2.
        base=(torch.zeros(1,3,257),torch.zeros(1,3,257))
        hist=torch.zeros(1,3,257)
        hist[...,65]=1
        cov=torch.ones(1,3)
        hits=torch.ones(1,3)
        updated=rail.future_forward(base,(hist,cov,hits)*2)
        for prior,after in zip(base,updated):
            self.assertGreater(after[...,65].mean().item(),0)
            self.assertTrue(torch.equal(prior[...,32],after[...,32]))
            self.assertTrue(torch.equal(prior[...,256],after[...,256]))
            mask=torch.ones(257,dtype=torch.bool)
            mask[[32,256]]=False
            self.assertTrue(torch.allclose(torch.logsumexp(after[...,mask],-1),
                                           torch.logsumexp(prior[...,mask],-1),atol=1e-5))

    def test_real_book_scorer_routes_future_memory_into_horizons(self):
        torch.set_num_threads(1)
        torch.manual_seed(31)
        cfg=Config(dim=16,layers=2,heads=2,pairs=4,topk_pairs=1,
                   expert_hidden=16,context=16,moe_every=2,symbiotic_rank=8,
                   micro_meta=True,simplex_transfer=True,dispatch_mode='gather',
                   balance_weight=0.,pair_successor_copy=True,
                   pair_successor_conditional=True,pair_successor_joint=True,
                   pair_successor_future=True)
        model=MirrorLM(cfg).eval()
        with torch.no_grad(): model.pair_successor.future_weight[:,0]=2.
        book=torch.tensor(([65,32,66,67]*26)+[256],dtype=torch.long)
        result=evaluate_causal_book(model,book,'R69_BOOK',context=16,
                                    chunk=4,history=80,key_mode='pair_byte',
                                    score_start=8,score_positions=64)
        self.assertNotEqual(result['exact']['ce_2'],result['neutral']['ce_2'])
        self.assertNotEqual(result['exact']['ce_3'],result['neutral']['ce_3'])
        self.assertEqual(result['exact']['positions_scored'],64)


if __name__=='__main__':
    unittest.main()
