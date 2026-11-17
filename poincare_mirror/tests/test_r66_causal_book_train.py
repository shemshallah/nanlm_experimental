"""The stream-trained gate must match DEV and preserve the inherited model."""
from pathlib import Path
import sys
import unittest

import torch
from torch.nn import functional as F

root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
sys.path.insert(0,str(root/'scripts'))
from book_stream_eval import evaluate_causal_book
from poincare_frontier import Config,MirrorLM
from train_r66_causal_book import cache_causal_book,train_joint_only


class CausalTrainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    @staticmethod
    def model():
        torch.manual_seed(66)
        cfg=Config(dim=16,layers=2,heads=2,pairs=4,topk_pairs=1,
                   expert_hidden=16,context=16,moe_every=2,symbiotic_rank=8,
                   micro_meta=True,simplex_transfer=True,dispatch_mode='gather',
                   balance_weight=0.,pair_successor_copy=True,
                   pair_successor_conditional=True)
        model=MirrorLM(cfg).eval()
        opt=torch.optim.AdamW(model.parameters(),lr=.001)
        model.enable_joint_pair_successor(opt,key_mode='hierarchical')
        return model,opt

    @staticmethod
    def book():
        return torch.tensor(([65,32,66,67,68,32,65,66]*6)+[69,32,256],
                            dtype=torch.long)

    def test_unique_targets_and_exact_live_stream_ce(self):
        model,_=self.model()
        book=self.book()
        with torch.no_grad():
            model.pair_successor.joint_weight.copy_(torch.tensor([.18,.12,.18,-.76]))
        records=cache_causal_book(model,book,'TRAIN',context=16,
                                   chunk=4,warmup=8,positions=20,history=100)
        self.assertEqual(sum(len(r[-1]) for r in records),20)
        score=0.
        with torch.no_grad():
            for logits,pairs,ids,hist,cov,hits,lo,hi,target in records:
                out,_=model.pair_successor(logits,pairs,ids,
                                           joint_memory_override=(hist,cov,hits))
                score+=float(F.cross_entropy(out[0,lo:hi],target,reduction='sum'))
        live=evaluate_causal_book(model,book,'DEV',context=16,chunk=4,
                                  history=100,score_start=8,score_positions=20)
        self.assertAlmostEqual(score/20,live['exact']['ce_1'],places=6)

    def test_future_suffix_does_not_change_features(self):
        model,_=self.model()
        book=self.book()
        modified=book.clone()
        modified[31:40]=90
        kwargs=dict(context=16,chunk=4,warmup=8,positions=16,history=100)
        a=cache_causal_book(model,book,'A',**kwargs)
        b=cache_causal_book(model,modified,'B',**kwargs)
        for first,second in zip(a,b):
            for j in list(range(6))+[-1]:
                self.assertTrue(torch.equal(first[j],second[j]))

    def test_only_new_trust_weight_changes(self):
        model,opt=self.model()
        original={n:t.clone() for n,t in model.state_dict().items()
                  if n!='pair_successor.joint_weight'}
        records=cache_causal_book(model,self.book(),'TRAIN',context=16,
                                   chunk=4,warmup=8,positions=20,history=100)
        stats=train_joint_only(model,opt,records,updates=8,lr=.005)
        self.assertEqual(stats['updates'],8)
        for n,t in original.items():
            self.assertTrue(torch.equal(t,model.state_dict()[n]),n)


if __name__=='__main__':
    unittest.main()
