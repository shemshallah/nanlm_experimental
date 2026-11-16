"""R63: sequential FA2 book scorer with matched counterfactual histories."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from book_stream_eval import ARMS,METRICS,evaluate_causal_book,evaluate_disjoint_books
from poincare_frontier import Config,MirrorLM


class RealBookScorerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    @staticmethod
    def small_model():
        torch.manual_seed(630)
        cfg=Config(dim=16,layers=2,heads=2,pairs=4,topk_pairs=1,
                   expert_hidden=16,context=16,moe_every=2,symbiotic_rank=8,
                   micro_meta=True,simplex_transfer=True,dispatch_mode="gather",
                   balance_weight=0.,pair_successor_copy=True,
                   pair_successor_conditional=True,pair_successor_joint=True)
        return MirrorLM(cfg).eval()

    @staticmethod
    def book():
        return torch.tensor(([65,32,66,67,68,32,65,66]*5)+[69,32,256],
                            dtype=torch.long)

    def test_all_five_arms_and_metrics_exact_zero_initialized_parity(self):
        model=self.small_model()
        before={k:v.clone() for k,v in model.state_dict().items()}
        book=self.book()
        result=evaluate_causal_book(model,book,"dev-001",context=16,chunk=4,
                                    history=64,key_mode="hierarchical")
        self.assertEqual(tuple(result),ARMS)
        self.assertEqual(result["exact"]["positions_scored"],book.numel()-1)
        self.assertEqual(result["exact"]["raw_target_bytes"],book.numel()-2)
        for arm in ARMS:
            for metric in METRICS:
                self.assertIn(metric,result[arm])
                self.assertTrue(torch.isfinite(torch.tensor(result[arm][metric])))
                self.assertAlmostEqual(result[arm][metric],
                                       result["neutral"][metric],places=6)
            self.assertGreater(result[arm]["bits_per_raw_byte"],0)
            self.assertGreater(result[arm]["evaluated_tokens_per_second"],0)
        for name,weights in before.items():
            self.assertTrue(torch.equal(weights,model.state_dict()[name]))

    def test_nonzero_rail_replay_determinism_across_book_ids(self):
        model=self.small_model()
        with torch.no_grad():
            model.pair_successor.joint_weight.copy_(
                torch.tensor([1.0,.2,.0,.0]))
        book=self.book()
        a=evaluate_causal_book(model,book,"A",context=16,chunk=4,
                               history=64)
        b=evaluate_causal_book(model,book,"B",context=16,chunk=4,
                               history=64)
        for arm in ARMS:
            for metric in METRICS:
                self.assertAlmostEqual(a[arm][metric],b[arm][metric],places=7)
        # No histories are shared when the order of independent books flips.
        other=torch.where(book.eq(65),torch.tensor(66),book)
        books={"A":book,"B":other}
        left=evaluate_disjoint_books(model,books,context=16,chunk=4,history=64)
        right=evaluate_disjoint_books(model,dict(reversed(list(books.items()))),
                                      context=16,chunk=4,history=64)
        for key in books:
            for arm in ARMS:
                for metric in METRICS:
                    self.assertAlmostEqual(left[key][arm][metric],
                                           right[key][arm][metric],places=7)

    def test_shorter_chunks_preserve_predictions_for_same_positions(self):
        model=self.small_model()
        book=self.book()
        # The *neural* context differs only at block boundaries; avoid
        # falsely asserting bitwise parity for distinct left-context truncations.
        one=evaluate_causal_book(model,book,"A",context=16,chunk=4,
                                 history=64)
        two=evaluate_causal_book(model,book,"A",context=16,chunk=8,
                                 history=64)
        self.assertEqual(one["exact"]["positions_scored"],
                         two["exact"]["positions_scored"])
        self.assertEqual(one["exact"]["raw_target_bytes"],
                         two["exact"]["raw_target_bytes"])
        for result in (one,two):
            self.assertLess(result["exact"]["acc_1"],1.00001)
            self.assertLess(result["exact"]["top5_1"],1.00001)

    def test_invalid_context_and_missing_joint_rail_refused(self):
        model=self.small_model()
        book=self.book()
        with self.assertRaisesRegex(ValueError,"context"):
            evaluate_causal_book(model,book,"a",context=32,chunk=4)
        with self.assertRaisesRegex(ValueError,"context"):
            evaluate_causal_book(model,book,"a",context=16,chunk=17)
        with self.assertRaisesRegex(ValueError,"eval"):
            evaluate_causal_book(model.train(),book,"a",context=16)
        model.eval()
        model.pair_successor.joint_weight=None
        with self.assertRaisesRegex(ValueError,"joint"):
            evaluate_causal_book(model,book,"a",context=16)

    def test_metrics_denominator_counts_EOS_once(self):
        model=self.small_model()
        book=self.book()
        result=evaluate_causal_book(model,book,"A",context=16,chunk=4)
        target_bytes=int(book[1:].ne(256).sum())
        self.assertEqual(result["exact"]["raw_target_bytes"],target_bytes)
        # Next token CE is normalized over *all* targets incl EOS.
        self.assertAlmostEqual(
            result["exact"]["bits_per_raw_byte"],
            (result["exact"]["ce_1"]*(book.numel()-1)/
             (target_bytes*torch.log(torch.tensor(2.)).item())),
            places=5)


if __name__=="__main__":
    unittest.main()
