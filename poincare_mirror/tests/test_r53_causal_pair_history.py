"""R53 exact-address history: strict causality, key controls, gradient, parity."""
import sys
from pathlib import Path
import unittest

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parents[1] / 'scripts'))
from causal_pair_history import retrieve_past_pair_values
from poincare_frontier import SymbioticGeometry
from scripts.probe_r53_pair_history import pair_history_intervention


class PairHistoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    @staticmethod
    def sample():
        # Two active edges, repeated identity 7 at t=2 and t=3.
        ids = torch.tensor([[[7,8],[9,10],[7,11],[7,8]]],dtype=torch.long)
        val = torch.tensor([[[[1.,2.],[3.,4.]], [[5.,6.],[7.,8.]],
                             [[9.,10.],[11.,12.]], [[13.,14.],[15.,16.]]]])
        return ids,val

    def test_prior_exact_match_and_no_current_or_future_value(self):
        ids,val=self.sample()
        read,hits=retrieve_past_pair_values(ids,val,16)
        self.assertTrue(torch.equal(read[0,0],torch.zeros(2)))
        self.assertTrue(torch.equal(read[0,1],torch.zeros(2)))
        self.assertTrue(torch.equal(read[0,2],torch.tensor([1.,2.])))
        self.assertTrue(torch.allclose(read[0,3],torch.tensor([4.,5.])))
        self.assertEqual(hits[0].tolist(),[0.,0.,1.,2.])

    def test_right_context_mutation_does_not_change_past_reads(self):
        ids,val=self.sample()
        old,_=retrieve_past_pair_values(ids,val)
        newer=val.clone(); newer[:,3].fill_(99999.)
        now,_=retrieve_past_pair_values(ids,newer)
        self.assertTrue(torch.equal(old,now))
        corrupt=ids.clone(); corrupt[:,3].fill_(42)
        changed,_=retrieve_past_pair_values(corrupt,val)
        self.assertTrue(torch.equal(old[:,:3],changed[:,:3]))

    def test_horizon_and_query_only_negative_control(self):
        ids,val=self.sample()
        one,_=retrieve_past_pair_values(ids,val,1)
        self.assertTrue(torch.equal(one[0,2],torch.zeros(2)))
        self.assertTrue(torch.equal(one[0,3],torch.tensor([9.,10.])))
        remap=torch.arange(32640,dtype=torch.long).roll(100)
        wrong,_=retrieve_past_pair_values(ids,val,16,remap)
        self.assertTrue(torch.equal(wrong,torch.zeros_like(wrong)))

    def test_gradients_flow_to_prior_values(self):
        ids,val=self.sample()
        val.requires_grad_()
        read,_=retrieve_past_pair_values(ids,val)
        read[0,2].sum().backward()
        self.assertGreater(val.grad[0,0,0].abs().sum().item(),0)
        self.assertEqual(val.grad[0,2].abs().sum().item(),0)
        self.assertEqual(val.grad[0,3].abs().sum().item(),0)

    def test_invalid_shapes_and_horizon_rejected(self):
        ids,val=self.sample()
        with self.assertRaises(ValueError):
            retrieve_past_pair_values(ids.float(),val)
        with self.assertRaises(ValueError):
            retrieve_past_pair_values(ids,val,horizon=0)
        with self.assertRaises(ValueError):
            retrieve_past_pair_values(ids,val[:,:,:1])

    def test_intervention_restores_original_state_after_exception(self):
        class Fake:
            simplex=object()
            pair_history_gain=0.0
            pair_history_query_map=None
            def modules(self):
                return [self]
        fake=Fake()
        remap=torch.arange(32640,dtype=torch.long).roll(1)
        with self.assertRaisesRegex(RuntimeError,'forced'):
            with pair_history_intervention(fake,.1,remap):
                self.assertEqual(fake.pair_history_gain,.1)
                self.assertIs(fake.pair_history_query_map,remap)
                raise RuntimeError('forced')
        self.assertEqual(fake.pair_history_gain,0.)
        self.assertIsNone(fake.pair_history_query_map)

    def test_existing_model_logits_and_tensors_equal_with_zero_gain(self):
        torch.manual_seed(42)
        model=SymbioticGeometry(16,8,micro_meta=False,simplex_transfer=True).eval()
        tokens=torch.randn(1,6,16)
        expert=torch.randn_like(tokens)
        sign=torch.sigmoid(torch.randn(1,6,1))
        before={key:v.clone() for key,v in model.state_dict().items()}
        with torch.no_grad():
            original=model(tokens,expert,sign)
            model.pair_history_gain=0.0
            repeated=model(tokens,expert,sign)
            self.assertTrue(torch.equal(original,repeated))
            model.pair_history_gain=0.05
            influenced=model(tokens,expert,sign)
            self.assertTrue(torch.isfinite(influenced).all())
            self.assertGreater((influenced-original).abs().max().item(),0)
            model.pair_history_gain=0.0
            restored=model(tokens,expert,sign)
            self.assertTrue(torch.equal(original,restored))
        for key,value in before.items():
            self.assertTrue(torch.equal(value,model.state_dict()[key]),key)


if __name__=='__main__':
    unittest.main()
