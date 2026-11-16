"""R7.44 causal normalization, zero-fallback, gradient and donor controls."""
import unittest

import torch

from fa1_r744_causal_word_backoff import CausalWordBackoff


class CausalWordBackoffTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(744)
        torch.set_num_threads(2)
        self.module = CausalWordBackoff()
        self.data = {
            'word_ids':torch.tensor([[1,3,4,3],[1,1,1,1]]),
            'diamond_backoff_recent_mask':torch.tensor([[False,True,True,True],
                                                       [False,False,False,False]]),
            'memory_modal_ids':torch.tensor([[4,3],[9,7]]),
            'memory_modal_mask':torch.tensor([[True,True],[False,False]]),
            'near_count':torch.tensor([[2.,1.]]),
        }
        self.trace={'triangle_attention':torch.full((2,3),1/3),
                    'trust_logit':torch.zeros(2),
                    'apex_std':torch.tensor(.5)}

    def test_zero_gate_and_no_history_are_exact_parent(self):
        alpha,q,info=self.module(self.data,self.trace,torch.tensor([3,9]))
        self.assertTrue(torch.equal(alpha,torch.zeros(2)))
        with torch.no_grad():self.module.gain.fill_(.4)
        alpha,q,info=self.module(self.data,self.trace,torch.tensor([3,9]))
        self.assertGreater(float(alpha[0]),0.)
        self.assertEqual(float(alpha[1]),0.)
        self.assertGreater(float(q[0]),0.)
        self.assertEqual(float(q[1]),0.)

    def test_route_never_uses_target_or_masked_modal_content(self):
        with torch.no_grad():self.module.gain.fill_(.5)
        alpha,q,_=self.module(self.data,self.trace,torch.tensor([3,9]))
        changed_alpha,other_q,_=self.module(self.data,self.trace,torch.tensor([7,4]))
        self.assertTrue(torch.equal(alpha,changed_alpha))
        self.assertNotEqual(float(q[0]),float(other_q[0]))
        donor=dict(self.data)
        donor['memory_modal_ids']=self.data['memory_modal_ids'].clone()
        donor['memory_modal_ids'][1,:]=torch.tensor([200,201])
        same_alpha,same_q,_=self.module(donor,self.trace,torch.tensor([3,9]))
        self.assertTrue(torch.equal(alpha,same_alpha))
        self.assertTrue(torch.equal(q,same_q))

    def test_probability_mass_is_normalized_and_bounded(self):
        with torch.no_grad():self.module.gain.fill_(10.)
        alphas=[];q=[]
        for target in range(7):
            a,p,_=self.module(self.data,self.trace,torch.tensor([target,target]))
            alphas.append(a)
            q.append(p)
        self.assertTrue(all(torch.equal(alphas[0],a) for a in alphas[1:]))
        self.assertAlmostEqual(float(torch.stack(q).sum(0)[0]),1.,places=6)
        self.assertLessEqual(float(alphas[0].max()),.125)

    def test_target_supervision_reaches_backoff_without_target_routing(self):
        parent=torch.tensor([4.,4.])
        alpha,q,_=self.module(self.data,self.trace,torch.tensor([3,9]))
        with torch.no_grad():self.module.gain.fill_(.3)
        alpha,q,_=self.module(self.data,self.trace,torch.tensor([3,9]))
        loss=-torch.logaddexp(torch.log1p(-alpha)-parent,
                             (alpha*q).clamp_min(1e-38).log()).mean()
        loss.backward()
        self.assertIsNotNone(self.module.gain.grad)
        self.assertTrue(torch.isfinite(self.module.gain.grad))
        self.assertNotEqual(float(self.module.gain.grad),0.)


if __name__=='__main__':
    unittest.main()
