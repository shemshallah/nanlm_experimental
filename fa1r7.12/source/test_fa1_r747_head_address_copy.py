"""R7.47 exact S+/S- address, probability normalization and causal tests."""
import unittest

import torch
from torch import nn

from fa1_r745_geometric_value_read import GeometricWordBackoff
from fa1_r747_head_address_copy import (HeadAddressWordBackoff,
                                         head_address_features,
                                         exact_head_copy_disabled)


class HeadAddressTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(747)
        self.heads=torch.tensor([[0,1],[2,5],[5,2],[3,7],[6,9],[1,4],
                                 [0,2],[2,8],[0,0],[3,5],[5,3],[7,6]])
        self.active=torch.tensor([[0,2,5,1,3,4,6,7],
                                  [0,5,2,1,3,4,6,7]])

    def test_signed_pair_orientation_uses_exact_point_order(self):
        ids=torch.tensor([[1,2,8],[1,2,8]])
        value=head_address_features(ids,self.active,self.heads)
        self.assertEqual(value.shape,(2,3,5))
        self.assertEqual(value[0,0].tolist(),[1,1,1,1,0])
        self.assertEqual(value[0,1].tolist(),[1,1,1,0,1])
        self.assertEqual(value[1,0].tolist(),[1,1,1,0,1])
        self.assertEqual(value[1,1].tolist(),[1,1,1,1,0])
        # A repeated head is not an oriented pair; its two endpoints coincide.
        self.assertEqual(value[0,2].tolist(),[1,1,1,0,0])

    def _module(self):
        module=HeadAddressWordBackoff()
        module.geometric_query=nn.Linear(32,96,bias=False)
        module.geometric_gain=nn.Parameter(torch.tensor(.4))
        module.address_coefficients=nn.Parameter(torch.zeros(5))
        module.address_gain=nn.Parameter(torch.zeros(()))
        module.address_copy_gain=nn.Parameter(torch.zeros(()))
        module.address_route_bias=nn.Parameter(torch.tensor(-1.5))
        module.address_route_weight=nn.Parameter(torch.zeros(()))
        module.register_buffer('word_head_lookup',self.heads)
        return module

    def _data(self):
        return {
            'word_ids':torch.tensor([[1,2,3,5],[2,3,4,5]]),
            'diamond_backoff_recent_mask':torch.ones(2,4,dtype=torch.bool),
            'vertex_ids':self.active,
        }

    def _trace(self):
        return {'triangle_attention':torch.full((2,56),1/56),
                'trust_logit':torch.zeros(2),'apex_std':torch.tensor(0.),
                'descended_point_summary':torch.randn(2,32)}

    def test_zero_migration_is_bitwise_inherited(self):
        module=self._module()
        data,trace=self._data(),self._trace()
        target=torch.tensor([2,3])
        embedding=torch.randn(12,96)
        a0,q0,_=GeometricWordBackoff.forward(module,data,trace,target,embedding)
        a1,q1,_=module(data,trace,target,embedding)
        self.assertTrue(torch.equal(a0,a1))
        self.assertTrue(torch.equal(q0,q1))

    def test_closed_read_restores_inherited_forward(self):
        module=self._module()
        with torch.no_grad():
            module.address_gain.fill_(.4)
            module.address_copy_gain.fill_(.3)
            module.address_coefficients.copy_(torch.tensor([.4,-.4,.8,.9,-.9]))
        data,trace=self._data(),self._trace()
        target=torch.tensor([2,3])
        embedding=torch.randn(12,96)
        with exact_head_copy_disabled(type('Model',(),{'word_backoff':module})()):
            closed=module(data,trace,target,embedding)
        inherited=GeometricWordBackoff.forward(module,data,trace,target,embedding)
        self.assertTrue(torch.equal(closed[0],inherited[0]))
        self.assertTrue(torch.equal(closed[1],inherited[1]))
        self.assertGreater(float(module.address_gain),0)

    def test_prediction_distribution_normalized_and_target_free(self):
        module=self._module()
        with torch.no_grad():
            module.gain.fill_(.5)
            module.address_gain.fill_(.65)
            module.address_copy_gain.fill_(.45)
            module.address_coefficients.copy_(torch.tensor([.4,-.4,.8,.9,-.9]))
        data,trace=self._data(),self._trace()
        embedding=torch.randn(12,96)
        results=[]
        for y in range(12):
            alpha,q,telemetry=module(data,trace,torch.full((2,),y),embedding)
            self.assertTrue(bool(((alpha>=0)&(alpha<=.25)).all()))
            self.assertTrue(bool(((q>=0)&(q<=1)).all()))
            self.assertTrue(bool((telemetry['r747_head_attention'].sum(1)>0.99999).all()))
            results.append(q)
        probs=torch.stack(results,dim=-1)
        self.assertTrue(torch.allclose(probs.sum(1),torch.ones(2),atol=1.e-6))

    def test_foreign_addresses_change_signed_rail_read(self):
        module=self._module()
        with torch.no_grad():
            module.address_gain.fill_(.8)
            module.address_coefficients.copy_(torch.tensor([0.,0.,0.,2.,-2.]))
        data,trace=self._data(),self._trace()
        embedding=torch.randn(12,96)
        target=torch.tensor([1,1])
        _,q,_=module(data,trace,target,embedding)
        swapped=dict(data,vertex_ids=data['vertex_ids'].flip(0))
        _,other,_=module(swapped,trace,target,embedding)
        self.assertGreater(float((q-other).abs().sum()),1.e-5)


if __name__=='__main__':
    unittest.main()
