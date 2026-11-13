"""R7.30 source tests: causality, identity, sparse routing and optimizer history."""
import copy
import unittest
from unittest.mock import patch

import torch
from torch import nn

from fa1_r724_geometry import OrientedSimplicialGeometry, TRIPLES
from fa1_r729_contextual_geometry import ContextualGeometry
from fa1_r730_factorized_geometry import (
    FACTOR_TOPK, FactorizedContextualGeometry, attach_factorized)


class FactorizedTests(unittest.TestCase):
    @staticmethod
    def scene():
        torch.manual_seed(730)
        geometry = FactorizedContextualGeometry()
        geometry.register_parameter('context_read_gain', nn.Parameter(torch.zeros(())))
        tri = torch.randn(3, 56, 32)
        vertices = torch.stack([torch.randperm(256)[:8] for _ in range(3)])
        tri_ids = torch.randint(0, TRIPLES, (3, 56))
        state = {'context_query':torch.randn(3, 32),
                 'face_curvature':torch.randn(3, 56), 'metrics':{}}
        return geometry, vertices, tri_ids, tri, state

    def test_identity_at_zero_gains(self):
        geometry,vertices,ids,tri,state=self.scene()
        with patch.object(OrientedSimplicialGeometry,'triangles_up',return_value=(tri,state)):
            got,_=geometry.triangles_up(vertices,ids,tri,state)
        self.assertTrue(torch.equal(tri,got))
        residual=geometry.decoder_residual(state,torch.randn(3,32),
                                           torch.randn(3,32),torch.randn(3,32))
        self.assertTrue(torch.equal(residual,torch.zeros_like(residual)))

    def test_sparse_attention_and_branch_gradients(self):
        geometry,vertices,ids,tri,state=self.scene()
        with torch.no_grad():
            geometry.factor_up_gain.fill_(.2)
            geometry.factor_decoder_gain.fill_(.35)
        with patch.object(OrientedSimplicialGeometry,'triangles_up',return_value=(tri,state)):
            got,_=geometry.triangles_up(vertices,ids,tri,state)
        touched=((got-tri).abs().sum(-1)>1.e-10).sum().item()
        self.assertGreater(touched,0)
        self.assertLessEqual(touched,3*FACTOR_TOPK)
        points=torch.randn(3,32)
        apex=torch.randn(3,32)
        query=torch.randn(3,32)
        correction=geometry.decoder_residual(state,points,apex,query)
        self.assertEqual(tuple(correction.shape),(3,96))
        (got.square().mean()+correction.square().mean()).backward()
        for parameter in (geometry.factor_key.weight,geometry.factor_value.weight,
                          geometry.factor_query.weight,geometry.factor_up_gain,
                          geometry.factor_decoder_gain,geometry.factor_readout[0].weight):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()),0.)

    def test_decoder_requires_both_downpath_and_apex(self):
        geometry,vertices,ids,tri,state=self.scene()
        with torch.no_grad():geometry.factor_decoder_gain.fill_(.35)
        with patch.object(OrientedSimplicialGeometry,'triangles_up',return_value=(tri,state)):
            geometry.triangles_up(vertices,ids,tri,state)
        point=torch.randn(3,32)
        apex=torch.randn(3,32)
        query=torch.randn(3,32)
        self.assertTrue(torch.equal(geometry.decoder_residual(
            state,torch.zeros_like(point),apex,query),torch.zeros(3,96)))
        self.assertTrue(torch.equal(geometry.decoder_residual(
            state,point,torch.zeros_like(apex),query),torch.zeros(3,96)))

    def test_attach_keeps_old_parameters_and_adam_moments(self):
        torch.manual_seed(731)
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.diamond=nn.Module()
                self.diamond.geometry=ContextualGeometry()
                self.diamond.geometry.register_parameter('context_read_gain',
                                                         nn.Parameter(torch.zeros(())))
        model=Model()
        name_by_id={id(p):n for n,p in model.named_parameters()}
        old_params=list(model.parameters())
        class Optimizer:
            pass
        opt=Optimizer()
        opt._name_by_id=dict(name_by_id)
        opt._group_names=[{'algorithm':'adamw','names':list(name_by_id.values())}]
        opt._adamw=torch.optim.AdamW(old_params,lr=.001)
        opt._legacy_groups=[]
        opt._source_group_counts=[1]
        opt.param_groups=opt._adamw.param_groups
        # A real Adam moment must not be invalidated when the group is added.
        old_params[0].grad=torch.ones_like(old_params[0])
        opt._adamw.step()
        saved=copy.deepcopy(opt._adamw.state[old_params[0]])
        attach_factorized(model,opt)
        self.assertEqual(len(opt.param_groups),2)
        self.assertEqual(len(opt._group_names[-1]['names']),10)
        registered={id(p) for p in model.parameters()}
        self.assertTrue(all(id(p) in registered for p in old_params))
        for key,value in saved.items():
            self.assertTrue(torch.equal(value,opt._adamw.state[old_params[0]][key]))
        with self.assertRaises(TypeError):
            attach_factorized(model,opt)


if __name__=='__main__':
    unittest.main()
