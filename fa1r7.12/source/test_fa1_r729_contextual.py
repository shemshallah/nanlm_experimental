"""Fast, source-only context-addressed geometry tests (no private weights)."""
import unittest
from unittest.mock import patch

import torch
from torch import nn

from fa1_r724_geometry import OrientedSimplicialGeometry,TRIPLES
from fa1_r729_contextual_geometry import ContextualGeometry,READ_TOPK


class ContextMemoryTests(unittest.TestCase):
    @staticmethod
    def scene():
        torch.manual_seed(729)
        model=ContextualGeometry()
        model.register_parameter('context_read_gain',nn.Parameter(torch.zeros(())))
        tri=torch.randn(3,56,32)
        state={'context_query':torch.randn(3,32),
               'face_curvature':torch.randn(3,56),
               'metrics':{}}
        ids=torch.randint(0,TRIPLES,(3,56))
        return model,tri,state,ids

    def test_zero_gain_exact_parity(self):
        model,tri,state,ids=self.scene()
        with patch.object(OrientedSimplicialGeometry,'triangles_up',return_value=(tri,state)):
            actual,_=model.triangles_up(None,ids,tri,state)
        self.assertTrue(torch.equal(tri,actual))

    def test_learned_read_enters_triangles_and_has_gradient(self):
        model,tri,state,ids=self.scene()
        with torch.no_grad():model.context_read_gain.fill_(.2)
        with patch.object(OrientedSimplicialGeometry,'triangles_up',return_value=(tri,state)):
            actual,trace=model.triangles_up(None,ids,tri,state)
        changed=(actual-tri).abs().sum(-1)>1.e-10
        self.assertLessEqual(int(changed.sum()),3*READ_TOPK)
        self.assertGreater(int(changed.sum()),0)
        self.assertIn('context_read_entropy',trace['metrics'])
        actual.square().mean().backward()
        self.assertTrue(torch.isfinite(model.context_read_gain.grad))
        self.assertNotEqual(float(model.context_read_gain.grad),0.)


if __name__=='__main__':
    unittest.main()
