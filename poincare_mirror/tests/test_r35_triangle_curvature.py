"""R35 hyperbolic triangle trigonometry, rotary storage, migration and causality."""
import copy
import math
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from poincare_frontier import (Config, MirrorLM, TriangleCurvatureMemory,
                              expand_architecture, count_parameters_exact)


def config(curvature=False):
    return Config(vocab_size=48, dim=32, heads=4, layers=2, pairs=4,
                  topk_pairs=2, expert_hidden=24, context=16,
                  moe_every=2, future_mode='rollout', symbiotic_rank=8,
                  micro_meta=True, simplex_transfer=True,
                  triangle_curvature=curvature)


class TestTriangleCurvature(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_gauss_bonnet_defect_and_rigid_rotation(self):
        # True hyperbolic triangles have positive angular defect, while
        # rotating the entire Poincare disk is a hyperbolic isometry.
        disk = torch.tensor([[[.15,.05],[-.4,.15],[.05,.55]]])
        faces = torch.tensor([[0,1,2]])
        a,phase,defect = TriangleCurvatureMemory.triangle_features(disk,faces)
        self.assertGreater(defect.item(), .01)
        self.assertLess(defect.item(), math.pi)
        rot = torch.tensor([[math.cos(.73),-math.sin(.73)],
                            [math.sin(.73), math.cos(.73)]])
        b,phase2,defect2 = TriangleCurvatureMemory.triangle_features(disk@rot.T,faces)
        self.assertTrue(torch.allclose(a,b,atol=2e-6,rtol=0))
        self.assertTrue(torch.allclose(phase,phase2,atol=2e-6,rtol=0))
        self.assertTrue(torch.allclose(defect,defect2,atol=2e-6,rtol=0))

    def test_complex_rotation_inverse_and_norm(self):
        x = torch.randn(3,5,8,dtype=torch.float64)
        theta = torch.randn(3,5,dtype=torch.float64)
        rotated = TriangleCurvatureMemory.rotate(x,theta)
        self.assertTrue(torch.allclose(rotated.square().sum(-1),x.square().sum(-1),atol=1e-12))
        self.assertTrue(torch.allclose(TriangleCurvatureMemory.rotate(rotated,-theta),x,atol=1e-12))

    def test_strict_prior_memory_and_batch_isolation(self):
        torch.manual_seed(17)
        memory = TriangleCurvatureMemory(8,48)
        face_indices = torch.combinations(torch.arange(8),r=3)
        points = torch.randn(2*4,8,8)
        triangles = torch.randn(2*4,56,8)
        weights = torch.ones(8,8)/8
        initial = memory(points,triangles,face_indices,weights,2,4)
        self.assertTrue(torch.equal(initial[:,0],torch.zeros_like(initial[:,0])))
        # Alter only the *future* of document 1 and all of document 2.
        points2=points.clone(); triangles2=triangles.clone()
        points2[2:]=torch.randn_like(points2[2:])
        triangles2[2:]=torch.randn_like(triangles2[2:])
        changed = memory(points2,triangles2,face_indices,weights,2,4)
        self.assertTrue(torch.allclose(initial[0,:2],changed[0,:2],atol=2e-6,rtol=0))
        self.assertTrue(torch.isfinite(changed).all())

    def test_exact_logit_migration_and_adam_states(self):
        torch.manual_seed(19)
        old_cfg=config(False)
        old=MirrorLM(old_cfg)
        opt=torch.optim.AdamW(old.parameters(),lr=.001)
        ids=torch.randint(0,48,(1,11))
        old.loss_with_future(ids)['loss'].backward()
        opt.step()
        saved={'model':copy.deepcopy(old.state_dict()),
               'optimizer':copy.deepcopy(opt.state_dict())}
        new=MirrorLM(config(True)); newopt=torch.optim.AdamW(new.parameters(),lr=.001)
        expand_architecture(new,newopt,saved,old_cfg)
        old.eval(); new.eval()
        with torch.no_grad():
            logits0,h0,_=old(ids)
            logits1,h1,_=new(ids)
            self.assertTrue(torch.equal(logits0,logits1))
            for a,b in zip(old.future_logits(h0),new.future_logits(h1)):
                self.assertTrue(torch.equal(a,b))
        self.assertEqual(len(opt.state),len(newopt.state))
        self.assertEqual(count_parameters_exact(new.cfg),sum(p.numel() for p in new.parameters()))

    def test_curvature_features_receive_supervised_gradients(self):
        torch.manual_seed(23)
        model=MirrorLM(config(True))
        curve=model.blocks[-1].ff.symbiotic.curvature
        sequence=torch.randint(0,48,(1,11))
        losses=model.loss_with_future(sequence)
        self.assertIn('curvature_aux_ce', losses)
        self.assertTrue(torch.isfinite(losses['loss']))
        losses['loss'].backward()
        for p in (curve.coordinates.weight,curve.angles.weight,curve.write.weight,
                  curve.read.weight,curve.aux_head.weight):
            self.assertIsNotNone(p.grad)
            self.assertTrue(torch.isfinite(p.grad).all())
            self.assertGreater(p.grad.abs().sum().item(),0.)
        self.assertIsNotNone(curve.gate.grad)

    def test_nonzero_curvature_path_cannot_read_future_bytes(self):
        torch.manual_seed(29)
        model=MirrorLM(config(True)).eval()
        with torch.no_grad():
            curve=model.blocks[-1].ff.symbiotic.curvature
            curve.gate.fill_(.2)
            model.blocks[-1].ff.symbiotic.to_hidden.weight.normal_(0,.02)
            first=torch.randint(0,48,(1,11))
            second=first.clone();second[:,6:]=torch.randint(0,48,second[:,6:].shape)
            a=model(first)[0][:,:6]
            b=model(second)[0][:,:6]
            self.assertTrue(torch.allclose(a,b,atol=3e-5,rtol=0))


if __name__=='__main__':
    unittest.main()
