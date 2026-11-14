"""Causal phase-selective Fourier memory and non-destructive R35 migration."""
import copy
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from poincare_frontier import (Config, MirrorLM, HolonomyAssociativeMemory,
                              expand_architecture, count_parameters_exact)


def cfg(use_holonomy=False):
    return Config(vocab_size=48,dim=32,heads=4,layers=2,pairs=4,
                  topk_pairs=2,expert_hidden=24,context=16,moe_every=2,
                  symbiotic_rank=8,micro_meta=True,simplex_transfer=True,
                  triangle_curvature=True,holonomy_retrieval=use_holonomy)


class TestHolonomyRetrieval(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_phase_addressed_retrieval_prefers_matching_history(self):
        memory=HolonomyAssociativeMemory(8,48).eval()
        with torch.no_grad():
            memory.key.weight.zero_()
            memory.query.weight.zero_()
            memory.value.weight.copy_(torch.eye(8))
            memory.output.weight.copy_(torch.eye(8))
            values=torch.zeros(1,4,8)
            values[0,0,0]=1.
            values[0,1,1]=1.
            phases=torch.tensor([[0.,.25,0.,.25]])
            read=memory(values,phases,1,4)
            self.assertTrue(torch.equal(read[:,0],torch.zeros_like(read[:,0])))
            # Both queries see the same two earlier writes; only phase changes.
            self.assertGreater(float(read[0,2,0]-read[0,2,1]), .2)
            self.assertGreater(float(read[0,3,1]-read[0,3,0]), .2)

    def test_causal_prefix_and_independent_batch_rows(self):
        torch.manual_seed(701)
        memory=HolonomyAssociativeMemory(8,48).eval()
        content=torch.randn(2,7,8)
        phase=torch.randn(2,7)*.06
        x=memory(content,phase,2,7)
        modified=content.clone();modified[0,4:]=torch.randn_like(modified[0,4:])
        modified[1]=torch.randn_like(modified[1])
        p2=phase.clone();p2[0,4:]=torch.randn_like(p2[0,4:]);p2[1]=torch.randn_like(p2[1])
        y=memory(modified,p2,2,7)
        self.assertTrue(torch.allclose(x[0,:4],y[0,:4],atol=2e-6,rtol=0))
        self.assertTrue(torch.isfinite(x).all())

    def test_adding_fourier_memory_preserves_r35_logits_and_adam(self):
        torch.manual_seed(713)
        old_cfg=cfg(False); old=MirrorLM(old_cfg)
        opt=torch.optim.AdamW(old.parameters(),lr=.001)
        seq=torch.randint(0,48,(1,11))
        old.loss_with_future(seq)['loss'].backward();opt.step()
        saved={'model':copy.deepcopy(old.state_dict()),
               'optimizer':copy.deepcopy(opt.state_dict())}
        new=MirrorLM(cfg(True));new_opt=torch.optim.AdamW(new.parameters(),lr=.001)
        expand_architecture(new,new_opt,saved,old_cfg)
        old.eval();new.eval()
        with torch.no_grad():
            a,h0,_=old(seq)
            b,h1,_=new(seq)
            self.assertTrue(torch.equal(a,b))
            for x,y in zip(old.future_logits(h0),new.future_logits(h1)):
                self.assertTrue(torch.equal(x,y))
        self.assertEqual(len(opt.state),len(new_opt.state))
        self.assertEqual(count_parameters_exact(new.cfg),sum(p.numel() for p in new.parameters()))

    def test_direct_supervision_trains_fourier_memory_weights(self):
        torch.manual_seed(727)
        model=MirrorLM(cfg(True))
        ids=torch.randint(0,48,(1,11))
        metrics=model.loss_with_future(ids)
        self.assertIn('curvature_aux_ce',metrics)
        self.assertIn('holonomy_aux_ce',metrics)
        metrics['loss'].backward()
        memory=model.blocks[-1].ff.symbiotic.curvature.holonomy
        for parameter in (memory.query.weight,memory.key.weight,memory.value.weight,
                          memory.output.weight,memory.aux_head.weight):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()),0.)
        self.assertIsNotNone(memory.gate.grad)

    def test_nonzero_holonomy_gate_cannot_read_future_bytes(self):
        torch.manual_seed(739)
        model=MirrorLM(cfg(True)).eval()
        with torch.no_grad():
            g=model.blocks[-1].ff.symbiotic
            g.to_hidden.weight.normal_(0,.03)
            g.curvature.holonomy.gate.fill_(.15)
            x=torch.randint(0,48,(1,12))
            y=x.clone();y[:,5:]=torch.randint(0,48,y[:,5:].shape)
            a=model(x)[0][:,:5]
            b=model(y)[0][:,:5]
            self.assertTrue(torch.allclose(a,b,atol=3e-5,rtol=0))

    def test_auxiliary_only_control_does_not_feed_predictions(self):
        torch.manual_seed(743)
        model=MirrorLM(cfg(True))
        g=model.blocks[-1].ff.symbiotic
        with torch.no_grad():
            g.to_hidden.weight.normal_(0,.03)
            g.curvature.holonomy.gate.fill_(.2)
        ids=torch.randint(0,48,(1,11))
        g.holonomy_feedback_enabled=False
        no_feedback=model(ids)[0].detach()
        g.holonomy_enabled=False
        disabled=model(ids)[0].detach()
        self.assertTrue(torch.equal(no_feedback,disabled))
        g.holonomy_enabled=True
        loss=model.loss_with_future(ids)['loss']
        loss.backward()
        head=g.curvature.holonomy.aux_head.weight
        self.assertIsNotNone(head.grad)
        self.assertGreater(float(head.grad.abs().sum()),0.)

    def test_512d_mixed_attention_and_geometric_backward(self):
        # Match R34's mixed old/new attention head widths while bounding
        # expert count and depth for a deterministic local CPU smoke test.
        torch.manual_seed(751)
        c=Config(vocab_size=48,dim=512,heads=36,layers=2,pairs=4,
                 topk_pairs=2,expert_hidden=32,context=8,moe_every=2,
                 legacy_attention_dim=192,legacy_attention_heads=16,
                 symbiotic_rank=64,micro_meta=True,simplex_transfer=True,
                 triangle_curvature=True,holonomy_retrieval=True)
        model=MirrorLM(c)
        ids=torch.randint(0,48,(1,9))
        logits,_,metrics=model(ids[:,:-3],ids[:,1:-2])
        self.assertEqual(tuple(logits.shape),(1,6,48))
        self.assertTrue(torch.isfinite(metrics['loss']))
        metrics['loss'].backward()
        curve=model.blocks[-1].ff.symbiotic.curvature
        self.assertTrue(torch.isfinite(curve.holonomy.query.weight.grad).all())


if __name__=='__main__':
    unittest.main()
