"""R7.40 zero-drift normalization, optimizer continuity and causal masking."""
import copy
import unittest

import torch

from fa1_r739_signed_rails import SignedRailCross, attach_signed_cross_rail
from fa1_r740_rail_calibration import (attach_rail_calibration,
                                       calibrated_rails_disabled)
from fa1_r728_geometric_memory import candidate_priority
import test_fa1_r739_signed_rails as prior_fixture


class CalibratedRailsTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(740)
        torch.set_num_threads(2)

    def scene(self):
        model,optimizer,inputs,kwargs=prior_fixture.SignedRailsTests().scene()
        attach_signed_cross_rail(model,optimizer)
        return model,optimizer,inputs,kwargs

    def test_exact_migration_and_optimizer_moments(self):
        model,optimizer,inputs,kwargs=self.scene()
        first=next(model.parameters())
        first.grad=torch.ones_like(first)
        optimizer._adamw.step()
        moments=copy.deepcopy(optimizer._adamw.state[first])
        previous={id(p) for p in model.parameters()}
        with torch.no_grad():
            baseline=model.diamond(*inputs,**kwargs,return_trace=True)
        attach_rail_calibration(model,optimizer)
        with torch.no_grad():
            migrated=model.diamond(*inputs,**kwargs,return_trace=True)
        self.assertTrue(all(torch.equal(x,y) for x,y in zip(baseline[:3],migrated[:3])))
        self.assertEqual(len(optimizer.param_groups),6)
        self.assertEqual(optimizer._source_group_counts,[6])
        self.assertTrue(previous.issubset({id(p) for p in model.parameters()}))
        self.assertEqual(len(set(id(p) for g in optimizer.param_groups for p in g['params'])),
                         len(list(model.parameters())))
        for key in moments:
            self.assertTrue(torch.equal(moments[key],optimizer._adamw.state[first][key]))
        clone=copy.deepcopy(model)
        clone.load_state_dict(model.state_dict(),strict=True)
        with torch.no_grad():
            restored=clone.diamond(*inputs,**kwargs,return_trace=True)
        self.assertTrue(all(torch.equal(x,y) for x,y in zip(migrated[:3],restored[:3])))
        with self.assertRaisesRegex(ValueError,'already attached'):
            attach_rail_calibration(model,optimizer)

    def test_scale_recovers_suppressed_cross_signal_without_mask_leak(self):
        model,optimizer,inputs,kwargs=self.scene()
        attach_rail_calibration(model,optimizer)
        rail=model.diamond.geometry.rail_cross
        far=torch.randn(2,4,32)
        mask=torch.tensor([[True,False,True,False],[False,False,False,False]])
        plus=.017*torch.randn(2,8)
        minus=.017*torch.randn(2,8)
        with torch.no_grad():
            rail.gain.fill_(.08)
            inherited=rail(far,mask,plus,minus)
            rail.calibration_gain.fill_(.3)
            calibrated=rail(far,mask,plus,minus)
            self.assertGreater(float((calibrated[0]-inherited[0]).square().mean().sqrt()),
                               10*float(inherited[0].square().mean().sqrt()))
            changed=far.clone()
            changed[0,1]+=1e4
            changed[1]+=1e4
            self.assertTrue(torch.equal(calibrated,rail(changed,mask,plus,minus)))
            self.assertTrue(torch.equal(calibrated[1],torch.zeros_like(calibrated[1])))

    def test_live_word_byte_gradients_and_ablation_restoration(self):
        model,optimizer,inputs,kwargs=self.scene()
        attach_rail_calibration(model,optimizer)
        with torch.no_grad():
            model.diamond.geometry.history_writer.gain.fill_(.3)
            model.diamond.geometry.rail_cross.gain.fill_(.3)
            rail=model.diamond.geometry.rail_cross
            rail.calibration_gain.fill_(.4)
            active=model.diamond(*inputs,**kwargs)
            with calibrated_rails_disabled(model):
                off=model.diamond(*inputs,**kwargs)
        self.assertGreater(float((active[0]-off[0]).abs().max()),0.)
        self.assertGreater(float((active[1]-off[1]).abs().max()),0.)
        model.zero_grad(set_to_none=True)
        hidden,byte_logits,_,trace=model.diamond(*inputs,**kwargs,return_trace=True)
        labels=torch.tensor([[97,98,256],[99,100,256]])
        word_bytes=model.diamond.hierarchy.autoregressive(
            labels,trace['layered_byte_condition'])
        (hidden.square().mean()+byte_logits.square().mean()+word_bytes.square().mean()).backward()
        gain_grad=rail.calibration_gain.grad
        self.assertTrue(gain_grad is not None and torch.isfinite(gain_grad).all())
        self.assertGreater(float(gain_grad.abs().sum()),0.)
        with self.assertRaisesRegex(RuntimeError,'restore'):
            with calibrated_rails_disabled(model):
                raise RuntimeError('restore')
        self.assertAlmostEqual(float(rail.calibration_gain),.4,places=6)

    def test_zero_specificity_rejected_for_promotion(self):
        ref={'full_word_nll':5.,'conditional_spelling_bits_per_byte':5.5}
        candidate={'full_word_nll':4.9,'conditional_spelling_bits_per_byte':5.4,
                   'geometry_word_gain':.1,'branch_geometry_gain':.1,
                   'correct_book_advantage_nats_per_word':.1,'apex_word_gain':.1,
                   'pair_word_gain':.1,'book_geometry_gain':{'book_bootstrap_ci95':[.01,.2]},
                   'book_correct_history':{'book_bootstrap_ci95':[.01,.2]},
                   'r740_calibration_word_gain':0.,
                   'r740_calibration_branch_gain':0.,
                   'r740_calibration_byte_gain_nats_per_byte':0.,
                   'r740_calibration_specificity_nats_per_word':0.,
                   'r740_calibration_byte_specificity_nats_per_byte':0.}
        self.assertEqual(candidate_priority(ref,candidate,.0001)[0],0)


if __name__=='__main__':
    unittest.main()
