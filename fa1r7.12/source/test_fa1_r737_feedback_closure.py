"""R7.37 second-pass geometry: exact migration and no decoder bypass."""
import copy
import unittest

import torch

from fa1_r737_feedback_closure import ApexReentry, attach_feedback_closure
from fa1_r736_deep_wiring import attach_deep_wiring, gradient_path_health
from fa1_r728_geometric_memory import apex_reentry_disabled
from test_fa1_r736_deep_wiring import DeepWiringTests


class ClosedFeedbackTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(737)
        torch.set_num_threads(2)

    @staticmethod
    def scene():
        model,optimizer,inputs,kw=DeepWiringTests.scene()
        attach_deep_wiring(model,optimizer)
        return model,optimizer,inputs,kw

    def test_second_pass_identity_optimizer_and_state_roundtrip(self):
        model,optimizer,inputs,kw=self.scene()
        old_ids={id(p) for p in model.parameters()}
        old_names=list(optimizer._name_by_id.values())
        old_state=copy.deepcopy(optimizer._adamw.state_dict())
        with torch.no_grad():
            before=model.diamond(*inputs,**kw,return_trace=True)
        attach_feedback_closure(model,optimizer)
        with torch.no_grad():
            after=model.diamond(*inputs,**kw,return_trace=True)
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(before[:3],after[:3])))
        self.assertEqual(len(optimizer.param_groups),3)
        self.assertEqual(optimizer._source_group_counts,[3])
        self.assertEqual(len(set(id(p) for g in optimizer.param_groups for p in g['params'])),
                         len(list(model.parameters())))
        self.assertTrue(set(old_names).issubset(set(optimizer._name_by_id.values())))
        self.assertTrue(old_ids.issubset({id(p) for p in model.parameters()}))
        state=optimizer._adamw.state_dict()
        self.assertEqual(old_state['state'],state['state'])
        clone=copy.deepcopy(model)
        clone.load_state_dict(model.state_dict(),strict=True)
        with torch.no_grad():
            restored=clone.diamond(*inputs,**kw)
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(after[:3],restored)))
        with self.assertRaisesRegex(ValueError,'already attached'):
            attach_feedback_closure(model,optimizer)

    def test_cycle_affects_word_and_byte_and_has_gradients(self):
        model,optimizer,inputs,kw=self.scene()
        attach_feedback_closure(model,optimizer)
        with torch.no_grad():
            closed=model.diamond(*inputs,**kw)
            model.diamond.hierarchy.apex_reentry_gain.fill_(.25)
            opened=model.diamond(*inputs,**kw)
        self.assertGreater(float((closed[0]-opened[0]).abs().max()),1.e-9)
        model.zero_grad(set_to_none=True)
        hidden,byte_logits,_,trace=model.diamond(*inputs,**kw,return_trace=True)
        labels=torch.tensor([[97,98,256],[99,100,256]])
        byte_ar=model.diamond.hierarchy.autoregressive(
            labels,trace['layered_byte_condition'])
        (hidden.square().mean()+byte_logits.square().mean()+byte_ar.square().mean()).backward()
        health=gradient_path_health(model)
        self.assertTrue(health['apex_reentry']['finite'])
        self.assertTrue(health['apex_reentry']['active'])
        self.assertTrue(health['apex_reentry_gate']['active'])
        with torch.no_grad():
            with apex_reentry_disabled(model):
                restored=model.diamond(*inputs,**kw)
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(closed,restored)))

    def test_ablation_cannot_be_bypassed_by_feedback(self):
        model,optimizer,inputs,kw=self.scene()
        attach_feedback_closure(model,optimizer)
        with torch.no_grad():
            point=torch.randn(2,32)
            loop=model.diamond.hierarchy.apex_reentry
            self.assertTrue(torch.equal(loop(point,torch.zeros_like(point)),torch.zeros_like(point)))
            self.assertTrue(torch.equal(loop(torch.zeros_like(point),point),torch.zeros_like(point)))
            model.diamond.hierarchy.apex_reentry_gain.fill_(.7)
            for ablate in ('without_apex','without_pair'):
                active=model.diamond(*inputs,**kw,**{ablate:True})
                with apex_reentry_disabled(model):
                    removed=model.diamond(*inputs,**kw,**{ablate:True})
                self.assertTrue(all(torch.equal(a,b) for a,b in zip(active,removed)),ablate)


if __name__=='__main__':
    unittest.main()
