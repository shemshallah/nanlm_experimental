"""R7.35 causal sparse writer, apex-only routing, and optimizer continuity."""
import copy
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

from fa1_r723_diamond import CombinatorialDiamond
from fa1_r730_factorized_geometry import FactorizedContextualGeometry
from fa1_r735_history_values import (ROUND, FarHistoryValueWriter,
                                      attach_history_values, open_new_causal_gates,
                                      different_book_writer_values)


class HistoryValueTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(735)
        torch.set_num_threads(2)

    def test_actual_values_and_causal_eligibility(self):
        writer=FarHistoryValueWriter()
        keys=torch.randn(2,8,32)
        context=torch.randn(2,32)
        far=torch.randn(2,4,32)
        mask=torch.tensor([[True,False,True,False],[False,False,False,False]])
        first=writer(keys,context,far,mask)
        swapped=far.clone()
        swapped[:,0] += 4
        changed=writer(keys,context,swapped,mask)
        self.assertGreater(float((first[0]-changed[0]).abs().sum()),0)
        self.assertTrue(torch.equal(first[1],torch.zeros_like(first[1])))
        # A masked token must not influence any value, even if changed greatly.
        swapped=far.clone()
        swapped[:,1] += 1000
        self.assertTrue(torch.equal(first,writer(keys,context,swapped,mask)))
        self.assertEqual(tuple(first.shape),(2,8,32))

    @staticmethod
    def diamond_scene():
        class Trunk:
            E=nn.Parameter(torch.randn(256,96))
            H=nn.Parameter(torch.randn(256,96))
            P=nn.Parameter(torch.randn(32640,64))
            M=nn.Parameter(torch.randn(32640,64))
        model=CombinatorialDiamond()
        geo=FactorizedContextualGeometry()
        geo.register_parameter('context_read_gain',nn.Parameter(torch.zeros(())))
        model.geometry=geo
        batch=2
        features=(torch.randn(batch,96),torch.randint(0,256,(batch,32)),
                  torch.randint(0,256,(batch,128)),torch.randint(0,256,(batch,8)),
                  torch.stack([torch.randperm(256)[:8] for _ in range(batch)]),Trunk())
        modal=torch.randint(0,256,(batch,12))
        mask=torch.ones(batch,12,dtype=torch.bool)
        return model,features,{'modal_ids':modal,'modal_mask':mask}

    def test_zero_gate_identity_and_apex_only_route(self):
        model,inputs,kwargs=self.diamond_scene()
        with torch.no_grad():
            old=model(*inputs,**kwargs,return_trace=True)
        model.geometry.history_writer=FarHistoryValueWriter()
        with torch.no_grad():
            inherited=model(*inputs,**kwargs,return_trace=True)
        for a,b in zip(old[:3],inherited[:3]):
            self.assertTrue(torch.equal(a,b))
        with torch.no_grad():
            model.geometry.history_writer.gain.fill_(.5)
            changed=model(*inputs,**kwargs,return_trace=True)
            baseline_no_apex=model(*inputs,**kwargs,without_apex=True)
            model.geometry.history_writer.gain.zero_()
            inherited_no_apex=model(*inputs,**kwargs,without_apex=True)
        self.assertGreater(float((changed[0]-old[0]).abs().sum()),0)
        self.assertTrue(torch.equal(baseline_no_apex[0],inherited_no_apex[0]))
        self.assertTrue(torch.equal(baseline_no_apex[1],inherited_no_apex[1]))
        # The original unmodified triangle->pair route survives unchanged.
        with torch.no_grad():
            model.geometry.history_writer.gain.fill_(.5)
            with_pair=model(*inputs,**kwargs)
            no_pair=model(*inputs,**kwargs,without_pair=True)
        self.assertGreater(float((with_pair[0]-no_pair[0]).abs().sum()),0)

    def test_donor_changes_only_history_values_not_query(self):
        model,inputs,kwargs=self.diamond_scene()
        model.geometry.history_writer=FarHistoryValueWriter()
        torch.manual_seed(7352)
        kwargs['modal_ids'][:,:]=torch.randint(0,256,(2,12))
        with torch.no_grad():
            model.geometry.history_writer.gain.fill_(.25)
        data={'memory_modal_ids':kwargs['modal_ids'],
              'memory_modal_mask':kwargs['modal_mask'],
              'book_ids':torch.tensor([0,1]),'target':torch.tensor([1,2])}
        swapped,donors=different_book_writer_values(data)
        self.assertTrue(torch.equal(swapped['memory_modal_ids'],data['memory_modal_ids']))
        self.assertTrue(torch.equal(donors,torch.tensor([1,0])))
        with torch.no_grad():
            true=model(*inputs,**kwargs)
            wrong=model(*inputs,**kwargs,
                        history_modal_ids=swapped['diamond_history_value_ids'],
                        history_modal_mask=swapped['diamond_history_value_mask'])
            model.geometry.history_writer.gain.zero_()
            disabled=model(*inputs,**kwargs)
            disabled_wrong=model(*inputs,**kwargs,
                                 history_modal_ids=swapped['diamond_history_value_ids'],
                                 history_modal_mask=swapped['diamond_history_value_mask'])
        self.assertGreater(float((true[0]-wrong[0]).abs().sum()),0)
        self.assertTrue(torch.equal(disabled[0],disabled_wrong[0]))
        with self.assertRaisesRegex(ValueError,'four causal modal'):
            model(*inputs,**kwargs,history_modal_ids=torch.zeros(2,3,dtype=torch.long))

    def test_sparse_gradient_and_checkpoint_roundtrip(self):
        model,inputs,kwargs=self.diamond_scene()
        model.geometry.history_writer=FarHistoryValueWriter()
        with torch.no_grad():
            model.geometry.history_writer.gain.fill_(.25)
        hidden,_,_=model(*inputs,**kwargs)
        hidden.square().mean().backward()
        w=model.geometry.history_writer
        for param in (w.gain,w.query.weight,w.key.weight,w.value.weight,w.write[0].weight):
            self.assertIsNotNone(param.grad)
            self.assertGreater(float(param.grad.abs().sum()),0)
        opt=torch.optim.AdamW(model.parameters(),lr=.0001)
        opt.step()
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'candidate.pt'
            torch.save({'model':model.state_dict(),'optimizer':opt.state_dict()},path)
            restored=copy.deepcopy(model)
            clone_opt=torch.optim.AdamW(restored.parameters(),lr=.0001)
            saved=torch.load(path,map_location='cpu',weights_only=True)
            restored.load_state_dict(saved['model'],strict=True)
            clone_opt.load_state_dict(saved['optimizer'])
            self.assertTrue(all(torch.equal(a,b) for a,b in
                                zip(model.state_dict().values(),restored.state_dict().values())))
            self.assertEqual(len(clone_opt.state),len(opt.state))

    def test_inplace_optimizer_extension_preserves_moments(self):
        class Wrapper(nn.Module):
            def __init__(self):
                super().__init__()
                self.diamond=nn.Module()
                self.diamond.geometry=FactorizedContextualGeometry()
        model=Wrapper()
        old=list(model.parameters())
        optimizer=type('HybridShell',(),{})()
        optimizer._name_by_id={id(p):n for n,p in model.named_parameters()}
        optimizer._group_names=[{'algorithm':'adamw','names':list(optimizer._name_by_id.values())}]
        optimizer._adamw=torch.optim.AdamW(old,lr=.001)
        optimizer._legacy_groups=[]
        optimizer._source_group_counts=[1]
        optimizer.param_groups=optimizer._adamw.param_groups
        old[0].grad=torch.ones_like(old[0]);optimizer._adamw.step()
        moments=copy.deepcopy(optimizer._adamw.state[old[0]])
        attach_history_values(model,optimizer)
        self.assertEqual(len(optimizer.param_groups),2)
        self.assertTrue(set(map(id,old)).issubset(set(map(id,model.parameters()))))
        for key,value in moments.items():
            self.assertTrue(torch.equal(value,optimizer._adamw.state[old[0]][key]))
        with self.assertRaisesRegex(ValueError,'already attached'):
            attach_history_values(model,optimizer)

    def test_warmup_opens_only_new_or_explicitly_resumed_gates(self):
        model,_,_=self.diamond_scene()
        model.geometry.history_writer=FarHistoryValueWriter()
        model.hierarchy=nn.Module()
        model.hierarchy.apex_feedback_gain=nn.Parameter(torch.zeros(()))
        class Wrapper:
            diamond=model
        fresh=open_new_causal_gates(Wrapper,'FA1_R7_32_CALIBRATED_DIAMOND_V1')
        self.assertEqual(fresh['writer_before'],0.)
        self.assertGreater(fresh['writer_after'],0.)
        self.assertGreater(fresh['apex_after'],0.)
        with torch.no_grad():
            model.geometry.history_writer.gain.fill_(-.0004)
            model.hierarchy.apex_feedback_gain.fill_(.0005)
        resumed=open_new_causal_gates(Wrapper,ROUND)
        self.assertAlmostEqual(resumed['writer_after'],-.0004,places=7)
        self.assertAlmostEqual(resumed['apex_after'],.0005,places=7)
        overridden=open_new_causal_gates(Wrapper,ROUND,resume_override=True)
        self.assertGreater(overridden['writer_after'],0.)
        self.assertTrue(overridden['resumed_override'])


if __name__ == '__main__':
    unittest.main()
