"""One-backbone geometry: addresses, exact cold migration, optimizer, causality."""
import copy
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from poincare_frontier import Config, MirrorLM, SymbioticGeometry, expand_architecture


class TestSymbioticGeometry(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    @staticmethod
    def config(rank=0):
        return Config(vocab_size=48,dim=32,heads=4,layers=2,pairs=4,
                      topk_pairs=2,expert_hidden=24,context=16,
                      moe_every=2,future_mode='rollout',symbiotic_rank=rank)

    def test_all_canonical_mirror_addresses_are_unique(self):
        addresses = []
        for a in range(256):
            for b in range(a+1,256):
                addresses.append(int(SymbioticGeometry.pair_address(torch.tensor(a),torch.tensor(b))))
        self.assertEqual(len(addresses),32640)
        self.assertEqual(set(addresses),set(range(32640)))
        self.assertEqual(torch.combinations(torch.arange(8),r=3).shape[0],56)
        self.assertEqual(torch.combinations(torch.arange(8),r=4).shape[0],70)

    def test_old_logits_and_all_adam_moments_survive_expansion(self):
        torch.manual_seed(74)
        oldcfg=self.config()
        old=MirrorLM(oldcfg)
        opt=torch.optim.AdamW(old.parameters(),lr=.001)
        seq=torch.randint(0,48,(1,11))
        old.loss_with_future(seq).get('loss').backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        saved={'model':copy.deepcopy(old.state_dict()),'optimizer':copy.deepcopy(opt.state_dict())}
        new=MirrorLM(self.config(8))
        newopt=torch.optim.AdamW(new.parameters(),lr=.001)
        expand_architecture(new,newopt,saved,oldcfg)
        old.eval(); new.eval()
        with torch.no_grad():
            x=seq[:,:-3]
            oldlog,oldh,_=old(x)
            newlog,newh,_=new(x)
            self.assertTrue(torch.equal(oldlog,newlog))
            self.assertTrue(torch.equal(oldh,newh))
            for a,b in zip(old.future_logits(oldh),new.future_logits(newh)):
                self.assertTrue(torch.equal(a,b))
        old_names=list(dict(old.named_parameters()))
        new_names=list(dict(new.named_parameters()))
        old_ids=saved['optimizer']['param_groups'][0]['params']
        new_ids=newopt.state_dict()['param_groups'][0]['params']
        oldstates=saved['optimizer']['state']
        newstates=newopt.state_dict()['state']
        byold=dict(zip(old_names,old_ids)); bynew=dict(zip(new_names,new_ids))
        self.assertEqual(len(oldstates),len(newstates))
        for name in old_names:
            old_id, new_id = byold[name], bynew[name]
            # AdamW has no moments for parameters that never received a gradient.
            # Preserve both populated states and that absent-state distinction.
            if old_id not in oldstates:
                self.assertNotIn(new_id, newstates, name)
                continue
            self.assertIn(new_id, newstates, name)
            self.assertEqual(set(oldstates[old_id]),set(newstates[new_id]),name)
            for key,val in oldstates[old_id].items():
                newval=newstates[new_id][key]
                if isinstance(val,torch.Tensor):
                    self.assertTrue(torch.equal(val,newval),(name,key))
                else:
                    self.assertEqual(val,newval)

        new.loss_with_future(seq)['loss'].backward()
        geom=new.blocks[-1].ff.symbiotic
        self.assertTrue(torch.isfinite(geom.to_hidden.weight.grad).all())
        self.assertGreater(geom.to_hidden.weight.grad.abs().sum().item(),0)
        newopt.step()
        newopt.zero_grad(set_to_none=True)
        new.loss_with_future(seq)['loss'].backward()
        self.assertIsNotNone(geom.point_values.grad)
        self.assertTrue(torch.isfinite(geom.point_values.grad).all())

    def test_nonzero_feedback_never_sees_future_tokens(self):
        torch.manual_seed(9)
        model=MirrorLM(self.config(8)).eval()
        with torch.no_grad():
            model.blocks[-1].ff.symbiotic.to_hidden.weight.normal_(0,.002)
            x=torch.randint(0,48,(1,12))
            y=x.clone()
            y[:,5:]=torch.randint(0,48,y[:,5:].shape)
            ax=model(x)[0][:,:5]
            ay=model(y)[0][:,:5]
        self.assertTrue(torch.allclose(ax,ay,atol=2e-5,rtol=0))


if __name__=='__main__':
    unittest.main()
