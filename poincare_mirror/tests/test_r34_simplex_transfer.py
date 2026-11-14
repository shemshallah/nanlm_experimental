"""Bounded causal concept storage and exact oriented simplex transport."""
import copy
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from poincare_frontier import Config, MirrorLM, SparseSimplexTransfer, expand_architecture


def config(simplex=False):
    return Config(vocab_size=48, dim=32, heads=4, layers=2, pairs=4,
                  topk_pairs=2, expert_hidden=24, context=16,
                  moe_every=2, future_mode='rollout', symbiotic_rank=8,
                  micro_meta=True, simplex_transfer=simplex)


class TestSimplexTransfer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_boundary_chain_and_address_ranges(self):
        s = SparseSimplexTransfer(8)
        self.assertTrue(torch.equal(s.d2 @ s.d1, torch.zeros(56, 8)))
        self.assertTrue(torch.equal(s.d3 @ s.d2, torch.zeros(70, 28)))
        self.assertEqual(int(s.tetra_address(torch.tensor(0),torch.tensor(1),
                                             torch.tensor(2),torch.tensor(3))), 0)
        self.assertEqual(int(s.tetra_address(torch.tensor(252),torch.tensor(253),
                                             torch.tensor(254),torch.tensor(255))), 174792639)
        self.assertEqual(s.pair_shared.weight.shape, (32640,8))
        self.assertLess(s.tetra_table.num_embeddings, 174792640)

    def test_cold_migration_preserves_outputs_and_all_optimizer_moments(self):
        torch.manual_seed(71)
        cfg = config(False)
        old = MirrorLM(cfg)
        opt = torch.optim.AdamW(old.parameters(), lr=.001)
        ids = torch.randint(0,48,(1,11))
        old.loss_with_future(ids)['loss'].backward()
        opt.step()
        saved = {'model':copy.deepcopy(old.state_dict()),
                 'optimizer':copy.deepcopy(opt.state_dict())}
        new = MirrorLM(config(True))
        new_opt = torch.optim.AdamW(new.parameters(), lr=.001)
        expand_architecture(new,new_opt,saved,cfg)
        old.eval(); new.eval()
        with torch.no_grad():
            a,ha,_ = old(ids)
            b,hb,_ = new(ids)
            self.assertTrue(torch.equal(a,b))
            for x,y in zip(old.future_logits(ha),new.future_logits(hb)):
                self.assertTrue(torch.equal(x,y))
        self.assertEqual(len(opt.state),len(new_opt.state))

    def test_gradients_flow_into_exact_pair_and_hashed_tetra_storage(self):
        torch.manual_seed(79)
        model = MirrorLM(config(True))
        geom = model.blocks[-1].ff.symbiotic
        with torch.no_grad():
            geom.to_hidden.weight.normal_(0,.02)
            geom.simplex.gate.fill_(.1)
        ids = torch.randint(0,48,(1,11))
        model.loss_with_future(ids)['loss'].backward()
        for parameter in (geom.simplex.pair_shared.weight,
                          geom.simplex.pair_anti.weight,
                          geom.simplex.tetra_table.weight,
                          geom.simplex.apex_bridge.weight):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(),0.)

    def test_meta_to_tetra_feedback_is_strictly_causal(self):
        torch.manual_seed(83)
        model = MirrorLM(config(True)).eval()
        with torch.no_grad():
            geom = model.blocks[-1].ff.symbiotic
            geom.to_hidden.weight.normal_(0,.02)
            geom.simplex.gate.fill_(.1)
            x = torch.randint(0,48,(1,12))
            y = x.clone()
            y[:,6:] = torch.randint(0,48,y[:,6:].shape)
            a = model(x)[0][:,:6]
            b = model(y)[0][:,:6]
            self.assertTrue(torch.allclose(a,b,atol=3e-5,rtol=0))


if __name__ == '__main__':
    unittest.main()
