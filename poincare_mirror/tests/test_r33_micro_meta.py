"""Sparse 256-node microtransformer/triangle-cloud/apex-MoE continuation checks."""
import copy
import itertools
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from poincare_frontier import Config, MirrorLM, SymbioticGeometry, expand_architecture, widen_config


class TestMicroMeta(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    @staticmethod
    def cfg(rank=0, micro=False):
        return Config(vocab_size=48, dim=32, layers=2, heads=4, pairs=4,
                      topk_pairs=2, expert_hidden=24, context=16,
                      moe_every=2, future_mode='rollout',
                      symbiotic_rank=rank, micro_meta=micro)

    def test_implicit_triangle_cloud_has_full_unique_address_space(self):
        # Every 0<=a<b<c<256 receives an exact address; no dense C(256,3)
        # tensor of trainable weights is allocated.
        all_addresses = []
        for a, b, c in itertools.combinations(range(256), 3):
            all_addresses.append(c * (c - 1) * (c - 2) // 6 + b * (b - 1) // 2 + a)
        self.assertEqual(len(all_addresses), 2763520)
        self.assertEqual(len(set(all_addresses)), 2763520)
        self.assertEqual(min(all_addresses), 0)
        self.assertEqual(max(all_addresses), 2763519)

    def test_function_preserving_512_style_growth_and_optimizer(self):
        torch.manual_seed(113)
        old_cfg = self.cfg()
        old = MirrorLM(old_cfg)
        opt = torch.optim.AdamW(old.parameters(), lr=.001)
        seq = torch.randint(0, 48, (1, 10))
        old.loss_with_future(seq)['loss'].backward()
        opt.step()
        saved = {'model': copy.deepcopy(old.state_dict()),
                 'optimizer': copy.deepcopy(opt.state_dict())}
        new_cfg = copy.deepcopy(old_cfg)
        widen_config(new_cfg, 64)  # same integral-head migration as 192->512's preserved old heads
        new_cfg.symbiotic_rank = 8
        new_cfg.micro_meta = True
        new = MirrorLM(new_cfg)
        newopt = torch.optim.AdamW(new.parameters(), lr=.001)
        expand_architecture(new, newopt, saved, old_cfg, widen_noise=0.)
        old.eval(); new.eval()
        with torch.no_grad():
            x = seq[:, :-3]
            before, h_before, _ = old(x)
            after, h_after, _ = new(x)
            self.assertTrue(torch.allclose(before, after, atol=1e-5, rtol=0))
            for a, b in zip(old.future_logits(h_before), new.future_logits(h_after)):
                self.assertTrue(torch.allclose(a, b, atol=1e-5, rtol=0))
        self.assertEqual(len(newopt.state), len(opt.state))
        self.assertEqual(len(new.blocks[-1].ff.symbiotic.meta_up), 256)
        self.assertEqual(new.blocks[-1].ff.symbiotic.triangle_table.weight.shape[0], 8192)

    def test_triangle_feed_and_micro_adapters_receive_gradients(self):
        torch.manual_seed(131)
        model = MirrorLM(self.cfg(8, True))
        geom = model.blocks[-1].ff.symbiotic
        with torch.no_grad():
            geom.to_hidden.weight.normal_(0, .01)
        ids = torch.randint(0, 48, (1, 10))
        model.loss_with_future(ids)['loss'].backward()
        for weight in (geom.micro_up, geom.micro_down,
                       geom.meta_up, geom.meta_down, geom.triangle_table.weight):
            self.assertIsNotNone(weight.grad)
            self.assertTrue(torch.isfinite(weight.grad).all())
            self.assertGreater(weight.grad.abs().sum().item(), 0)

    def test_causality_with_nonzero_path(self):
        torch.manual_seed(157)
        model = MirrorLM(self.cfg(8, True)).eval()
        with torch.no_grad():
            model.blocks[-1].ff.symbiotic.to_hidden.weight.normal_(0, .01)
            ids = torch.randint(0, 48, (1, 11))
            modified = ids.clone()
            modified[:, 5:] = torch.randint(0, 48, modified[:, 5:].shape)
            a = model(ids)[0][:, :5]
            b = model(modified)[0][:, :5]
            self.assertTrue(torch.allclose(a, b, atol=3e-5, rtol=0))


if __name__ == '__main__':
    unittest.main()
