"""R31: 192D -> 256D, single optimizer and causal multiscale feedback."""
import argparse
import copy
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from poincare_frontier import (Config, MirrorLM, count_parameters_exact,
                               expand_architecture, morph_tensor, widen_config,
                               save_checkpoint, grow_checkpoint)
from eval_external_bytes import benchmark


class TestR31Multiscale256(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def config(self, **kwargs):
        cfg = Config(vocab_size=257, dim=32, heads=4, layers=2,
                     pairs=4, topk_pairs=2, expert_hidden=16,
                     context=16, moe_every=2, future_mode='offset', **kwargs)
        return cfg

    def test_mixed_192_to_256_retains_all_old_moments_and_three_horizons(self):
        torch.manual_seed(31)
        old = Config(vocab_size=257, dim=192, heads=16, layers=2,
                     pairs=4, topk_pairs=2, expert_hidden=24,
                     context=16, moe_every=2)
        model = MirrorLM(old)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        ids = torch.randint(0, 257, (1, 19))
        optimizer.zero_grad()
        model.loss_with_future(ids)['loss'].backward()
        optimizer.step()
        saved = {'model': copy.deepcopy(model.state_dict()),
                 'optimizer': copy.deepcopy(optimizer.state_dict())}

        new = copy.deepcopy(old)
        widen_config(new, 256)
        new.hierarchy_rank = 16
        new.symbiotic_rank = 8
        self.assertEqual((new.heads, new.legacy_attention_dim,
                          new.legacy_attention_heads), (20, 192, 16))
        candidate = MirrorLM(new)
        candidate_optimizer = torch.optim.AdamW(candidate.parameters(), lr=3e-4)
        expand_architecture(candidate, candidate_optimizer, saved, old, widen_noise=0.)
        model.eval()
        candidate.eval()
        with torch.no_grad():
            old_logits, old_h, _ = model(ids[:, :-3])
            new_logits, new_h, _ = candidate(ids[:, :-3])
            olds = (old_logits,) + model.future_logits(old_h)
            news = (new_logits,) + candidate.future_logits(new_h)
            for a, b in zip(olds, news):
                self.assertLess((a-b).abs().max().item(), 1e-4)
        old_names = list(dict(model.named_parameters()))
        new_names = list(dict(candidate.named_parameters()))
        old_st = saved['optimizer']['state']
        new_st = candidate_optimizer.state_dict()['state']
        old_idx = dict(zip(old_names, saved['optimizer']['param_groups'][0]['params']))
        new_idx = dict(zip(new_names, candidate_optimizer.state_dict()['param_groups'][0]['params']))
        self.assertEqual(set(old_st), set(old_idx.values()))
        for name in old_names:
            for field in ('exp_avg', 'exp_avg_sq'):
                source = old_st[old_idx[name]][field]
                actual = new_st[new_idx[name]][field]
                expected = torch.empty_like(actual)
                morph_tensor(name, expected, source, old, new)
                self.assertTrue(torch.equal(actual, expected), (name, field))
            self.assertTrue(torch.equal(old_st[old_idx[name]]['step'],
                                        new_st[new_idx[name]]['step']))
        self.assertEqual(count_parameters_exact(new), sum(p.numel() for p in candidate.parameters()))

    def test_future_tokens_never_leak_into_prior_hierarchy_states(self):
        torch.manual_seed(51)
        cfg = self.config(hierarchy_rank=8)
        model = MirrorLM(cfg).eval()
        with torch.no_grad():
            model.hierarchy.to_hidden.weight.normal_(0, 0.02)
        a = torch.randint(0, 255, (1, 16))
        b = a.clone()
        b[:, 9:] = torch.randint(0, 255, (1, 7))
        a[0, 4], b[0, 4] = 32, 32  # word boundary
        a[0, 6], b[0, 6] = 46, 46  # sentence boundary
        with torch.no_grad():
            left = model(a)[0][:, :9]
            right = model(b)[0][:, :9]
        self.assertTrue(torch.allclose(left, right, atol=1e-6))

    def test_hierarchy_wakes_after_zero_initialized_migration(self):
        torch.manual_seed(61)
        cfg = self.config(hierarchy_rank=8)
        model = MirrorLM(cfg)
        self.assertEqual(model.hierarchy.to_hidden.weight.abs().sum().item(), 0.)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        ids = torch.randint(0, 257, (1, 19))
        opt.zero_grad()
        model.loss_with_future(ids)['loss'].backward()
        self.assertGreater(model.hierarchy.to_hidden.weight.grad.abs().sum().item(), 0.)
        opt.step()
        opt.zero_grad()
        model.loss_with_future(ids)['loss'].backward()
        self.assertGreater(model.hierarchy.scale_qkv.weight.grad.abs().sum().item(), 0.)

    def test_second_mixed_growth_preserves_trained_extra_heads(self):
        cfg = self.config()
        widen_config(cfg, 48)
        self.assertEqual((cfg.heads, cfg.legacy_attention_heads), (5, 4))
        trained = MirrorLM(cfg)
        optimizer = torch.optim.AdamW(trained.parameters(), lr=3e-4)
        ids = torch.randint(0, 257, (1, 19))
        trained.loss_with_future(ids)['loss'].backward()
        optimizer.step()
        saved = {'model': copy.deepcopy(trained.state_dict()),
                 'optimizer': copy.deepcopy(optimizer.state_dict())}
        new_cfg = copy.deepcopy(cfg)
        widen_config(new_cfg, 80)
        self.assertEqual((new_cfg.heads, new_cfg.legacy_attention_heads), (7, 4))
        next_model = MirrorLM(new_cfg)
        next_optimizer = torch.optim.AdamW(next_model.parameters())
        expand_architecture(next_model, next_optimizer, saved, cfg, widen_noise=0.)
        with torch.no_grad():
            old_logits = trained(ids[:, :-3])[0]
            new_logits = next_model(ids[:, :-3])[0]
        self.assertTrue(torch.allclose(old_logits, new_logits, atol=1e-5))
        self.assertEqual(len(next_optimizer.state), len(optimizer.state))
        with self.assertRaisesRegex(ValueError, 'whole inherited'):
            widen_config(new_cfg, 81)

    def test_external_three_horizon_top5_bpb_report(self):
        torch.manual_seed(71)
        model = MirrorLM(self.config()).eval()
        data = np.frombuffer((b'This is a small test sentence. ' * 30), dtype=np.uint8)
        results, speed, sampled_sha = benchmark(model, data, context=16,
                                                positions=16, batch_size=4, seed=19)
        self.assertEqual(list(results), ['+1', '+2', '+3'])
        self.assertEqual(len(sampled_sha), 64)
        self.assertGreater(speed, 0)
        for result in results.values():
            self.assertGreaterEqual(result['top5'], result['top1'])
            self.assertAlmostEqual(result['bits_per_byte'],
                                   result['ce_nats_per_byte'] / np.log(2), places=5)

    def test_atomic_grow_cli_preserves_checkpoint_lineage(self):
        torch.manual_seed(81)
        old = self.config()
        model = MirrorLM(old)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        ids = torch.randint(0, 257, (1, 19))
        model.loss_with_future(ids)['loss'].backward()
        optimizer.step()
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder) / 'latest.pt'
            metadata = {'vocab_size':257, 'unit':'UTF8_bytes_plus_eos',
                        'tokenizer_model':None, 'train_sha256':'fixed'}
            save_checkpoint(checkpoint, model, optimizer, 2801, 12345,
                            metadata, np.random.default_rng(3), old)
            saved = torch.load(checkpoint, weights_only=False)
            args = argparse.Namespace(run_dir=folder, widen_dim=48, add_layers=0,
                                      grow_pairs=None, grow_expert_hidden=None,
                                      symbiotic_rank=8, hierarchy_rank=8,
                                      set_context=None, dispatch_mode=None,
                                      gather_chunk=None, initial_pair_bias=-12.,
                                      widen_noise=0., grow_warmup=100,
                                      max_logit_drift=1e-4, threads=1, report=None)
            grow_checkpoint(args)
            migrated = torch.load(checkpoint, weights_only=False)
            self.assertEqual(migrated['step'], 2801)
            self.assertEqual(migrated['seen_tokens'], 12345)
            self.assertEqual(migrated['data'], metadata)
            self.assertTrue(torch.equal(migrated['rng_torch'], saved['rng_torch']))
            self.assertEqual(migrated['rng_numpy'], saved['rng_numpy'])
            self.assertEqual(migrated['config']['dim'], 48)
            self.assertEqual(migrated['config']['hierarchy_rank'], 8)
            self.assertEqual(migrated['config']['symbiotic_rank'], 8)
            self.assertEqual(list(Path(folder).iterdir()), [checkpoint])


if __name__ == '__main__':
    unittest.main()
