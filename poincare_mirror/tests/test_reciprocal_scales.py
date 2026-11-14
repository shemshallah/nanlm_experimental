"""Five causal byte resolutions into the existing signed simplex and meta geometry."""
import argparse
import copy
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from poincare_frontier import (Config, MirrorLM, ReciprocalScaleGeometry,
                              count_parameters_exact, expand_architecture,
                              grow_checkpoint, save_checkpoint)


def config(reciprocal=False, *, isolated=False):
    return Config(vocab_size=257, dim=32, layers=2, heads=4,
                  pairs=4, topk_pairs=2, expert_hidden=16, context=16,
                  moe_every=2, hierarchy_rank=8, symbiotic_rank=8,
                  micro_meta=True, simplex_transfer=True,
                  reciprocal_scales=reciprocal,
                  future_mode='isolated' if isolated else 'offset',
                  future_adapter_rank=8 if isolated else 0)


class ReciprocalScalesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_required_existing_geometry_and_exact_parameter_count(self):
        cfg = config(True)
        self.assertEqual(count_parameters_exact(cfg), sum(p.numel() for p in MirrorLM(cfg).parameters()))
        cfg.micro_meta = False
        with self.assertRaises(AssertionError):
            MirrorLM(cfg)

    def test_five_byte_aligned_scales_have_no_right_context(self):
        torch.manual_seed(392)
        z = torch.randn(2, 13, 8)
        ids = torch.randint(0, 255, (2, 13))
        levels = ReciprocalScaleGeometry.causal_scales(z, ids)
        altered = z.clone(); altered[:, 7:] += 100
        changed = ids.clone(); changed[:, 7:] = 33
        other = ReciprocalScaleGeometry.causal_scales(altered, changed)
        self.assertEqual(levels.shape, (2, 13, 5, 8))
        self.assertTrue(torch.equal(levels[:, :7], other[:, :7]))
        self.assertTrue(torch.equal(levels[:, :, 0], z))

    def test_migration_preserves_three_horizons_and_all_prior_adam_moments(self):
        torch.manual_seed(393)
        old_cfg = config(isolated=True)
        old = MirrorLM(old_cfg)
        old_opt = torch.optim.AdamW(old.parameters())
        ids = torch.randint(0, 256, (1, 19))
        old.loss_with_future(ids)['loss'].backward(); old_opt.step()
        saved = {'model':copy.deepcopy(old.state_dict()),
                 'optimizer':copy.deepcopy(old_opt.state_dict())}
        new_cfg = config(True, isolated=True)
        migrated = MirrorLM(new_cfg)
        new_opt = torch.optim.AdamW(migrated.parameters())
        expand_architecture(migrated, new_opt, saved, old_cfg)
        old.eval(); migrated.eval()
        with torch.no_grad():
            a, ah, _ = old(ids[:, :-3]); b, bh, _ = migrated(ids[:, :-3])
            for left,right in ((a,b), *zip(old.future_logits(ah),migrated.future_logits(bh))):
                self.assertTrue(torch.equal(left,right))
        self.assertEqual(len(old_opt.state),len(new_opt.state))
        for name, old_tensor in saved['model'].items():
            self.assertTrue(torch.equal(old_tensor,migrated.state_dict()[name]),name)
        for a,b in zip(old_opt.state.values(),new_opt.state.values()):
            for k in ('step','exp_avg','exp_avg_sq'):
                self.assertTrue(torch.equal(a[k],b[k]))

    def test_enabled_bridge_is_causal_and_has_real_feedback(self):
        torch.manual_seed(394)
        model = MirrorLM(config(True)).eval()
        model.reciprocal.gate.data.fill_(0.4)
        ids = torch.randint(0, 256, (1, 13))
        ids2 = ids.clone(); ids2[:, 8:] = torch.randint(0, 256, (1,5))
        with torch.no_grad():
            a = model(ids)[0]; b = model(ids2)[0]
            model.reciprocal.enabled = False
            no_feedback = model(ids)[0]
        self.assertTrue(torch.allclose(a[:, :8], b[:, :8], atol=2e-5, rtol=0))
        self.assertGreater((a-no_feedback).abs().max().item(),1e-7)

    def test_gradients_reach_scales_signed_pairs_triangles_and_meta(self):
        torch.manual_seed(395)
        model = MirrorLM(config(True))
        model.reciprocal.gate.data.fill_(0.2)
        ids = torch.randint(0, 256, (1, 19))
        loss = model.loss_with_future(ids)['loss']; loss.backward()
        g = model.blocks[1].ff.symbiotic
        for name,param in [('scale',model.reciprocal.scale_qkv.weight),
                           ('sign',model.reciprocal.signed_orientation.weight),
                           ('point',g.point_values),('pair',g.simplex.pair_shared.weight),
                           ('triangle',g.triangle_table.weight),
                           ('meta',g.meta_up)]:
            self.assertIsNotNone(param.grad,name)
            self.assertGreater(param.grad.abs().sum().item(),0,name)

    def test_every_geometric_return_edge_has_measurable_effect(self):
        torch.manual_seed(432)
        cfg = config(True)
        cfg.layers = 4  # two geometric MoE blocks and two ordinary attention blocks
        model = MirrorLM(cfg).eval()
        model.reciprocal.gate.data.fill_(.4)
        ids = torch.randint(0,256,(1,15))
        with torch.no_grad():
            reference = model(ids)[0]
            for name in ('pair_to_micro_enabled','cloud_to_micro_enabled',
                         'meta_to_micro_enabled','meta_tokenizer_enabled'):
                setattr(model.reciprocal,name,False)
                reduced = model(ids)[0]
                setattr(model.reciprocal,name,True)
                self.assertGreater((reference-reduced).abs().max().item(),1e-6,name)

    def test_two_geometric_cycles_and_token_feedback_at_all_four_blocks(self):
        torch.manual_seed(433)
        cfg = config(True); cfg.layers = 4
        model = MirrorLM(cfg)
        model.reciprocal.gate.data.fill_(.2)
        ids = torch.randint(0,256,(1,15))
        outputs = []
        original_emit = model.reciprocal.emit
        def capture_feedback(*args,**kwargs):
            result = original_emit(*args,**kwargs)
            outputs.append(result)
            return result
        with patch.object(model.reciprocal,'ingest',wraps=model.reciprocal.ingest) as ingest, \
             patch.object(model.reciprocal,'couple',wraps=model.reciprocal.couple) as couple, \
             patch.object(model.reciprocal,'emit',side_effect=capture_feedback) as emit:
            model(ids)
            self.assertEqual(ingest.call_count,4)
            self.assertEqual(emit.call_count,4)
            self.assertEqual(couple.call_count,2)
        self.assertTrue(all(float(v.abs().sum()) > 0 for v in outputs))

    def test_512d_mixed_head_reciprocal_forward_backward(self):
        torch.manual_seed(397)
        cfg = Config(vocab_size=257, dim=512, layers=2, heads=36,
                     legacy_attention_dim=192, legacy_attention_heads=16,
                     pairs=2, topk_pairs=1, expert_hidden=16, context=8,
                     moe_every=2, hierarchy_rank=32, symbiotic_rank=32,
                     micro_meta=True, simplex_transfer=True, reciprocal_scales=True)
        model = MirrorLM(cfg)
        self.assertEqual(count_parameters_exact(cfg),sum(p.numel() for p in model.parameters()))
        model.reciprocal.gate.data.fill_(.1)
        result = model.loss_with_future(torch.randint(0,256,(1,9)))
        self.assertTrue(torch.isfinite(result['loss']))
        result['loss'].backward()
        self.assertGreater(model.reciprocal.scale_to_geo.weight.grad.abs().sum().item(),0)

    def test_atomic_cli_growth_records_flag_without_training_or_optimizer_loss(self):
        torch.manual_seed(396)
        cfg = config(isolated=True)
        model = MirrorLM(cfg)
        optimizer = torch.optim.AdamW(model.parameters())
        ids = torch.randint(0, 256, (1, 19))
        model.loss_with_future(ids)['loss'].backward(); optimizer.step()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'latest.pt'
            save_checkpoint(path,model,optimizer,2865,100000,
                            {'vocab_size':257,'unit':'UTF8_bytes_plus_eos',
                             'split_unit':'whole_book'},
                            np.random.default_rng(42),cfg)
            before = torch.load(path,weights_only=False)
            args = argparse.Namespace(run_dir=tmp,widen_dim=None,add_layers=0,
                grow_pairs=None,grow_expert_hidden=None,symbiotic_rank=None,
                hierarchy_rank=None,set_context=None,attention_window=None,
                geometry_memory_chunk=None,micro_meta=False,simplex_transfer=False,
                triangle_curvature=False,holonomy_retrieval=False,
                reciprocal_scales=True,future_adapter_rank=None,future_mode=None,
                dispatch_mode=None,gather_chunk=None,initial_pair_bias=-12.,
                widen_noise=0.,grow_warmup=100,max_logit_drift=1e-6,
                threads=1,report=None)
            grow_checkpoint(args)
            after = torch.load(path,weights_only=False)
            self.assertTrue(after['config']['reciprocal_scales'])
            self.assertEqual(after['step'],before['step'])
            self.assertEqual(after['seen_tokens'],before['seen_tokens'])
            self.assertTrue(torch.equal(after['rng_torch'],before['rng_torch']))
            self.assertEqual(after['rng_numpy'],before['rng_numpy'])
            for name,p in before['model'].items():
                self.assertTrue(torch.equal(p,after['model'][name]),name)
            old_names = list(dict(model.named_parameters()))
            new_names = list(dict(MirrorLM(Config(**after['config'])).named_parameters()))
            old_ids = dict(zip(old_names,before['optimizer']['param_groups'][0]['params']))
            new_ids = dict(zip(new_names,after['optimizer']['param_groups'][0]['params']))
            for name in old_names:
                old_state = before['optimizer']['state'].get(old_ids[name],{})
                new_state = after['optimizer']['state'].get(new_ids[name],{})
                for key,val in old_state.items():
                    self.assertTrue(torch.equal(val,new_state[key]),(name,key))
            self.assertGreater(after['growth_history'][-1]['adam_moment_fields_verified'],0)


if __name__ == '__main__':
    unittest.main()
