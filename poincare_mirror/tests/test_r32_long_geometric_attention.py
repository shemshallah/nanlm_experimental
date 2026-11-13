"""R32: bounded causal attention + content-addressed completed-chunk geometry."""
import copy
import sys
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from poincare_frontier import (CausalAttention, CausalGeometricMemory, Config,
                               MirrorLM, apply_rope, count_parameters_exact,
                               expand_architecture)


class TestR32LongGeometricAttention(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_sliding_attention_matches_full_explicit_causal_mask(self):
        torch.manual_seed(31)
        att = CausalAttention(32, 4, window=8).eval()
        data = torch.randn(2, 23, 32)
        actual = att(data)
        q, k, v = att.qkv(data).reshape(2, 23, 3, 4, 8).unbind(2)
        q = apply_rope(q.transpose(1, 2))
        k = apply_rope(k.transpose(1, 2))
        v = v.transpose(1, 2)
        pos = torch.arange(23)
        allowed = (pos[None, :] <= pos[:, None]) & (
            pos[None, :] > pos[:, None] - 8)
        reference = att.out(F.scaled_dot_product_attention(
            q, k, v, attn_mask=allowed, dropout_p=0).transpose(1, 2).reshape(2, 23, 32))
        self.assertTrue(torch.allclose(actual, reference, atol=2e-6, rtol=1e-6))

    def test_completed_chunk_memory_no_future_information_or_gradient(self):
        torch.manual_seed(32)
        memory = CausalGeometricMemory(8, 8)
        memory.gate.data.fill_(1.)
        states = torch.randn(1, 48, 8, requires_grad=True)
        output = memory(states)
        self.assertTrue(torch.equal(output[:, :8], states[:, :8]))
        changed = states.detach().clone()
        changed[:, 21:] = torch.randn_like(changed[:, 21:])
        with torch.no_grad():
            replay = memory(changed)
        self.assertTrue(torch.allclose(output[:, :21], replay[:, :21], atol=1e-6))
        output[:, :21].sum().backward()
        self.assertEqual(states.grad[:, 21:].abs().max().item(), 0.)

    def test_geometric_memory_reaches_beyond_local_attention_window(self):
        torch.manual_seed(33)
        memory = CausalGeometricMemory(8, 8).eval()
        states = torch.randn(1, 80, 8)
        with torch.no_grad():
            baseline = memory(states)
            self.assertTrue(torch.equal(baseline, states))
            memory.gate.fill_(1.)
            informed = memory(states)
            self.assertGreater((informed[:, 32:] - baseline[:, 32:]).abs().max().item(), 1e-5)

    def test_migration_keeps_old_logits_and_every_populated_adam_state(self):
        torch.manual_seed(34)
        old_cfg = Config(vocab_size=48, dim=32, layers=2, heads=4,
                         pairs=4, topk_pairs=2, expert_hidden=16,
                         context=16, moe_every=2, symbiotic_rank=8)
        old = MirrorLM(old_cfg)
        optimizer = torch.optim.AdamW(old.parameters(), lr=1e-3)
        ids = torch.randint(0, 48, (1, 19))
        old.loss_with_future(ids)['loss'].backward()
        optimizer.step()
        saved = {'model': copy.deepcopy(old.state_dict()),
                 'optimizer': copy.deepcopy(optimizer.state_dict())}
        new_cfg = copy.deepcopy(old_cfg)
        new_cfg.context = 128
        new_cfg.attention_window = 16
        new_cfg.geometry_memory_chunk = 8
        new = MirrorLM(new_cfg)
        new_optimizer = torch.optim.AdamW(new.parameters(), lr=1e-3)
        expand_architecture(new, new_optimizer, saved, old_cfg)
        old.eval(); new.eval()
        with torch.no_grad():
            original, hidden, _ = old(ids[:, :-3])
            migrated, hidden_new, _ = new(ids[:, :-3])
            self.assertTrue(torch.equal(original, migrated))
            for a, b in zip(old.future_logits(hidden), new.future_logits(hidden_new)):
                self.assertTrue(torch.equal(a, b))
        old_params = list(dict(old.named_parameters()))
        new_params = list(dict(new.named_parameters()))
        a_ids = dict(zip(old_params, saved['optimizer']['param_groups'][0]['params']))
        b_ids = dict(zip(new_params, new_optimizer.state_dict()['param_groups'][0]['params']))
        for name in old_params:
            old_moment = saved['optimizer']['state'].get(a_ids[name])
            new_moment = new_optimizer.state_dict()['state'].get(b_ids[name])
            self.assertEqual(old_moment is None, new_moment is None)
            if old_moment is not None:
                for field, value in old_moment.items():
                    self.assertTrue(torch.equal(value, new_moment[field]), (name, field))
        self.assertEqual(count_parameters_exact(new_cfg), sum(p.numel() for p in new.parameters()))
        with torch.no_grad():
            for block in new.blocks:
                if block.use_moe:
                    block.ff.symbiotic.memory.gate.fill_(0.5)
                    block.ff.symbiotic.to_hidden.weight.normal_(0, .005)
            long_seq = torch.randint(0, 48, (1, 64))
            changed = long_seq.clone()
            changed[:, 40:] = torch.randint(0, 48, (1, 24))
            self.assertTrue(torch.allclose(new(long_seq)[0][:, :40],
                                           new(changed)[0][:, :40], atol=2e-5))


if __name__ == '__main__':
    unittest.main()
