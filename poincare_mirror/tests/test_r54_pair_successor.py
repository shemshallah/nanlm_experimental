"""R54 causal byte-copy memory: value controls, zero parity, optimizer lineage."""
from dataclasses import asdict
from pathlib import Path
import sys
import unittest

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pair_successor_memory import PairSuccessorTrust, retrieve_pair_successors
from poincare_frontier import Config, MirrorLM


class R54PairSuccessorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_exact_successors_not_current_or_future_targets(self):
        pairs = torch.tensor([[[7, 8], [5, 6], [7, 9], [7, 8]]])
        seen = torch.tensor([[1, 2, 3, 4]])
        hist, cover, count = retrieve_pair_successors(pairs, seen, vocab=8)
        self.assertTrue(torch.equal(hist[0, 0], torch.zeros(8)))
        self.assertTrue(torch.equal(hist[0, 1], torch.zeros(8)))
        self.assertEqual(hist[0, 2, 2].item(), 1.0)
        self.assertAlmostEqual(hist[0, 3, 2].item(), .75)
        self.assertAlmostEqual(hist[0, 3, 4].item(), .25)
        self.assertEqual(cover[0].tolist(), [0., 0., .5, 1.])
        self.assertEqual(count[0].tolist(), [0., 0., 1., 2.])
        extended = torch.cat([seen, torch.tensor([[7]])], dim=1)
        extended_pairs = torch.cat([pairs, torch.tensor([[[7, 8]]])], dim=1)
        more, _, _ = retrieve_pair_successors(extended_pairs, extended, vocab=8)
        self.assertTrue(torch.equal(hist, more[:, :4]))
        altered = seen.clone()
        altered[0, 3] = 5
        changed, _, _ = retrieve_pair_successors(pairs, altered, vocab=8)
        self.assertTrue(torch.equal(hist[:, :3], changed[:, :3]))

    def test_content_control_preserves_exact_hit_support(self):
        pairs = torch.tensor([[[7, 8], [5, 6], [7, 9], [7, 8]]])
        seen = torch.tensor([[1, 2, 3, 4]])
        exact, cov, n = retrieve_pair_successors(pairs, seen, vocab=8)
        rotated, cov2, n2 = retrieve_pair_successors(
            pairs, seen, vocab=8, value_control='rotated')
        self.assertTrue(torch.equal(cov, cov2))
        self.assertTrue(torch.equal(n, n2))
        self.assertFalse(torch.equal(exact[0, 3], rotated[0, 3]))
        shifted = torch.arange(32640).roll(100)
        wrong, _, _ = retrieve_pair_successors(
            pairs, seen, vocab=8, query_map=shifted)
        self.assertTrue(torch.equal(wrong, torch.zeros_like(wrong)))

    def test_successor_memory_synthetic_positive_control(self):
        # In this controlled stream, 0 is always followed by 1 and vice versa.
        # Pair IDs encode the *current* observed byte, not future labels.
        t = 28
        seen = torch.tensor([[i % 2 for i in range(t)]])
        pair_ids = torch.where(seen[..., None] == 0,
                               torch.tensor([7, 8]), torch.tensor([9, 10]))
        logits = torch.zeros((1, t, 2))
        trust = PairSuccessorTrust(horizon=16)
        with torch.no_grad():
            trust.weight[0] = 2.
        correct, _ = trust(logits, pair_ids, seen)
        trust.value_control = 'rotated'
        corrupted, _ = trust(logits, pair_ids, seen)
        targets = seen[:, 1:]
        correct_ce = F.cross_entropy(correct[:, 4:-1].reshape(-1, 2),
                                     targets[:, 4:].reshape(-1))
        corrupt_ce = F.cross_entropy(corrupted[:, 4:-1].reshape(-1, 2),
                                     targets[:, 4:].reshape(-1))
        self.assertLess(correct_ce.item(), .40)
        self.assertGreater(corrupt_ce.item(), correct_ce.item()+.30)
        self.assertTrue(torch.isfinite(correct).all())
        trust.value_control = 'exact'
        trust.zero_grad(set_to_none=True)
        grad_ce = F.cross_entropy(trust(logits, pair_ids, seen)[0][:, 4:-1].reshape(-1,2),
                                  targets[:, 4:].reshape(-1))
        grad_ce.backward()
        self.assertGreater(trust.weight.grad.abs().sum().item(), 0)

    def test_inherited_logits_and_optimizer_migrated_in_one_group(self):
        torch.manual_seed(54)
        cfg = Config(dim=16, layers=2, heads=2, pairs=4, topk_pairs=1,
                     expert_hidden=16, context=8, moe_every=2,
                     symbiotic_rank=8, micro_meta=True, simplex_transfer=True,
                     dispatch_mode='gather', balance_weight=0.)
        model = MirrorLM(cfg)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        seen = torch.tensor([[4, 5, 4, 5, 4, 5, 4, 5]])
        out, _, metrics = model(seen, seen)
        metrics['loss'].backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        model.eval()
        with torch.no_grad():
            before = model(seen)[0]
        inherited = {k: v.clone() for k, v in model.state_dict().items()}
        old_adam = {id(p): {k: v.clone() if torch.is_tensor(v) else v
                            for k, v in state.items()}
                    for p, state in optimizer.state.items()}
        old_count = len(optimizer.param_groups[0]['params'])
        model.enable_pair_successor(optimizer)
        self.assertEqual(len(optimizer.param_groups), 1)
        self.assertEqual(len(optimizer.param_groups[0]['params']), old_count + 1)
        with torch.no_grad():
            replay, _, details = model(seen)
        self.assertTrue(torch.equal(before, replay))
        self.assertIn('pair_successor_coverage', details)
        for key, tensor in inherited.items():
            self.assertTrue(torch.equal(tensor, model.state_dict()[key]), key)
        for param, state in optimizer.state.items():
            self.assertIn(id(param), old_adam)
            for key, value in state.items():
                previous = old_adam[id(param)][key]
                if torch.is_tensor(value):
                    self.assertTrue(torch.equal(value, previous))
                else:
                    self.assertEqual(value, previous)
        self.assertTrue(model.cfg.pair_successor_copy)
        reloaded = MirrorLM(Config(**asdict(model.cfg))).eval()
        reloaded.load_state_dict(model.state_dict(), strict=True)
        recovered_optimizer = torch.optim.AdamW(reloaded.parameters(), lr=.001)
        recovered_optimizer.load_state_dict(optimizer.state_dict())
        self.assertEqual(len(recovered_optimizer.state), len(optimizer.state))
        with torch.no_grad():
            self.assertTrue(torch.equal(reloaded(seen)[0], replay))
        with self.assertRaisesRegex(ValueError, 'already enabled'):
            model.enable_pair_successor(optimizer)


if __name__ == '__main__':
    unittest.main()
