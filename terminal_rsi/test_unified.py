"""CPU-only acceptance gates for the exact Terminal-RSI logic used on GPUs."""
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from unified import Preset, RSIController, TerminalRSI, loss_and_stats
from train import BookCorpus, evaluate, restore_checkpoint, save_checkpoint


def tiny():
    return Preset("cpu", 32, 2, 4, 24, 2, 2, 2e-3, 1)


class TerminalRSICPUTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)
        self.cfg = tiny()
        self.model = TerminalRSI(self.cfg)
        self.raw = torch.randint(0, 256, (2, 23), dtype=torch.long)

    def test_proper_three_horizon_losses_and_gradient(self):
        out = self.model(self.raw[:, :-3])
        loss, stats = loss_and_stats(out, self.raw)
        self.assertEqual(out["logits"].shape, (3, 2, 20, 256))
        self.assertTrue(math.isfinite(loss.item()))
        self.assertTrue(all(math.isfinite(v.item()) for v in stats.values()))
        loss.backward()
        critical = {
            "byte": self.model.byte.weight.grad,
            "attn": self.model.backbone[0].qkv.weight.grad,
            "router": self.model.router.weight.grad,
            "critic": self.model.critic[-1].weight.grad,
            "feedback": self.model.feedback.weight.grad,
            "output": self.model.output[0].weight.grad,
        }
        self.assertTrue(all(v is not None and torch.isfinite(v).all() for v in critical.values()),
                        {k: v is None for k, v in critical.items()})
        for k, v in critical.items():
            self.assertGreater(v.abs().sum().item(), 0.0, k)

    def test_future_bytes_cannot_change_past_predictions(self):
        self.model.eval()
        prefix = self.raw[:, :9]
        altered = self.raw.clone()
        altered[:, 9:] = torch.randint(0, 256, altered[:, 9:].shape)
        with torch.no_grad():
            whole = self.model(self.raw[:, :-3])["logits"][:, :, :8]
            perturb = self.model(altered[:, :-3])["logits"][:, :, :8]
            short = self.model(prefix)["logits"][:, :, :8]
        self.assertTrue(torch.allclose(whole, perturb, atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(whole, short, atol=1e-5, rtol=1e-5))

    def test_recursion_is_functionally_active(self):
        self.model.eval()
        with torch.no_grad():
            one = self.model(self.raw[:, :-3], passes=1)["logits"]
            two = self.model(self.raw[:, :-3], passes=2)["logits"]
        self.assertGreater((one - two).abs().max().item(), 1e-7)

    def test_training_changes_predictions(self):
        optim = torch.optim.AdamW(self.model.parameters(), lr=2e-3)
        with torch.no_grad():
            before = self.model(self.raw[:, :-3])["logits"].clone()
        loss, _ = loss_and_stats(self.model(self.raw[:, :-3]), self.raw)
        loss.backward()
        optim.step()
        with torch.no_grad():
            after = self.model(self.raw[:, :-3])["logits"]
        self.assertGreater((after - before).abs().max().item(), 1e-6)

    def test_book_disjoint_causal_padding_checkpoint_exact_restart(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for split, text in (("train", b"abc " * 15),
                                ("dev", b"abc " * 12),
                                ("test", b"different book..." * 4)):
                path = root / split / "book.txt"
                path.parent.mkdir(parents=True)
                path.write_bytes(text)
            corpus = BookCorpus(root)
            self.assertEqual(len(corpus.books["train"]), 1)
            self.assertEqual(len(corpus.books["dev"]), 1)
            self.assertEqual(len(corpus.books["test"]), 1)
            import random
            raw, valid = corpus.batch("train", 2, self.cfg.context,
                                      random.Random(11), torch.device("cpu"))
            self.assertEqual(raw.shape, (2, 27))
            self.assertTrue(valid.any())
            optim = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.lr)
            controller = RSIController(self.cfg)
            before = evaluate(self.model, corpus, torch.device("cpu"),
                              self.cfg.context, windows=1)
            self.assertTrue(math.isfinite(before["h1_bits"]))
            controller.observe(optim, 1, before["h1_bits"])
            check = root / "candidate.pt"
            save_checkpoint(check, self.model, optim, controller, 1, corpus)
            fresh = TerminalRSI(self.cfg)
            new_opt = torch.optim.AdamW(fresh.parameters(), lr=self.cfg.lr)
            new_ctrl = RSIController(self.cfg)
            self.assertEqual(restore_checkpoint(check, fresh, new_opt, new_ctrl, corpus), 1)
            with torch.no_grad():
                a = self.model(self.raw[:, :-3])["logits"]
                b = fresh(self.raw[:, :-3])["logits"]
            self.assertTrue(torch.equal(a, b))
            self.assertEqual(new_ctrl.history, controller.history)

    def test_controller_self_modifies_learning_policy(self):
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.lr)
        ctrl = RSIController(self.cfg)
        self.assertTrue(ctrl.observe(opt, 1, 3.0))
        self.assertFalse(ctrl.observe(opt, 2, 3.5))
        self.assertFalse(ctrl.observe(opt, 3, 3.4))
        self.assertEqual(ctrl.history[-1]["action"], "halve_lr")
        self.assertEqual(opt.param_groups[0]["lr"], self.cfg.lr * 0.5)

    def test_preset_shapes_and_identity(self):
        from unified import PRESETS
        self.assertEqual(list(PRESETS), ["cpu", "t4", "blackwell"])
        # Execution profiles, not model sizes: the exact v3 graph everywhere.
        self.assertEqual({p.width for p in PRESETS.values()}, {256})
        self.assertEqual({p.layers for p in PRESETS.values()}, {3})
        self.assertEqual({p.context for p in PRESETS.values()}, {256})
        self.assertLess(PRESETS["cpu"].microbatch, PRESETS["t4"].microbatch)
        self.assertLess(PRESETS["t4"].microbatch, PRESETS["blackwell"].microbatch)
        self.assertTrue(all(p.width % p.heads == 0 for p in PRESETS.values()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
