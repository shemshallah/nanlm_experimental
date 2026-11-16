"""CPU gates for capacity expansion, including structured Q/K/V and Adam slices."""
import unittest
import torch

from terminal_rsi.unified import Preset, TerminalRSI, loss_and_stats
from terminal_rsi.migrate import migration_summary


class ExpansionTests(unittest.TestCase):
    def test_cpu_to_accelerator_preserves_specialists_and_adam_history(self):
        small = Preset("cpu", 32, 1, 4, 24, 1, 2, 1e-3, 1)
        big = Preset("t4", 64, 2, 4, 32, 1, 2, 1e-3, 1)
        torch.manual_seed(1701)
        old = TerminalRSI(small)
        old_opt = torch.optim.AdamW(old.parameters(), lr=1e-3)
        raw = torch.randint(0, 256, (1, 17))
        loss, _ = loss_and_stats(old(raw[:, :-3]), raw)
        loss.backward()
        old_opt.step()
        source = {"format": "terminal-rsi-unified-v1",
                  "config": vars(small),
                  "model": old.state_dict(),
                  "optimizer": old_opt.state_dict()}
        new = TerminalRSI(big)
        new_opt = torch.optim.AdamW(new.parameters(), lr=1e-3)
        audit = migration_summary(source, new, new_opt)
        self.assertGreater(audit["weights_migrated"], 10)
        self.assertGreater(audit["adam_states_migrated"], 10)
        self.assertFalse(audit["function_preserving"])
        self.assertTrue(torch.equal(new.byte.weight[:, :32], old.byte.weight))
        self.assertTrue(torch.equal(new.output[0].weight[:, :32],
                                    old.output[0].weight))
        q0 = old.backbone[0].qkv.weight.view(3, 4, 8, 32)
        q1 = new.backbone[0].qkv.weight.view(3, 4, 16, 64)
        self.assertTrue(torch.equal(q1[:, :, :8, :32], q0))
        self.assertEqual(new.backbone[1].project.weight.abs().max().item(), 0)
        self.assertEqual(new.backbone[1].down.weight.abs().max().item(), 0)
        self.assertTrue(new_opt.state[new.byte.weight]["exp_avg"].shape ==
                        new.byte.weight.shape)
        self.assertGreater(new_opt.state[new.byte.weight]["exp_avg"][:, :32].abs().sum().item(), 0)
        self.assertEqual(new_opt.state[new.byte.weight]["exp_avg"][:, 32:].abs().max().item(), 0)
        loss, _ = loss_and_stats(new(raw[:, :-3]), raw)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(new.byte.weight.grad).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
