"""CPU simulation of T4/Blackwell RSI capacity migration, without GPU."""
from __future__ import annotations
from unittest.mock import patch
import unittest

import torch

from terminal_rsi.unified import PRESETS, Preset
from terminal_rsi.verbatim_v103 import VerbatimV103RSI, build_v103_optimizer
from terminal_rsi.train_verbatim import train_one_batch
from terminal_rsi.migrate_verbatim import SCHEMA, migrate_checkpoint


class HistoricToBlackwellCPUTests(unittest.TestCase):
    def test_original_online_state_and_both_adam_groups_survive_growth(self):
        small = Preset("cpu", 32, 1, 4, 64, 2, 2, 1e-3, 1)
        bigger = Preset("t4", 64, 2, 4, 128, 2, 2, 1e-4, 1)
        with patch.dict(PRESETS, {"cpu": small, "t4": bigger}):
            torch.manual_seed(103)
            original = VerbatimV103RSI("cpu")
            opt = build_v103_optimizer(original)
            history = torch.randint(0, 256, (2, 32))
            target = torch.randint(0, 256, (2,))
            aux = original.original._default_aux(history).to(torch.int32)
            train_one_batch(original, opt, history, target, aux)
            source = {"schema": SCHEMA, "config": vars(small),
                      "model": original.state_dict(),
                      "optimizer": opt.state_dict()}
            grown, migrated, report = migrate_checkpoint(source, "t4")
            self.assertEqual(report["new_mode"], "t4")
            self.assertEqual(len(migrated.param_groups), 4)
            self.assertFalse(report["function_preserving"])
            for key, tensor in original.original.state_dict().items():
                self.assertTrue(torch.equal(tensor,
                                            grown.original.state_dict()[key]),
                                "Original RSI parameter/buffer "+key)
            self.assertTrue(torch.equal(original.residual_gate,
                                        grown.residual_gate))
            self.assertEqual(len(migrated.state),
                             len(opt.state))
            for group_index in (0, 1):
                old_group = opt.param_groups[group_index]
                new_group = migrated.param_groups[group_index]
                self.assertEqual(old_group["lr"], new_group["lr"])
                for before, after in zip(old_group["params"], new_group["params"]):
                    self.assertEqual(set(opt.state.get(before, {})),
                                     set(migrated.state.get(after, {})))
                    for key, value in opt.state.get(before, {}).items():
                        other = migrated.state[after][key]
                        if torch.is_tensor(value):
                            self.assertTrue(torch.equal(value, other))
                        else:
                            self.assertEqual(value, other)

if __name__ == "__main__":
    unittest.main(verbosity=2)
