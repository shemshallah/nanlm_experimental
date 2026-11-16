"""Exact original-v103 subprocess command safety and semantic controls."""
import tempfile
import unittest
from pathlib import Path

from terminal_rsi.run_original_v103 import build_original_v103_command


class OriginalV103LaunchTests(unittest.TestCase):
    def test_verbatim_cpu_invocation_carries_all_historical_switches(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = root / "original_v103.pt"
            train = root / "train_10000000.bin"
            dev = root / "win_dev.bin"
            bpe = root / "nancm_v103_bpe.txt"
            for path in (old, train, dev, bpe):
                path.write_bytes(b"test")
            target = root / "candidate_only.pt"
            cmd = build_original_v103_command(
                checkpoint=old, train=train, dev=dev, bpe=bpe,
                output=target, seconds=0, causal_copy=True)
            for flag in ("--allow-cpu-test", "--v103-pathways",
                         "--causal-copy", "--holdout-dev", "--bpe-merges",
                         "--resume", "--seconds"):
                self.assertIn(flag, cmd)
            self.assertIn("legacy_v103_reference", cmd[2])
            self.assertIn("main(sys.argv[1:])", cmd[2])
            self.assertNotIn("--eval-test", cmd)
            self.assertNotIn("--holdout-test", cmd)
            self.assertEqual(cmd[cmd.index("--resume")+1], str(old))
            self.assertEqual(cmd[cmd.index("--output")+1], str(target))
            with self.assertRaisesRegex(ValueError, "overwrite"):
                build_original_v103_command(checkpoint=old, train=train,
                      dev=dev, bpe=bpe, output=old)

if __name__ == "__main__":
    unittest.main(verbosity=2)
