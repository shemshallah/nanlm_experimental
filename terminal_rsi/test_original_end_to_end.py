"""End-to-end CPU replay of THE archived v103 main(), not a surrogate model.

Synthetic fixtures only: this validates runtime wiring and original checkpoint
continuation, NEVER claims recovered real 10M/C-model benchmark parity.
"""
from __future__ import annotations

import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

import torch


class OriginalCPUReplayEndToEnd(unittest.TestCase):
    def test_real_unmodified_main_runs_then_resumes_with_rsi_state(self):
        from terminal_rsi.legacy_v103_reference import PARITY_SCHEMA

        with tempfile.TemporaryDirectory(prefix="terminal_rsi_original_") as root_str:
            root = Path(root_str)
            corpus = root / "synthetic_train.bin"
            corpus.write_bytes((b"Terminal-RSI causal bytes and verified recursive feedback.\n" * 48)[:1536])
            # Native <II count,skip>, <I book length>, then EXACT book bytes.
            # A one-position DEV keeps CI bounded, and is entirely disjoint.
            dev = root / "synthetic_dev.bin"
            book = b"unrelated book for the original test!"
            dev.write_bytes(struct.pack("<II", 1, 32) +
                            struct.pack("<I", len(book)) + book)
            bpe = Path(__file__).resolve().parent / "reference/nancm_v103_bpe.txt"
            donor = root / "synthetic_original_v103.pt"
            continued = root / "synthetic_original_cpu_resume.pt"
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
            # First create a REAL checkpoint by calling the unmodified archived
            # training function; >0 seconds forces at least one AdamW step.
            initial = [
                sys.executable, "-c",
                "import sys; from terminal_rsi.legacy_v103_reference "
                "import main; main(sys.argv[1:])",
                "--corpus", str(corpus), "--output", str(donor),
                "--holdout-dev", str(dev), "--bpe-merges", str(bpe),
                "--v103-pathways", "--causal-copy",
                "--allow-cpu-test", "--seconds", "0.01", "--batch", "4",
                "--checkpoint-every", "0",
            ]
            first = subprocess.run(initial, capture_output=True,
                                   text=True, timeout=360, env=env)
            self.assertEqual(first.returncode, 0,
                             (first.stdout + first.stderr)[-12000:])
            before = torch.load(donor, map_location="cpu", weights_only=True)
            self.assertEqual(before["schema"], PARITY_SCHEMA)
            self.assertGreater(before["steps"], 0)
            self.assertTrue(before["optimizer"]["state"])
            self.assertEqual(before["holdout_digests"]["test"], None)
            # Exercise the public CPU launcher, including metadata inspection,
            # original parser, original feature builder, Adam and terminal RSI.
            cmd = [
                sys.executable, "-m", "terminal_rsi.run_original_v103",
                "--checkpoint", str(donor),
                "--train-stream", str(corpus), "--dev-bin", str(dev),
                "--bpe-merges", str(bpe), "--output", str(continued),
                "--seconds", "0",
            ]
            second = subprocess.run(cmd, capture_output=True,
                                    text=True, timeout=360, env=env)
            self.assertEqual(second.returncode, 0,
                             (second.stdout + second.stderr)[-12000:])
            self.assertIn("EXACT coverage cursor:", second.stdout)
            self.assertIn("Verified exact GPU checkpoint prediction round-trip: PASS",
                          second.stdout)
            after = torch.load(continued, map_location="cpu", weights_only=True)
            self.assertEqual(after["schema"], before["schema"])
            self.assertEqual(after["steps"], before["steps"])
            self.assertEqual(after["sampler"], before["sampler"])
            self.assertEqual(after["source_sha256"], before["source_sha256"])
            self.assertEqual(after["v103_feature_manifest"],
                             before["v103_feature_manifest"])
            # Exactly zero new optimizer steps, so every inherited neural
            # tensor and Adam moment is preserved. Only the verified online
            # terminal-feedback buffers are permitted to evolve.
            self.assertEqual(after["optimizer"]["param_groups"],
                             before["optimizer"]["param_groups"])
            for key, value in before["model"].items():
                if key in ("feedback_state", "verified_events"):
                    continue
                self.assertTrue(torch.equal(after["model"][key], value), key)
            for param_id, moment_dict in before["optimizer"]["state"].items():
                for key, value in moment_dict.items():
                    seen = after["optimizer"]["state"][param_id][key]
                    if isinstance(value, torch.Tensor):
                        self.assertTrue(torch.equal(seen, value),
                                        f"Adam parameter {param_id}: {key}")
                    else:
                        self.assertEqual(seen, value)
            self.assertGreater(
                int(after["model"]["verified_events"]),
                int(before["model"]["verified_events"]))
            self.assertGreater(len(after["events"]), len(before["events"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
