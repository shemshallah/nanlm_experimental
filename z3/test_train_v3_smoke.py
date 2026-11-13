#!/usr/bin/env python3
"""Smoke tests for the z3 V3 training harness.

Run directly:

    python z3/test_train_v3_smoke.py -v

or with discovery:

    python -m unittest discover -s z3 -p 'test_train_v3_smoke.py' -v
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve().with_name("train_v3.py")

TINY_CONFIG = [
    "--vocab-size", "256",
    "--ctx", "32",
    "--d-model", "64",
    "--n-layers", "2",
    "--n-heads", "4",
    "--swiglu", "128",
    "--window", "16",
    "--global-every", "2",
    "--moe-layers", "1",
    "--n-experts", "4",
    "--topk", "2",
    "--shared-experts", "1",
    "--expert-hidden", "128",
    "--mtp", "3",
    "--dropout", "0.0",
    "--batch-size", "2",
    "--eval-batches", "1",
]


class TrainV3SmokeTest(unittest.TestCase):
    def run_cli(self, *args: str) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        return subprocess.run(
            [sys.executable, str(SCRIPT), *args],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=300,
            env=env,
        )

    def assert_ok(self, proc: subprocess.CompletedProcess[str]) -> None:
        if proc.returncode != 0:
            self.fail(
                f"command failed with exit code {proc.returncode}\n"
                f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
            )

    def test_help(self) -> None:
        proc = self.run_cli("--help")
        self.assert_ok(proc)
        for flag in (
            "--random-smoke",
            "--init-from",
            "--eval-before",
            "--max-seconds",
            "--steps",
            "--device",
            "--amp",
            "--mtp-weight",
        ):
            self.assertIn(flag, proc.stdout)

    def test_factorized_tied_mtp_smoke(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v3_tied_smoke_") as tmp:
            out = Path(tmp) / "run"
            proc = self.run_cli(
                "--random-smoke",
                *TINY_CONFIG,
                "--factorized-embedding",
                "--embedding-rank", "16",
                "--mtp-tie-heads",
                "--steps", "1",
                "--threads", "1",
                "--device", "cpu",
                "--out", str(out),
            )
            self.assert_ok(proc)
            result = json.loads((out / "result.json").read_text(encoding="utf-8"))
            config = json.loads((out / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(result["status"], "completed")
            self.assertTrue(result["mtp"]["enabled"])
            self.assertEqual(result["mtp"]["heads"], 2)
            self.assertEqual(config["resolved"]["mtp_heads"], 2)
            self.assertTrue(config["factorized_embedding"])
            self.assertTrue(config["mtp_tie_heads"])

    def test_lr_horizon_uses_total_steps_for_warmup(self) -> None:
        """A short smoke must stay on the long run's warmup curve."""
        with tempfile.TemporaryDirectory(prefix="v3_lr_horizon_") as tmp:
            out = Path(tmp) / "run"
            proc = self.run_cli(
                "--random-smoke",
                *TINY_CONFIG,
                "--steps", "2",
                "--warmup-steps", "1",
                "--lr-total-steps", "10",
                "--threads", "1",
                "--device", "cpu",
                "--out", str(out),
            )
            self.assert_ok(proc)
            result = json.loads((out / "result.json").read_text(encoding="utf-8"))
            config = json.loads((out / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(result["lr_schedule"]["warmup_steps"], 1)
            self.assertAlmostEqual(result["lr_schedule"]["warmup_frac"], 0.1)
            self.assertGreater(
                result["lr_schedule"]["last_lr"],
                0.9 * config["lr"],
            )

    def test_random_smoke_artifacts_and_metrics(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v3_train_smoke_") as tmp:
            out = Path(tmp) / "run"
            proc = self.run_cli(
                "--random-smoke",
                *TINY_CONFIG,
                "--steps", "3",
                "--threads", "1",
                "--seed", "7",
                "--device", "cpu",
                "--out", str(out),
            )
            self.assert_ok(proc)
            self.assertNotIn("Traceback", proc.stderr)

            expected = (
                "best.pt",
                "final.pt",
                "ema.pt",
                "last.pt",
                "result.json",
                "history.json",
                "config.json",
            )
            for name in expected:
                path = out / name
                self.assertTrue(path.is_file(), f"missing artifact: {path}")
                self.assertGreater(path.stat().st_size, 0, f"empty artifact: {path}")

            result = json.loads((out / "result.json").read_text(encoding="utf-8"))
            self.assertTrue(result["random_smoke"])
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["steps_completed"], 3)
            self.assertEqual(result["steps_this_run"], 3)
            self.assertGreater(result["params"], 0)

            for key in ("final", "final_ema", "best"):
                self.assertIsNotNone(result[key])
            self.assertTrue(math.isfinite(result["final"]["loss"]))
            self.assertTrue(math.isfinite(result["final_ema"]["loss"]))
            self.assertTrue(math.isfinite(result["best"]["loss"]))
            self.assertGreater(result["final"]["tokens"], 0)
            self.assertTrue(math.isfinite(result["final"]["mtp_loss"]))
            self.assertTrue(math.isfinite(result["final"]["load_balance"]))
            self.assertTrue(math.isfinite(result["final"]["router_z"]))
            self.assertIn(result["best"]["source"], ("model", "ema"))
            self.assertLessEqual(
                result["best"]["loss"], result["final"]["loss"] + 1e-9
            )

            self.assertTrue(result["mtp"]["enabled"])
            self.assertEqual(result["mtp"]["heads"], 2)
            self.assertEqual(result["objective"]["mtp_lags"], [2, 3])
            self.assertAlmostEqual(result["objective"]["mtp_weight"], 0.3)
            self.assertEqual(result["data"]["mode"], "random")

            for key in (
                "best_weights",
                "final_weights",
                "ema_weights",
                "last_checkpoint",
                "result",
                "history",
                "config",
            ):
                self.assertIn(key, result["artifacts"])
                self.assertTrue(Path(result["artifacts"][key]).is_file())

            history = json.loads((out / "history.json").read_text(encoding="utf-8"))
            self.assertGreaterEqual(len(history), 3)
            self.assertEqual(len(history), result["history_entries"])
            self.assertTrue(math.isfinite(history[-1]["val"]["loss"]))
            self.assertIsNotNone(history[-1]["val_ema"])
            self.assertTrue(math.isfinite(history[-1]["train"]["loss"]))

            config = json.loads((out / "config.json").read_text(encoding="utf-8"))
            self.assertTrue(config["random_smoke"])
            self.assertEqual(config["ctx"], 32)
            self.assertEqual(config["resolved"]["mtp_heads"], 2)

            best = torch.load(
                out / "best.pt", map_location="cpu", weights_only=False
            )
            self.assertIn("model", best)
            self.assertIn(best["source"], ("model", "ema"))
            self.assertEqual(best["step"], result["best"]["step"])
            self.assertGreater(
                sum(value.numel() for value in best["model"].values()), 0
            )

            final = torch.load(
                out / "final.pt", map_location="cpu", weights_only=False
            )
            ema = torch.load(
                out / "ema.pt", map_location="cpu", weights_only=False
            )
            self.assertEqual(final["step"], 3)
            self.assertEqual(ema["source"], "ema")
            self.assertEqual(final["model"].keys(), ema["model"].keys())


if __name__ == "__main__":
    unittest.main()
