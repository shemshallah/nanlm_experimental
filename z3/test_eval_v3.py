#!/usr/bin/env python3
"""Tests for z3/eval_v3.py.

Run from the repository root:

    python -m unittest z3/test_eval_v3.py -v
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import eval_v3  # noqa: E402
from v3_model import V3Config, V3LM  # noqa: E402


BPE_PATH = ROOT.parent / "z1" / "build" / "bpe32k_new.json"


def tiny_config(vocab_size: int = 128, ctx: int = 16, mtp: int = 3) -> V3Config:
    return V3Config(
        vocab_size=vocab_size,
        ctx=ctx,
        d_model=32,
        n_layers=3,
        n_heads=4,
        swiglu=64,
        window=8,
        global_every=2,
        moe_layers=2,
        n_experts=8,
        topk=2,
        shared_experts=1,
        expert_hidden=64,
        mtp=mtp,
        dropout=0.0,
    )


class EvalV3Tests(unittest.TestCase):
    def test_metrics_keys_and_ranges(self) -> None:
        torch.manual_seed(0)
        cfg = tiny_config()
        model = V3LM(cfg)
        token_ids = [(i * 37) % cfg.vocab_size for i in range(40)]

        metrics = eval_v3.evaluate_ids(
            model,
            token_ids,
            cfg,
            ctx=16,
            batch_size=2,
            batches=3,
            seed=5,
            device="cpu",
            name="unit",
        )

        self.assertEqual(metrics["tokens"], 6)
        self.assertEqual(metrics["tokens_available"], len(token_ids))
        self.assertGreater(metrics["loss"], 0.0)
        self.assertTrue(math.isfinite(metrics["bits_per_token"]))
        self.assertGreater(metrics["bits_per_token"], 0.0)
        self.assertGreaterEqual(metrics["top1"], 0.0)
        self.assertLessEqual(metrics["top1"], metrics["top5"])
        self.assertLessEqual(metrics["top5"], 1.0)
        self.assertEqual(set(metrics["mtp_top1"]), {"t+2", "t+3"})
        for value in metrics["mtp_top1"].values():
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)
        self.assertEqual([item["offset"] for item in metrics["mtp"]], [2, 3])

        router = metrics["router_utilization"]
        self.assertIsNotNone(router)
        expected_router_layers = cfg.moe_layers + (
            cfg.mtp - 1 if cfg.mtp_block else 0
        )
        self.assertEqual(len(router["layers"]), expected_router_layers)
        for layer in router["layers"]:
            self.assertEqual(layer["n_experts"], cfg.n_experts)
            self.assertEqual(len(layer["utilization"]), cfg.n_experts)
            self.assertAlmostEqual(sum(layer["utilization"]), 1.0, places=5)
            self.assertGreaterEqual(layer["max_share"], 1.0 / cfg.n_experts - 1e-9)
            self.assertLessEqual(layer["dead_experts"], cfg.n_experts)

        self.assertIsNotNone(metrics["aux_load_balance"])
        self.assertIsNotNone(metrics["aux_router_z"])
        positions = metrics["val_positions"]
        self.assertEqual(positions["count"], 6)
        self.assertFalse(positions["replacement"])
        self.assertGreaterEqual(positions["min"], positions["allowed_min"])
        self.assertLessEqual(positions["max"], positions["allowed_max"])
        self.assertEqual(len(positions["digest"]), 16)

    def test_fixed_positions_are_deterministic(self) -> None:
        first, meta_first = eval_v3.make_positions(40, 16, 6, 3, 5, "unit")
        second, meta_second = eval_v3.make_positions(40, 16, 6, 3, 5, "unit")
        self.assertEqual(first, second)
        self.assertEqual(meta_first["digest"], meta_second["digest"])

        other, meta_other = eval_v3.make_positions(40, 16, 6, 3, 6, "unit")
        self.assertNotEqual(meta_first["digest"], meta_other["digest"])
        self.assertEqual(len(other), 6)

    def test_checkpoint_dict_and_config_json(self) -> None:
        torch.manual_seed(1)
        cfg = tiny_config()
        model = V3LM(cfg)

        with tempfile.TemporaryDirectory() as temp:
            temp_path = Path(temp)
            checkpoint = temp_path / "checkpoint.pt"
            torch.save(
                {"model": model.state_dict(), "config": asdict(cfg), "step": 7},
                checkpoint,
            )

            loaded, loaded_cfg, metadata = eval_v3.load_checkpoint(checkpoint)
            self.assertEqual(loaded_cfg, cfg)
            self.assertEqual(metadata["step"], 7)
            self.assertEqual(metadata["config_source"], "checkpoint")
            with torch.no_grad():
                logits, mtp, _aux = loaded(torch.zeros(1, 4, dtype=torch.long))
            self.assertEqual(tuple(logits.shape), (1, 4, cfg.vocab_size))
            self.assertEqual(len(mtp), cfg.mtp - 1)

            bare = temp_path / "bare.pt"
            torch.save({"model": model.state_dict()}, bare)
            with self.assertRaises(ValueError):
                eval_v3.load_checkpoint(bare)

            config_json = temp_path / "config.json"
            config_json.write_text(
                json.dumps({"model": asdict(cfg)}), encoding="utf-8"
            )
            loaded, loaded_cfg, metadata = eval_v3.load_checkpoint(
                bare, config_json=config_json
            )
            self.assertEqual(loaded_cfg, cfg)
            self.assertEqual(metadata["config_source"], "config-json")
            with torch.no_grad():
                logits, _mtp, _aux = loaded(torch.zeros(1, 4, dtype=torch.long))
            self.assertEqual(tuple(logits.shape), (1, 4, cfg.vocab_size))

    def test_multi_corpus_cli(self) -> None:
        torch.manual_seed(2)
        cfg = tiny_config(vocab_size=256, ctx=8, mtp=2)
        model = V3LM(cfg)

        with tempfile.TemporaryDirectory() as temp:
            temp_path = Path(temp)
            checkpoint = temp_path / "checkpoint.pt"
            torch.save(
                {"model": model.state_dict(), "config": asdict(cfg)}, checkpoint
            )
            alpha = temp_path / "alpha.txt"
            beta = temp_path / "beta.txt"
            alpha.write_text("abcde" * 500, encoding="utf-8")
            beta.write_text("zyxwv" * 500, encoding="utf-8")
            out = temp_path / "nested" / "eval.json"

            code = eval_v3.main(
                [
                    "--checkpoint",
                    str(checkpoint),
                    "--corpus",
                    f"alpha={alpha}",
                    "--corpus",
                    f"beta={beta}",
                    "--tokenizer",
                    "bytes",
                    "--ctx",
                    "8",
                    "--batch-size",
                    "2",
                    "--batches",
                    "2",
                    "--device",
                    "cpu",
                    "--out",
                    str(out),
                ]
            )
            self.assertEqual(code, 0)
            self.assertTrue(out.is_file())
            result = json.loads(out.read_text(encoding="utf-8"))

        self.assertEqual(result["mode"], "checkpoint")
        self.assertEqual(result["ctx"], 8)
        self.assertEqual(set(result["corpora"]), {"alpha", "beta"})
        self.assertEqual(result["tokens"], 8)
        self.assertEqual(set(result["mtp_top1"]), {"t+2"})
        for name in ("alpha", "beta"):
            metrics = result["corpora"][name]
            self.assertEqual(metrics["tokens"], 4)
            self.assertEqual(metrics["tokens_available"], 2500)
            self.assertEqual(metrics["tokenizer"], "bytes")
            self.assertEqual(metrics["tokenizer_vocab_size"], 256)
            self.assertEqual(metrics["raw_bytes"], 2500)
            self.assertEqual(metrics["encoded_tokens"], 2500)
            self.assertAlmostEqual(metrics["bytes_per_token"], 1.0)
            self.assertAlmostEqual(
                metrics["bits_per_byte"], metrics["bits_per_token"]
            )
            self.assertGreater(metrics["bits_per_token"], 0.0)
            self.assertGreaterEqual(metrics["top1"], 0.0)
            self.assertLessEqual(metrics["top1"], metrics["top5"])
            self.assertLessEqual(metrics["top5"], 1.0)
            self.assertEqual(set(metrics["mtp_top1"]), {"t+2"})
            self.assertIsNotNone(metrics["router_utilization"])

    def test_random_smoke_cli(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "smoke" / "eval.json"
            code = eval_v3.main(
                [
                    "--random-smoke",
                    "--batches",
                    "2",
                    "--batch-size",
                    "2",
                    "--device",
                    "cpu",
                    "--seed",
                    "11",
                    "--out",
                    str(out),
                ]
            )
            self.assertEqual(code, 0)
            self.assertTrue(out.is_file())
            result = json.loads(out.read_text(encoding="utf-8"))

        self.assertEqual(result["mode"], "random-smoke")
        self.assertIsNone(result["checkpoint"])
        self.assertEqual(result["tokens"], 4)
        self.assertEqual(result["config"]["mtp"], 3)
        self.assertGreater(result["parameter_count"], 0)
        self.assertEqual(set(result["mtp_top1"]), {"t+2", "t+3"})
        self.assertEqual(set(result["corpora"]), {"random-smoke"})

        smoke = result["corpora"]["random-smoke"]
        self.assertEqual(smoke["tokens"], 4)
        self.assertEqual(len(smoke["mtp"]), 2)
        self.assertTrue(math.isfinite(smoke["bits_per_token"]))
        self.assertGreater(smoke["bits_per_token"], 0.0)
        self.assertGreaterEqual(smoke["top1"], 0.0)
        self.assertLessEqual(smoke["top1"], smoke["top5"])
        self.assertLessEqual(smoke["top5"], 1.0)
        self.assertEqual(set(smoke["mtp_top1"]), {"t+2", "t+3"})
        self.assertIsNotNone(smoke["router_utilization"])
        smoke_cfg = result["config"]
        expected_router_layers = smoke_cfg["moe_layers"] + (
            smoke_cfg["mtp"] - 1 if smoke_cfg["mtp_block"] else 0
        )
        self.assertEqual(
            len(smoke["router_utilization"]["layers"]), expected_router_layers
        )
        for layer in smoke["router_utilization"]["layers"]:
            self.assertAlmostEqual(sum(layer["utilization"]), 1.0, places=5)

    def test_layered_tokenizer_loader(self) -> None:
        path = ROOT / "tokenizers" / "layered_v1" / "layered_tokenizer.json"
        if not path.is_file():
            self.skipTest("layered tokenizer artifact not present")
        tokenizer = eval_v3.load_tokenizer("layered", layered_model=path)
        ids = tokenizer.encode("Hello layered tokenizer! Second sentence.")
        self.assertGreater(len(ids), 0)
        self.assertEqual(tokenizer.vocab_size, 64000)
        self.assertTrue(all(0 <= int(i) < tokenizer.vocab_size for i in ids))

    @unittest.skipUnless(BPE_PATH.is_file(), "z1 BPE tokenizer JSON not present")
    def test_bpe_tokenizer_loader(self) -> None:
        tokenizer = eval_v3.load_tokenizer("bpe", BPE_PATH)
        first = tokenizer.encode("Hello world! The quick brown fox.")
        second = tokenizer.encode("Hello world! The quick brown fox.")
        self.assertEqual(first, second)
        self.assertGreater(len(first), 0)
        self.assertTrue(all(0 <= token_id < tokenizer.vocab_size for token_id in first))


if __name__ == "__main__":
    unittest.main(verbosity=2)
