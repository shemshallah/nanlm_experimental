#!/usr/bin/env python3
"""Preset and CLI-default contract tests for the 115M z3 model."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from presets import PRESET_PARAM_COUNTS, get_preset
from train_v3 import apply_preset, build_parser, make_config
from v3_model import V3LM


class PresetTests(unittest.TestCase):
    def test_ngram_memory_is_off_by_default_and_opt_in(self):
        parser = build_parser()
        self.assertFalse(
            parser.parse_args(["--corpus", "x"]).ngram_memory
        )
        self.assertTrue(
            parser.parse_args(
                ["--corpus", "x", "--ngram-memory"]
            ).ngram_memory
        )
        self.assertFalse(
            parser.parse_args(
                ["--corpus", "x", "--no-ngram-memory"]
            ).ngram_memory
        )

    def test_115m_preset_parameter_count_and_layout(self):
        parser = build_parser()
        argv = ["--preset", "115m", "--corpus", "x"]
        args = parser.parse_args(argv)
        apply_preset(args, argv)
        self.assertFalse(args.ngram_memory)
        self.assertTrue(args.document_framing)
        self.assertEqual(args.moe_branches, 4)
        cfg = make_config(args, 64060)
        model = V3LM(cfg)
        self.assertEqual(
            model.parameter_count(), PRESET_PARAM_COUNTS["115m"]
        )
        self.assertFalse(cfg.mtp_attention_gate)
        self.assertTrue(cfg.mtp_attention_global)
        self.assertTrue(cfg.mtp_position_embed)
        self.assertTrue(cfg.mtp_gate)
        self.assertFalse(cfg.use_ngram_memory)
        self.assertTrue(cfg.feedback_token_residual)
        self.assertTrue(cfg.feedback_route_concept)
        self.assertTrue(cfg.feedback_object_carry)
        self.assertTrue(cfg.feedback_memory_mtp)
        self.assertTrue(cfg.feedback_mtp_carry)
        trunk = model.blocks[-1].mlp.inner
        self.assertEqual((trunk.branches, trunk.experts_per_branch), (4, 3))
        self.assertEqual(trunk.topk, 2)
        self.assertFalse(trunk.branch_degenerate)
        mtp = model.mtp_blocks[0].moe
        self.assertEqual((mtp.branches, mtp.experts_per_branch), (2, 4))
        self.assertFalse(mtp.branch_degenerate)

    def test_kaggle10m_preset_keeps_115m_geometry(self):
        parser = build_parser()
        argv = ["--preset", "kaggle10m", "--corpus", "x"]
        args = parser.parse_args(argv)
        apply_preset(args, argv)
        self.assertEqual(args.max_seconds, 600.0)
        self.assertEqual(args.steps, 1000)
        self.assertEqual(args.batch_size, 4)
        self.assertEqual(args.warmup_steps, 30)
        self.assertEqual(args.feedback_gate_warmup_steps, 30)
        self.assertEqual(args.mtp_weight, 0.1)
        self.assertEqual(args.mtp_aux_weight, 0.0)
        self.assertEqual(args.byte_sequence_weight, 0.0)
        self.assertTrue(args.no_ema)
        cfg = make_config(args, 64060)
        model = V3LM(cfg)
        self.assertEqual(
            model.parameter_count(), PRESET_PARAM_COUNTS["115m"]
        )
        self.assertEqual(
            PRESET_PARAM_COUNTS["kaggle10m"], PRESET_PARAM_COUNTS["115m"]
        )
        self.assertEqual(cfg.concept_top_vertices, 4)
        self.assertEqual(cfg.concept_top_pairs, 2)
        self.assertEqual(cfg.concept_top_triples, 0)

    def test_next_move_experiment_presets(self):
        parser = build_parser()
        for name in ("kaggle10m_obj", "kaggle10m_route", "kaggle10m_mtp64"):
            argv = ["--preset", name, "--corpus", "x"]
            args = parser.parse_args(argv)
            apply_preset(args, argv)
            self.assertEqual(args.ctx, 512)
            self.assertEqual(args.warmup_steps, 30)
        argv = ["--preset", "kaggle10m_obj", "--corpus", "x"]
        args = parser.parse_args(argv); apply_preset(args, argv)
        self.assertAlmostEqual(args.object_gate_init, 0.05)
        argv = ["--preset", "kaggle10m_route", "--corpus", "x"]
        args = parser.parse_args(argv); apply_preset(args, argv)
        self.assertTrue(args.gate_route_attention)
        argv = ["--preset", "kaggle10m_mtp64", "--corpus", "x"]
        args = parser.parse_args(argv); apply_preset(args, argv)
        self.assertEqual(args.mtp_output_rank, 64)
        cfg = make_config(args, 64060)
        model = V3LM(cfg)
        self.assertEqual(
            model.parameter_count(), PRESET_PARAM_COUNTS["kaggle10m_mtp64"]
        )
        self.assertEqual(model.mtp_output.shape[-1], 64)
        self.assertEqual(model.mtp_heads[0].out_features, 64)

    def test_lr_sweep_presets(self):
        parser = build_parser()
        for name, lr, steps in (
            ("kaggle10m_lr3e4", 3e-4, 1000),
            ("kaggle10m_lr3e4_decay", 3e-4, 600),
            ("kaggle10m_lr4e4", 4e-4, 1000),
        ):
            argv = ["--preset", name, "--corpus", "x"]
            args = parser.parse_args(argv)
            apply_preset(args, argv)
            self.assertAlmostEqual(args.lr, lr)
            self.assertEqual(args.steps, steps)
            self.assertEqual(args.batch_size, 4)
            self.assertEqual(args.warmup_steps, 30)

    def test_explicit_flag_beats_preset(self):
        parser = build_parser()
        argv = [
            "--preset", "115m", "--corpus", "x", "--steps", "7",
            "--moe-branches", "6",
        ]
        args = parser.parse_args(argv)
        apply_preset(args, argv)
        self.assertEqual(args.steps, 7)
        self.assertEqual(args.moe_branches, 6)
        self.assertEqual(get_preset("115m")["steps"], 10_000)

    def test_negative_boolean_flag_beats_preset(self):
        parser = build_parser()
        argv = [
            "--preset", "115m", "--corpus", "x",
            "--no-document-framing", "--ngram-memory",
        ]
        args = parser.parse_args(argv)
        apply_preset(args, argv)
        self.assertFalse(args.document_framing)
        self.assertTrue(args.ngram_memory)


if __name__ == "__main__":
    unittest.main(verbosity=2)
