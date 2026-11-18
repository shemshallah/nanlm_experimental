"""Real single-update CPU smoke of fresh Terminal-RSI architecture and new DEV."""
from __future__ import annotations

import contextlib
import io
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

from terminal_rsi.build_fresh_dev import build
from terminal_rsi.test_build_fresh_dev import english_book
from terminal_rsi.unified import PRESETS, Preset
from terminal_rsi.train_verbatim import main as train_main


class FreshOneUpdateEndToEnd(unittest.TestCase):
    def test_disposable_rsi_cpu_training_resumes_full_original_architecture(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            corpus_dir = root / "corpora"
            corpus_dir.mkdir()
            for bid in (1101, 1102, 1103, 1104):
                (corpus_dir / f"{bid}--unknown--story.txt").write_bytes(english_book(bid))
            train = root / "train.bin"
            train.write_bytes(english_book(1101))
            excluded = root / "train_ids.json"
            excluded.write_text('{"book_ids":[1101]}')
            dev = root / "dev"
            build(corpus_dir=corpus_dir, train_stream=train, output_dir=dev,
                  book_count=2, skip=64, positions_per_book=80,
                  train_book_ids=excluded)
            original_bpe = Path("terminal_rsi/reference/nancm_v103_bpe.txt")
            small = Preset("cpu", 32, 1, 4, 64, 1, 2, 3e-4, 1)
            output = root / "fresh_rsi_candidate.pt"
            args = [
                "--fresh-start", "--mode", "cpu",
                "--cpu-threads", "1",
                "--train-stream", str(train),
                "--bpe-merges", str(original_bpe),
                "--dev-bin", str(dev / "win_dev.bin"),
                "--dev-manifest", str(dev / "win_dev.manifest.json"),
                "--output", str(output),
                "--champion-dir", str(root / "protected_champion"),
                "--updates", "1", "--batch", "2",
                "--rsi-cycle-updates", "1", "--rsi-probe-positions", "2",
                "--max-dev-positions", "2", "--topk-probes", "1",
                "--upgrade-context", "32",
            ]
            with patch.dict(PRESETS, {"cpu": small}):
                with contextlib.redirect_stdout(io.StringIO()):
                    report = train_main(args)
            saved = torch.load(output, map_location="cpu", weights_only=True)
            self.assertEqual(saved["steps"], 1)
            self.assertEqual(report["actual_byte_updates_this_round"], 1)
            self.assertEqual(report["cpu_threads"], 1)
            self.assertTrue(report["fresh_start_untrained_v103"])
            self.assertEqual(len(saved["optimizer"]["param_groups"]), 4)
            self.assertTrue(saved["optimizer"]["state"])
            self.assertEqual(report["terminal_events"], [])
            self.assertEqual(report["terminal_reward_mode"], "predictive_dual_train_bpb")
            self.assertTrue(report["legacy_terminal_demo_disabled"])
            self.assertTrue(report["old_v103_logits_identical_at_gate_zero"])
            self.assertEqual(report["new_dev_books"], 2)
            self.assertEqual(report["rsi_mutations_verified"], 1)
            self.assertEqual(len(report["rsi_self_construction"]["events"]), 1)
            self.assertEqual(saved["training_state"]["rsi_self_builder"]["proposals"], 1)
            self.assertEqual(report["champion_guard"]["status"], "initialized")
            self.assertTrue(Path(report["champion_guard"]["champion_checkpoint"]).is_file())
            self.assertEqual(report["rsi_self_construction"]["schema"],
                             "terminal-rsi-self-builder-v3")


if __name__ == "__main__":
    unittest.main()
