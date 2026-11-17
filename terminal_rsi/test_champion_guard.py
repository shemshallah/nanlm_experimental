"""Checkpoint-protection regression: real files; never overwrite older winner."""
from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from terminal_rsi.champion_guard import consider_candidate, sha256


def receipt(score=1.33, digest="english-book-dev-v1", updates=1000):
    return {
        "old_v103_logits_identical_at_gate_zero": True,
        "new_dev_id_level_train_disjoint_proven": True,
        "test_untouched": True,
        "book_disjoint_dev_digest": digest,
        "new_dev_books": 64,
        "new_dev_scored_positions": 65536,
        "upgrade_context": 256,
        "after": {"bpb": score, "evaluated_bytes": 1024},
        "final_step": updates,
        "rsi_self_construction": {"proposals": 20, "accepted": 6},
    }


class ChampionGuardTests(unittest.TestCase):
    def test_protect_1p33_and_refuse_worse_bpb(self):
        with TemporaryDirectory() as temp:
            home = Path(temp)
            checkpoint = home / "candidate.pt"
            folder = home / "champions"
            checkpoint.write_bytes(b"trained weights and full AdamW at 1.33")
            first = consider_candidate(checkpoint, receipt(), folder)
            self.assertEqual(first["status"], "initialized")
            best = json.loads((folder / "best.json").read_text())
            preserved = folder / best["checkpoint"]
            protected_digest = sha256(preserved)
            checkpoint.write_bytes(b"new weights but regression to 3.34")
            next_run = consider_candidate(checkpoint, receipt(3.34, updates=2000), folder)
            self.assertEqual(next_run["status"], "retained")
            self.assertEqual(sha256(preserved), protected_digest)
            self.assertEqual(json.loads((folder / "best.json").read_text())["bpb"], 1.33)

    def test_promote_strict_gain_preserves_older_immutable_champion(self):
        with TemporaryDirectory() as temp:
            home = Path(temp)
            checkpoint = home / "candidate.pt"
            folder = home / "champions"
            checkpoint.write_bytes(b"old checkpoint")
            consider_candidate(checkpoint, receipt(2.0), folder)
            old = json.loads((folder / "best.json").read_text())
            checkpoint.write_bytes(b"new checkpoint")
            result = consider_candidate(checkpoint, receipt(1.80, updates=2000), folder)
            self.assertEqual(result["status"], "improved")
            self.assertEqual((folder / old["checkpoint"]).read_bytes(), b"old checkpoint")
            latest = json.loads((folder / "best.json").read_text())
            self.assertEqual(latest["previous_champion"], old["checkpoint"])
            self.assertEqual((folder / latest["checkpoint"]).read_bytes(), b"new checkpoint")

    def test_refuse_mismatched_dev_and_corrupted_champion(self):
        with TemporaryDirectory() as temp:
            home = Path(temp)
            checkpoint = home / "candidate.pt"
            folder = home / "champions"
            checkpoint.write_bytes(b"valid initial checkpoint")
            consider_candidate(checkpoint, receipt(), folder)
            with self.assertRaisesRegex(ValueError, "DIFFERENT DEV"):
                consider_candidate(checkpoint, receipt(0.1, digest="other-books"), folder)
            winner = json.loads((folder / "best.json").read_text())
            (folder / winner["checkpoint"]).write_bytes(b"corrupted")
            with self.assertRaisesRegex(ValueError, "missing or corrupted"):
                consider_candidate(checkpoint, receipt(0.1), folder)

    def test_refuse_unverified_or_nonfinite_scores(self):
        with TemporaryDirectory() as temp:
            home = Path(temp)
            checkpoint = home / "candidate.pt"
            checkpoint.write_bytes(b"valid")
            folder = home / "champions"
            invalid = receipt()
            invalid["new_dev_id_level_train_disjoint_proven"] = False
            with self.assertRaisesRegex(ValueError, "disjointness"):
                consider_candidate(checkpoint, invalid, folder)
            with self.assertRaisesRegex(ValueError, "finite"):
                consider_candidate(checkpoint, receipt(float("nan")), folder)


if __name__ == "__main__":
    unittest.main()
