"""R32 history counterfactuals and joint geometry ablation regression tests."""
import contextlib
import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from eval_context_ladder import ce_improvement, main, unrelated_history_transform
from poincare_frontier import Config, MirrorLM


class TestR32ContextLadderControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_counterfactual_preserves_recent_context_and_targets(self):
        donor = np.arange(180, dtype=np.uint8)
        positions = np.array([96, 128])
        starts = np.array([5, 44])
        original = np.array([[5] * 16, [7] * 16], dtype=np.int64)
        replace = unrelated_history_transform(donor, positions, starts, 32, 16, 4)
        counterfactual = replace(original, positions)
        np.testing.assert_array_equal(counterfactual[:, -4:], original[:, -4:])
        np.testing.assert_array_equal(counterfactual[0, :-4], donor[21:33])
        np.testing.assert_array_equal(counterfactual[1, :-4], donor[60:72])
        np.testing.assert_array_equal(original, np.array([[5] * 16, [7] * 16]))
        with self.assertRaises(ValueError):
            unrelated_history_transform(donor, positions, starts, 32, 16, 16)
        with self.assertRaises(ValueError):
            unrelated_history_transform(donor, positions, [149, 44], 32, 16, 4)

    def test_positive_advantage_means_full_model_is_better(self):
        intact = {h: {'ce_nats_per_byte': 2.0} for h in ('+1', '+2', '+3')}
        disabled = {h: {'ce_nats_per_byte': 2.25} for h in ('+1', '+2', '+3')}
        self.assertEqual(ce_improvement(disabled, intact)['+1'], 0.25)

    def test_end_to_end_same_labels_and_all_ablation_branches(self):
        torch.manual_seed(38)
        cfg = Config(vocab_size=257, dim=32, layers=2, heads=4, context=32,
                     pairs=4, topk_pairs=2, expert_hidden=16, moe_every=2,
                     attention_window=16, symbiotic_rank=8, geometry_memory_chunk=8)
        model = MirrorLM(cfg)
        with torch.no_grad():
            geometry = model.blocks[-1].ff.symbiotic
            geometry.to_hidden.weight.normal_(0, .003)
            geometry.memory.gate.fill_(0.3)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            checkpoint = root / 'model.pt'
            test_file, donor_file = root / 'test.bin', root / 'donor.bin'
            torch.save({'config':vars(cfg), 'model':model.state_dict(), 'step':7},checkpoint)
            test_file.write_bytes(bytes(range(200)) * 3)
            donor_file.write_bytes(bytes(reversed(range(200))) * 3)

            def sha(file):
                return hashlib.sha256(file.read_bytes()).hexdigest()

            arguments = ['eval_context_ladder.py', '--checkpoint', str(checkpoint),
                         '--raw-test', str(test_file), '--expected-test-sha256', sha(test_file),
                         '--expected-checkpoint-sha256', sha(checkpoint),
                         '--unrelated-history', str(donor_file),
                         '--expected-unrelated-sha256', sha(donor_file),
                         '--contexts', '8,16,32', '--preserve-recent-bytes', '8',
                         '--positions', '2', '--batch', '1', '--threads', '1',
                         '--ablations']
            with patch.object(sys, 'argv', arguments), contextlib.redirect_stdout(io.StringIO()) as out:
                main()
            report = json.loads(out.getvalue())
            self.assertEqual(report['checkpoint_step'], 7)
            self.assertEqual(report['positions_per_context'], 2)
            self.assertEqual(set(report['results_by_context']), {'8','16','32'})
            self.assertNotIn('unrelated_history', report['results_by_context']['8'])
            for size in ('8','16','32'):
                row = report['results_by_context'][size]
                for name in ('results','no_geometric_memory','no_symbiotic_geometry',
                             'temporal_memory_ce_gain_nats','symbiotic_geometry_ce_gain_nats'):
                    self.assertIn(name, row)
                self.assertEqual(set(row['results']), {'+1','+2','+3'})
                if size != '8':
                    self.assertIn('correct_history_ce_advantage_nats', row)
                    self.assertIn('unrelated_history', row)


if __name__ == '__main__':
    unittest.main()
