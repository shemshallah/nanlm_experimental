"""R7.34 byte-normalized book bootstrap and output-path safeguards."""
import tempfile
import unittest
from pathlib import Path

import torch

from fa1_r734_byte_bootstrap import book_byte_ratio_bootstrap
from fa1_r734_checkpoint_guard import validate_artifact_paths


class BookWeightedByteBootstrapTests(unittest.TestCase):
    def test_pool_matches_literal_byte_metric_not_word_average(self):
        # Two small positive words, one large negative word, across books.
        # Pooled byte gain: (2*1 + 2*1 - 10*.5) / 14 = -1/14.
        delta = torch.tensor([1., 1., -.5])
        counts = torch.tensor([2, 2, 10])
        books = torch.tensor([1, 1, 2])
        report = book_byte_ratio_bootstrap(delta, counts, books, 734)
        self.assertAlmostEqual(report['book_pooled_estimate'], -1. / 14.)
        self.assertAlmostEqual(report['book_average'], .25)
        self.assertEqual(report['books'], 2)
        self.assertNotEqual(report['book_pooled_estimate'], float(delta.mean()))

    def test_whole_book_resamples_preserve_ratios_and_seed(self):
        delta = torch.tensor([.7, .2, -.1, .4, .8])
        counts = torch.tensor([1, 10, 4, 3, 8])
        books = torch.tensor([4, 4, 8, 8, 12])
        a = book_byte_ratio_bootstrap(delta, counts, books, 10)
        b = book_byte_ratio_bootstrap(delta, counts, books, 10)
        self.assertEqual(a, b)
        self.assertEqual(a['resampling_unit'], 'book')
        self.assertAlmostEqual(a['book_pooled_estimate'],
                               float((delta * counts).sum() / counts.sum()), places=6)
        self.assertTrue(all(-.100001 <= v <= .800001 for v in a['book_bootstrap_ci95']))

    def test_zero_byte_targets_do_not_affect_estimate(self):
        a = book_byte_ratio_bootstrap(torch.tensor([.4, -.5, 3.]),
                                      torch.tensor([2, 4, 0]),
                                      torch.tensor([1, 2, 3]), 7)
        b = book_byte_ratio_bootstrap(torch.tensor([.4, -.5]),
                                      torch.tensor([2, 4]),
                                      torch.tensor([1, 2]), 7)
        self.assertEqual(a, b)

    def test_rejects_invalid_shapes_and_counts(self):
        with self.assertRaisesRegex(ValueError, 'identical'):
            book_byte_ratio_bootstrap(torch.zeros(2), torch.ones(3),
                                       torch.ones(2), 1)
        with self.assertRaisesRegex(ValueError, 'nonnegative'):
            book_byte_ratio_bootstrap(torch.ones(2), torch.tensor([1, -1]),
                                       torch.tensor([1, 2]), 1)
        with self.assertRaisesRegex(ValueError, 'finite'):
            book_byte_ratio_bootstrap(torch.tensor([float('nan'), 1.]),
                                       torch.ones(2), torch.tensor([1, 2]), 1)


class ArtifactSafetyTests(unittest.TestCase):
    def test_separate_successor_and_report_allowed(self):
        with tempfile.TemporaryDirectory() as root:
            base = Path(root)
            parent = base / 'accepted.pt'
            parent.write_bytes(b'protected')
            validate_artifact_paths(parent, base/'r734.json', base/'unpromoted.pt')
            self.assertEqual(parent.read_bytes(), b'protected')

    def test_same_path_and_symlink_or_hardlink_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            base = Path(root)
            parent = base / 'accepted.pt'
            parent.write_bytes(b'protected')
            for report, candidate in (
                    (parent, ''),
                    (base/'report.json', parent),
                    (base/'same.pt', base/'same.pt')):
                with self.assertRaisesRegex(ValueError, 'alias'):
                    validate_artifact_paths(parent, report, candidate)
            alias = base / 'alias.pt'
            alias.symlink_to(parent)
            with self.assertRaisesRegex(ValueError, 'alias'):
                validate_artifact_paths(parent, base/'report.json', alias)
            alias.unlink()
            alias.hardlink_to(parent)
            with self.assertRaisesRegex(ValueError, 'alias'):
                validate_artifact_paths(parent, base/'report.json', alias)


if __name__ == '__main__':
    unittest.main()
