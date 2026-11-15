"""v102 byte specialist transfers algorithms into RSI without donor checkpoint."""
from pathlib import Path
import struct
import tempfile
import unittest
import numpy as np

from terminal_rsi.byte_specialist import nancm_v102 as donor
from terminal_rsi.byte_specialist.controller import (
    SOURCE_SHA256, assign_books, audit_records, digest, select_windows, write_windows,
)
from terminal_rsi.byte_specialist.residual import ResidualPolicy, instrumented_source


class SpecialistTests(unittest.TestCase):
    def test_exact_v102_source_recovered(self):
        self.assertEqual(digest(Path(donor.__file__).read_bytes()), SOURCE_SHA256)

    def test_complete_book_split_and_duplicate_suppression(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for i in range(80):
                (root / f'book-{i:03}.txt').write_bytes((f'Book {i:03} line words.\n' * 140).encode())
            (root / 'duplicate.txt').write_bytes((root / 'book-000.txt').read_bytes())
            parts = assign_books(root)
            names = [name for group in parts.values() for name, _ in group]
            self.assertEqual(len(names), 80)
            self.assertEqual(len(names), len(set(names)))
            self.assertEqual(parts, assign_books(root))
            self.assertTrue(all(len(parts[k]) > 2 for k in ('train','dev','test')))

    def test_byte_window_provenance_and_exact_scoring_scope(self):
        books = [(f'book-{i}', (f'English book {i} with words.\n' * 140).encode())
                 for i in range(10)]
        bpe = donor.BPE([])
        windows = select_windows(books, bpe, count=4, width=512, skip=32, seed=3)
        self.assertEqual(len({name for name, _ in windows}), 4)
        with tempfile.TemporaryDirectory() as tmp:
            meta = write_windows(Path(tmp), 'dev', windows, bpe, skip=32)
            self.assertEqual(meta['scored_bytes'], 4 * 480)
            self.assertEqual(meta['books'], 4)
            self.assertEqual(len((Path(tmp) / 'win_dev.u').read_bytes()), 4 * 512)
            self.assertEqual(meta, write_windows(Path(tmp), 'dev', windows, bpe, skip=32))

    def test_independent_record_auditor_rejects_bad_denominators(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            p = work / 'specialist_trial_dev_frozen.bin'
            p.write_bytes(struct.pack('<fBBBB', 0.69314718, 1, 1, 0, 0) * 8)
            manifest = dict(width=6, skip=2, development=dict(scored_bytes=8))
            result = audit_records(work, 'trial', 'dev', 'frozen', manifest)
            self.assertEqual((result['scored_bytes'],result['books']), (8,2))
            self.assertAlmostEqual(result['bpc'], 1., places=5)
            self.assertEqual(result['top1'],1.)
            p.write_bytes(p.read_bytes()[:-8])
            with self.assertRaises(ValueError):
                audit_records(work, 'trial', 'dev', 'frozen', manifest)

    def test_donor_instrumentation_preserves_original_and_is_guarded(self):
        original = donor.CSRC
        output = instrumented_source()
        self.assertEqual(original, donor.CSRC)
        self.assertIn('if(dist_out && !online) fwrite(pr256',output)
        self.assertIn('else if(!strcmp(argv[i],"-distout"))',output)

    def test_residual_real_adam_continuation_and_metrics(self):
        rng=np.random.default_rng(123)
        targets=np.full(400,11,dtype=np.int64)
        p=rng.uniform(.01,1.,(400,256))
        p/=p.sum(axis=1,keepdims=True)
        model=ResidualPolicy()
        before=model.score(p,targets,40)
        first=model.train(p,targets,steps=32,lr=.02,phase='RSR')
        previous_m=model.m.copy()
        previous_weights=model.weights.copy()
        second=model.train(p,targets,steps=32,lr=.005,phase='RPR')
        after=model.score(p,targets,40)
        self.assertEqual(model.t,64)
        self.assertEqual(first[-1]['adam_step'],32)
        self.assertEqual(second[-1]['adam_step'],64)
        self.assertTrue(np.any(previous_m!=model.m))
        self.assertTrue(np.any(previous_weights!=model.weights))
        self.assertLess(after['bpc'],before['bpc'])
        self.assertGreaterEqual(after['top1'],before['top1'])
        self.assertEqual(model.state()['adam_step'],64)
        with tempfile.TemporaryDirectory() as tmp:
            saved=Path(tmp)/'residual.npz'
            model.save_checkpoint(saved)
            restored=ResidualPolicy.load_checkpoint(saved)
            self.assertEqual(restored.state(),model.state())
            self.assertEqual(restored.score(p,targets,40),model.score(p,targets,40))
            # Resume from exactly the stored moments; no replacement model.
            restored.train(p,targets,steps=8,lr=.001,phase='RPR')
            self.assertEqual(restored.t,72)


if __name__ == '__main__':
    unittest.main()
