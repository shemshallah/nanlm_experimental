"""v102 byte specialist transfers algorithms into RSI without donor checkpoint."""
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest
import numpy as np

from terminal_rsi.byte_specialist import nancm_v102 as donor
from terminal_rsi.byte_specialist.controller import (
    SOURCE_SHA256, assign_books, audit_records, digest, select_windows, write_windows,
)
from terminal_rsi.byte_specialist.residual import (ResidualPolicy, build_executable,
                                                   instrumented_source)
from terminal_rsi.byte_specialist.contextual import ContextualResidual, previous_byte_classes


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
        self.assertIn('else if(!strcmp(argv[i],"-diskctx")) ctxdir=argv[++i];',output)

    def test_disk_backed_ctx_byte_exact_against_original_allocation(self):
        """Same v102 table indices and bit updates in both memory modes."""
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            train=(b'Only previously observed bytes update v102.\n' * 240)
            window=(b'New tokens, punctuation, and repeated letters.\n' * 7)[:256]
            (root/'train.bin').write_bytes(train)
            (root/'eval.bin').write_bytes(
                struct.pack('<II',1,32)+struct.pack('<I',len(window))+window)
            exe=build_executable(root)
            outputs=[]
            for mode in ('anonymous','disk'):
                stem=root/mode
                cmd=[str(exe),'-tshift','8','-eval',str(root/'eval.bin'),
                     '-o',str(stem),'-noonline','-distout',str(stem)+'.prob']
                if mode=='disk':cmd.extend(('-diskctx',str(root)))
                cmd.append(str(root/'train.bin'))
                result=subprocess.run(cmd,capture_output=True,text=True,timeout=60)
                self.assertEqual(result.returncode,0,result.stderr[-500:])
                self.assertIn('FINAL frozen',result.stdout)
                outputs.append(((root/(mode+'_frozen.bin')).read_bytes(),
                                (root/(mode+'.prob')).read_bytes()))
            self.assertEqual(outputs[0],outputs[1])

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

    def test_function_preserving_context_extension_and_optimizer_mapping(self):
        rng=np.random.default_rng(555)
        probs=rng.uniform(.001,1.,(80,256));probs/=probs.sum(axis=1,keepdims=True)
        targets=rng.integers(0,256,80);categories=rng.integers(0,7,80)
        parent=ResidualPolicy()
        parent.train(probs,targets,steps=16,lr=.01,phase='RSR')
        old_state=parent.state()
        extended=ContextualResidual(parent)
        self.assertEqual(extended.score(probs,targets,20,categories),
                         parent.score(probs,targets,20))
        self.assertEqual(extended.t,16)
        extended.train(probs,targets,categories,steps=8,lr=.002)
        self.assertEqual((extended.t,extended.ct),(24,8))
        self.assertEqual(parent.state(),old_state)
        guarded=extended.score_guarded(probs,targets,20,categories)
        original=parent.score(probs,targets,20)
        self.assertEqual(guarded['top1'],original['top1'])
        self.assertEqual(guarded['top5'],original['top5'])
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'contextual.npz'
            extended.checkpoint(path)
            restored=ContextualResidual.from_checkpoint(path)
            self.assertTrue(np.array_equal(restored.context,extended.context))
            self.assertTrue(np.array_equal(restored.cm,extended.cm))
            self.assertEqual(restored.t,24)
            self.assertEqual(restored.ct,8)
            self.assertEqual(restored.score(probs,targets,20,categories),
                             extended.score(probs,targets,20,categories))
            self.assertEqual(restored.score_guarded(probs,targets,20,categories),guarded)

    def test_context_classes_use_only_previous_observed_byte(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            left=b'A 0\n!'+b'b'*15
            right=b'Z 9\n?'+b'c'*15
            payload=struct.pack('<II',2,2)
            payload+=struct.pack('<I',len(left))+left
            payload+=struct.pack('<I',len(right))+right
            (root/'win_dev3.bin').write_bytes(payload)
            cats=previous_byte_classes(root,'dev3')
            self.assertEqual(len(cats),2*(len(left)-2))
            self.assertEqual(int(cats[0]),3)  # previous byte is space
            self.assertEqual(int(cats[len(left)-2]),3)


if __name__ == '__main__':
    unittest.main()
