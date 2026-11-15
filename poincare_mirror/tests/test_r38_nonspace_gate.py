"""R38 follow-up: guard against gains dominated by the ASCII space byte."""
import argparse
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from poincare_frontier import Config, MirrorLM, content_balanced_ce, prepare_books
from scripts.compare_geometry_books import (
    METRICS, bootstrap_gain, measure_books, passes_content_gate)


class SpaceBiasedPredictor:
    """Deterministic distribution, deliberately better on space than content."""
    def __init__(self, space_logit):
        self.space_logit = space_logit

    def eval(self):
        return self

    def __call__(self, ids):
        logits = torch.zeros(*ids.shape, 257)
        logits[..., 32] = self.space_logit
        return logits, logits, {}

    def future_logits(self, hidden):
        return hidden, hidden


class NonSpaceObjectiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_default_matches_exact_inherited_cross_entropy(self):
        logits = torch.randn(2, 5, 257, generator=torch.Generator().manual_seed(9))
        targets = torch.tensor([[32, 65, 32, 100, 32], [34, 32, 90, 20, 32]])
        inherited = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        self.assertTrue(torch.equal(content_balanced_ce(logits, targets), inherited))

    def test_content_weight_has_expected_normalization_and_gradients(self):
        logits = torch.zeros(1, 4, 257, requires_grad=True)
        with torch.no_grad():
            logits[..., 32] = 3.0
        targets = torch.tensor([[32, 65, 32, 65]])
        per = F.cross_entropy(logits.flatten(0,1), targets.flatten(), reduction='none')
        expected = (per[0] + 2*per[1] + per[2] + 2*per[3])/6
        actual = content_balanced_ce(logits, targets, 2.0)
        self.assertTrue(torch.allclose(actual, expected))
        actual.backward()
        self.assertTrue(torch.isfinite(logits.grad).all().item())
        self.assertGreater(logits.grad.abs().sum().item(), 0)

    def test_rejects_invalid_weights(self):
        logits = torch.zeros(1, 2, 257)
        labels = torch.tensor([[32, 65]])
        for weight in (float('nan'), float('inf'), -1.0, 0.5):
            with self.subTest(weight=weight), self.assertRaises(ValueError):
                content_balanced_ce(logits, labels, weight)

    def test_unweighted_validation_ce_remains_separate_from_training_loss(self):
        cfg = Config(vocab_size=257,dim=32,heads=4,layers=2,pairs=4,
                     topk_pairs=2,expert_hidden=24,context=8,moe_every=2,
                     future2_weight=0.,future3_weight=0.)
        model = MirrorLM(cfg).eval()
        batch = torch.tensor([[65, 32, 65, 32, 65, 32, 65, 32, 65, 32, 65]])
        original = model.loss_with_future(batch)
        weighted = model.loss_with_future(batch, nonspace_weight=2.0)
        self.assertTrue(torch.equal(original['next_ce'], weighted['next_ce']))
        self.assertTrue(torch.equal(original['loss'], original['next_ce'] +
                                    cfg.balance_weight * original['balancing']))
        self.assertTrue(torch.isfinite(weighted['content_balanced_ce']).item())

    def test_cpu_cli_resumes_same_adam_history_with_content_weighting(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            books = root/'books'
            books.mkdir()
            for index in range(6):
                (books/f'{index:02}.txt').write_text(
                    (f'Book {index} contains different content bytes. '*12), encoding='utf-8')
            data = root/'data'
            prepare_books(argparse.Namespace(
                source_dir=str(books),out_dir=str(data),valid_fraction=.25,
                min_bytes=100,max_doc_bytes=10000,seed=2026,tokenizer_model=None))
            cli = [sys.executable, str(Path(__file__).resolve().parents[1]/'poincare_frontier.py'),
                   'train','--data-dir',str(data),'--run-dir',str(root/'run'),
                   '--steps','2','--batch','1','--context','8','--dim','32',
                   '--layers','2','--heads','4','--pairs','4','--topk','2',
                   '--expert-hidden','16','--moe-every','2',
                   '--train-context','8','--eval-context','8','--cpu','--cpu-threads','1',
                   '--eval-batches','1','--eval-every','2','--save-every','2',
                   '--future2','0','--future3','0','--nonspace-loss-weight','1.5',
                   '--report-byte-metrics']
            for _ in range(2):
                proc = subprocess.run(cli,capture_output=True,text=True,timeout=90)
                self.assertEqual(proc.returncode,0,proc.stdout+'\n'+proc.stderr)
                report = next(json.loads(line) for line in proc.stdout.splitlines()
                              if line.startswith('{"step":'))
                self.assertEqual(set(report['byte_metrics']),{'next','plus2','plus3'})
                self.assertEqual(report['byte_metrics']['next']['all']['positions'],8)
                self.assertGreaterEqual(report['byte_metrics']['next']['all']['top5_percent'],
                                        report['byte_metrics']['next']['all']['top1_percent'])
                self.assertAlmostEqual(report['byte_metrics']['next']['all']['ce_nats_per_byte'],
                                       report['validation_ce'],places=5)
            saved = torch.load(root/'run'/'latest.pt',map_location='cpu',weights_only=False)
            self.assertEqual(saved['step'],4)
            self.assertGreater(len(saved['optimizer']['state']),0)
            moments = [v['step'].item() for v in saved['optimizer']['state'].values()
                       if 'step' in v]
            self.assertTrue(moments)
            self.assertTrue(all(n == 4 for n in moments))
            records = [h for h in saved['growth_history']
                       if h['kind']=='nonspace_target_weighted_next_byte_loss']
            self.assertEqual([h['step'] for h in records],[0,2])


class BookLevelContentGateTests(unittest.TestCase):
    def test_space_only_gain_is_detected_and_rejected(self):
        # 12 independent books, 75% space targets, 25% harder content bytes.
        books = [np.array([32,32,32,65]*64,dtype=np.int64) for _ in range(12)]
        mem = np.concatenate(books)
        records = [np.array([i*256 + p for p in (0,20,40,60)])
                   for i in range(12)]
        original = measure_books(SpaceBiasedPredictor(0),mem,records,8)
        candidate = measure_books(SpaceBiasedPredictor(3),mem,records,8)
        self.assertEqual(set(original[0]),set(METRICS))
        self.assertLess(candidate[0]['ce_1'],original[0]['ce_1'])
        self.assertGreater(candidate[0]['ce_nonspace_1'],original[0]['ce_nonspace_1'])
        self.assertLess(candidate[0]['ce_space_1'],original[0]['ce_space_1'])
        gain = bootstrap_gain([x['ce_nonspace_1'] for x in original],
                              [x['ce_nonspace_1'] for x in candidate],resamples=100)
        self.assertLess(gain['ci95'][1],0)
        comparison = {
            'ce_1':{'ci95':[0.1,0.2]},
            'ce_nonspace_1':{'ci95':gain['ci95']},
            'acc_nonspace_1':{'mean_gain':0},
            'top5_nonspace_1':{'mean_gain':0},
            'top5_1':{'ci95':[0.,0.01]},
            'ce_2':{'mean_gain':0},
            'ce_3':{'mean_gain':0}}
        feedback = {'ce_1':{'ci95':[.01,.03]},
                    'ce_nonspace_1':{'ci95':[.01,.03]}}
        self.assertFalse(passes_content_gate(comparison,feedback,'valid'))
        comparison['ce_nonspace_1']={'ci95':[.001,.02]}
        self.assertTrue(passes_content_gate(comparison,feedback,'valid'))
        self.assertFalse(passes_content_gate(comparison,feedback,'test'))

    def test_missing_space_subset_fails_closed(self):
        mem = np.array([65]*100,dtype=np.int64)
        with self.assertRaisesRegex(ValueError,'space and non-space'):
            measure_books(SpaceBiasedPredictor(0),mem,[np.array([0])],8)


if __name__ == '__main__':
    unittest.main()
