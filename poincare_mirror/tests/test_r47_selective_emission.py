"""R47 frozen-weights selective-emission and paired-book regression tests."""
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from poincare_frontier import Config, MirrorLM
from compare_geometry_books import METRICS
from probe_r47_selective_emission import (book_comparison, diagnostic_gate,
                                          emission_variants, literal_byte_count)


class R47EmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    @staticmethod
    def model():
        torch.manual_seed(47)
        cfg = Config(vocab_size=257, dim=32, layers=2, heads=4,
            pairs=4, topk_pairs=2, expert_hidden=16, context=12,
            moe_every=2, symbiotic_rank=8, hierarchy_rank=8,
            micro_meta=True, simplex_transfer=True, reciprocal_scales=True,
            reciprocal_content_trust=True)
        model = MirrorLM(cfg).eval()
        with torch.no_grad():
            model.reciprocal.gate.fill_(.04)
        return model

    def test_prespecified_variants(self):
        variants = emission_variants(6)
        self.assertEqual(variants['full'], None)
        self.assertEqual(variants['coupling_only'], (0.,) * 6)
        self.assertEqual(variants['early_only'], (1.,1.,1.,0.,0.,0.))
        self.assertEqual(variants['late_only'], (0.,0.,0.,1.,1.,1.))
        self.assertEqual(set(variants), {'full','coupling_only','quarter',
                                          'half','early_only','late_only'})
        with self.assertRaises(ValueError):
            emission_variants(1)

    def test_rejects_malformed_gains_and_restores(self):
        model = self.model()
        bridge = model.reciprocal
        for bad in ((0.,), (0.,2.), (float('nan'),0.), (-.1,0.), (float('inf'),0.)):
            with self.assertRaises(ValueError):
                bridge.set_emission_gains(bad, model.cfg.layers)
        self.assertIsNone(bridge.emission_gains)
        bridge.set_emission_gains((0.,1.), model.cfg.layers)
        self.assertEqual(bridge.emission_gains, (0.,1.))
        bridge.set_emission_gains(None, model.cfg.layers)
        self.assertIsNone(bridge.emission_gains)

    def test_original_logits_and_checkpoint_state_preserved(self):
        model = self.model()
        inputs = torch.tensor([[65, 66, 32, 67, 68, 69]])
        original_weights = {key: weight.clone() for key, weight in model.state_dict().items()}
        with torch.no_grad():
            reference = model(inputs)[0]
            model.reciprocal.set_emission_gains((1.,1.), 2)
            all_ones = model(inputs)[0]
            model.reciprocal.set_emission_gains((0.,0.), 2)
            coupling_only = model(inputs)[0]
            model.reciprocal.set_emission_gains(None, 2)
            restored = model(inputs)[0]
        self.assertTrue(torch.equal(reference, all_ones))
        self.assertTrue(torch.equal(reference, restored))
        self.assertGreater((reference-coupling_only).abs().max().item(), 0.)
        self.assertEqual(set(original_weights), set(model.state_dict()))
        self.assertTrue(all(torch.equal(value, model.state_dict()[key])
                            for key, value in original_weights.items()))

    def test_no_future_input_leakage(self):
        model = self.model()
        base = torch.tensor([[65, 66, 32, 67, 68, 69]])
        altered = base.clone()
        altered[0,-1] = 99
        with torch.no_grad():
            for gains in ((0.,0.), (.5,.5), (1.,0.), (0.,1.)):
                model.reciprocal.set_emission_gains(gains, 2)
                a = model(base)[0]
                b = model(altered)[0]
                self.assertTrue(torch.allclose(a[:,:-1], b[:,:-1], atol=1e-5))
        model.reciprocal.set_emission_gains(None, 2)

    def test_book_bootstrap_and_screening(self):
        baseline = [{key: 2. if key.startswith('ce_') else .4 for key in METRICS}
                    for _ in range(12)]
        candidate = [{key: v-.1 if key.startswith('ce_') else v+.01
                      for key,v in row.items()} for row in baseline]
        gains = book_comparison(baseline, candidate, 500, 47, comparisons=5)
        self.assertGreater(gains['ce_1']['ci95_bonferroni'][0], 0.)
        self.assertGreater(gains['top5_1']['mean_gain'], 0.)
        self.assertTrue(diagnostic_gate(gains))
        candidate[0]['ce_nonspace_1'] += 5.
        bad = book_comparison(baseline, candidate, 500, 47, comparisons=5)
        self.assertFalse(diagnostic_gate(bad))
        with self.assertRaises(ValueError):
            book_comparison(baseline, candidate[:10])

    def test_raw_byte_denominator_excludes_eos(self):
        import numpy as np
        mem = np.array([65, 66, 256, 67, 68, 69, 70], dtype=np.uint16)
        self.assertEqual(literal_byte_count(mem, [np.array([0])], 4), 3)


if __name__ == '__main__':
    unittest.main()
