"""Frozen M3 R21 independent-test software checks; only synthetic bytes."""
import unittest
from unittest.mock import patch
import numpy as np

from m3_r21 import locked_default_test as m
from m3_default.predictor import M3R21Predictor as M3DefaultPredictor
from m3_r20.signed_mirror_memory import SignedMirrorSuccessorMemory


class DefaultTestChecks(unittest.TestCase):
    def test_new_test_is_after_all_training_and_dev(self):
        self.assertEqual(m.R20_DEV_END, 132)
        self.assertEqual(m.NEW_TEST_BOOKS, 24)
        self.assertEqual(m.TOTAL_BOOKS, 156)

    def test_canonical_r20_has_frozen_scales_and_fallback(self):
        p = M3DefaultPredictor(np.ones(256))
        self.assertEqual(p.mode, "r20")
        p.observe(65)
        all_predictions = p.predict_all()
        self.assertEqual(set(m.ARMS), set(all_predictions) - {"oscillator"})
        self.assertTrue(np.all(np.isfinite(all_predictions["r20"])))
        self.assertAlmostEqual(np.exp(all_predictions["r20"]).sum(),1.0,places=12)

    def test_pair_and_triangle_distribution_match_conventional_count_reference(self):
        prior = np.ones(256)/256
        mem = SignedMirrorSuccessorMemory(prior,horizon=128)
        pair_count = np.zeros((256,256), dtype=np.int32)
        tris = {}
        seq = b"ababababcababcab"
        for i in range(1,len(seq)):
            p = seq[i-2] if i>=2 else None
            a = seq[i-1]
            b = seq[i]
            pair,triangle = mem.predict(p,a)
            base = (pair_count[a]+8*prior)/(pair_count[a].sum()+8)
            np.testing.assert_allclose(pair,base,rtol=0,atol=0)
            tri_count = tris.get((p,a)) if p is not None else None
            conventional = (tri_count+4*base)/(tri_count.sum()+4) if tri_count is not None else base
            np.testing.assert_allclose(triangle,conventional,rtol=0,atol=0)
            mem.observe(p,a,b)
            pair_count[a,b] += 1
            if p is not None:
                tris.setdefault((p,a),np.zeros(256,dtype=np.int32))[b]+=1

    def test_synthetic_score_fixture_has_no_fake_improvement(self):
        class FakePredictor:
            def __init__(self, prior):
                self.n = 0
            def predict_all(self):
                uniform = np.full(256,-np.log(256))
                return {name:uniform for name in m.ARMS}
            def observe(self,byte):
                self.n += 1
        books = [b"aaaaaa a"]*24
        with patch.object(m,"M3DefaultPredictor",FakePredictor), patch.object(m,"WARMUP",4):
            result=m.score_streamed_test(books,[f"fixture-{i}" for i in range(24)],
                                         np.ones(256)/256)
        self.assertEqual(result["test_positions"],96)
        self.assertAlmostEqual(result["test"]["r19"]["bpb"],8.0,places=12)
        self.assertAlmostEqual(result["test"]["r20"]["gain_bpb_vs_r19"],0.0,places=12)
        self.assertFalse(result["test"]["r20"]["independent_test_gate"])


if __name__=="__main__":
    unittest.main()
