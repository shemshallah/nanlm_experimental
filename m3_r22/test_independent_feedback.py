"""R22 synthetic structure, mathematical baselines and causal protocol tests."""
import unittest
import numpy as np

from m3_r20.signed_mirror_memory import SignedMirrorSuccessorMemory
from m3_r22.independent_feedback import (
    ConventionalCountNgram, feedback_at_prefix, signed_flux, candidate_logp,
)
from m3_r22 import evaluate as e


class R22Tests(unittest.TestCase):
    def setUp(self):
        self.prior=np.ones(256,dtype=np.float64)/256

    def test_fresh_splits_do_not_reuse_independent_test(self):
        self.assertEqual((e.FIT_START,e.GATE_START,e.DEV_START,e.TOTAL_BOOKS),
                         (156,172,176,188))
        self.assertEqual(len(e.R21_TEST_HASHES),24)
        self.assertTrue(all(len(h)==64 for h in e.R21_TEST_HASHES))
        self.assertEqual(len(e.GRID),27)

    def test_conventional_ngram_is_exactly_the_same_count_representation(self):
        geo=SignedMirrorSuccessorMemory(self.prior,horizon=32)
        ref=ConventionalCountNgram(self.prior,horizon=32)
        rng=np.random.default_rng(20261117)
        seq=rng.integers(0,12,size=100)
        for i in range(1,len(seq)):
            a,b=int(seq[i-1]),int(seq[i])
            p=int(seq[i-2]) if i>1 else None
            pg,tg=geo.predict(p,a)
            ps,ts=ref.predict(p,a)
            np.testing.assert_allclose(pg,ps,atol=0,rtol=0)
            np.testing.assert_allclose(tg,ts,atol=0,rtol=0)
            geo.observe(p,a,b)
            ref.observe(p,a,b)
        self.assertEqual(geo.invariants()["retained_edges"],32)

    def test_signed_flux_is_antisymmetric_and_zero_on_diagonal(self):
        geo=SignedMirrorSuccessorMemory(self.prior)
        for a,b in [(1,2),(1,2),(2,1),(2,3),(3,3)]:
            geo.observe(None,a,b)
        self.assertAlmostEqual(signed_flux(geo,1)[2],-signed_flux(geo,2)[1],places=15)
        self.assertEqual(signed_flux(geo,3)[3],0)
        self.assertTrue(np.all(np.isfinite(signed_flux(geo,2))))

    def test_pair_triangle_have_independent_residuals_and_zero_support_gates(self):
        geo=SignedMirrorSuccessorMemory(self.prior)
        base=np.log(self.prior)
        initial=feedback_at_prefix(geo,1,2,base)
        np.testing.assert_array_equal(initial["pair_residual"],np.zeros(256))
        np.testing.assert_array_equal(initial["triangle_residual"],np.zeros(256))
        for t in (3,3,3,4):
            geo.observe(1,2,t)
        active=feedback_at_prefix(geo,1,2,base)
        self.assertGreater(active["pair_count"],0)
        self.assertGreater(active["triangle_count"],0)
        self.assertGreater(np.abs(active["pair_residual"]).max(),0)
        self.assertGreater(np.abs(active["triangle_residual"]).max(),0)

    def test_function_preserving_frozen_default_and_normalization(self):
        base=np.log(self.prior)
        a=np.linspace(-1,1,256)
        b=np.cos(np.arange(256))
        c=np.sin(np.arange(256))
        self.assertIs(candidate_logp(base,a,b,c,(0,0,0)),base)
        actual=candidate_logp(base,a,b,c,(0.25,0.5,-0.25))
        self.assertAlmostEqual(float(np.exp(actual).sum()),1.0,places=12)
        self.assertTrue(np.isfinite(actual).all())

    def test_train_only_scale_selection_and_zero_fallback(self):
        fit=[{"scales":list(scales),"bpb":4.0-g,
              "gain_bpb_vs_r20":g,"nonspace_gain_bpb_vs_r20":g}
             for scales in e.GRID
             for g in [0.002 if scales==(0.5,0,0) else 0.0]]
        gate=[dict(r) for r in fit]
        chosen=e.select_train_only(fit,gate)
        self.assertEqual(chosen["scales"],[0.5,0,0])
        gate=[{**r,"bpb":4.0+abs(r["gain_bpb_vs_r20"]),
               "gain_bpb_vs_r20":-abs(r["gain_bpb_vs_r20"]),
               "nonspace_gain_bpb_vs_r20":-abs(r["nonspace_gain_bpb_vs_r20"])}
              for r in fit]
        chosen=e.select_train_only(fit,gate)
        self.assertEqual(chosen["scales"],[0.0,0.0,0.0])

    def test_causal_book_collection_matches_frozen_r20(self):
        # ASCII is valid UTF-8 and includes non-space/space targets in scoring.
        book=(b"separate mirror pair triangles with causal evidence. "*28)[:1400]
        collected=e.collect_book(book,self.prior)
        self.assertEqual(len(collected["data"]["targets"]),1400-e.WARMUP)
        self.assertLessEqual(collected["ngram_max_abs_diff"],2e-10)
        self.assertLessEqual(collected["frozen_alignment_max_abs_diff"],2e-10)
        np.testing.assert_array_equal(
            e.candidate_logp(collected["r20"],collected["pair_residual"],
                             collected["triangle_residual"],collected["flux"],(0,0,0)),
            collected["r20"])


if __name__=="__main__":
    unittest.main()
