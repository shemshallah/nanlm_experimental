"""Strict causal, migration, normalization and fallback checks for R26."""
from pathlib import Path
import tempfile
import unittest

import numpy as np

from m3_default.predictor import M3DefaultPredictor
from m3_r25.temporal import M3R25Predictor
from m3_r25.train_dev import FIT
from m3_r26.hybrid import (ByteContextTree, M3R26Predictor,
                           hybrid_readout, pack_features, WEIGHTS)
from m3_r26.train_dev import (ExpandedAdam, PARENT_M, PARENT_STEP,
                              PARENT_THETA, PARENT_V)


class R26Tests(unittest.TestCase):
    def test_migration_recovers_r25_bitwise_and_canonical_r24(self):
        prior = np.arange(1,257,dtype=float)
        zero = M3DefaultPredictor(prior)
        r25 = M3R25Predictor(prior, weights=PARENT_THETA)
        theta = np.zeros(WEIGHTS)
        theta[:3] = PARENT_THETA
        r26 = M3R26Predictor(prior, theta)
        for byte in b"alpha alpha gamma alpha alpha gamma" * 2:
            for p in (zero, r25, r26):
                p.observe(byte)
            np.testing.assert_array_equal(r26.predict_logp("r24"), zero.predict_logp())
            np.testing.assert_array_equal(r26.predict_logp(), r25.predict_logp())
            base, f = r26.base_and_hybrid_features()
            np.testing.assert_array_equal(f, r26.hybrid_features_from_base(base))
            np.testing.assert_array_equal(
                r26.base_and_hybrid_features(wrong_values=True)[1],
                r26.hybrid_features_from_base(base, wrong_values=True))
        r26.theta[:] = 0
        np.testing.assert_array_equal(r26.predict_logp(), zero.predict_logp())

    def test_hierarchical_tree_is_causal_and_normalized(self):
        prior = np.log(np.arange(1,257)/np.arange(1,257).sum())
        tree = ByteContextTree(near_horizon=12,long_horizon=24)
        for b in b"and here and now and here and now":
            tree.observe(b)
        count = tree.observed
        before = tree.features(prior)
        after = tree.features(prior)
        np.testing.assert_array_equal(before,after)
        self.assertEqual(tree.observed,count)
        self.assertEqual(before.shape,(4,256))
        self.assertTrue(np.isfinite(before).all())
        self.assertFalse(np.array_equal(before,tree.features(prior,wrong_values=True)))
        packed=pack_features(np.zeros((3,256)),before)
        self.assertEqual(packed.shape,(WEIGHTS,256))
        self.assertTrue(np.all(packed[7:,32]==0))
        p=hybrid_readout(prior,packed,np.full(WEIGHTS,.1))
        self.assertAlmostEqual(float(np.exp(p).sum()),1.0,places=12)
        for n in (1,2,3,4,6,8):
            for kind,limit in tree.horizons.items():
                self.assertLessEqual(sum(sum(row.values()) for row in tree.counts[kind][n].values()),limit)

    def test_missing_context_is_zero_feedback(self):
        base=np.log(np.ones(256)/256)
        tree=ByteContextTree()
        np.testing.assert_array_equal(tree.features(base),np.zeros((4,256)))
        with self.assertRaises(ValueError):
            tree.observe(256)
        with self.assertRaises(ValueError):
            pack_features(np.zeros((4,256)),np.zeros((4,256)))

    def test_optimizer_migrates_all_moments_without_reset(self):
        with tempfile.TemporaryDirectory() as d:
            checkpoint=Path(d)/"checkpoint.npz"
            np.savez(checkpoint, theta=PARENT_THETA, adam_m=PARENT_M,
                     adam_v=PARENT_V, adam_step=PARENT_STEP,
                     prior=np.ones(256)/256,fit_indices=np.array(FIT))
            optimizer=ExpandedAdam(checkpoint)
            np.testing.assert_array_equal(optimizer.theta[:3],PARENT_THETA)
            np.testing.assert_array_equal(optimizer.m[:3],PARENT_M)
            np.testing.assert_array_equal(optimizer.v[:3],PARENT_V)
            np.testing.assert_array_equal(optimizer.steps[:3],np.full(3,PARENT_STEP))
            self.assertFalse(np.any(optimizer.theta[3:]))
            self.assertFalse(np.any(optimizer.steps[3:]))
            optimizer.update(np.ones(WEIGHTS)*.02)
            np.testing.assert_array_equal(optimizer.steps[:3],np.full(3,PARENT_STEP+1))
            np.testing.assert_array_equal(optimizer.steps[3:],np.ones(WEIGHTS-3))
            self.assertTrue(np.isfinite(optimizer.theta).all())


if __name__=="__main__":
    unittest.main()
