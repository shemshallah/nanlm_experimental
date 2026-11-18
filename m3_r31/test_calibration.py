"""R31 function-preserving migration, class targeting and purity tests."""
import unittest

import numpy as np

from m3_default.predictor import M3DefaultPredictor
from m3_r30.router import M3R30Predictor
from m3_r31.calibration import (M3R31Predictor, R30_DELTA,
                                class_features, calibrated_readout, SPACE)


class R31Tests(unittest.TestCase):
    def test_zero_init_exactly_preserves_promoted_r29_distribution(self):
        prior = np.arange(1,257,dtype=np.float64)
        accepted = M3DefaultPredictor(prior)
        trial = M3R31Predictor(prior)
        for byte in b"the yellow bird flew over the garden. " * 3:
            accepted.observe(byte)
            trial.observe(byte)
            np.testing.assert_array_equal(accepted.predict_logp(),trial.predict_logp())
            np.testing.assert_array_equal(accepted.predict_logp("r26"),trial.predict_logp("r26"))
        accepted.reset()
        trial.reset()
        accepted.observe(65)
        trial.observe(65)
        np.testing.assert_array_equal(accepted.predict_logp(),trial.predict_logp())

    def test_inherited_r30_path_is_preserved_with_exact_frozen_gains(self):
        prior = np.ones(256)
        old = M3R30Predictor(prior,R30_DELTA)
        new = M3R31Predictor(prior)
        for byte in b"low bit probability low bit probability ":
            old.observe(byte)
            new.observe(byte)
        np.testing.assert_array_equal(old.predict_logp(),new.predict_logp("r30"))

    def test_class_gating_stays_causal_and_normalized(self):
        p = M3R31Predictor(np.ones(256))
        for byte in b"green tree green tree green tr":
            p.observe(byte)
        count=p.lexical.observed
        base,features=p.r29_base_and_calibration()
        self.assertEqual(features.shape,(3,256))
        np.testing.assert_array_equal(features,p.r29_base_and_calibration()[1])
        self.assertEqual(p.lexical.observed,count)
        self.assertTrue(np.all(features[0,32]==0))
        self.assertTrue(np.all(features[1,:32]==0))
        self.assertEqual(SPACE.sum(),1)
        logp=calibrated_readout(base,features,np.array([.1,.1,.1]))
        self.assertAlmostEqual(np.exp(logp).sum(),1.,places=12)
        np.testing.assert_array_equal(calibrated_readout(base,features,np.zeros(3)),base)

    def test_invalid_dimensions(self):
        with self.assertRaises(ValueError):
            M3R31Predictor(np.ones(256),np.zeros(4))
        with self.assertRaises(ValueError):
            class_features(np.zeros((7,256)),0.)
        with self.assertRaises(ValueError):
            calibrated_readout(np.zeros(256),np.zeros((3,256)),np.zeros(4))


if __name__=="__main__":
    unittest.main()
