"""R23 structural, state, causal and numerical parity verification."""
import unittest

import numpy as np

from m3_default.predictor import M3DefaultPredictor
from m3_r23.apex import (
    AdamReadout, M3R23Predictor, RecurrentTriangleApex, SuffixApexMemory,
)


class ApexTests(unittest.TestCase):
    def setUp(self):
        self.prior=np.ones(256)/256

    def test_zero_residual_equals_frozen_r21_for_every_byte(self):
        old=M3DefaultPredictor(self.prior)
        new=M3R23Predictor(self.prior)
        seq=b"an actual geometric apex has recurrent state and causal memory. "*3
        for b in seq:
            old.observe(b)
            new.observe(b)
            np.testing.assert_array_equal(old.predict_logp("r20"),new.predict_logp())
        self.assertEqual(new.seen_bytes,len(seq))

    def test_read_only_predictions_and_shared_memory(self):
        model=M3R23Predictor(self.prior,theta=np.array([.3,.1,-.2,.2,.1]))
        for b in b"mirrors are not equivalent to an independent network":
            model.observe(b)
        a,f=model.base_and_features()
        b,g=model.base_and_features()
        np.testing.assert_array_equal(a,b)
        np.testing.assert_array_equal(f,g)
        self.assertEqual(f.shape,(5,256))
        self.assertAlmostEqual(float(np.exp(model.predict_logp()).sum()),1.,places=12)

    def test_suffix_retrieval_is_causal_and_wrong_content_changes_it(self):
        mem=SuffixApexMemory(horizon=256)
        base=np.full(256,-np.log(256))
        for b in b"abcdefgh":
            mem.observe(b)
        np.testing.assert_array_equal(mem.residual(base),np.zeros(256))
        for b in b"abcdefghiabcdefg":
            mem.observe(b)
        self.assertGreater(np.abs(mem.residual(base)).max(),0)
        self.assertGreater(np.abs(mem.residual(base)-mem.residual(base,True)).max(),0)

    def test_standing_wave_dynamics_and_shuffled_geometry_differ(self):
        torus=RecurrentTriangleApex("torus")
        shuffled=RecurrentTriangleApex("shuffled")
        field=np.zeros((256,256))
        field[1,2]=2
        field[2,1]=-2
        for pos in range(128):
            b=(pos*11)%256
            a=(pos*7)%256
            p=(pos*3)%256
            torus.observe(b,a,p,field)
            shuffled.observe(b,a,p,field)
        t=torus.features(1,2)
        s=shuffled.features(1,2)
        self.assertEqual(t.shape,(4,256))
        self.assertTrue(np.isfinite(t).all())
        self.assertTrue(np.isfinite(s).all())
        self.assertGreater(np.max(np.abs(t-s)),1e-4)

    def test_adam_readout_retains_accumulated_moments(self):
        adam=AdamReadout(lr=.01)
        adam.update(np.array([1.,-.5,.2,.1,-.1]))
        adam.update(np.array([.1,.3,-.4,.5,1.]))
        self.assertEqual(adam.steps,2)
        self.assertTrue(np.any(adam.m))
        self.assertTrue(np.any(adam.v))
        self.assertEqual(adam.theta.shape,(5,))

    def test_reset_preserves_readout_and_resets_all_causal_fields(self):
        m=M3R23Predictor(self.prior,theta=np.array([0.,.3,0.,0.,0.]))
        for b in b"a lengthy context":
            m.observe(b)
        initial=m.theta.copy()
        m.reset()
        np.testing.assert_array_equal(m.theta,initial)
        self.assertEqual(m.seen_bytes,0)
        self.assertEqual(len(m.suffix.context),0)
        self.assertEqual(m.apex.u.sum(),0)


if __name__=="__main__":
    unittest.main()
