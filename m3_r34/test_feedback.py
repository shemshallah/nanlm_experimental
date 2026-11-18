"""Protect the one M3 default while testing zero-initialized feedback."""
import tempfile
from pathlib import Path
import unittest
import numpy as np
from m3_default.predictor import M3DefaultPredictor,R32_CHECKPOINT,DEFAULT_MODE
from m3_r32.anchor import load_r32_checkpoint
from m3_r34.feedback import M3R34Predictor,feedback_readout
from m3_r34.optimizer import R34ContinuationAdam


class FeedbackRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parent=load_r32_checkpoint(R32_CHECKPOINT)

    def test_exact_r32_zero_gain_parity_through_reset(self):
        self.assertEqual(DEFAULT_MODE,"r32")
        prior=np.arange(1,257,dtype=np.float64)
        default=M3DefaultPredictor(prior)
        candidate=M3R34Predictor(prior,self.parent.anchor,self.parent.anchor_theta)
        for byte in b"a local word asks the anchor to remember, again. "*2:
            default.observe(byte)
            candidate.observe(byte)
            np.testing.assert_array_equal(default.predict_logp(),candidate.predict_logp())
        for p in (default,candidate):
            p.reset()
            p.observe(65)
        np.testing.assert_array_equal(default.predict_logp(),candidate.predict_logp())

    def test_read_only_causal_feedback_and_normalization(self):
        x=M3R34Predictor(self.parent.prior,self.parent.anchor,self.parent.anchor_theta,
                         np.array([.1,-.1,.04,.15]))
        for byte in b"a red fox came back to the same home.":
            x.observe(byte)
        local_count=x.lexical.observed
        anchor_count=x.anchor.observed
        p=x.predict_logp()
        np.testing.assert_array_equal(p,x.predict_logp())
        self.assertAlmostEqual(float(np.exp(p).sum()),1.,places=12)
        self.assertEqual(local_count,x.lexical.observed)
        self.assertEqual(anchor_count,x.anchor.observed)

    def test_four_gain_channel_is_function_preserving(self):
        logp=np.log(np.ones(256)/256.)
        features=np.arange(1024,dtype=np.float64).reshape(4,256)/500.
        np.testing.assert_array_equal(feedback_readout(logp,features,np.zeros(4)),logp)
        changed=feedback_readout(logp,features,np.array([.1,.2,.3,-.4]))
        self.assertAlmostEqual(float(np.exp(changed).sum()),1.,places=12)
        with self.assertRaises(ValueError):
            feedback_readout(logp,features,np.zeros(3))

    def test_four_new_adam_groups_preserve_all_32_inherited_states(self):
        opt=R34ContinuationAdam(R32_CHECKPOINT)
        opt.update(np.array([.2,-.1,.05,.3]))
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"candidate.npz"
            opt.save(path,(540,556),(556,560),(560,572),(572,596))
            with np.load(path,allow_pickle=False) as candidate:
                for key in ("theta","adam_m","adam_v","adam_steps"):
                    np.testing.assert_array_equal(candidate[key][:32],opt.immutable[key])
                    self.assertEqual(candidate[key].shape,(36,))
                np.testing.assert_array_equal(candidate["background_bytes"],
                                              opt.parent["background_bytes"])
                self.assertEqual(candidate["r34_sealed_test_indices"].tolist(),[572,596])


if __name__=="__main__":
    unittest.main()
