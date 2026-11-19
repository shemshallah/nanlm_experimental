"""R35 causal, normalization, frozen parent, and two-gain migration tests."""
import tempfile
from pathlib import Path
import unittest
import numpy as np
from m3_default.predictor import M3DefaultPredictor,R32_CHECKPOINT,DEFAULT_MODE
from m3_r32.anchor import load_r32_checkpoint
from m3_r35.local_feedback import M3R35Predictor,local_readout
from m3_r35.optimizer import R35ContinuationAdam
from m3_r35.train_dev import get_r34_reference,gate


class LocalFeedbackRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parent=load_r32_checkpoint(R32_CHECKPOINT)

    def test_exact_accepted_r32_zero_gain_parity_and_reset(self):
        self.assertEqual(DEFAULT_MODE,"r32")
        reference=M3DefaultPredictor(self.parent.prior)
        candidate=M3R35Predictor(self.parent.prior,self.parent.anchor,
                                 self.parent.anchor_theta)
        for byte in b"a local word asks a frozen anchor to remember. "*2:
            reference.observe(byte)
            candidate.observe(byte)
            np.testing.assert_array_equal(reference.predict_logp(),candidate.predict_logp())
        for obj in (reference,candidate):
            obj.reset()
            obj.observe(97)
        np.testing.assert_array_equal(reference.predict_logp(),candidate.predict_logp())

    def test_two_local_gains_causal_readonly_and_normalized(self):
        candidate=M3R35Predictor(self.parent.prior,self.parent.anchor,
                                 self.parent.anchor_theta,np.array([.2,-.1]))
        for byte in b"the fox jumped over the hill. the fox returned.":
            candidate.observe(byte)
        local_n=candidate.lexical.observed
        anchor_n=candidate.anchor.observed
        logp=candidate.predict_logp()
        np.testing.assert_array_equal(logp,candidate.predict_logp())
        self.assertAlmostEqual(float(np.exp(logp).sum()),1.,places=12)
        self.assertEqual(local_n,candidate.lexical.observed)
        self.assertEqual(anchor_n,candidate.anchor.observed)

    def test_local_readout_dimensions_and_exact_parity(self):
        base=np.log(np.ones(256)/256.)
        features=np.arange(512,dtype=np.float64).reshape(2,256)/600.
        np.testing.assert_array_equal(local_readout(base,features,np.zeros(2)),base)
        alt=local_readout(base,features,np.array([.2,-.3]))
        self.assertAlmostEqual(float(np.exp(alt).sum()),1.,places=12)
        with self.assertRaises(ValueError):
            local_readout(base,features,np.zeros(4))

    def test_optimizer_inherits_all_32_groups_exactly(self):
        opt=R35ContinuationAdam(R32_CHECKPOINT)
        opt.update(np.array([.2,-.1]))
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/"candidate.npz"
            opt.save(p,(540,556),(556,560),(596,608))
            with np.load(p,allow_pickle=False) as candidate:
                for key in ("theta","adam_m","adam_v","adam_steps"):
                    self.assertEqual(candidate[key].shape,(34,))
                    np.testing.assert_array_equal(candidate[key][:32],opt.immutable[key])
                np.testing.assert_array_equal(candidate["background_bytes"],
                                              opt.parent["background_bytes"])
                self.assertEqual(candidate["r34_sealed_test_indices"].tolist(),[572,596])

    def test_r34_reference_is_frozen_and_parent_matched(self):
        gains=get_r34_reference(Path("m3_r34/results/R34_36_group_candidate.npz"))
        self.assertEqual(gains.shape,(2,))
        self.assertTrue(np.isfinite(gains).all())

    def test_no_automatic_promotion_on_failed_gate(self):
        metrics={"arms":{"r32":{"bpb":2.,"nonspace_bpb":2.,"top1":.3,"top5":.6},
                         "r35":{"bpb":2.1,"nonspace_bpb":2.2,"top1":.2,"top5":.6,
                                "books_improved":0,"whole_book_bootstrap_95pct_bits":[-.1,.01]}}}
        self.assertFalse(all(gate(metrics,9,True).values()))


if __name__=="__main__":
    unittest.main()
