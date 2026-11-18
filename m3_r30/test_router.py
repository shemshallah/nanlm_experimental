"""Zero-init parity, read purity, and conditional trust feature tests."""
import unittest

import numpy as np

from m3_default.predictor import M3DefaultPredictor, R29_THETA
from m3_r29.lexical import WordSuccessorMemory, pack_lexical
from m3_r30.router import (M3R30Predictor, router_features, router_readout,
                           support, ROUTER_FEATURES)
from m3_r30.train_dev import gate_pass, FrozenR29Adam


class R30Tests(unittest.TestCase):
    def test_zero_init_exact_canonical_r29_all_steps_and_reset(self):
        prior = np.arange(1, 257, dtype=np.float64)
        promoted = M3DefaultPredictor(prior, mode="r29")
        candidate = M3R30Predictor(prior)
        for byte in b"the green lake and the blue lake, the green lake. " * 2:
            promoted.observe(byte)
            candidate.observe(byte)
            np.testing.assert_array_equal(promoted.predict_logp(), candidate.predict_logp())
            np.testing.assert_array_equal(promoted.predict_logp("r26"),candidate.predict_logp("r26"))
            np.testing.assert_array_equal(promoted.predict_logp("r24"),candidate.predict_logp("r24"))
        promoted.reset()
        candidate.reset()
        promoted.observe(65)
        candidate.observe(65)
        np.testing.assert_array_equal(promoted.predict_logp(),candidate.predict_logp())

    def test_read_only_three_channels_normalized(self):
        p = M3R30Predictor(np.ones(256))
        for b in b"a bright green lake a bright green l":
            p.observe(b)
        before = p.lexical.observed
        base, features = p.r29_base_and_router_features()
        self.assertEqual(features.shape, (ROUTER_FEATURES, 256))
        np.testing.assert_array_equal(features,p.r29_base_and_router_features()[1])
        self.assertEqual(p.lexical.observed,before)
        result=router_readout(base,features,np.array([.1,.1,.1]))
        self.assertAlmostEqual(np.exp(result).sum(),1.,places=12)
        np.testing.assert_array_equal(router_readout(base,features,np.zeros(3)),base)

    def test_matched_history_control_uses_current_query_not_future(self):
        actual, unrelated = WordSuccessorMemory(), WordSuccessorMemory()
        for a,b in zip(b"red lake red lake red l",b"red lane red lane red l"):
            actual.observe(a)
            unrelated.observe(b)
        observed = actual.observed
        u = support(unrelated,query=actual)
        self.assertTrue(0<=u<=1)
        base = np.log(np.ones(256)/256)
        other=pack_lexical(unrelated.features(base,query=actual))
        f=router_features(other,u)
        self.assertEqual(f.shape,(3,256))
        self.assertEqual(actual.observed,observed)
        self.assertEqual(unrelated.observed,observed)

    def test_invalid_shapes_and_coefficients(self):
        with self.assertRaises(ValueError):
            M3R30Predictor(np.ones(256),np.zeros(2))
        with self.assertRaises(ValueError):
            router_features(np.zeros((7,256)),0.)
        with self.assertRaises(ValueError):
            router_readout(np.zeros(256),np.zeros((3,256)),np.zeros(4))
        with self.assertRaises(ValueError):
            FrozenR29Adam(__import__("pathlib").Path("no-parent.npz"))

    def test_gate_requires_correct_history_and_accuracy_preservation(self):
        base=dict(bpb=3.5,nonspace_bpb=4.,top1=.39,top5=.70)
        candidate=dict(bpb=3.49,nonspace_bpb=3.99,top1=.40,top5=.71,
                       books_improved=4)
        a={"r29":base,"r30":candidate,"wrong_successor":{"bpb":3.6},
           "unrelated_matched":{"bpb":3.55},"unrelated_full":{"bpb":3.48}}
        self.assertTrue(gate_pass({"arms":a},3))
        a["unrelated_matched"]["bpb"]=3.48
        self.assertFalse(gate_pass({"arms":a},3))


if __name__ == "__main__":
    unittest.main()
