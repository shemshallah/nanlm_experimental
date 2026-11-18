"""R32 causal anchor, exact R31 fallback, and acceptance guard tests."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from m3_default.predictor import R31_THETA
from m3_r31.calibration import M3R31Predictor
from m3_r32.anchor import (M3R32Predictor, anchor_features,
                            anchored_readout, make_anchor)
from m3_r32.train_dev import (BACKGROUND, FIT, GATE, DEV, SEALED_TEST,
                               FrozenR31Adam, gate_pass, selective_books)


class AnchorTests(unittest.TestCase):
    def test_zero_init_exact_r31_and_all_earlier_modes(self):
        prior=np.arange(1,257,dtype=np.float64)
        anchor=make_anchor([b"the brown dog ran the brown dog.",
                            b"the blue fox ate the blue fish."])
        trial=M3R32Predictor(prior,anchor)
        frozen=M3R31Predictor(prior,R31_THETA)
        for b in b"the green dog ran the green fox. " * 3:
            trial.observe(b)
            frozen.observe(b)
            np.testing.assert_array_equal(trial.predict_logp(),frozen.predict_logp())
            for mode in ("r31","r30","r29","r26","r24","r20"):
                np.testing.assert_array_equal(trial.predict_logp(mode),frozen.predict_logp(mode))
        trial.reset()
        frozen.reset()
        trial.observe(65)
        frozen.observe(65)
        np.testing.assert_array_equal(trial.predict_logp(),frozen.predict_logp())

    def test_anchor_immutable_and_wrong_successors_are_distinguishable(self):
        anchor=make_anchor([b"blue bird flies blue bird flies blue bird flies "]*3)
        p=M3R32Predictor(np.ones(256),anchor)
        for b in b"blue bird f":
            p.observe(b)
        before=anchor.observed
        r26,lex=p.base_and_lexical_features()
        f=anchor_features(r26,lex,p.lexical,anchor)
        wrong=anchor_features(r26,lex,p.lexical,anchor,True)
        self.assertEqual(f.shape,(4,256))
        self.assertEqual(anchor.observed,before)
        self.assertGreater(np.max(np.abs(f-wrong)),0)
        base=p.predict_logp("r31")
        np.testing.assert_array_equal(anchored_readout(base,f,np.zeros(4)),base)
        self.assertAlmostEqual(np.exp(anchored_readout(base,f,np.ones(4)*.1)).sum(),1.,places=12)

    def test_parent_checkpoint_and_split_are_protected(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError,"checkpoint"):
                FrozenR31Adam(Path(d)/"missing.npz")
        for a,b in zip((BACKGROUND,FIT,GATE,DEV),(FIT,GATE,DEV,SEALED_TEST)):
            self.assertEqual(a[1],b[0])
        with self.assertRaises(ValueError):
            make_anchor([])
        with self.assertRaises(ValueError):
            M3R32Predictor(np.ones(256),make_anchor([b"abc"]),np.ones(3))

    def test_selective_loader_leaves_unneeded_test_bytes_unread(self):
        import m3_r32.train_dev as trainer
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for i in range(9):
                (root/f"book_{i}.txt").write_bytes(bytes([65+i])*10)
            with patch.object(trainer,"BYTES_PER_BOOK",4):
                paths,loaded=selective_books(root,9,{0,1,8})
            self.assertEqual(len(paths),9)
            self.assertEqual(set(loaded),{0,1,8})
            self.assertTrue(all(len(v)==4 for v in loaded.values()))

    def test_gate_rejects_unrelated_advantage_and_accuracy_loss(self):
        a={"r29":{"bpb":3.5,"top1":.4},
           "r31":{"bpb":3.4,"top1":.41,"top5":.70,"nonspace_bpb":4.2},
           "r32":{"bpb":3.39,"top1":.42,"top5":.71,"nonspace_bpb":4.1,
                  "books_improved":9},
           "wrong_anchor":{"bpb":3.6},"unrelated_matched":{"bpb":3.45}}
        self.assertTrue(gate_pass({"arms":a},9))
        a["unrelated_matched"]["bpb"]=3.38
        self.assertFalse(gate_pass({"arms":a},9))
        a["unrelated_matched"]["bpb"]=3.45
        a["r32"]["top1"]=.405
        self.assertFalse(gate_pass({"arms":a},9))


if __name__=="__main__":
    unittest.main()
