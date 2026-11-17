"""R26 frozen-candidate integrity, causal parity and one-shot gates."""
import json
from pathlib import Path
import unittest

import numpy as np

from m3_r26.hybrid import M3R26Predictor
from m3_r27.locked_test import (LOCK, TEST_FIRST, TEST_END,
                                pass_locked_test)


class R27LockedProtocol(unittest.TestCase):
    def test_frozen_theta_and_prior_are_well_formed(self):
        lock = json.loads(LOCK.read_text(encoding="utf-8"))
        self.assertEqual(lock["sealed_test_indices"], [276, 300])
        self.assertEqual(lock["training_indices"], [244, 260])
        self.assertEqual(lock["gate_indices"], [260, 264])
        self.assertEqual(lock["dev_indices"], [264, 276])
        self.assertEqual(len(lock["theta"]), 14)
        self.assertEqual(len(lock["optimizer_steps"]), 14)
        self.assertEqual(len(lock["training_checkpoint_sha256"]), 64)
        self.assertEqual(len(lock["training_dev_report_sha256"]), 64)
        self.assertTrue(np.isfinite(np.asarray(lock["theta"])).all())

    def test_frozen_r26_is_unchanged_by_scoring(self):
        lock = json.loads(LOCK.read_text(encoding="utf-8"))
        theta = np.array(lock["theta"], dtype=np.float64)
        p = M3R26Predictor(np.ones(256), theta)
        for byte in b"the first history is already written, the first history is written":
            p.observe(byte)
        snapshot = p.tree.observed
        first = p.predict_logp()
        second = p.predict_logp()
        np.testing.assert_array_equal(first, second)
        self.assertEqual(p.tree.observed, snapshot)
        self.assertAlmostEqual(float(np.exp(first).sum()), 1.0, places=12)

    def test_predeclared_requirements_reject_borderline_results(self):
        lock = json.loads(LOCK.read_text(encoding="utf-8"))
        base={"bpb":3.5,"nonspace_bpb":4.2,"top1":.35,"top5":.68}
        row={"bpb":3.4,"nonspace_bpb":4.1,"top1":.36,"top5":.69,
             "books_improved":20,"whole_book_bootstrap_95pct_bits":[.02,.08]}
        m={"books":TEST_END-TEST_FIRST,"arms":{
            "r24":base,"r25":{"bpb":3.48},"r26":row,
            "wrong_successor":{"bpb":3.8},"unrelated_history":{"bpb":3.7}}}
        self.assertTrue(pass_locked_test(m,lock))
        m["arms"]["r26"]={**row,"nonspace_bpb":4.21}
        self.assertFalse(pass_locked_test(m,lock))
        m["arms"]["r26"]={**row,"books_improved":17}
        self.assertFalse(pass_locked_test(m,lock))
        m["arms"]["r26"]={**row,"whole_book_bootstrap_95pct_bits":[-.001,.08]}
        self.assertFalse(pass_locked_test(m,lock))


if __name__ == "__main__":
    unittest.main()
