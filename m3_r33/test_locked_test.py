"""Independent TEST criteria fixed before any R32 TEST book is read."""
import json
import unittest

import numpy as np

from m3_r33.locked_test import LOCK_PATH,evaluate_acceptance,report_json


class R32PrelockedAcceptance(unittest.TestCase):
    def test_lock_has_training_provenance_and_sealed_indices(self):
        d=json.loads(LOCK_PATH.read_text())
        self.assertEqual(d["sealed_test_indices"],[516,540])
        self.assertEqual(d["optimizer_groups"],32)
        self.assertEqual(d["new_adam_updates"],768)
        self.assertTrue(d["test_unread_at_lock"])

    def test_strict_acceptance_checks_all_controls(self):
        lock=json.loads(LOCK_PATH.read_text())
        base={"bpb":3.2,"nonspace_bpb":4.0,"top1":.4,"top5":.7}
        arms={key:base.copy() for key in ("r29","r31","wrong_anchor",
                                            "unrelated_matched","unrelated_full")}
        arms["r32"]={"bpb":3.0,"nonspace_bpb":3.8,"top1":.45,"top5":.75,
                     "books_improved":20,"whole_book_bootstrap_95pct_bits":[.1,.3]}
        measured={"arms":arms,"books":24}
        passed,detail=evaluate_acceptance(measured,lock)
        self.assertTrue(passed)
        self.assertTrue(all(detail.values()))
        arms["unrelated_full"]["bpb"]=2.99
        self.assertFalse(evaluate_acceptance(measured,lock)[0])
        arms["unrelated_full"]["bpb"]=3.2
        arms["r32"]["books_improved"]=17
        self.assertFalse(evaluate_acceptance(measured,lock)[0])

    def test_numpy_scalar_reporting_retains_values_and_types(self):
        raw={"passed":np.bool_(True),"count":np.int64(24),
             "bpb":np.float64(3.0973586126),"bytes":[1,2,3]}
        restored=json.loads(report_json(raw))
        self.assertIs(restored["passed"],True)
        self.assertEqual(restored["count"],24)
        self.assertEqual(restored["bpb"],3.0973586126)
        self.assertEqual(restored["bytes"],[1,2,3])
