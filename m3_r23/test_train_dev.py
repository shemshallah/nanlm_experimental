"""R23 synthetic training, disjointness, gate and checkpoint-shape probes."""
import unittest

import numpy as np

from m3_r23 import train_dev as m


class DriverTests(unittest.TestCase):
    def test_disjoint_training_gate_dev_and_old_test(self):
        self.assertEqual((m.FIT_START,m.GATE_START,m.DEV_START,m.TOTAL_BOOKS),
                         (188,204,208,220))
        self.assertGreater(m.FIT_START,156)
        self.assertEqual(m.TRAIN_BATCH,48)

    def test_small_real_forward_and_training_update(self):
        prior=np.ones(256)/256
        book=(b"once upon a recurrent triangle with memory and geometry. "*24)[:1150]
        self.assertEqual(len(book),1150)
        theta,adam,details=m.train_readout([book],prior,"torus")
        self.assertEqual(details["train_positions"],len(book)-m.WARMUP)
        self.assertGreater(adam.steps,0)
        self.assertTrue(np.any(theta))
        self.assertTrue(np.isfinite(theta).all())
        result=m.evaluate_books([book],["synthetic_ascii"],prior,theta,np.zeros(5))
        self.assertEqual(result["positions"],len(book)-m.WARMUP)
        self.assertEqual(result["arms"]["apex_shuffled"]["bpb"],
                         result["arms"]["r20"]["bpb"])

    def test_selection_prefers_real_geometric_gain_over_copy_and_shuffled(self):
        def row(bpb,ns):
            return {"bpb":bpb,"nonspace_bpb":ns}
        arms={"r20":row(3.5,4.2),"apex_torus":row(3.4,4.1),
              "apex_shuffled":row(3.45,4.15),"copy_only":row(3.47,4.16)}
        d=m.choose_gate({"arms":arms})
        self.assertEqual(d["chosen"],"apex_torus")
        arms["apex_shuffled"]=row(3.35,4.0)
        self.assertEqual(m.choose_gate({"arms":arms})["chosen"],"copy_only")
        arms["copy_only"]=row(3.6,4.3)
        self.assertEqual(m.choose_gate({"arms":arms})["chosen"],"r20")


if __name__=="__main__":
    unittest.main()
