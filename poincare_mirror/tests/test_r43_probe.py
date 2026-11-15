"""R43 bridge-strength probe is bounded and has exact control settings."""
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from probe_r43_feedback_strength import replacement_gate


class ReciprocalProbeTests(unittest.TestCase):
    def test_learned_and_zero_controls(self):
        original = -.0008993984083645046
        self.assertAlmostEqual(replacement_gate(original, 1.), original, places=12)
        self.assertEqual(replacement_gate(original, 0.), 0.)

    def test_sign_and_amplitude(self):
        original = -.0008993984083645046
        amplified = math.tanh(8. * replacement_gate(original, 4.))
        reversed_gain = math.tanh(8. * replacement_gate(original, -4.))
        self.assertAlmostEqual(amplified, 4. * math.tanh(8. * original), places=10)
        self.assertAlmostEqual(reversed_gain, -amplified, places=10)

    def test_saturation_and_bad_input(self):
        self.assertTrue(math.isfinite(replacement_gate(.5, 10000.)))
        with self.assertRaises(ValueError):
            replacement_gate(0., float('nan'))


if __name__ == '__main__':
    unittest.main()
