"""Deterministic mathematical regression tests for M3's genuine wave core."""

import unittest

import numpy as np

from mirror_wave import M3MirrorWave, N, PAIR_COUNT, WaveConfig, nan16_address, nan16_pair


class MirrorWaveTests(unittest.TestCase):
    def test_topology(self):
        w = M3MirrorWave()
        self.assertEqual((N, PAIR_COUNT), (256, 32640))
        self.assertEqual(w.neighbors.shape, (256, 4))
        self.assertTrue(np.all(w.adjacent == w.adjacent.T))
        self.assertTrue(np.all(np.sum(w.adjacent, axis=1) == 4))
        self.assertTrue(np.all(np.diag(w.degree) == 0))
        self.assertEqual(int(w.degree[0, 1]), 6)
        self.assertEqual(int(w.degree[0, 128]), 8)

    def test_nan16_bit_reversal_exact(self):
        def rev16(v):
            return int(f"{v:016b}"[::-1], 2)

        fixed = []
        for a in range(256):
            for b in range(256):
                address = nan16_address(a, b)
                self.assertEqual(nan16_pair(address), (a, b))
                self.assertEqual(rev16(address), nan16_address(b, a))
                if address == rev16(address):
                    fixed.append(address)
        self.assertEqual(len(fixed), 256)

    def test_constant_is_zero_frequency_mode(self):
        w = M3MirrorWave(WaveConfig(bridge=0))
        uniform = np.ones((N, N))
        np.fill_diagonal(uniform, 0)
        self.assertLess(np.max(np.abs(w.pair_laplacian(uniform, 0))), 1e-14)
        self.assertLess(np.max(np.abs(w.head_laplacian(np.ones(N)))), 1e-14)

    def test_true_finite_speed_wavefront(self):
        w = M3MirrorWave(WaveConfig(bridge=0, damping=0))
        w.inject_transition(0, 128)
        w.step()
        self.assertEqual(w.pair_u[0, 1, 128], 0.0)
        w.step()
        self.assertGreater(abs(w.pair_u[0, 1, 128]), 1e-7)
        self.assertEqual(w.pair_u[0, 15, 240], 0.0)
        uncoupled = M3MirrorWave(WaveConfig(bridge=0, pair_speed=0))
        uncoupled.inject_transition(0, 128)
        uncoupled.step(2)
        self.assertEqual(uncoupled.pair_u[0, 1, 128], 0.0)

    def test_pair_wave_rings_and_changes_sign(self):
        w = M3MirrorWave(WaveConfig(dt=0.2, bridge=0))
        w.inject_transition(0, 128)
        signal = []
        for _ in range(48):
            w.step()
            signal.append(w.pair_u[0, 0, 128])
        self.assertGreater(max(signal), 0.5)
        self.assertLess(min(signal), -0.5)

    def test_matched_onsite_oscillator_control(self):
        cfg = WaveConfig(pair_speed=0, head_speed=0, bridge=0, onsite_frequency=1)
        w = M3MirrorWave(cfg)
        w.inject_transition(0, 128)
        w.step(8)
        self.assertGreater(abs(w.pair_u[0, 0, 128]), 0.5)
        self.assertEqual(w.pair_u[0, 1, 128], 0)
        self.assertLess(abs(w.energy() / 1.5 - 1), 0.02)

    def test_signed_rail_orientation_and_conservation(self):
        w = M3MirrorWave(WaveConfig(bridge=0.3, shared_head_fraction=0.25))
        w.inject_transition(24, 79)
        self.assertEqual(w.pair_v[0, 24, 79], w.pair_v[0, 79, 24])
        self.assertEqual(w.pair_v[1, 24, 79], -w.pair_v[1, 79, 24])
        w.step(32)
        self.assertLess(w.parity_error(), 1e-12)
        self.assertTrue(np.all(w.pair_u[:, np.arange(N), np.arange(N)] == 0))

    def test_energy_stays_bounded_without_damping(self):
        w = M3MirrorWave(WaveConfig(dt=0.15, bridge=0.4))
        w.inject_transition(0, 127)
        initial = w.energy()
        self.assertGreater(initial, 0)
        w.step(120)
        self.assertLess(abs(w.energy() / initial - 1.0), 0.005)

    def test_energy_decays_with_damping(self):
        w = M3MirrorWave(WaveConfig(dt=0.15, damping=0.2, bridge=0.4))
        w.inject_transition(0, 127)
        initial = w.energy()
        w.step(80)
        self.assertLess(w.energy(), initial * 0.2)

    def test_nontrivial_standing_mode(self):
        w = M3MirrorWave(WaveConfig(dt=0.25, bridge=0, pair_speed=0))
        # A known nonzero-eigenvalue standing mode of the 16x16 torus.
        xx = np.arange(N).reshape(16, 16) // 16
        original = np.cos(2 * np.pi * xx / 16).ravel()
        w.head_u[0] = original
        eigenvalue = 0.5 * (1 - np.cos(2 * np.pi / 16))
        self.assertLess(np.linalg.norm(w.head_laplacian(original) - eigenvalue * original), 1e-12)
        w.step(round(np.pi / (np.sqrt(eigenvalue) * w.config.dt)))
        self.assertLess(np.dot(w.head_u[0], original) / np.dot(original, original), -0.98)

    def test_two_way_spring_coupling(self):
        w = M3MirrorWave(WaveConfig(bridge=1.0, head_speed=0))
        w.pair_u[0, 10, 90] = w.pair_u[0, 90, 10] = 1.0
        w.step(5)
        self.assertGreater(abs(w.head_u[0, 10]), 0)
        self.assertGreater(abs(w.head_u[0, 90]), 0)

    def test_exact_superposition(self):
        cfg = WaveConfig(bridge=0.6, shared_head_fraction=0.2)
        left, right, both = [M3MirrorWave(cfg) for _ in range(3)]
        left.inject_transition(12, 32, 0.4)
        right.inject_transition(32, 12, -0.7)
        both.inject_transition(12, 32, 0.4)
        both.inject_transition(32, 12, -0.7)
        for w in (left, right, both):
            w.step(8)
        np.testing.assert_allclose(both.pair_u, left.pair_u + right.pair_u, atol=1e-12)
        np.testing.assert_allclose(both.head_u, left.head_u + right.head_u, atol=1e-12)

    def test_causal_readout_does_not_mutate(self):
        w = M3MirrorWave()
        before = w.energy()
        self.assertEqual(w.features24().shape, (24,))
        self.assertEqual(w.energy(), before)
        self.assertEqual(w.ticks, 0)
        w.advance_observation(1, 1)
        self.assertEqual(w.ticks, 1)
        self.assertGreater(w.energy(), before)

    def test_byte_and_config_validation(self):
        w = M3MirrorWave()
        with self.assertRaises(ValueError):
            w.inject_transition(-1, 40)
        with self.assertRaises(ValueError):
            w.inject_transition(200, 256)
        with self.assertRaises(ValueError):
            w.step(-1)
        with self.assertRaises(ValueError):
            WaveConfig(dt=2)


if __name__ == "__main__":
    unittest.main()
