"""M3 R13 experimental conservative graph-wave dynamics.

This is a new wave engine, not a restored M3 R12 trained checkpoint.
The 256 heads form a periodic 16x16 lattice.  The C(256,2) mirrors form
the unordered two-particle configuration space of that lattice.  Every
mirror is coupled to spatially adjacent mirrors sharing one endpoint.

Both head and mirror fields have displacement and velocity.  The +/-
mirror rails obey symmetric/antisymmetric address parity respectively.
Velocity-Verlet integration, energy-consistent head/pair springs, and an
optional exact signed shared-head Laplacian supply true wave dynamics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


N = 256
SIDE = 16
PAIR_COUNT = N * (N - 1) // 2


def reverse8(value: int) -> int:
    """Eight-bit reversal; the NaN16 mirror involution swaps pair axes."""
    if not (0 <= value < N):
        raise ValueError("reverse8 requires a byte")
    value = ((value & 0xF0) >> 4) | ((value & 0x0F) << 4)
    value = ((value & 0xCC) >> 2) | ((value & 0x33) << 2)
    return ((value & 0xAA) >> 1) | ((value & 0x55) << 1)


def nan16_address(a: int, b: int) -> int:
    """Exact NaN16 address for the ordered (a,b) geometric location."""
    if not (0 <= a < N and 0 <= b < N):
        raise ValueError("address endpoints must be bytes")
    return (a << 8) | reverse8(b)


def nan16_pair(address: int) -> tuple[int, int]:
    """Decode an ordered NaN16 address back into its two head labels."""
    if not (0 <= address <= 0xFFFF):
        raise ValueError("NaN16 address must lie in [0, 65535]")
    return address >> 8, reverse8(address & 0xFF)


@dataclass(frozen=True)
class WaveConfig:
    dt: float = 0.25
    pair_speed: float = 1.0
    head_speed: float = 1.0
    damping: float = 0.0
    bridge: float = 0.2
    shared_head_fraction: float = 0.0
    onsite_frequency: float = 0.0

    def __post_init__(self) -> None:
        if not (0.0 < self.dt <= 0.5):
            raise ValueError("dt must lie in (0, 0.5]")
        if not (0.0 <= self.pair_speed <= 2.0 and 0.0 <= self.head_speed <= 2.0):
            raise ValueError("wave speeds must lie in [0, 2]")
        if not (0.0 <= self.damping <= 2.0 and 0.0 <= self.bridge <= 2.0):
            raise ValueError("damping and bridge must lie in [0, 2]")
        if not (0.0 <= self.onsite_frequency <= 2.0):
            raise ValueError("onsite_frequency must lie in [0, 2]")
        if not (0.0 <= self.shared_head_fraction <= 1.0):
            raise ValueError("shared_head_fraction must lie in [0, 1]")


class M3MirrorWave:
    """Second-order wave on the 256 heads and 32,640 unordered mirrors.

    Arrays are stored in ordered 256x256 form to make linear algebra simple.
    The diagonal is absent; each off-diagonal mirror occurs twice.  The
    `plus` rail is symmetric and `minus` is antisymmetric.  In a streaming
    predictor, call `features24()` before the target byte is observed, then
    `advance_observation(previous_byte, observed_byte)` afterward.
    """

    def __init__(self, config: WaveConfig = WaveConfig()) -> None:
        self.config = config
        lattice = np.arange(N).reshape(SIDE, SIDE)
        self.neighbors = np.stack(
            [np.roll(lattice, shift, axis).ravel()
             for axis, shift in ((0, 1), (0, -1), (1, 1), (1, -1))], axis=1
        )
        indices = np.arange(N)
        self.adjacent = np.zeros((N, N), dtype=bool)
        self.adjacent[indices[:, None], self.neighbors] = True
        self.degree = 8.0 - 2.0 * self.adjacent.astype(np.float64)
        np.fill_diagonal(self.degree, 0.0)
        # Two rail channels, each with the full 256-head + pair state.
        self.pair_u = np.zeros((2, N, N), dtype=np.float64)
        self.pair_v = np.zeros_like(self.pair_u)
        self.head_u = np.zeros((2, N), dtype=np.float64)
        self.head_v = np.zeros_like(self.head_u)
        self.ticks = 0

    @staticmethod
    def _zero_diagonal(state: np.ndarray) -> None:
        state[:, np.arange(N), np.arange(N)] = 0.0

    def pair_laplacian(self, rail_state: np.ndarray, rail: int) -> np.ndarray:
        if rail not in (0, 1) or rail_state.shape != (N, N):
            raise ValueError("rail must be 0 or 1; state must be (256, 256)")
        # Move one pair endpoint one lattice hop, leaving the other fixed.
        # Moves onto the forbidden pair diagonal are suppressed.
        # The pair coordinate is a 16x16 head lattice times another 16x16
        # lattice. Four spatial axes with +/- unit shifts exactly reproduce
        # the original shared-endpoint adjacency, without slow gathers.
        pair_lattice = rail_state.reshape(SIDE, SIDE, SIDE, SIDE)
        nearby = np.zeros_like(pair_lattice)
        for axis in range(4):
            nearby += np.roll(pair_lattice, 1, axis=axis)
            nearby += np.roll(pair_lattice, -1, axis=axis)
        nearby = nearby.reshape(N, N)
        local = (self.degree * rail_state - nearby) / 8.0
        if self.config.shared_head_fraction:
            marginal = rail_state.sum(axis=1)
            # Antisymmetric rail uses oriented incidence.  Using S_i+S_j
            # here would violate its parity and is mathematically wrong.
            incidence = (marginal[:, None] + marginal[None, :]) if rail == 0 else (
                marginal[:, None] - marginal[None, :]
            )
            global_lap = (2 * (N - 1) * rail_state - incidence) / (2 * (N - 1))
            alpha = self.config.shared_head_fraction
            local = (1.0 - alpha) * local + alpha * global_lap
        np.fill_diagonal(local, 0.0)
        return local

    def head_laplacian(self, state: np.ndarray) -> np.ndarray:
        if state.shape != (N,):
            raise ValueError("head field must have 256 entries")
        lattice = state.reshape(SIDE, SIDE)
        adjacent = (np.roll(lattice, 1, axis=0) + np.roll(lattice, -1, axis=0)
                    + np.roll(lattice, 1, axis=1) + np.roll(lattice, -1, axis=1))
        return ((4.0 * lattice - adjacent) / 4.0).ravel()

    def _residuals(self) -> np.ndarray:
        head_p, head_m = self.head_u
        pair_targets = np.stack((
            0.5 * (head_p[:, None] + head_p[None, :]),
            0.5 * (head_m[:, None] - head_m[None, :]),
        ))
        residual = self.pair_u - pair_targets
        self._zero_diagonal(residual)
        return residual

    def _accelerations(self) -> tuple[np.ndarray, np.ndarray]:
        pair_accel = (
            np.stack([-(self.config.pair_speed ** 2) * self.pair_laplacian(self.pair_u[r], r)
                      for r in (0, 1)])
            if self.config.pair_speed else np.zeros_like(self.pair_u)
        )
        head_accel = (
            np.stack([-(self.config.head_speed ** 2) * self.head_laplacian(self.head_u[r])
                      for r in (0, 1)])
            if self.config.head_speed else np.zeros_like(self.head_u)
        )
        if self.config.onsite_frequency:
            omega2 = self.config.onsite_frequency ** 2
            pair_accel -= omega2 * self.pair_u
            head_accel -= omega2 * self.head_u
        if self.config.bridge:
            residual = self._residuals()
            spring = self.config.bridge / (N - 1)
            pair_accel -= spring * residual
            head_accel += 0.5 * spring * residual.sum(axis=2)
        return pair_accel, head_accel

    def step(self, count: int = 1) -> None:
        """Advance the conservative wave and optional Rayleigh damping."""
        if count < 0:
            raise ValueError("step count cannot be negative")
        dt = self.config.dt
        damp = np.exp(-0.5 * self.config.damping * dt)
        for _ in range(count):
            self.pair_v *= damp
            self.head_v *= damp
            ap, ah = self._accelerations()
            self.pair_v += 0.5 * dt * ap
            self.head_v += 0.5 * dt * ah
            self.pair_u += dt * self.pair_v
            self.head_u += dt * self.head_v
            ap, ah = self._accelerations()
            self.pair_v += 0.5 * dt * ap
            self.head_v += 0.5 * dt * ah
            self.pair_v *= damp
            self.head_v *= damp
            self._zero_diagonal(self.pair_u)
            self._zero_diagonal(self.pair_v)
            self.ticks += 1

    def inject_transition(self, previous: int, observed: int, impulse: float = 1.0) -> None:
        """Apply *observed* directed transition as a velocity impulse.

        This function must never be called with a byte being predicted.
        No update happens in `features24`, `energy`, or the readout methods.
        """
        if not (0 <= previous < N and 0 <= observed < N):
            raise ValueError("transition endpoints must be bytes")
        if not np.isfinite(impulse):
            raise ValueError("impulse must be finite")
        a, b = previous, observed
        if a != b:
            self.pair_v[0, a, b] += impulse
            self.pair_v[0, b, a] += impulse
            self.pair_v[1, a, b] += impulse
            self.pair_v[1, b, a] -= impulse
            self.head_v[0, a] += 0.5 * impulse
            self.head_v[0, b] += 0.5 * impulse
            self.head_v[1, a] += 0.5 * impulse
            self.head_v[1, b] -= 0.5 * impulse
        else:
            # The 256 fixed points are heads, not discarded self-pairs.
            self.head_v[0, a] += impulse

    def advance_observation(self, previous: int, observed: int, impulse: float = 1.0) -> None:
        """Causal stream update after the true target has been revealed."""
        self.step()
        self.inject_transition(previous, observed, impulse)

    def energy(self) -> float:
        """Conservative Hamiltonian; decreases approximately under damping.

        Pair matrices double-count unordered edges, giving their 1/4 factor.
        """
        pair_kin = 0.25 * float(np.sum(self.pair_v ** 2))
        head_kin = 0.5 * float(np.sum(self.head_v ** 2))
        pair_potential = 0.25 * self.config.pair_speed ** 2 * sum(
            float(np.sum(self.pair_u[r] * self.pair_laplacian(self.pair_u[r], r)))
            for r in (0, 1)
        )
        head_potential = 0.5 * self.config.head_speed ** 2 * sum(
            float(np.dot(self.head_u[r], self.head_laplacian(self.head_u[r])))
            for r in (0, 1)
        )
        onsite_potential = self.config.onsite_frequency ** 2 * (
            0.25 * float(np.sum(self.pair_u ** 2))
            + 0.5 * float(np.sum(self.head_u ** 2))
        )
        residual = self._residuals()
        bridge_potential = (self.config.bridge / (4.0 * (N - 1))) * float(
            np.sum(residual ** 2)
        )
        return pair_kin + head_kin + pair_potential + head_potential + onsite_potential + bridge_potential

    def features24(self) -> np.ndarray:
        """A new deterministic 24D spectral probe, NOT R12's trained basis.

        Six low-frequency complex Fourier components per rail yield
        2 x 6 x (real,imag) = 24 values. Includes reciprocal head/pair state.
        """
        indices = ((0, 1), (1, 0), (1, 1), (0, 2), (2, 0), (1, 2))
        output = []
        for rail in (0, 1):
            marginal = self.head_u[rail] + self.pair_u[rail].sum(axis=1) / (N - 1)
            spectrum = np.fft.fft2(marginal.reshape(SIDE, SIDE), norm="ortho")
            for row, col in indices:
                output.extend((spectrum[row, col].real, spectrum[row, col].imag))
        return np.asarray(output, dtype=np.float64)

    def parity_error(self) -> float:
        """Maximum violation of the +/- mirror addressing identities."""
        return max(
            float(np.max(np.abs(self.pair_u[0] - self.pair_u[0].T))),
            float(np.max(np.abs(self.pair_u[1] + self.pair_u[1].T))),
            float(np.max(np.abs(self.pair_v[0] - self.pair_v[0].T))),
            float(np.max(np.abs(self.pair_v[1] + self.pair_v[1].T))),
        )
