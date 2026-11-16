"""Reproducible numerical wave diagnostics, independent of the R12 decoder.

Usage: python3 m3_r13/wave_diagnostics.py
No results generated here should be called text-model BPB or BPC.
"""

import json

import numpy as np

from mirror_wave import M3MirrorWave, N, PAIR_COUNT, WaveConfig


def diagnostics() -> dict:
    conservative = M3MirrorWave(WaveConfig(dt=0.15, bridge=0.4))
    conservative.inject_transition(0, 127)
    e0 = conservative.energy()
    conservative.step(120)
    e1 = conservative.energy()

    damped = M3MirrorWave(WaveConfig(dt=0.15, bridge=0.4, damping=0.2))
    damped.inject_transition(0, 127)
    damped.step(80)

    front = M3MirrorWave(WaveConfig(bridge=0))
    front.inject_transition(0, 128)
    front.step()
    neighbor_one_tick = float(front.pair_u[0, 1, 128])
    front.step()
    neighbor_two_ticks = float(front.pair_u[0, 1, 128])
    distant_two_ticks = float(front.pair_u[0, 15, 240])

    standing = M3MirrorWave(WaveConfig(dt=0.25, bridge=0, pair_speed=0))
    xx = np.arange(N).reshape(16, 16) // 16
    mode = np.cos(2 * np.pi * xx / 16).ravel()
    standing.head_u[0] = mode
    eigenvalue = 0.5 * (1 - np.cos(2 * np.pi / 16))
    standing.step(round(np.pi / (np.sqrt(eigenvalue) * standing.config.dt)))

    pair_ringing = M3MirrorWave(WaveConfig(dt=0.2, bridge=0))
    pair_ringing.inject_transition(0, 128)
    pair_trace = []
    for _ in range(48):
        pair_ringing.step()
        pair_trace.append(float(pair_ringing.pair_u[0, 0, 128]))

    uncoupled = M3MirrorWave(WaveConfig(bridge=0, pair_speed=0))
    uncoupled.inject_transition(0, 128)
    uncoupled.step(2)

    return {
        "architecture": "M3 R13 spatial mirror graph wave candidate",
        "source_round": "new, no R12 decoder/checkpoint loaded",
        "heads": N,
        "unordered_mirrors": PAIR_COUNT,
        "signed_pair_rails": 2,
        "conservative_energy_initial": e0,
        "conservative_energy_after_120_steps": e1,
        "conservative_relative_drift": (e1 / e0 - 1),
        "damped_energy_fraction_after_80_steps": damped.energy() / e0,
        "neighbor_amplitude_after_1_step": neighbor_one_tick,
        "neighbor_amplitude_after_2_steps": neighbor_two_ticks,
        "distant_amplitude_after_2_steps": distant_two_ticks,
        "uncoupled_neighbor_after_2_steps": float(uncoupled.pair_u[0, 1, 128]),
        "pair_ringing_min_48_steps": min(pair_trace),
        "pair_ringing_max_48_steps": max(pair_trace),
        "standing_mode_half_period_correlation": float(np.dot(standing.head_u[0], mode) / np.dot(mode, mode)),
        "pair_parity_error": conservative.parity_error(),
        "text_bpb": None,
        "text_bpc": None,
        "checkpoint_promotion": "none",
    }


if __name__ == "__main__":
    print(json.dumps(diagnostics(), indent=2, sort_keys=True))
