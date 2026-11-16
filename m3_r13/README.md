# M3 R13 — Genuine Mirror-Wave Engine (candidate)

**Status:**  New additive engine on `master`, **not** an M3 R12 checkpoint
migration or a demonstrated BPB/BPC improvement.  M3 stays independent of
FA1, FA2, Terminal-RSI, and the NaNLang virtual-machine core.

## Mathematical object

- `N=256` fixed-point heads occupy a periodic 16×16 lattice.
- Exact native address map: `addr(a,b)=(a<<8)|reverse8(b)`. Reversing all
  16 address bits swaps `(a,b)` and `(b,a)`, leaving exactly 256 fixed
  diagonal points. This is an addressing identity, not a new NaNLang VM.
- `C(256,2)=32640` undirected mirror addresses each carry two wave rails:
  `S+` is symmetric under swapping head labels; `S-` is antisymmetric.
- Each mirror `(a,b)` has up to eight spatially adjacent mirror neighbors:
  move `a` by one lattice edge while fixing `b`, or vice versa.  A forbidden
  diagonal collision removes two neighbors, giving degree six instead of eight.
- This is a **spatially local shared-head graph**: mirror neighbors have one
  endpoint in common, but a disturbance cannot jump instantly across all
  32,640 mirrors.  It is the unordered two-particle configuration space of
  the 16×16 torus, with the collision diagonal excluded.
  Reciprocal head/pair springs optionally provide additional short paths
  through the head nodes; strict spatial-front diagnostics turn them off.

With `u` the wave displacement and `v` its velocity, the discrete system is

```
du/dt = v
dv/dt = -c² (D-A)u/8 - gamma*v - dV_bridge/du
```

The head layer independently obeys the 16×16 toroidal graph wave equation.
The head and mirror layers exchange energy through a reciprocal spring
potential, with `k=bridge/255`:

```
V_bridge = k/2 * sum_{a<b} [
  (S+_ab - (H+_a + H+_b)/2)^2
  + (S-_ab - (H-_a - H-_b)/2)^2
]
```

The signed direction rail uses oriented incidence.  Using ordinary symmetric
incidence on `S-` would break its anti-transposition parity.  An optional
complete shared-head line-graph Laplacian is included as a controlled
ablation (`shared_head_fraction`).  Its global connections remove strictly
local wavefronts, so the default keeps it off.

### What makes this a wave (rather than an oscillator label)

1. Displacement **and** momentum are state variables; a second-order
   spatial Laplacian couples their evolution.
2. Distant mirrors have zero response until the wavefront arrives.
3. The linear conservative system supports superposition and interference.
4. Pair-source amplitudes repeatedly change sign; nontrivial head Fourier
   eigenmodes exhibit standing-wave half-period reversal.
5. With `gamma=0`, the derived total Hamiltonian stays approximately
   constant under bounded-step velocity-Verlet integration; with damping,
   energy decays.  The exact continuum/discrete-time physical simulation
   analogy is mathematical, not a claim of experimentally observed waves.

### Causal stream operation

Read `features24()` before the target is observed.  Only **after** it is
revealed, call `advance_observation(previous, observed)` to evolve the wave
and inject the observed directed transition.  Self-transitions excite the
256 fixed-point heads instead of being dropped.  `features24()` is a NEW
fixed Fourier probe (12 real dimensions per rail); it is not byte-for-byte
compatible with the prior R12 trained 24D spectral basis or 384-feature
ridge readout without a verified adapter.

### Reproducible numerical validation

```bash
python3 -m pip install numpy
python3 -m unittest discover -s m3_r13 -p 'test_*.py' -v
python3 m3_r13/wave_diagnostics.py
```

The tests verify exact topology, parity, finite-speed wavefront against
uncoupled controls, signed orientations, nontrivial standing modes, ringing,
two-way energy exchange, conservative bounded-energy behavior, damped decay,
superposition, and readout causality.

## Promotion / data integrity

- Keep R12 measurements as historical baselines; no scores are transferred.
- R12 source and trained ridge checkpoint were **not recovered** during this
  development.  Do not represent this code as a loadable successor until a
  verified migration and same-book evaluation pass.
- Independent NaNROUTER book-disjoint BPB/BPC/top-1/top-5, matched graph-off
  and oscillator-only controls, and throughput are mandatory for R13
  checkpoint promotion.  The numerical wave tests establish wave dynamics,
  **not predictive superiority**.
- Training against development books must not reuse their target bytes in
  feature construction or in same-position gate calibration.
