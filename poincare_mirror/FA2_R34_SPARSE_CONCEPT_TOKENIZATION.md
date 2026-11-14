# FA2 R34 — Sparse geometric concept tokens and tetrahedral transfer

The experiment extends the **same FA2 model and Adam history**, with no FA1
changes. Its starting R30 checkpoint is step 2,801, SHA-256
`9ad6a368a9e68f4c1ccff366bee88047f7475f93ccb4d1da1c999db7bddee902`.
The R33 intermediate at step 2,833 is an **unpromoted** 512D pilot.

## Typed latent concept tokens (not a replacement for the 257-symbol byte codec)

Each token in a causal window routes to 8 of 256 lower point addresses. Each
lower point has a shared two-loop microattention kernel and its own trained
low-rank FF adapter. The 8 points generate the exact combinatorial subsets:

| Type | Address space | Active per byte position | Representation/storage |
| --- | ---: | ---: | --- |
| point | 256 | 8 | explicit point vectors, per-point micro-FF |
| oriented S+/S− mirror pair | C(256,2)=32,640 | 28 | exact signed address; shared/anti pair embeddings |
| triangle concept | C(256,3)=2,763,520 | 56 | exact integer ID; 8,192-row hash code + Fourier address + oriented pair boundary |
| tetra concept | C(256,4)=174,792,640 | 70 | exact integer ID; 8,192-row hash code + Fourier address + oriented face boundary |
| meta apex MoE | 256 experts | 8 | triangle-fed sparse expert FF and two feedback loops |

For increasing addresses `a<b<c<d`, the exact combinadic IDs are:

* triangle: `C(c,3) + C(b,2) + a`;
* tetra: `C(d,4) + C(c,3) + C(b,2) + a`.

The pair level retains the repository's original lexicographic
`a*(511-a)//2+b-a-1` address. The full millions-wide spaces are **addressable**
but not allocated as millions of independent trainable vectors; hash-table
collisions remain possible and are supplemented with exact-address Fourier
features. The sparse point/pair/triangle/tetra tensor sizes are independent
of the full combinatorial cardinalities.

## Information transfer and storage

1. The selected point microtransformers communicate by two same-position
   self-attention rounds; no future text tokens enter their representation.
2. Oriented pair tokens include distinct symmetric and antisymmetric learned
   weights, using the same S+/S− sign probability as the original mirrored MoE.
3. A triangle receives the signed/unsigned incidence messages of its three
   edges, then a tetra receives the messages of its four oriented triangular
   faces. These use fixed simplicial boundary matrices `d1,d2,d3`, checked by
   `d2@d1=0` and `d3@d2=0`. The unsigned part avoids the trivial cancellation
   caused by applying a boundary twice.
4. Sparse tetra and triangle content reaches the 256 apex experts via the
   learned geometric residual. The apex MoE itself is selected by triangular
   abstract concept state; only 8/256 experts execute for a position.
5. The meta layer returns through a **meta-anchored tetra**: each of 56 lower
   triangles treats the current meta vector as a fourth vertex. A learned
   downward pass aggregates tetra -> face -> edge -> intake-point features.
6. A causal cumulative prefix of simplex writes gives an O(context)-time
   within-window memory. It resets at every independent document window; it
   is **not** yet a persistent cross-window external memory.

All R34 transfers are controlled by a zero-initialized learned residual gate.
The step-2,833 -> step-2,833 migration verified **zero** +1/+2/+3 logit drift
and remapped all 230 inherited Adam moment tensors by name. R34 is opt-in:
`grow --run-dir RUN --simplex-transfer`. No trained simplex path can be
removed by a later capacity migration.

## Actual CPU experiments

Data: checksum-verified `corpora-short-stories-0004` (615 unique books;
491 training, 62 development, 62 held-out test), UTF-8 byte + EOS vocabulary.
Train context mix `256:8,512:3,1024:1`, batch one, four CPU threads. Evaluation
is same 2,048 development byte positions, seed 2026, context 256.

| DEV metric | Original 192D step 2,801 | 512D R33 step 2,833 | R34 step 2,849 |
| --- | ---: | ---: | ---: |
| CE +1 (lower better) | 2.195293 | 2.169560 | 2.173589 |
| CE +2 | 2.804915 | 2.795661 | 2.799840 |
| CE +3 | 3.020161 | 3.015114 | 3.008189 |
| next-byte top-1 | 37.06% | 38.53% | 38.92% |
| next-byte top-5 | 73.97% | 74.90% | 74.61% |
| bits/byte | 3.167138 | 3.130013 | 3.135825 |

R33: 32 actual updates after migration. R34: 16 more actual updates on the
expanded model. R34 has 76,476,019 parameters, 36 mixed-dimension physical
attention heads (16 inherited), and 151 populated Adam states (115 inherited
before R34's new weights got gradients).

### Causal pathway ablations at step 2,849

All disabled pathways use the **same** weights and fixed development positions.
The positive number is how much +1 CE worsens when a component is disabled.

| Disabled component | CE without - full CE | Interpretation |
| --- | ---: | --- |
| point microtransformers | +0.001113 | modest positive contribution |
| token hierarchy | +0.001539 | modest positive contribution |
| entire geometric residual | +0.000488 | positive on this DEV sample |
| triangle abstract address | -0.000039 | no established benefit |
| triangle-fed meta MoE | +0.000003 | no meaningful contribution yet |
| new simplex transfer | <0.000001 | no measurable contribution yet |
| antisymmetric mirror weights | +0.094836 | strongly useful inherited pathway |

The simplex gates moved away from zero (~ -0.00027 to -0.00037), but the
new transfer is **not yet predictively useful** at the current precision.
Longer or better-targeted training is required. These ablations are associative
measurements at a fixed checkpoint, not proof of causal semantic understanding.

## Promotion status

The first R33 independent test-book gate **failed**: +1 test CE improved from
3.265911 to 3.255024, but +2 CE regressed 3.704141 -> 3.713745. These 2,048
test positions cover only the earliest portion of the first test book; broader
book coverage is required. R34 has not been sent through a fresh sealed test
promotion gate. The canonical R30 checkpoint and FA1 remain unchanged.

Further priorities: prove a nontrivial simplex ablation effect, add a
target-conditioned auxiliary geometric prediction objective, use stratified
held-out whole-book evaluation, and implement explicit persistent cross-window
concept state with reset/isolation checks. Do not claim frontier quality.
