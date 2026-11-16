# M3 R15 — Phase-Invariant Learned Wave Readout

M3 is an independent NaN16 Mirror-Wave Memory Predictor experiment.  This
round extends R14's real wave without modifying the historical M3 R12
checkpoint (which is not available for migration).

## Why R14 was rejected

The NaNROUTER R14 benchmark passed CI but selected a zero signed-wave gain
for every arm: memory-only BPB was 3.876296 on 2,048 DEV bytes and all
wave arms tied it.  The signed displacement prediction alone did not show
useful generalization. R14 therefore remains a historical negative control.

## New testable mechanism

R15 keeps the 256 heads, 32,640 C(256,2) mirrors, opposite-sign rails,
16×16 spatial adjacency, symplectic wave dynamics and causal updates.
It adds five distinct wave-to-byte features:

1. Directional displacement (+ and − rails combined causally).
2. Directional momentum (velocity).
3. Forward power, `u² + (v/omega)²`, insensitive to phase sign.
4. Forward-minus-reverse power contrast.
5. Reciprocal head displacement and momentum.

The readout is a five-weight regularized, convex multinomial-logistic
residual.  All five coefficients are trained by damped Newton optimization
using **TRAIN targets only**, and then frozen on DEV.  The initial zero
weights reproduce memory-only exactly. This is a predictive candidate;
there is no inherited R12 neural optimizer state and no checkpoint
promotion based on source-level math alone.

Both the oscillator-only and spatial-wave arms have the **same** on-site
frequency, damping, wave impulse, and four-byte wave-clock stride. Only
the graph Laplacian and optional head–pair bridge differ.  Slower wave
clocking and 4,096-byte book windows expose longer temporal structures
than R14's short 768-byte pilot. The new window memory is still exactly
128 near + 2,048 older transitions.

## Independent verification

```bash
python3 -m unittest discover -s m3_r15 -p 'test_*.py' -v
python3 -m m3_r15.learned_wave_readout \
  --corpus-root /path/to/NaNROUTER/short-stories \
  --train-books 4 --dev-books 4 --bytes-per-book 4096 \
  --warmup 1024 --output m3_r15/results/R15_NaNROUTER_DEV.json
```

The GitHub workflow downloads NaNROUTER's public-domain short-story batch
0004, checks its SHA-256, uses book-disjoint TRAIN/DEV, and publishes the
results as a workflow artifact. TEST remains untouched. Every reported
gain must be separately compared to the stronger oscillator-only arm;
otherwise it cannot be attributed to spatial wave propagation.

Promotion requires a positive reproducible DEV result, independent held-out
books, longer-context gates, and a verified loadable readout/checkpoint.
