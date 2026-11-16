# M3 R21 — Predeclared frozen independent TEST for canonical R20 default

R20 DEV selected pair_scale=0.5 and triangle_scale=0.5 from fresh TRAIN
books alone; DEV achieved +0.3086173262 BPB over inherited R19 on 12
additional books. This is the first independent TEST of that exact frozen
selection and of the canonical M3 streaming inference implementation.

## Complete provenance / split boundaries

- NaNROUTER release: corpora-short-stories-0004 with verified SHA-256.
- Selection seed: 160016; exact UTF-8-aligned 4096-byte prefixes;
  1024-byte warmup.
- Index 0–7: R16 original TRAIN prior; immutable.
- Index 8–51: R16 DEV / R17 TEST / R18 TEST; SHA-256-verified, never
  selected for tuning or scored in R21.
- Index 52–83: R19 new TRAIN/DEV; never used for R21 fitting or testing.
- Index 84–99: reserved R19 independent TEST; never used in R21.
- Index 100–131: R20 TRAIN/DEV; never used for R21 selection or testing.
- Index 132–155: **24 previously untouched TEST documents**; score exactly
  once with weights/scales fixed above. No fitting, fallback choice or
  checkpoint promotion may be conditioned on these target bytes.

## Numeric parity and independent controls

The same M3DefaultPredictor streams observed bytes to:
1. The untouched R16 oscillator, frozen R19 class gate, directed signed
   pair address memory and sparse ordered-triangle successor memory.
2. An exact R19 fallback distribution.
3. The frozen selected R20 default distribution.
4. A wrong-address control shifting both pair and triangle successor
   logits by 37 candidate positions.
5. A wrong-context control shifting only the triangle logits.
6. A triangle-only diagnostic.

All probability distributions are computed before observing the current
target byte. No model state or optimizer is trained or reset at target
scoring time; only document boundaries reset the causal stream.

The primary statistic is whole TEST BPB reduction over frozen R19, with
24-book bootstrap interval and per-book signs. Also report non-space BPB,
UTF-8 bits/character, top-1/top-5 and throughput. A positive bootstrap
lower bound is descriptive, not a formal frequentist confidence claim
under adaptive research. A successful TEST supports the *causal successor
memory engineering result*, not a unique advantage over equally calibrated
conventional trigram models.

Workflow .github/workflows/m3-r21-default-locked-test.yml, master only.
