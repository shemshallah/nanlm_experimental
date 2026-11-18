# M3 R31 — explicit directed default promotion

On 2026-11-17 (America/Los_Angeles), the user instructed: **“r31 make
default and continue”**. R31 therefore becomes the unified default M3
prediction path, while retaining selectable, bitwise-equivalent R30/R29/R26/
R24/R21 fallbacks. This is a **directed engineering promotion** with an
acknowledged failed scientific control, not an independent-TEST acceptance.

## Trained and frozen state

- R31 full 28-group Adam artifact: GitHub Actions run `44227555060`, artifact
  `m3-r31-28-group-optimizer-and-class-dev`, SHA-256 checkpoint
  `dd8fbaa93a17e5568bf2dc774b5599e5761824dee74db8f71242039f743a12ff`.
- Parent R30 checkpoint: SHA-256
  `a757433f6b72b4b444763de2f3158eea960a530c94fe613af58de247647b3421`.
- New Adam state: 768 steps on each of three class-calibration weights.
  The inherited 25 weights/moments/ages remain untouched in the recoverable
  28-group optimizer. All 28 are available to later training continuations.
- Frozen R30 coefficients:
  `[-0.0583874445684363, -0.3092941951187244, -0.257904760223574]`.
- Frozen R31 coefficients:
  `[1.2195544888726837, 0.03536605045213109, -0.8597706306738135]`.

## Evidence and limitation

R31 GATE books 428–431: R29 **3.394872** BPB, R31 **3.387715**, matched
unrelated history **3.377471**. R31 DEV books 432–443: R29 **3.410224**, R31
**3.394562**, matched unrelated history **3.386495**. On DEV, R31 nonspace
BPB improved 4.057942→3.898143 and top-5 improved 70.793%→70.923%; top-1
regressed 40.487%→40.435%; 7/12 books improved. Thus R31 **failed** its
GATE/DEV promotion criteria. The sealed R31 TEST books 444–467 were not read.

Promotion is authorized by the user's explicit override, not by a claim of
predictive or geometric superiority. Preserve the R29 reference and compare
against unrelated-history controls in subsequent rounds. Never relabel this
stage as independent TEST certification.
