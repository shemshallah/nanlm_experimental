# M3 R32 certified promotion — exact checkpoint and provenance

Date: 2026-11-18. Branch: `master` only. Lineage: M3, separate from FA1,
FA2, NaNLMind and Terminal-RSI.

## Default selection

The user explicitly requested promotion of trained R32 and continued
architecture upgrades. `M3DefaultPredictor(prior)` therefore selects `r32`.
The former default `mode="r31"` and exact `r30`, `r29`, `r26`, `r24` and
R20/legacy modes remain available through the same causal state machine.

The initial promotion was user-directed. Subsequently, the prelocked
independent 24-book TEST **passed all seven prespecified criteria**, so R32
is now independently TEST-certified. R26 previously passed its own TEST;
R29 and R31 remain directed promotions, not independent certifications.

## Source-of-truth integrity

- Original 32-group CPU checkpoint: `m3_default/r32_accepted_checkpoint.npz`
- SHA-256: `67e7417fb38c8d914714cee243fb080da6e63cf5f9e9c9737e28a5b168c1da1a`
- Frozen DEV JSON: `m3_r33/R32_FROZEN_DEV.json`
- DEV JSON SHA-256: `bf0f2d5a17863b78c7925e037901521e543a315d66c5865eefb0678d248f92c2`
- R31 parent: `dd8fbaa93a17e5568bf2dc774b5599e5761824dee74db8f71242039f743a12ff`
- 28 inherited optimizer groups + 4 new R32 groups = 32 groups preserved,
  including first and second Adam moments and step ages.
- Original fitted gains: `[0.5927959113137101, -1.5,
  -0.20722426183204995, 0.294518336989888]`.
- 16 separate TRAIN books (468–483) contribute the embedded 65,536-byte,
  individually SHA-checked immutable background. FIT 484–499, GATE
  500–503, DEV 504–515. TEST 516–539 was unopened at the time of lock.
- [Precommitted TEST lock](../m3_r33/r32_frozen_candidate.json) and
  [original successful CPU training](https://github.com/shemshallah/nanlm_experimental/actions/runs/44233287739).

## Measured training and development results

768 new CPU updates on 49,150 positions; all inherited parameters and
optimizer histories retained. No train/test cross-document leakage.

| Same DEV positions, 12 disjoint books | R31 | R32 |
| --- | ---: | ---: |
| Bits/byte | 3.303148 | **3.097359** |
| Nonspace bits/byte | 3.956602 | **3.777566** |
| Top-1 byte accuracy | 42.4828% | **45.1847%** |
| Top-5 byte accuracy | 71.8057% | **75.3188%** |

R32 improved 12/12 DEV books and 4/4 GATE books. On DEV, matched unrelated
history scored 3.292148 BPB, full unrelated 3.280915 BPB, and wrong
successor values 3.524173 BPB. This supports the lexical anchor's value
within the tested distribution, now corroborated by independent TEST.

## Independently verified TEST — all seven acceptance criteria passed

Source: [GitHub Actions run 44375790318](https://github.com/shemshallah/nanlm_experimental/actions/runs/44375790318).
Frozen test indices 516–539; 24 untouched NaNROUTER short-story books,
73,728 scored byte positions. No weights or criteria were changed after lock.

| Measure | R29 | R31 | R32 |
| --- | ---: | ---: | ---: |
| BPB ↓ | 3.430948 | 3.420187 | **3.212654** |
| Nonspace BPB ↓ | 4.026989 | 3.876766 | **3.704590** |
| Byte top-1 ↑ | 39.5684% | 39.7244% | **41.8443%** |
| Byte top-5 ↑ | 70.5702% | 70.6706% | **74.2961%** |

R32 improved **24/24** TEST books over R31, a gain of **0.207533 BPB**
(6.07% relative). The descriptive whole-book bootstrap interval is
**[+0.183534,+0.229933] bits/byte**. R32 defeated wrong successor values
(3.658268 BPB), unrelated matched history (3.412642 BPB), and unrelated
full history (3.400951 BPB). Every precommitted test condition is true,
including nonspace loss, top-1, top-5 and positive book-bootstrap lower
bound.

Complete frozen report: [`m3_r33/results/R32_INDEPENDENT_TEST.json`](../m3_r33/results/R32_INDEPENDENT_TEST.json),
SHA-256 `047f4c4a44fea7c5056fa058339c78166b7e922eaca35b5c07fa6831337762f6`.

## Independent TEST transport and non-peeking policy

Frozen workflow run `44355232997` failed **before TEST reading** because
the external GitHub Actions checkpoint artifact download returned HTTP 503
egress over account limit. The subsequent transport repair embeds the exact
hash-matched checkpoint and DEV report directly in GitHub `master`. It does
not alter coefficients, precommitted TEST indices, or acceptance thresholds.
The NaNROUTER corpus is still the source for all new TEST data.
The second TEST workflow run `44356262187` scored all 24 books but failed
*after* inference on a NumPy Boolean JSON serialization bug. The reporting
fix (`a1201ba`) changed no weights, splits, criteria, or comparisons;
run `44375790318` replayed the exact locked evaluation and wrote the
successful report. The replay is disclosed for reproducibility.

## Regression and recoverability

The unified default is checked for bitwise parity with `M3R32Predictor`
on a causal byte stream and following reset. It is also checked against
the historical R31/R30 and earlier exact fallback predictions. No
promotion or future architecture experiment may discard or rewrite the
32-group source checkpoint. New candidates go in new folders on `master`
and may only replace the default after explicit evaluation or user direction.
