# WikiText-2 CPU architecture report

This is a GitHub-hosted CPU training gate, not a frontier-quality run.  The
model is deliberately small (~31M params), the corpus is WikiText-2 raw, and
the tokenizer is the bundled 64k layered artifact.

## Data

- train tokens: 2,450,404
- valid tokens: 259,326
- bytes/token: 4.4059
- vocab: 64,060

## 200-step matrix

Run: <https://github.com/shemshallah/nanlm_experimental/actions/runs/43755456161>  
All rows completed 200 steps with zero non-finite skips.

| Row | Loss | Bits/byte | Params |
|---|---:|---:|---:|
| baseline_no_concept | 7.20094 | **2.35793** | 30,847,070 |
| concept_no_composition | 7.24315 | 2.37175 | 31,729,760 |
| concept_no_sidecar | 7.25700 | 2.37628 | 31,731,104 |
| concept_no_mtp_structure | 7.26325 | 2.37833 | 31,731,104 |
| concept_pair_only | 7.26429 | 2.37867 | 31,731,104 |
| concept_full | 7.26501 | 2.37891 | 31,731,104 |

At 200 steps the no-concept baseline is best.  The gap is only ~0.02 BPB and
the run sees only ~51k training tokens (about 2% of one pass over the training
split), so this is not evidence that the concept architecture is worse; it is
evidence that the current concept injection is not free at very short horizons.

A second 200-step run tested `--concept-loop-scale 0.1`:

| Row | Loss | Bits/byte |
|---|---:|---:|
| baseline_no_concept | 7.20125 | 2.35803 |
| concept_loopscale10 | 7.26789 | 2.37985 |

Scaling the loop down did not help.  The composition auxiliary loss is the most
visible single cost in the short run: disabling the composition head improves
BPB from 2.3789 to 2.3718.  Removing the sidecar does **not** recover the
baseline, so the cost is in the concept/object loop rather than tokenizer
alignment alone.

A third control isolated the attention residual:

| Row | Loss | Bits/byte |
|---|---:|---:|
| baseline_no_concept | 7.20094 | 2.35793 |
| concept_router_only | 7.11532 | 2.32989 |
| concept_router_tokens | **7.11338** | **2.32926** |

Router-only concept feedback is the first concept variant to beat the baseline
at the same token budget.  Token-aligned router state was slightly better at
200 steps; its 1,000-step confirmation is below.

Token-aligned routing is now the default:
`--concept-router-tokens` is on, `--no-concept-router-tokens` restores the
global-context control, and full attention injection remains available as
`--concept-loop-attention` for ablation.

## Longer convergence run

A 1,000-step baseline-vs-concept run is dispatched separately on the same
data and configuration:

<https://github.com/shemshallah/nanlm_experimental/actions/runs/43756422506>

## 1,000-step convergence

| Row | Loss | Bits/byte | Params |
|---|---:|---:|---:|
| baseline_no_concept | 6.24743 | 2.04570 | 30,847,070 |
| concept_router_only (global context) | 6.22357 | 2.03789 | 31,731,104 |
| **concept_router_tokens** | **6.15627** | **2.01585** | 31,731,104 |
| concept_router_no_composition (global) | 6.22435 | 2.03815 | 31,729,760 |
| concept_full (attention injection) | 7.04421 | 2.30661 | 31,731,104 |
| concept_no_composition (full attention) | 7.05788 | 2.31108 | 31,729,760 |

The result is consistent across horizons:

- 200 steps: global router 2.3299, token router 2.3293, baseline 2.3580.
- 1,000 steps: token router **2.01585**, global router 2.03789, baseline 2.04570.

Full attention injection is clearly harmful at this scale and duration
(2.3066 at 1,000 steps).  Router-only concept feedback is therefore the
promoted default; `--concept-loop-attention` remains the explicit control flag.

Removing the composition auxiliary head in router-only mode changes BPB only
from 2.03789 to 2.03815, so the composition head is neutral to slightly helpful
at this horizon and stays enabled.

The token-router gain over baseline is ~0.030 BPB at 1,000 steps, still on a
small model and one run.  It is a validated architecture signal, not a frontier
claim.
