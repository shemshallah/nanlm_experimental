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

The composition auxiliary loss is the most visible single cost in this short
run: disabling the composition head improves BPB from 2.3789 to 2.3718.
Removing the sidecar does **not** recover the baseline, so the cost is in the
concept/object loop rather than tokenizer alignment alone.

## Longer convergence run

A 1,000-step baseline-vs-concept run is dispatched separately on the same
data and configuration:

<https://github.com/shemshallah/nanlm_experimental/actions/runs/43756422506>

The next architecture decision waits for that slope.  If concept full remains
behind baseline at 1,000 steps, the first targeted change is a gated concept
loop injection (small initial loop scale) rather than adding more experts.
