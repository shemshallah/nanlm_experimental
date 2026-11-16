# FA2 R47 — Selective reciprocal emission, read-only causal ablation

**2026-11-15 · `master` / `poincare_mirror` · checkpoint protection in force**

## Verified research starting point

The previously reported R46 DEV comparison sampled 8 causal windows from
each of 12 disjoint whole NaNROUTER DEV books: 12,288 matched prediction
positions per configuration. **This is not exhaustive full-book scoring.**
A JSON transcription of
the **user-provided summary**, with its precise provenance, is preserved at
`reports/fa2_r46_reported_dev_metrics.json`.

| DEV metric | Protected step-2993 parent | R44 full bridge | R44 coupling only |
|---|---:|---:|---:|
| Next-byte CE (nats) ↓ | **2.383392** | 2.384695 | 2.383520 |
| Bits/literal byte ↓ | **3.438508** | 3.440388 | 3.438692 |
| Top-1 (%) ↑ | 35.70 | **35.72** | 35.69 |
| Top-5 (%) ↑ | 69.88 | 69.79 | **69.93** |
| +2-byte CE ↓ | **2.926668** | 2.928017 | 2.926931 |
| +3-byte CE ↓ | 3.106511 | **3.105153** | 3.105255 |
| Non-space CE ↓ | 2.650380 | 2.649764 | **2.648795** |

The coupling-only next-byte CE gain relative to the protected parent was
reported as −0.000127 nats/byte, with a descriptive 95% whole-book bootstrap
interval [−0.002407, +0.001926]. The point estimate is *worse* than the
parent, and the interval crosses zero. This is not a promoted result.

**Provenance limitation:** the R45–R46 raw per-book score arrays and R44
candidate checkpoint were not present in the checked-in source. The table
above transcribes the preceding research report, not a newly executed R47
measurement. R47 adds the missing reproducible ablation mechanism; no new
large-scale predictive result is claimed here.

## Why per-layer emission is the next experiment

The same existing reciprocal geometry performs **two distinguishable jobs**:

1. *Internal coupling:* causal token resolutions affect mirror-pair,
   triangle and apex microtransformers within the shared network.
2. *Outer emission:* `reciprocal.emit` injects a hidden-state residual after
   each of the six inherited backbone blocks.

The R46 coupling-only advantage in non-space CE makes the first job
interesting, but the second job damaged overall likelihood in R44. We now
isolate the **location and magnitude of outer emission** while retaining the
first job. This is an ablation of the existing model, not a new language head.

The R47 implementation introduces a runtime-only `emission_gains` tuple.
`None` or all ones preserves the original logits. All zeros is a true
internal-coupling-only path, **not** the fully disabled reciprocal control.
Early-only and late-only settings divide the six blocks at the midpoint;
quarter and half use the same gain at all blocks. There are no new trainable
parameters, checkpoint keys, optimizer groups, altered moments or future
token accesses.

## DEV-only preregistered sweep

Six fixed configurations are evaluated on exactly the same causal windows:

| Label | Per-layer outer emission gain (six-block architecture) |
|---|---|
| `full` | Original checkpoint computation |
| `coupling_only` | 0, 0, 0, 0, 0, 0 |
| `quarter` | 0.25 at all layers |
| `half` | 0.5 at all layers |
| `early_only` | 1, 1, 1, 0, 0, 0 |
| `late_only` | 0, 0, 0, 1, 1, 1 |

The program checks the DEV split and manifest hashes, whole-book boundaries, model/data
manifest equality and context bounds. Each arm outputs next/+2/+3 CE,
top-1, top-5, space and non-space CE/accuracy; bits/byte are exactly
`CE / ln(2)` for literal bytes. Pairwise book-resampled confidence intervals
are Bonferroni-adjusted across five alternatives to the full or coupling-only
reference. The protected-parent comparison includes all six emission arms,
so its interval uses a six-arm correction. The new
report distinguishes **bits per byte/EOS target** (`CE / ln(2)`) from **bits
per literal byte**, which removes EOS targets from the denominator; it also
reports actual raw bytes/token. Earlier R46 summaries used CE/ln(2) for their
published bits/byte column and must be compared on that same convention.
The confidence intervals resample books as clusters of sampled windows;
they do not imply that every position of each book was evaluated.

```bash
cd poincare_mirror
python scripts/probe_r47_selective_emission.py \
  --checkpoint /path/to/R44_candidate/latest.pt \
  --parent /path/to/protected_step2993/latest.pt \
  --data-dir /path/to/verified_38book_bytes \
  --context 128 --windows-per-book 8 --seed 2026 \
  --bootstrap 2000 --threads 2 \
  --output /path/to/r47_dev.json
```

The parent, candidate and training optimizer remain untouched. The script
intentionally does **not** offer a `--split test` option: choose the model
and fixed emission schedule only on DEV, lock it, then run a separate
single-shot sealed TEST evaluation and aligned context ladder (128/256/1024/
4096 token input, matched target positions) before considering promotion.

## Admission rule and unresolved evidence

The R47 diagnostic screen requires a strictly positive Bonferroni-adjusted
lower book-bootstrap bound for both next-byte CE and non-space CE against the
protected parent, no mean +2/+3 CE regression, no non-space top-1/top-5
regression, and no material whole-byte top-5 regression. A pass still **does
not automatically promote** anything. Check for long-context regressions,
independent TEST, repeated data-split sensitivity and full recovery of
model and Adam moments first.

The user-supplied R46 numbers support preserving internal geometry as a
research hypothesis but do not justify altering the canonical FA2 lineage.
No FA1 or Terminal-RSI architecture or optimizer has been changed.
