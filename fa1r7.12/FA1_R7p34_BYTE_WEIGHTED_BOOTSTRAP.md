# FA1 R7.34 — conditional-byte evaluation repair and immutable checkpoint guard

**Scope:** existing FA1 R7.33 layered, calibrated Diamond on `master` only.
No architecture split, new parameter, optimizer group, or automatic checkpoint
promotion. FA2 is unchanged. R7.33's multi-horizon curriculum remains intact.

## Why this is necessary before further training

The conditional spelling metric is a **ratio of total UTF-8 byte loss to
literal bytes scored**, excluding EOW. Earlier R7.31–R7.33 diagnostic code
reported a byte-weighted full-model point estimate but formed several 95%
book-bootstrap intervals from unweighted **per-word mean byte losses**.
Short and long target words therefore had inconsistent influence on the
point estimates and the promotion confidence bounds. A gain with many
correct short words can have a positive word-mean bootstrap even when its
literal-byte-weighted gain is negative.

R7.34 uses a cluster bootstrap of the **same pooled-byte ratio estimator**:

1. For each scored word, multiply its per-byte NLL difference by the number
   of literal bytes scored (exclude EOW and ignore zero-byte targets).
2. Sum both byte-NLL differences and byte counts within each held-out book.
3. Resample whole books with replacement (1,024 times, seeded), summing the
   selected books' numerators and denominators before forming each ratio.
4. Use the 2.5% and 97.5% quantiles for the 95% book-bootstrap interval.

This applies to conditional-byte geometric gain, R7.32 matched-history
byte-specific gain, and R7.33 worst-horizon far-history byte gain and byte
geometric specificity. All four existing promotion gate keys remain stable.
Reports additionally expose `book_pooled_estimate`, `resampling_unit` and
`estimand` so the units and exact weighted estimand are auditable.

For example, three target words with mean per-byte changes `[+1,+1,-0.5]`
and literal-byte counts `[2,2,10]` have per-word mean `+0.5` but pooled-byte
gain **−1/14 = −0.07143**. R7.34 correctly reports the latter as its pooled
point estimate. The small synthetic case is a unit test, not an LM benchmark.

## Protected recovery state

Before loading an ancestor or starting updates, R7.34 rejects:

- Report output resolving to the same file as the source checkpoint.
- Candidate `.pt` resolving to the same file as the source checkpoint.
- Report output and candidate checkpoint referring to the same file.
- Symlink and existing hard-link aliases for these cases.

After writing an optional *unpromoted* candidate, R7.34 hashes the original
source checkpoint a second time and fails if it changed. The accepted
step-14,416 ancestor, inherited parameters and optimizer histories are not
altered. New helper functions carry **no learnable state**, so R7.33
checkpoint compatibility is unchanged.

## Validation and research status

Run the source verification with:

```bash
PYTHONPATH=fa1r7.12/source python -m unittest discover \
  -s fa1r7.12/source -p 'test_fa1_r734_*.py' -q
```

The added tests cover byte/word disagreement, book-cluster reproducibility,
zero-byte positions, invalid shapes and numerical inputs, allowed separate
artifacts, and same-path/symlink/hard-link protection. Existing R7.27–R7.33
source regressions also run in GitHub Actions.

**No new held-out FA1 model benchmark is claimed.** Previous R7.31–R7.33
conditional-byte confidence intervals must be recomputed using the same
frozen candidate and book-disjoint DEV positions; prior word/bootstrap CI
figures are not interchangeable with R7.34's byte ratio CIs. A strict DEV
pass, including real older-history specificity and original word/byte gates,
is still needed before any sealed TEST evaluation or checkpoint promotion.
