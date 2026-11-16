# WikiText-2 CPU architecture report

This is a GitHub-hosted CPU training gate, not a frontier-quality run.  The
model is deliberately small (~31M params), the corpus is WikiText-2 raw, and
the tokenizer is the bundled 64k layered artifact.

## Data and metric denominators

- train tokens: 2,450,404
- validation tokens: 259,326
- training bytes/token: 4.4059
- fixed evaluation windows: 510 target tokens, 2,206 UTF-8 bytes,
  2,202 Unicode characters
- evaluation bytes/token: 4.32549
- evaluation chars/token: 4.31765

The first version of this report used the training bytes/token average for
validation BPB.  That understated BPB/BPC by ~1.8%.  The evaluator now sums
exact byte and character lengths from the aligned sidecar for the same causal
target positions as the CE loss.  The tables below use the exact denominator.

## 1,000-step convergence

| Row | Loss | Bits/byte | Bits/char | Params |
|---|---:|---:|---:|---:|
| baseline_no_concept | 6.24743 | 2.08372 | 2.08751 | 30,847,070 |
| concept_router_only (global) | 6.22357 | 2.07577 | 2.07954 | 31,731,104 |
| **concept_router_tokens** | **6.15627** | **2.05332** | **2.05705** | 31,731,104 |
| concept_router_no_composition (global) | 6.22435 | 2.07603 | 2.07980 | 31,729,760 |
| concept_full (attention injection) | 7.04421 | 2.34948 | 2.35375 | 31,731,104 |
| concept_no_composition (full attention) | 7.05788 | 2.35404 | 2.35831 | 31,729,760 |

Token-aligned router state is best: **2.05332 BPB / 2.05705 BPC**, versus
baseline **2.08372 / 2.08751**.  The improvement is ~0.030 BPB.

Full attention injection is clearly harmful at this scale and duration
(2.34948 BPB).  Router-only concept feedback is therefore the promoted default;
`--concept-loop-attention` remains the explicit control.

Removing the composition auxiliary head in router-only mode changes global
router BPB only from 2.07577 to 2.07603, so the composition head is neutral and
stays enabled.

A 200-step control also wired token-aligned concept state into the MTP branch
context.  It regressed exact BPB from 2.37255 to 2.37564, so MTP context remains
global while the trunk router stays token-aligned.

A second control removed the pooled concept component from the trunk router,
leaving only token state:

| Row | BPB | BPC |
|---|---:|---:|
| global + token router | **2.37274** | **2.37705** |
| token-only router | 2.37434 | 2.37866 |

The pooled concept component is retained; token-only is slightly worse.

## 200-step controls (exact denominators)

| Row | Loss | Bits/byte | Bits/char |
|---|---:|---:|---:|
| baseline_no_concept | 7.20094 | 2.40175 | 2.40612 |
| concept_router_only (global) | 7.11836 | 2.37421 | 2.37852 |
| **concept_router_tokens** | **7.11338** | **2.37255** | **2.37686** |
| concept_loopscale10 | 7.26789 | 2.42408 | 2.42849 |
| concept_full (attention) | 7.26501 | 2.42312 | 2.42752 |

The ordering is consistent across horizons.  Scaling the concept loop down did
not help.

## Timed 1,000-step multi-resolution run

Run: <https://github.com/shemshallah/nanlm_experimental/actions/runs/43795909157>  
Commit: `0889234`  
Architecture: token-aligned router, global+token context, global MTP context,
composition head, byte/word probe heads.

| Metric | Value |
|---|---:|
| Training steps | 1,000 |
| Parameters | 34,852,448 |
| Non-finite skips | 0 |
| Train time | 30m21.9s |
| Row wall time | 30m27.1s |
| Full workflow wall time | 31m49s |
| Final loss | 6.16333 |
| BPB | 2.05568 |
| BPC | 2.05941 |
| Subword/token top-1 | 20.00% |
| Subword/token top-5 | 32.16% |
| Byte top-1 (next token first byte) | 94.31% |
| Byte top-5 | 95.10% |
| Word top-1 (next token lexical word) | 12.08% |
| Word top-5 | 24.47% |
| Byte probe loss | 0.37756 |
| Word probe loss | 6.62169 |

The probe heads add ~3.1M parameters and slightly increase the main BPC versus
the no-probe 1,000-step run (2.05941 vs 2.05705). They are diagnostic heads, not
part of the default generation path. Byte accuracy means the first byte of the
next token; word accuracy means the lexical word identity of the next token.

## Decision

- Router-only concept feedback: default.
- Token-aligned concept/object router state: default.
- Attention residual: off by default, explicit ablation only.
- Composition head: kept; neutral at 1,000 steps.
- Gains are small and single-seed; not a frontier claim.
