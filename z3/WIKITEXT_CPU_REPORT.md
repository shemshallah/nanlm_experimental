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

## 300-step timed smoke

Run: <https://github.com/shemshallah/nanlm_experimental/actions/runs/43804560939>

| Metric | Value |
|---|---:|
| Steps | 300 |
| Parameters | 34,852,448 |
| Train time | 528.19s (8m48.2s) |
| Row wall time | 532.26s (8m52.3s) |
| Step time | 1.7606s |
| Loss | 6.77685 |
| BPB | 2.26030 |
| BPC | 2.26441 |
| Subword/token top-1 | 15.10% |
| Subword/token top-5 | 30.78% |
| Byte top-1 | 88.04% |
| Byte top-5 | 95.69% |
| Word top-1 | 7.25% |
| Word top-5 | 18.73% |

At 1.7606 s/step, a 4-hour CPU window is about 8,179 steps and
2.09M tokens. With a 4.4 bytes/token corpus average, the raw slice floor is
about 9.2 MB; a 64 MB FineWeb-Edu slice provides roughly a 7x safety margin.

## Frontier gate and 4-hour FineWeb-Edu lane

The `concept_frontier` row adds learned route/pooled/token router mixing,
an exact 32,640-row hard pair-ID embedding, detached probe heads, and the
hardest-negative pointer margin.  It passed a 20-step WikiText-2 wiring gate:

| Metric | Value |
|---|---:|
| Steps | 20 |
| Parameters | 36,953,706 |
| Train time | 38.27s |
| Step time | 1.9137s |
| Non-finite skips | 0 |
| Loss | 10.60622 |
| BPB | 3.53753 |
| BPC | 3.54395 |

This is a wiring/timing gate, not a quality result.

The FineWeb-Edu lane downloads only the first 64 MiB of a public Kaggle shard
through an anonymous HTTP byte range, falls back to the public HuggingFace
sample parquet if needed, and prepares `uint16` bins on the runner.  The timed
run is capped at 8,000 steps or 14,400 seconds and uses the same 36.95M-param
row.  At 1.9137 s/step the 4-hour window is about 7,524 steps / 1.93M tokens.
After training, the checkpoint is evaluated zero-shot over **every** WT2 and
WT103 target token with exact target-byte denominators and window-level BPB
mean/std.

Run: <https://github.com/shemshallah/nanlm_experimental/actions/runs/43812135296>  
Commit: `59e34d1`

### Integrated `concept_frontier_all` row

The consolidated row now includes all nine requested upgrades:

| # | Upgrade | Integration |
|---:|---|---|
| 1 | Detached multi-resolution probes | probe hidden state is detached by default |
| 2 | Full-byte probes | per-token 48-byte sequence head and exact next-token byte targets |
| 3 | Cross-resolution distillation | token distribution → first-byte marginal and first-byte head → token marginal |
| 4 | Learned router-context gate | positive learned route/pooled/token gates, initialized to identity |
| 5 | Hard object-ID stream | exact 32,640-row pair-ID embedding selected from hard object IDs |
| 6 | Speculative MTP decode | main draft + MTP t+2 draft + verification forward |
| 7 | Full-validation uncertainty | every target token scored; window BPB mean/std |
| 8 | Hard negatives | hardest-negative margin on pair/triple pointer logits |
| 9 | Long-context path | 8k RoPE headroom, YaRN auto, activation checkpointing at ctx ≥ 1024 |

Short CPU gates on WikiText-2:

| Gate | Steps | Params | Step time | BPB | Full-byte top-1/top-5 | Cross-res loss | Non-finite |
|---|---:|---:|---:|---:|---:|---:|---:|
| ctx 256 | 20 | 39,313,002 | 2.464s | 3.55972 | 5.26% / 21.89% | 3.35195 | 0 |
| ctx 512 | 8 | 39,313,002 | 5.263s | 3.56735 | 1.22% / 4.48% | 3.59188 | 0 |

These are wiring/timing gates, not quality results. The 99-test suite and
five-row ablation matrix pass. Speculative decode is a correctness-preserving
inference path; it is not trained as a separate objective.

### Equal-wall context probe (3 minutes training)

The same `concept_frontier_all` row was trained for the same 180s wall budget
at three context lengths on the same WikiText-2 bins:

| ctx | Steps | Tokens | Step time | BPB | BPC | Top-1 | Top-5 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 296 | 37,888 | 0.610s | **2.28514** | **2.29392** | 19.29% | 31.10% |
| 256 | 103 | 26,368 | 1.763s | 2.44355 | 2.44799 | 6.27% | 25.10% |
| 512 | 47 | 24,064 | 3.919s | 2.74159 | 2.74403 | 1.47% | 11.15% |

This is an **equal-wall** result, not an equal-token or equal-update context
ablation. Context 128 processed 44% more tokens and 2.9x more optimizer steps
than 256 in the same three minutes, which explains most or all of its BPB
advantage. It does not prove that shorter context is intrinsically better.
The earlier 256-vs-512 comparison used 20 vs 8 steps and was not comparable.

The 512 path had no NaNs, no router discontinuity, and no connectivity failure.
Its deficit is consistent with fewer optimizer updates, fewer tokens, a larger
local-attention window, and a short single-seed run. A clean context ablation
must equalize tokens and updates, keep the local window/global frequency fixed,
and use multiple seeds.

A consolidated 4-hour FineWeb-Edu ctx-512 run is active:
<https://github.com/shemshallah/nanlm_experimental/actions/runs/43939945177>

### Completed 4-hour FineWeb-Edu run

| Metric | Value |
|---|---:|
| Steps completed | 8,000 |
| Wall-cap | 14,400s (not reached) |
| Train time | 14,163.25s (3h56m03s) |
| Step time | 1.7704s |
| Tokens seen | 2,048,000 |
| Parameters | 36,953,706 |
| Non-finite skips | 0 |
| Train tokens available | 13,579,553 |
| Validation tokens | 562,286 |
| Training bytes/token | 4.7225 |
| FineWeb valid loss | 6.51515 |
| FineWeb valid BPB | 1.78936 |
| FineWeb valid first-byte top-1 / top-5 | 85.29% / 94.12% |
| FineWeb valid word top-1 / top-5 | 16.67% / 29.22% |
| FineWeb valid subword top-1 / top-5 | 16.27% / 29.22% |

The FineWeb validation result is from the two fixed validation batches used by
the CPU row, so its exact denominator is small; it is not a full FineWeb
validation estimate.

Checkpoint artifact (weights remain in GitHub storage):
<https://github.com/shemshallah/nanlm_experimental/actions/runs/43812135296/artifacts/14214040692>

### Zero-shot WT2 and WT103, full coverage

This evaluation teacher-forces every target token, not 128 sampled positions.
BPB is exact for the scored target bytes; BPB uncertainty is the standard
deviation of per-window BPB.

| Corpus | Targets | Eval bytes | Eval chars | Bits/token | BPB | BPC | Top-1 | Top-5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| WikiText-2 test | 299,552 | 1,256,448 | 1,255,018 | 11.96834 | **2.85339** | **2.85664** | 7.26% | 14.48% |
| WikiText-103 test | 285,826 | 1,287,654 | 1,285,622 | 12.09453 | **2.68467** | **2.68892** | 7.42% | 15.95% |
| Combined | 585,378 | 2,544,102 | 2,540,640 | 12.02995 | **2.76800** | **2.77177** | 7.34% | 15.20% |

Window BPB was 2.88734 ± 0.36075 on WT2 and 2.71088 ± 0.32440 on WT103.
MTP t+2 top-1 was 3.36% on WT2 and 4.10% on WT103.  The full evaluation took
5,094.9s (84m55s).  The BPB denominator is the exact byte count of the scored
target tokens.  The BPC denominator is the full corpus character count
(2,540,640 characters); the full evaluation scores every token except the
first, so this differs from the scored-target character denominator by at most
that one token.  The corpus byte denominators match the raw files to within the
first-token offset.

Evaluation artifact:
<https://github.com/shemshallah/nanlm_experimental/actions/runs/43812135296/artifacts/14219723033>

BPC derivation run:
<https://github.com/shemshallah/nanlm_experimental/actions/runs/43918342570>

**This is not a frontier result.**  WT2 zero-shot BPB 2.853 is worse than the
old 115M byte reference (1.6969) and worse than the 1,000-step WikiText-2
2.05332 BPB.  The model is 36.95M parameters trained for four CPU hours on
FineWeb-Edu; the full-corpus zero-shot protocol and the FineWeb-only training
mix are also harder than the earlier fixed-window WikiText validation.

## Path toward 1.6 BPB

The current 36.95M checkpoint saw 2.05M tokens; the 5-minute probes see 24k–38k
tokens. That is far below the compute/data scale needed for 1.6 BPB:

- The 37M model has seen about 0.055 tokens/parameter. A Chinchilla-style
  budget is roughly 20 tokens/parameter, or ~740M tokens.
- The current 4-hour CPU run is ~370x short of that token budget.
- The old 115M byte reference reached 1.6969 BPB; matching or beating it needs
  a model in that class plus orders of magnitude more tokens, not another
  diagnostic head.

Priority order:

1. Train on 0.5–2B tokens with a GPU, using gradient accumulation for an
   effective batch of 8–32.
2. Add a prose/Wikipedia/books mix for the WT2/WT103 target distribution;
   FineWeb-Edu alone is out of domain.
3. Keep the token-aligned router and hard object IDs, but treat full-byte,
   cross-resolution, and speculative paths as diagnostics/inference unless a
   matched-token ablation shows a main-loss gain.
4. Use a 128/256-token curriculum first, then extend to 512/1024 with the local
   window and global-layer frequency held fixed.
5. Report full WT2/WT103 BPB and BPC, fixed-window BPB, and multi-seed spread;
   do not compare token CE to byte BPB.

## Decision

- Router-only concept feedback: default.
- Token-aligned concept/object router state: default.
- Attention residual: off by default, explicit ablation only.
- Composition head: kept; neutral at 1,000 steps.
- Probe heads are detached by default so diagnostics cannot perturb the trunk.
- Full validation is teacher-forced over every target token; `bits_per_byte`
  uses the bytes of those scored targets, not the corpus-average ratio.
- The 4-hour FineWeb checkpoint reached 1.78936 valid BPB but only 2.85339 /
  2.68467 zero-shot BPB on WT2 / WT103. It does not beat the old byte
  reference or the 1,000-step WT2 result.
- 96 unit/smoke tests and the 5-row ablation matrix pass.
- Gains are small and single-seed; not a frontier claim.
