# MTP and tokenizer smoke results (2026-11-13)

All runs are small diagnostics. No 6-hour or frontier-scale training was
started after these results.

## Rollback

- Git tag: `pre-mtp-tokenizer-fix`
- Archive: `~/meta/backups/z3_pre_mtp_tokenizer_fix.tar.gz`
- The tag points at the last code state before the tokenizer v4 and MTP
  adapter experiments.

## Tokenizer v4 smoke — pass

Script: `z3/train_tokenizer_v4.py`

```text
train bytes      20,000,408
eval bytes        2,006,676
eval tokens         421,432
bytes/token         4.7616
vocab               64,000
specials            <pad>,<bos>,<eos>,<doc>,<now>,<future_1..3>
```

This clears the ≥3.8 bytes/token gate for the smoke sample. It is trained on
FineWeb-Edu only; the frontier tokenizer still needs the mixed corpus.

Output: `~/meta/kaggle_tokenizer_v4_out/tokenizer_v4/tokenizer_v4_64k.json`

## MTP experiments — all failed to beat the original 1.5484

Original FineWeb-6h best MTP loss: **1.54843**. Main CE: **0.98063**.

### Frozen trunk, MTP head only (500 steps)

```text
CE   0.98063 -> 0.98063
MTP  1.54843 -> 1.54856
```

### Frozen trunk + low-rank MTP adapter, scale 1e-3 (100 steps)

```text
MTP  1.54843 -> 1.54856
```

The scale stayed too small to move the adapter.

### Frozen trunk + adapter, scale 0.1 (500 steps)

```text
MTP  1.54880 -> 1.54856
CE   0.98063 -> 0.98063
```

### Last block unfrozen + adapter (500 steps)

```text
MTP  1.54880 -> 1.54966
CE   0.98063 -> 0.98155
```

The last-block path did not help MTP and slightly hurt CE.

## Interpretation

The frozen trunk does not contain a linearly accessible t+2 signal for the
current byte-level objective. Thawing the last block alone is insufficient.
The next MTP test should change the **branch and objective**, not just the head:

1. MTP-specific branch with its own decoder block/cross-attention.
2. Now-slot / copy gate using a real `<now>` embedding.
3. Higher MTP weight or a two-pass future-query token.
4. BPE/subword MTP; byte-level t+2 is only two characters.

The tokenizer v4 64k smoke is the first part that passed. Do not start a full
cycle until the MTP branch passes a 500-step gate.

## t+1-conditioned MTP block (2026-11-13)

Implemented a one-block MTP branch with its own attention and 8-expert router,
t+1 conditioning, `<now>` vector, and a concat projection.

50-step diagnostics (trunk frozen, LR 5e-5):

| Variant | Eval MTP | Correct-vs-wrong t+1 delta |
|---|---:|---:|
| teacher t+1 = 1.0 | 1.5701 | **-0.1792** |
| teacher t+1 = 0.5 | 1.5588 | -0.0299 |
| teacher t+1 = 0.0 | 1.5500 | 0.0000 |

The branch genuinely uses t+1 when it is given the **ground-truth** next
token: MTP drops to 1.3802 in the sensitivity probe and correct-vs-random t+1
delta is -0.179. But with the **predicted** t+1 used at eval, held-out MTP is
1.55–1.57, not below the 1.548 baseline and not near the 1.40 target.

This is the key remaining issue: MTP is conditioned on a t+1 guess whose top-1
is only about 70%, and the branch was not trained to be robust to that guess.
Simply training longer with ground-truth t+1 will not fix it.

No 200/1000-step follow-up was run because the gate was not met.

## Soft t+1 + confidence gate (100 steps)

The branch now receives a predicted t+1 distribution mixed with ground truth
(1.0 -> 0.7 over 100 steps), uses a concat projection, and blends the
t+1-conditioned logits with the old head through a confidence gate supervised
by t+1 correctness.

```text
                         MTP loss   correct-t1   wrong-t1   acceptance
baseline (old head)       1.5484      1.0769      2.6682      67.03%
soft+gate best (100)      1.5543      1.0673      2.7124      67.03%
soft+gate final           1.5682      1.0579      2.7813      66.85%
```

Teacher-forced sensitivity: correct 1.3779, random 1.5556, delta -0.1777.

Interpretation: the branch still uses t+1 when it is correct, but it does not
beat the old head on predicted t+1, and the gate did not lower the wrong-t1
cost. Acceptance is already 67% for the old head; the branch does not improve
it. Do not start the 5,000-step joint run.

## Frontier MTP branch, per-lag + soft + supervised gate (100 steps)

Per-lag MTP blocks now use QK-RMSNorm, RoPE, causal SDPA, and a separate
8-expert MoE.  Conditioning is the predicted t+1 distribution mixed with
ground truth (1.0 -> 0.7), and the confidence gate is BCE-supervised against
t+1 correctness.

```text
                         MTP loss   correct-t1   wrong-t1   acceptance
baseline old head         1.5484      1.0769      2.6682      67.02%
frontier best (100)       1.5526      1.0688      2.7023      67.10%
frontier final            1.5660      1.0594      2.7698      66.81%
```

Teacher-forced sensitivity after training: correct 1.3903, random 1.5445,
delta -0.1542.

The branch is now genuinely conditioned and the gate is supervised, but
predicted-t+1 MTP still does not beat the old head and the gate does not lower
the wrong-t+1 cost enough. No 200-step or 5,000-step run was started.
