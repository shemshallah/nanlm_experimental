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
