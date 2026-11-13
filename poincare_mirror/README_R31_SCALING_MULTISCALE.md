# R31 — 192D → 256D → 384D → 512D causal scaling (single checkpoint)

This change extends the canonical Poincaré Mirror `poincare_mirror/poincare_frontier.py`
on `master`. It does **not** start a new neural network, merge unrelated
FA1/FA2 optimizer histories, or promote candidate weights.

## Observed external-domain starting point

The previously recorded R30 baseline used the **192D, step-2,801** checkpoint,
2,048 SHA-256-verified WikiText-103 test target positions and 256-byte context.
These are inherited observations, **not results from the new architecture**:

| Target | Top-1 | Top-5 | CE (nats/byte) | Bits/byte | Per-byte perplexity |
| --- | ---: | ---: | ---: | ---: | ---: |
| Next byte | 32.42% | 64.70% | 2.72094 | 3.92549 | 15.19 |
| +2 bytes | 21.63% | 50.44% | 3.18607 | 4.59652 | 24.19 |
| +3 bytes | 18.16% | 47.02% | 3.36580 | 4.85582 | 28.96 |

Recorded two-thread CPU inference throughput: ~3,883 target bytes/s. The
fixed R30 checkpoint identity is SHA-256
`9ad6a368a9e68f4c1ccff366bee88047f7475f93ccb4d1da1c999db7bddee902`,
255,749,067 bytes, 63 Adam moment-bearing parameter states. The external
test does not prove book-disjoint generalization against all historical data.

## Capacity migration

- The original 16 heads retain **192 coordinates, 12 coordinates per head**,
  including exactly their trained RoPE frequencies.
- Four **new 16-dimensional heads** are appended, yielding **256 total
  coordinates and 20 physical attention heads**. The original heads are not
  secretly reshaped into 16D heads.
- Old Q/K/V, mirrored experts, tied byte head, future objectives, and the old
  RMSNorm divisor are embedded into the expanded coordinate system.
- AdamW first/second moments are migrated by parameter **name and coordinate**;
  the saved step, dataset identity and RNG streams remain unchanged.
- The 256D candidate remains one model with the original 257-way byte/EOS
  vocabulary. The old checkpoint is protected separately.

The mixed-head family can be extended again: 256D/20 heads, 384D/28 heads,
and 512D/36 heads all keep the original 16 × 12D attention heads exactly
intact. The newly added 16D head group grows by whole heads; learned new
heads are not changed during subsequent widening. Appending identity-initialized
transformer blocks is orthogonal to widening. The 512D/12-block variant has
134,670,556 parameters, versus 21,304,992 in the accepted 192D/6-block R30.

The 512D migration from the *real* R30 checkpoint was executed in two stages,
192D → 256D/6 blocks → 384D/8 blocks → 512D/12 blocks. After the last stage:

- Checkpoint step and token counter stayed at 2,801 and 1,128,496.
- All 63 inherited Adam states and 126 first/second-moment tensor fields were
  verified through each capacity migration.
- The maximum three-horizon logit drift across the final stage was below
  0.000002. Its new weights are untrained and **not promoted**.

At 512D/12 blocks, weights + gradients + FP32 Adam moments alone require
about 2.01 GiB; attention activations and transient allocations add to this.
The 9.7 GiB CPU host should use short training windows and batch 1. There is
no theoretical cap at 12 blocks, but increasing beyond the CPU budget without
held-out improvements is not justified.

## Hierarchical causal feedback

`--hierarchy-rank 32` enables a trainable rank-32 residual with four scales:

1. Byte representation from the inherited decoder stream.
2. A causal **two-byte pair proxy**.
3. Prefix-pooled representations since the last word delimiter.
4. Prefix-pooled representations since the last sentence delimiter.

Two shared cross-scale attention passes combine these four views **at each
current position** and feed the result back to the same decoder stream.
Temporal causality is still provided by the inherited self-attention backbone;
prefix means cannot access future characters. Its output projection starts at
exactly zero, so migration leaves inherited logits unchanged up to float
rounding, then new features become trainable during Adam updates.

**BPE means Byte Pair Encoding.** The pair scale here is a causal byte-pair
proxy, **not yet a fitted BPE tokenizer**. No new BPE vocabulary, BPE top-5
benchmark, or word/sentence classification accuracy should be claimed from
this implementation. Training a merge table exclusively on training books,
adding true BPE boundary annotations and multi-token targets, and measuring
compression are separate future tasks.

The symbiotic sparse 256-point/pair/triangle/tetra/apex path remains in the
same MoE blocks and can be enabled simultaneously with this hierarchy. An
`evaluate --ablations` run now reports `no_token_hierarchy` alongside the
existing `no_symbiotic_geometry` and signed-expert controls.

## Repeatable commands

First verify the canonical R30 checkpoint with `scripts/verify_lineage.py`.
With its exact file at `run/latest.pt`:

```bash
python poincare_frontier.py grow \
  --run-dir run --widen-dim 256 --hierarchy-rank 32 \
  --symbiotic-rank 32 --widen-noise 1e-5 \
  --max-logit-drift 5e-4 --threads 2 \
  --report reports/256d_migration.json
```

The `grow` action is atomic and will reject an old checkpoint whose logits
or inherited optimizer moments are not preserved. Do not run this command
against the sole copy of your canonical checkpoint: keep the original Release
immutable and perform growth on a candidate working copy.

To reproduce or compare the external raw-byte probe at **exactly the same
target positions** (after verifying the test file's SHA-256):

```bash
python scripts/eval_external_bytes.py \
  --checkpoint run/latest.pt --raw-test /path/to/wikitext103-test.raw \
  --expected-test-sha256 YOUR_VERIFIED_TEST_SHA256 \
  --context 256 --positions 2048 --batch 8 --threads 2 --seed 2026 \
  --output reports/external256.json
```

The report includes +1/+2/+3 cross-entropy, BPB, per-byte perplexity,
top-1/top-5 and measured inference throughput. It reports file hashes and
fails closed when an expected hash does not match. Corpus rendering, sampling
seed and target positions must agree before comparing its numbers with any
earlier benchmark. The inherited WikiText baseline did not publish all these
inputs here, so **numerical comparability has not yet been established**.

`evaluate` on held-out book splits additionally emits all three top-5
accuracies and BPB/perplexity at each horizon. The existing independent-book
and aligned 256-vs-4,096 context gates still determine eligibility for manual
promotion. The baseline aligned long-context penalty was +0.312404 BPB; do
not substitute unrelated short/long target windows.

The GitHub Actions workflow accepts `widen_dim=256` and
`hierarchy_rank=32`, and now defaults to `widen_dim=512` and `add_layers=6`
for the requested deeper candidate. It still requires the exact canonical R30
checkpoint to be available as a Release asset; source CI alone does **not**
restore or train that model. The new 256D architecture has not yet earned
a real held-out perplexity/accuracy improvement claim.
