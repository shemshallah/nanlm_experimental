# R32: causal long-context geometry in the existing Poincaré Mirror network

This is a **source-level candidate** on `master`, not a newly trained or
promoted checkpoint. R30 step 2,801 (63 original Adam states) remains the
accepted reference. The 512D/12-block step-2,865 pilot is **not promoted**:
next-byte BPB and top-1 worsened despite some top-5 and auxiliary gains.

## What changes

- `--set-context 32768` expands the declared byte context. The byte/EOS
  tokenizer, expert weights, signed S+/S- mirrors, and future heads are intact.
- `--attention-window 4096` retains **exactly the original dense attention
  computation for input sequences of up to 4,096 bytes**. Longer inputs use
  query-tiling and causal sliding attention, limiting the attention-score
  workspace instead of allocating a full 32K-by-32K score matrix.
- `--geometry-memory-chunk 128` gives each existing geometric MoE block
  **two shared apex-memory self-attention loops**. Queries at byte position
  *t* may read only fully completed earlier 128-byte chunks. The keys and
  values are pooled from that block's *own* point → signed pair → triangle →
  tetrahedron → apex trajectory; the result returns down through the same
  triangle/pair/point feedback path and shared language decoder.
- New temporal memory has a scalar **zero-initialized residual gate**. At
  migration, new attention is invisible and old predictions and optimizer
  moments must remain intact; the gate and memory projections can then learn.
- Existing `--hierarchy-rank` byte/pair-proxy/word/sentence two-loop feedback
  stays enabled when present. No separate tokenizer or model is introduced.

The new geometric bank stores *computed hidden-state summaries*, not a full
`C(256,3)` dense table. Each active token still selects eight of the 256
points (28 edges, 56 triangles, 70 tetrahedra). The apex concept keys remain
256-dimensional in count. Address geometry and memory are learned through the
same loss/Adam optimizer, with no new checkpoint lineage.

## Suggested controlled migration (do not alter the accepted release)

From a **working copy** of the unpromoted 512D candidate (if independently
recoverable), enable the temporal path and raise max context:

```bash
python poincare_mirror/poincare_frontier.py grow \
  --run-dir poincare_mirror/run_candidate \
  --set-context 32768 --attention-window 4096 \
  --geometry-memory-chunk 128 --threads 2 \
  --report poincare_mirror/reports/r32_migration.json
```

The geometry chunk flag requires pre-existing `symbiotic_rank` in that
checkpoint. When migrating **directly from R30** without geometry, include
`--symbiotic-rank 32` in the same `grow` command. If the inherited maximum
context exceeds 4,096, choose `--attention-window` at least that large to
protect old-context logits. A previously trained nonzero window cannot be
changed by this safe migration tool.

For 9 GiB RAM, do **not** start by backpropagating through 32K × 512D × 12
blocks. Begin with batch 1, 256–2,048-byte training windows and 4,096-byte
validation, then test 8K and 32K evaluation with controlled memory use:

```bash
python poincare_mirror/poincare_frontier.py train \
  --data-dir poincare_mirror/data --run-dir poincare_mirror/run_candidate \
  --steps 64 --batch 1 --grad-accum 1 --cpu --cpu-threads 2 \
  --context-mix '256:4,512:4,1024:2,2048:1' \
  --eval-context 4096 --eval-batches 1 --save-every 16
```

The training example requires verified document-safe NaNROUTER corpus files
and an already-migrated candidate checkpoint. This is a **proposed pilot**;
the results must be measured before calling it an improvement.

## Aligned context ladder and gate

The `scripts/eval_context_ladder.py` command measures identical target bytes
with 256/4,096/8,192/32,768 history (all horizons +1/+2/+3):

```bash
python poincare_mirror/scripts/eval_context_ladder.py \
  --checkpoint poincare_mirror/run_candidate/latest.pt \
  --raw-test /path/to/independent-test-book.txt \
  --expected-test-sha256 VERIFIED_HEX_SHA256 \
  --contexts 256,4096,8192,32768 --positions 16 --batch 1 \
  --ablate-geometric-memory \
  --output poincare_mirror/reports/r32_aligned_ladder.json
```

Compare the original R30 and R32 **on the same target-position hash**, not
unmatched short/long samples. Report top-1, top-5, cross-entropy, bits/byte,
per-byte perplexity and throughput at each prediction horizon. Run the
same-weight ablation with the new memory gate set to zero; further controls
must include correct document versus shuffled/unrelated histories. Promotion
requires lower independent-book next-byte BPB, no meaningful +2/+3 or 256-byte
regressions, and causal/optimizer/regression checks. A synthetic source test
does not establish language-model quality.

Tests: `python -m unittest discover -s poincare_mirror/tests -q`, including
`test_r32_long_geometric_attention.py`. The new suite checks bounded-window
attention against a dense explicit mask, future-byte/gradient isolation,
nontrivial memory feedback, and checkpoint/Adam migration.
