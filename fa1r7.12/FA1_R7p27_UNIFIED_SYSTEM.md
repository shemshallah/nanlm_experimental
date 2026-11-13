# FA1 R7.27 — unified model, optimizer, checkpoint and CPU test suite

**Branch:** existing `master`; **architecture:** existing PAL R7 only. FA2,
NaNLMind and Poincaré are separate lineages, not silently combined into R7.

**Accepted checkpoint:** historical R7 step **14,416** (`pal_active.pt`, SHA-256
`54e835b00642013d4f1bf486a8150eb1ff06c547565771bd21933e143d050f2d`).
**Starting experimental state:** R7.24 384-update geometric candidate SHA-256
`5d07b1ef66aa9f870d05c6fda257b1dcdf36ecd4623999dff4a347cf82e4ed65`.
Neither input is overwritten; R7.27 is **not promoted**.

## One learning system

R7.27 registers the R7.7 original word/tree prediction trunk, signed 256-point
S+/S− mirror memory, directed recurrence, near and far historical retrieval,
semantic and long-semantic readers, byte/word/sentence Diamond towers, geometric
triangle/tetrahedron operators, meta-specialist attention, and downward
pair/point feedback under the same R7 `nn.Module` (`UnifiedR723Diamond` with
`GeometricDiamond`). The original point/edge tensors are retained as learned
weights; geometry adds residuals and does not replace them.

| Sparse geometric component | Global addresses | Active per prediction |
| --- | ---: | ---: |
| Original points | 256 | 8 |
| Signed mirror pairs | 32,640 | 28 |
| Oriented triple addresses | 2,763,520 | 56 |
| Tetrahedron addresses | 174,792,640 | 70 |

These are *address spaces*, not 174M stored tetrahedron embedding vectors.
The semantic long reader can address up to 1M prior tokens; that does not
establish trained million-token reasoning.

### Optimizer migration

The inherited optimizer inventory is not five AdamW engines:

1. PAL6 `LegacyAccumulator`, an exact AdaGrad-style accumulated-square update.
2. Four AdamW engines, including the multi-group memory/semantic/Diamond path.

`source/fa1_r727_optimizer.py` provides **one** public `.step()`, `.zero_grad()`,
`.state_dict()` and `.load_state_dict()` and 13 parameter groups. The original
legacy math stays unchanged; all AdamW groups are merged into one native AdamW
backend with the same hyperparameters and per-parameter moments/steps. The
public checkpoint records **one optimizer state**. A direct real-weight
13-group one-step test found identical parameters and moments on both sides.

Legacy R7.24 candidates can omit five newer geometry tensors. Recovery now
reconstructs those *missing, untrained* tensors with a fixed seed (`724027`)
instead of the ambient random state. The stored weights and moments never
change during this migration.

`source/fa1_r727_checkpoint.py` writes a separate unpromoted R7.27 checkpoint
with a full model state, unified optimizer state, source digest, and three
vocabulary/tree metadata hashes. Its loader is independent of the original
R7.24 candidate file (the R7.13 corpus vocabulary/tree is still required).

## One test suite

Run the source-only suite in CI:

```bash
python fa1r7.12/source/fa1_r727_suite.py --output r727_source_suite.json
```

It checks seven layers: R7.22 semantic memory, R7.23 topology, R7.24 signed
geometry, R7.25 foreign-book controls, R7.26 read-only factor masks, R7.27
optimizer update/round-trip parity, and R7.27 word-tree probability/corpus
integrity. GitHub Actions: `.github/workflows/fa1-r727-unified-suite.yml`.

For actual checkpoint and NaNROUTER-book evaluation, first use the existing
`prepare_document_corpus.py` to build SHA-separated whole-book data. Then run:

```bash
python fa1r7.12/source/fa1_r727_suite.py \
  --root R713_FULL_ROOT \
  --candidate FA1_R7p24_GEOMETRIC_UNPROMOTED.pt \
  --corpus-dir PREPARED_NANROUTER_DIR \
  --targets 512 --seed 727 \
  --output r727_unified_suite_dev512.json
```

For a new **unpromoted** checkpoint containing one optimizer state:

```bash
python fa1r7.12/source/fa1_r727_migrate.py \
  --root R713_FULL_ROOT \
  --candidate FA1_R7p24_GEOMETRIC_UNPROMOTED.pt \
  --output FA1_R7p27_UNIFIED_UNPROMOTED.pt
```

The migration verifies round-trip preservation of every model tensor and
optimizer moment; it refuses to overwrite the source candidate or accepted
checkpoint. It performs **zero training updates**.

## CPU measurements (NaNROUTER batch 0004)

The first reproducible benchmark used the first 240 book files from batch
`short_stories-0004.tar.zst` (SHA-256
`d6ce3a23757e85fe1b9487ef984a5ba81db9c2dc6e962ac7c5df2960480e0054`).
The deterministic whole-book split contained 216 training, 12 development and
12 test books. Only development was evaluated; five long enough books supplied
512 eligible positions, selected with seed 727. No training was performed.

| Metric | Development result |
| --- | ---: |
| Full model vocabulary-target NLL | 5.098015 nats/word |
| Trunk-only vocabulary-target NLL | 5.321434 nats/word |
| Trunk word top-1 (normalized 32K tree) | 16.60% |
| Trunk word top-5 (normalized 32K tree) | 37.70% |
| First byte of next word top-1 | 15.63% |
| First byte of next word top-5 | 39.45% |
| First-byte auxiliary CE | 3.386091 nats/next word |
| First-byte auxiliary bits/next word | 4.885097 bits/next word |
| Diamond correct vs foreign-book history | +0.000056 nats/word |
| OOV vocabulary targets | 13/512 |

The first-byte head supplies a normalized distribution over 256 values, and
the inherited trunk supplies a normalized 32K-word tree distribution. The
complete near/far/Diamond mixture currently provides target-conditioned
vocabulary NLL but **not** a full-vocabulary ranking. Its top-1/top-5 cannot
be honestly reported yet. The auxiliary head does **not** predict every byte
in the raw stream: raw-stream bits/byte and next-byte top-5 are not measured.

`first_byte_target` now uses the original corpus spelling for unknown tokens
instead of incorrectly treating every OOV token as literal `<unk>`.

### Promotion and next tasks

- Correct-book history has no convincing advantage at 512 positions.
- The NaNROUTER 90/5/5 splits are disjoint by complete book SHA-256, but
  overlap with the **original** R7 ancestor's training corpus is not fully
  reconstructed; this is an external development diagnostic, not a sealed
  independent-ancestor test.
- Preserve all 13 optimizer groups. Add a token-independent normalized output
  head for the *complete* unified mixture, and a truly autoregressive byte
  decoder before claiming full-model token top-k or raw BPB.
- Require larger held-out book clusters, correct-versus-foreign history gains,
  legacy prediction regression and useful geometry/apex ablations before
  promoting any checkpoint. The accepted step-14,416 state remains frozen.
