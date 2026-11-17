# FA1 R7.54 — CPU successor pilot and strict causal DEV gates

Status: **source-integrated, unpromoted; checkpoint-backed training pending the exact recovery bundle.** Code lives on `master` inside `fa1r7.12`. FA2, M3, and Terminal-RSI are **not** merged into this FA1 experiment.

## What changed

R7.53 had the additive causal successor module but no actual checkpoint-backed training/evaluation caller. R7.54 now executes a real end-to-end path: book-balanced TRAIN and DEV positions → separate 512-word successor history → R7.49 bitwise zero-gain parity → four-tensor successor training → matched DEV ablations → whole-book bootstrap → 2048-word long-context gate → immutable unpromoted candidate save and strict reload.

The 29 inherited groups and their moments are frozen; only one appended AdamW group (four tensors) receives gradients. A SHA-256 of inherited parameter values and optimizer slots is checked before migration, after training, and after a fresh checkpoint reload. The candidate is **not promoted** regardless of DEV outcome. The independent TEST split is never read.

The no-apex ablation was also corrected to suppress only the 32-dimensional successor reliability input. It no longer replaces all successor confidence with a constant, which would confound apex attribution and support-count confidence.

## Exact parent contract

Protected R7.49 SHA-256:

`b8dabe54e2417a75defd01c776606367a3f3b0f18f479b032f8390b6fc87ddb5`

Accepted underlying FA1 step: **14,416**. R7.49 is an unpromoted research continuation, and the vanished R7.51/R7.52 transient checkpoints are **not** reconstructed or claimed intact. R7.54 extends R7.49 directly, preserves all 29 inherited optimizer groups, and appends group 30. One model and one optimizer.

The workflow cannot legitimately run the saved weights until the original checkpoint and the compatible R7 tokenizer/root and book-disjoint TRAIN/DEV artifacts are accessible to the runner. It **must not** fall back to random initial weights, a different R7 round, synthetic DEV data, or WikiText-103.

## CPU schedule

Default CPU thread settings: 1, 2, or 4 (4 by default).

1. Verify exact parent SHA-256, optimizer topology, 270 TRAIN books and 15 disjoint DEV books.
2. Sample 2,048 TRAIN positions across at least 200 independent books and 512 DEV positions across all 15 books. Only sample within-book causal targets.
3. Verify exact zero-gain word NLL parity with R7.49.
4. Run **256 lexical successor-only updates**, with the geometry gate closed.
5. Run **744 joint lexical/geometric successor updates**, opening the geometric gate after the lexical stage. Same parent and optimizer, no independent checkpoints.
6. At matched DEV positions, measure full, disabled, word-only, wrong address, wrong successor, foreign-book history, and isolated no-apex conditions. Bootstrap by **book**, not word.
7. Re-evaluate selected candidate at 2,048-word successor history, verify frozen weights/moments and exact saved/reloaded optimizer state; export a candidate and JSON metrics.

The optional additional **9,000 updates** are deliberately not launched without a successful 1,000-update measured gate and a separately recoverable candidate carrying the full 30-group optimizer state.

## Strict decision criterion

The lexical stage must improve parent R7.49 NLL and beat unrelated-book successor history with **positive 95% lower whole-book bootstrap confidence limits**.

Geometric attribution additionally requires positive 95% lower whole-book limits for:

- Full candidate versus R7.49.
- Full candidate versus word-only.
- Correct geometric addresses versus wrong addresses.
- Correct **geometric** successor values versus scrambled geometric values, with the lexical distribution unchanged.
- The geometric increment from correct-book history versus unrelated-book history, with the word-only increment subtracted.
- Conditioned apex trust versus the isolated no-apex successor gate.

The independent 2,048-word check must avoid NLL regression and retain a positive correct-book advantage. These are deliberately difficult criteria: mere activation of the geometric head or an improvement against an arbitrary baseline is insufficient.

## Metric honesty

The native R7.54 measurement is **next-word** NLL and loss attribution on identical target positions, plus support fractions and mixture weights. The inherited source does not provide exact full-mixture vocabulary top-5 or a genuine contiguous-byte autoregressive probability. Therefore it reports `full_mixture_word_top5_available=false` and `true_contiguous_byte_bpb_available=false`; neither number is invented.

Raw-byte next/+2/+3 top-1/top-5/BPB benchmarking requires a separately implemented and tested byte-level scoring head and conversion contract. Existing conditional spelling losses and inherited trunk word top-k must not be mislabeled as full R7.53 byte or vocabulary mixture scores.

## Recovery-bundle contract for GitHub Actions

Upload one recoverable release asset `FA1_R749_CPU_RECOVERY.zip` to a release tag (default `fa1-r749-cpu-recovery`). It must contain exactly these **relative paths** (additional files permitted):

```text
r749.pt
root/corpus/vocab.words.txt
root/corpus/token_heads.u16
root/corpus/tree.i32
root/checkpoint/pal_active.pt
processed/train_documents.json
processed/dev_documents.json
processed/train_temporal.i32
processed/dev_temporal.i32
processed/train_longctx2048.i8
processed/dev_longctx2048.i8
processed/train_char_offsets.u32
processed/dev_char_offsets.u32
processed/train_char_blob.bin
processed/dev_char_blob.bin
processed/train_doc_start.u32
processed/dev_doc_start.u32
metadata/train_documents.json
metadata/dev_documents.json
```

The R7.49 checkpoint must match the protected SHA-256 above. The processed corpus must be created with the original R7 tokenizer and the archived 270/15 independent-book split; the runner verifies the matching manifest. TEST files are neither required nor accessed.

Source-only tests run on `master` pushes, independent of the bundle. The manual dispatch workflow fails closed if the bundle is missing or inconsistent.

### Entry point

```bash
python fa1r7.12/source/fa1_r754_cpu_pilot.py \
  --root ROOT --corpus-dir PROCESSED --split-metadata METADATA \
  --checkpoint r749.pt --output-checkpoint R753_UNPROMOTED.pt \
  --report R754_DEV.json --threads 4 \
  --word-steps 256 --geo-steps 744 \
  --context 512 --long-context 2048
```

## Current claim ledger

- R7.53 source CI: **passed**.
- R7.54 source CI: see current GitHub Actions result.
- Checkpoint-backed R7.54 1,000-update CPU measurement: **not yet run**.
- Actual 30-group Adam moment preservation after a real training/reload: **not yet measured**, enforced in code.
- DEV NLL improvement and causal geometric contribution: **not yet established**.
- Independent TEST: **sealed**.
- Checkpoint promotion: **none**.
