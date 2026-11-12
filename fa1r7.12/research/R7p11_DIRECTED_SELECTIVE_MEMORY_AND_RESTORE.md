# PAL R7.11 — selective directed rank-64 retrieval (experimental, not promoted)

**Date:** 2026-11-12.  **Accepted model:** R7.7, 14,416 joint LM updates, SHA-256
`54e835b00642013d4f1bf486a8150eb1ff06c547565771bd21933e143d050f2d`.
**Experimental auxiliary updates attempted:** 320; **development-selected** auxiliary update: 80.
The 14,416 joint-LM step counter is unchanged: an auxiliary recall update is **not** a
joint language-model update. The full accepted checkpoint and five inherited optimizer
histories remain bitwise unchanged. The experiment is recoverable from a tiny overlay
on that parent, rather than a second independently evolving full checkpoint.

## Mechanism and CPU learning protocol

`source/r711_selective_writer.py` adds a directed adjacent-token association
writer inside the same rank-64 physical memory matrix. It reads the latest 768
of the *older* 6,144 historical tokens, which themselves end 2,048 tokens
before the NOW prediction. It forms causal directed pairs `(token[t], token[t+1])`,
normalizes inherited 96-D embeddings through newly trainable 64-D projections,
centers source and destination vectors to reduce unigram frequency bias, weights
pairs by recency and a learned selector, and accumulates one weighted
64×64 value⊗key matrix. A tied query/value readout scores 32 candidate
historical successor words. Reversing word order within 64-token blocks changes
the pair matrix and breaks directed associations; old averaged block initialization
provably erased that order.

The parameter migration is exactly zero-gated: when `dir_gate = 0` the inherited
R7.7 full-history matrix, NOW prediction, optimizer states, and weights are
unchanged to bitwise precision. No new independent memory state is persisted.
The new reader was optimized *only* for historical associations, **not** jointly
for next-word prediction. The nonzero-gate carry-forward update is not yet
consistent with exact sliding-window reinitialization, so **do not promote it**
into streaming LM inference.

Examples: original `source/r710_hard_negative.py` unique-key selection in older
history, positive real next-word and 31 competing historical successors ranked
by frozen accepted-model embedding similarity. All positives and negatives
preexist NOW; NONE uses NOW or future targets. Selection was made before
training for 512 train positions; dev/test holdouts were never backpropagated.
Two CPU threads, batch size 24, 320 auxiliary updates with the original five
optimizer objects, only adding the new parameter group to optimizer 4.
Legacy parameters are frozen during this targeted phase, while **all old Adam
moments are preserved**. The best development state AND its matched optimizer
moments were restored from update 80, not the worse step-320 continuation.

## Primary matched historical evaluation (512 queries per split, 32-way)

| Metric | Dev, untrained pair reader | Dev, update 80 | Test, update 80 |
|:--|--:|--:|--:|
| Correct-memory CE ↓ | 4.078073 | **3.305284** | **3.348275** |
| Unrelated-memory CE ↓ | 4.329570 | 3.892234 | 3.902403 |
| Within-64 reversal CE ↓ | 5.367093 | 4.754706 | 4.515306 |
| Correct-memory top-1 ↑ | 8.20% | **20.31%** | **16.80%** |
| Unrelated−correct CE advantage ↑ | 0.251497 | **0.586950** | **0.554128** |

Uniform 32-way chance: CE ln(32)=3.465736, top-1 3.125%. These numbers
measure **experimental directed retrieval**, not overall language NLL.
Original R7.10 readout scores were much worse, but those are a different
readout and must NOT be interpreted as a direct controlled training gain.

### Alternate precommitted query-selection seeds (512 per split)

| Metric | Development before → after | Historical test before → after |
|:--|--:|--:|
| Directed read CE ↓ | 4.25905 → **3.33101** | 4.47141 → **3.41256** |
| Top-1 ↑ | 7.03% → **17.97%** | 9.38% → **17.77%** |
| Unrelated−correct CE ↑ | 0.22527 → **0.52542** | 0.21111 → **0.62116** |

The alternate query seeds reduce dependence on one chosen query set but reuse
the SAME historical concatenated development/test corpora. They are not new
books. The packaged old corpora lack `*_doc_start.u32`: reliable book IDs,
author independence and **document-bootstrap intervals cannot be computed**.
Any suggested book-cluster confidence would be invalid. Report sampling overlap
and corpus contamination as unresolved.

## Language-model gate decision: no promotion

The full LM was evaluated at the same candidate adapter weights with the
zero-gated old memory and experimental matrix blending. Fixed historical
128-position DEV NLL:

| Pair blend gate | Dev whole-word NLL (lower better) |
|--:|--:|
| 0 | **5.5636130** |
| 0.05 | 5.5636248 |
| 0.20 | 5.5636602 |
| 0.50 | 5.5637320 |
| 0.90 | 5.5638284 |

All nonzero blends in this sweep made NLL slightly worse; it selected the
zero blend, leaving the accepted next-word predictor unchanged. This
128-position sample is different from the prior 4,096-position evaluation;
**do not compare absolute NLL across samples**. The historical test was not
consulted to choose the gate. Given zero gate, the model has **not yet**
demonstrated joint LM improvements or better generation.

## Checkpoint integrity / reproducibility

- There is exactly ONE full checkpoint: `checkpoint/pal_active.pt`, still at
  step 14,416. It retains all five original optimizer histories and RNG.
- `adaptation/r711_experimental_delta.pt` is a **173,477-byte experimental
  overlay**, not a second full model. It contains only the five new
  tensors, four new optimizer-moment entries, the selected auxiliary
  RNG state, provenance and versioned metadata. Its SHA-256 is
  `a7ff6f371f7a08474d80f74195c0a081e96ea71c183aa3d6f3b16c8a8898402c`.
- Applying this overlay to the verified accepted source recovers exactly
  **all 79 tensors and five optimizer state dictionaries** of the full
  selected experimental candidate; checksums and original-source identity
  fail closed. The source checkpoint and optimizer state were bitwise
  checked after 320 attempted updates.
- Ten additional `test_r711.py` integrity/causality checks passed; all
  45 inherited R7.7–R7.10 regression checks passed again. The independent
  overlay recall/gradient test also passed.
- The selected experimental update 80 does NOT appear in `joint_steps`:
  the auxiliary counter is separately recorded. A future true joint training
  phase must increment the joint counter only when its language-model
  objective is actually updated.

## Reproduction

From release root, with Python3, PyTorch, NumPy and SentencePiece:

```sh
export PYTHONPATH=source OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2
python source/test_r711.py
python source/r711_delta.py --root . --overlay adaptation/r711_experimental_delta.pt
python source/r711_validation.py --root . --count 512 --batch 16 --output /tmp/r711_validation.json
# Retrain from accepted checkpoint, reconstruct selectable experimental overlay:
python source/r711_selective_writer.py --root . --updates 320 --train-examples 512 \
  --dev-examples 512 --test-examples 512 --batch 24 --eval-every 40 \
  --output /tmp/r711_working_candidate.pt --report /tmp/r711_training.json
python source/r711_delta.py --root . --candidate /tmp/r711_working_candidate.pt \
  --overlay /tmp/r711_rebuilt_delta.pt
```

The training command above does not alter accepted `checkpoint/pal_active.pt`.
The original research `reports/R7p11_directed320.json` contains exact numbers,
including rejected updates through 320; the `.log` records progress.

## Next hard gates

1. Acquire provenance-verified NaNROUTER whole Gutenberg books; use distinct
   books AND authors for train/validation/sealed test, with explicit
   `doc_start` boundaries. Test 32/128-way random and hard negative recall,
   varied delays (2,048–16,384+) and source key frequency.
2. Make incremental updates equivalent to causal full reinitialization for
   the **same** directed rank-64 matrix, then train joint NOW NLL plus a modest
   historical-recall auxiliary objective with all five inherited optimizers.
3. Promote ONLY when the development/sealed test, memory-content ablation,
   order ablation, language NLL and optimizer-continuity gates pass. Current
   experiment cannot justify frontier-scale or fluent-generation claims.
