# FA1 R7.30 — factorized triangle memory with explicit branch readout

**One R7 on `master`; one stateful optimizer.** This is a source-level
continuation of the R7.29 unified model, not a replacement trunk or new branch.
The accepted step-14,416 checkpoint and every inherited optimizer history
remain protected. Research successors are never automatically promoted.

## Motivation and falsifiable hypothesis

The earlier 64-update R7.30 pilot reported unchanged full next-word NLL
(4.955495), first-byte top-1 (10.94%) and top-5 (32.81%), and a negative
correct-book advantage. Auxiliary factor learning was not translating into
predictive value. That pilot's complete source/checkpoint are not currently
stored on GitHub `master`, so these supplied numbers are **not** a new verified
training result of this commit.

The actionable hypothesis is that the existing geometric read loses most of
its signal in the triangle/apex/point channel and the subsequent weak
probability bridge. The code now tests this explicitly instead of optimizing
an auxiliary retrieval score alone.

## Integrated architecture

1. Each of the eight causal active points contributes learned 32D key and
   value factors. Every addressed triangle sums its three factors. This
   reuses real C(256,3) IDs and the trained exact-address S+/S- face values;
   **no** 2.76M by 32D dense tensor is allocated.
2. The current past-only query sparsely selects eight of 56 addressed
   triangles and retrieves a value. A zero-initialized `factor_up_gain`
   delivers it into the **existing** apex -> triangle -> pair -> point path.
3. A separate zero-initialized `factor_decoder_gain` proposes a correction
   to the existing 96D *branch* hidden state after the downward point path.
   The proposal vanishes if either the point descent or apex is ablated.
   The inherited R7 trunk, token tree and mixture stay unchanged.
4. The original R7.29 contextual read is retained. All new keys, values,
   query, trust and readout parameters append as the **15th group in the same
   R7.27 optimizer**, keeping the previous 14 groups and all historical
   moments. Checkpoint restoration reconstructs both R7.29 and R7.30 in
   sequence, then loads the exact full model and optimizer state.

## Diagnostics and acceptance

The DEV probe now reports *separately*:

| Stage | Counterfactual metric |
| --- | --- |
| New read, final system | `factor_word_gain` (factor-off minus full word NLL) |
| New read, branch only | `factor_branch_gain` (factor-off minus branch word NLL) |
| Branch versus full mixture | `branch_vs_mixture_gap` and `branch_beats_inherited_fraction` |
| Older R7.29 read | `contextual_read_word_gain` and its branch ablation |
| History | Full and branch correct-book advantage with foreign book context |
| Statistical reliability | Whole-book bootstrap 95% intervals for factor word and branch gains |
| First byte | Normalized 256-class top-1/top-5, cross-entropy bits **per next word** |

Selection remains DEV-only. Promotion still requires a positive full-system
NLL gain, positive geometric/apex/pair and factor-off causal ablations,
positive correct-book advantage, and strictly positive book-bootstrap lower
bounds. If only the branch improves but the final mixture does not, this is a
**routing/trust failure**, not a performance gain. This architecture does not
produce raw-byte bits/byte or full-mixture word top-5 yet; the 256-way head
only predicts the first byte of each next word.

## Reproduce when the full R7.29 checkpoint is available

```bash
python fa1r7.12/source/fa1_r728_geometric_memory.py \
  --factorized --root R713_ANCESTOR --checkpoint R729_UNPROMOTED.pt \
  --corpus-dir SHA_DISJOINT_300_BOOK_CORPUS --split dev \
  --train 1024 --dev 512 --steps 96 --minibatch 16 --batch 8 \
  --eval-every 16 --ablate-every 4 --factor-up-init .20 \
  --factor-decoder-init .35 --factor-weight .75 \
  --output R730_DEV.json --candidate-checkpoint R730_UNPROMOTED.pt
```

This writes only a distinct **unpromoted** R7.30 candidate; source and parent
checkpoint files are never overwritten. The 15 reserved TEST books must not
be used in training, checkpoint selection, or hyperparameter tuning.

GitHub Actions validates source-only structural and gradient tests without
private checkpoint data; a successful CI run is not evidence of held-out
predictive improvement. No GPU scaling is justified until that evidence exists.
