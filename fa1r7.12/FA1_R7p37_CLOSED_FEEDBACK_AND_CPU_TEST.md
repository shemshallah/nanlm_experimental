# FA1 R7.37 — finite deep feedback closure; real CPU training and TEST

**Status:** source-integrated, checkpoint-backed **unpromoted** experiment on
`master`. The accepted step-14,416 model is unchanged. FA1 and FA2 stay separate.

## Remaining feedback gap repaired

R7.36 used upward triangle/apex computation and downward apex→pair→point
reconstruction, plus query-conditioned apex-to-decoder attention. It did not
return the reconstructed point state to the apex for a second causal descent.

R7.37 adds exactly one finite cycle:

`point → pair → triangle → apex → pair → point → apex(return) → pair → point → word/byte`

- The return depends on the *reconstructed* point and original apex, with a
  product-of-norms route that vanishes when either direction is ablated.
- A zero-initialized scalar makes the attachment exactly function-preserving.
- The same inherited pair/point parameters run the second descent; there is
  no bypass around the older decoder or replacement architecture.
- The existing hybrid optimizer receives one additional AdamW group, bringing
  the total from 20 to 21, without reassigning old parameter objects or moments.
- Unit tests cover zero-drift migration, nonzero return-path gradients, strict
  state roundtrip, exclusion of bypasses, and preserved old optimizer state.

`PYTHONPATH=fa1r7.12/source python -m unittest discover -s fa1r7.12/source -p 'test_fa1_*.py' -q`

**Local result:** 77 tests passed, no failures.

## Provenance and corpus

The newest *recoverable* complete prior FA1 checkpoint found for this round was
`FA1_R733_V2_GEOLR96_UNPROMOTED.pt` (the saved architecture round is R7.32).
The earlier R7.35 and R7.36 source modules were attached in-place with
identity initialization, followed by R7.37. This run **does not claim** to
resume unavailable trained R7.35 weights.

NaNROUTER `corpora-short-stories-0004` was downloaded from the GitHub release;
SHA-256 `d6ce3a23757e85fe1b9487ef984a5ba81db9c2dc6e962ac7c5df2960480e0054`
matched the release manifest. The R7.37 selection script uses only book SHA
and word count, with 270 TRAIN, 15 DEV, 15 TEST books. DEV/TEST books each have
at least 65,536 words. All 300 chosen documents have distinct content hashes.
This proves disjointness **within this experiment**; overlap with unknown
earlier checkpoint-training books has not been completely audited.

## 48 CPU updates; development selection

Actual CPU backpropagation ran for 48 updates, with 240 TRAIN and 120 DEV
positions; minibatch 8; two CPU threads. Predeclared selection favored the
strict geometry/history gate first, then full-model word loss. No candidate
passed that strict gate. The selected and saved state is **update 1**; updates
24 and 48 improved conditional spelling but regressed in word loss relative to
update 1, so they were not selected.

| DEV measure | Source parent | Selected update 1 |
|---|---:|---:|
| Full word NLL (nats/word) ↓ | 5.210881 | 5.209484 |
| Conditional spelling BPB ↓ | 5.349677 | 5.315689 |
| Conditional spelling top-1 | 4.0733% | 4.8880% |
| Conditional spelling top-5 | 37.6782% | 38.2892% |
| Correct-book history advantage (nats/word) | −0.004839 | −0.006071 |
| Geometry-only word gain (nats/word) | −0.001048 | +0.000313 |
| Far-value writer word gain (nats/word) | 0 | −0.000074 |

At update 48 (not saved as selected), DEV conditional spelling BPB was 4.905836
but full word NLL was 5.212603. This is a tradeoff, not a strictly improved
replacement checkpoint.

All 21 optimizer groups and the original checkpoint SHA are preserved in the
new unpromoted 266,008,016-byte checkpoint, SHA-256
`74a4112c97ccd828b485a7a329dee01c8fc66a0b6cb42bb49f9da3cdf6026046`.

## One-shot reserved TEST, after DEV selection

The saved candidate was loaded independently and compared on **the same 120
positions from 15 reserved TEST books** with the parent migrated using
closed new gates. No TEST results were used to choose the model/update.

| TEST measure | Source parent | R7.37 saved candidate |
|---|---:|---:|
| Full word NLL (nats/word) ↓ | 5.240534 | 5.237765 |
| Conditional spelling BPB ↓ | 5.341905 | 5.302047 |
| Conditional spelling top-1 | 4.2506% | 4.6980% |
| Conditional spelling top-5 | 38.4787% | 38.7025% |
| First-byte top-1 | 11.6667% | 11.6667% |
| First-byte top-5 | 45.8333% | 45.0000% |

The full-model TEST word NLL point gain is **+0.002769 nats/word**. The
whole-book bootstrap 95% CI is **[−0.000420, +0.005610]**; it includes zero.
The far-value writer ablation remains negative (−0.000138 nats/word) and the
writer-only foreign-value advantage is also negative. The strict development
gate fails: **do not promote**.

The exact normalized inherited-trunk word top-1/top-5 on this TEST sample is
15.8333% / 34.1667%. It is **not** a full-mixture top-k metric. Conditional
spelling BPB uses only literal UTF-8 bytes within target words (3.725 bytes
per word here), excluding EOW from the denominator. Continuous raw-text BPB
and full-mixture next-word top-k are unavailable and must not be inferred.

## Reproducibility and recovery

Prepare the 300-book split via `fa1_r737_longbook_split.py`, then tokenize it
with `prepare_document_corpus.py --presplit`, preserving the R7.13 frozen
vocabulary/tree. Run `source/fa1.py` with `--steps 48 --train 240 --dev 120
--minibatch 8 --batch 8 --eval-every 24 --ablate-every 8 --threads 2
--lr 0.00012 --seed 737` and a separate `--candidate-checkpoint` path. Run
`fa1_r737_sealed_test.py` **only after** the DEV-selected checkpoint is frozen.

The result files `FA1_R737_CPU48_DEV.json`, `FA1_R737_SEALED_TEST.json`, and
`FA1_R737_CPU48_UNPROMOTED.pt` carry full per-path metrics, ablations,
gradient norms, provenance, and saved optimizer state. The GitHub source and
regression suite remain executable even when a GitHub Release checkpoint asset
has not yet been uploaded.
