# PAL R7.10 — transactional checkpoint integrity and hard-negative memory audit

**Status:** no checkpoint promotion; the only active checkpoint remains the R7.7
step-14,416 tensor/optimizer set inherited unchanged through R7.8–R7.10.
Accepted checkpoint SHA-256:
`54e835b00642013d4f1bf486a8150eb1ff06c547565771bd21933e143d050f2d`.
The five optimizer histories are present; this is the separate R7.7/R7.9
associative memory research lineage, **not** R10–R16 PAL and not NaNLMind.

## Changes performed

1. `source/r710_checkpoint_integrity.py` verifies the accepted checkpoint format,
   step, immutable tokenizer/corpus hashes, model tensor finiteness, **all five**
   nonempty optimizer states, and four independent RNG states. For experimental
   saves it uses a temporary file + fsync + atomic rename + SHA-256 sidecar,
   and expressly forbids writing `pal_active.pt`. Candidate promotion must be
   made separately after held-out gates; no such promotion has happened.
2. `source/r710_hard_negative.py` constructs reproducible 32-way, unique-key
   causal historical association questions. The positive is the next word
   after a key occurring exactly once in the older 6,144-word window, sampled
   from the latest 512 old words. All distractors are real alternative value
   tokens following other keys in that same *older* memory; none are future
   labels. The same query, target and 31 distractors are scored under correct,
   batch-swapped unrelated, within-64-block-reversed and zero memory states.
   `random` picks competitors randomly; `hard` prioritizes alternative pairs
   by 0.7 cosine(query,key) + 0.3 cosine(true value, competing value), with
   frozen accepted-model token embeddings. **Hard here means semantically
   similar historical competitors, not necessarily model-maximizing decoys.**
3. `source/test_r710.py` checks deterministic mining, distinct candidates,
   historical causality, preserved checkpoint/optimizer/RNG and original
   within-block order erasure. Inherited R7.7–R7.9 tests also reran.

## Measured read-only historical 512-query evaluation

The large 512-position evaluation is the primary report; 16/128-position
pilots are retained for audit. All questions are drawn from *previously
consulted* PAL development/test corpora. Book-level independent validation
has NOT been performed, and positions may overlap heavily in source context.
One query from one chronological older memory is not a new independently
sampled document.

| Split | Negative set | Correct memory CE ↓ | Unrelated memory CE ↓ | Correct minus unrelated benefit (unrelated − correct) ↑ | Correct memory top-1 ↑ |
|--|--|--:|--:|--:|--:|
| Development | Random | 4.129105 | 4.128519 | −0.000586 | 1.5625% |
| Development | Semantic hard | 3.867020 | 3.866386 | −0.000634 | 0.3906% |
| Test | Random | 3.936687 | 3.937702 | +0.001015 | 2.5391% |
| Test | Semantic hard | 3.739855 | 3.739821 | **−0.000035** | 0.7813% |

Chance-uniform (32 candidates): CE **ln(32)=3.465736**, expected top-1
**3.125%**. The current readout underperforms this diagnostic baseline in
CE and top-1 under all shown conditions. Although embedding-based hard
candidates happen to produce lower CE than random ones here, this does NOT
mean they are easier in a general sense; difficulty is model-dependent.

On the principal hard-negative test, the query-bootstrap 95% CI for correct
memory benefit is **[−0.001349, +0.001288] nats/query**. This interval is
illustrative only: historical contexts overlap and there are no per-book
sampling units. Reversed within-block history exactly reproduces original
R7.7 state to normal floating-point tolerance because the history initializer
averages each 64-word block. Zero memory gives chance CE and tie-aware 3.125%
accuracy, verifying that the top-1 metric does not treat first-index ties as
perfect predictions. Prior R7.9 results have different sample positions and
negative construction and **cannot** be compared as a matched improvement.

The negative result is specific: the accepted rank-64 memory is not a useful
key-to-successor associative reader with this probe. It does not show whether
all forms of context utility or exact retrieval are absent; those are separate
paths, and next-word NLL was not changed or remeasured in this read-only round.

## Checkpoint integrity results

- Parent/source checksum verified; accepted source still step **14,416**.
- Save/restore roundtrip: **all 74 model tensor entries exactly equal**; all
  five optimizer state dictionaries and torch/numpy/Python/batch RNG states
  exactly equal.
- Corrupt sidecar was rejected; active checkpoint write was blocked; original
  file SHA-256 unchanged. No training updates were executed in R7.10.
- **45/45 inherited/new structural checks passed**: R7.7 12, R7.8 order 6,
  R7.8 directed pair 5, R7.9 7, R7.10 15. Checkpoint roundtrip behavioral
  checks are additional.

## Reproduce

From the release root (Python + torch, numpy, sentencepiece):

```bash
export PYTHONPATH=source OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2
python source/r710_checkpoint_integrity.py --root . --scratch /tmp/r710_audit
python source/test_r710.py
python source/r710_hard_negative.py --root . --count 512 --batch 8 \
  --output reports/R7p10_hard_negative_512.json
```

The audit's temporary 94.65 MB roundtrip checkpoint intentionally lives OUTSIDE
this release; the release contains exactly one active/current checkpoint.

## Next scientific gate

Implement a zero-gated, **selective, order-sensitive directed key→value writer**
into the same rank-64 physical matrix, train on a causal recall objective with
hard negatives, and measure correct-vs-unrelated and correct-vs-reversed memory
benefits on author-disjoint, verified, unseen NaNROUTER Gutenberg books. Only
then test whether this learned state improves held-out next-token NLL while
keeping the exact/subword paths and all five inherited optimizer histories.
Neither negative random scores nor intra-split bootstrap establishes a
frontier-scale language-model claim.
