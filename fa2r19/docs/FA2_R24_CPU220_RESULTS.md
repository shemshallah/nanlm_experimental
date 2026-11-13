# FA2 R24 — unified single-checkpoint CPU continuation to step 220

Date: 2026-11-12 (US Pacific). **Status: experimental / NOT promoted.**
All new code is on `master`. The accepted R18 checkpoint is unchanged:
`dd040106568786e7fbad2a880f4010635b626b51c3697e2ab08712905a2de91f`.

## Execution lineage

This is **one PAL recurrent model** with signed-mirror sparse MoE,
R18/R21/R22 memory and trainable R23 contextual reader, and **one
nine-group AdamW optimizer**. The selected R24 candidate is not a new
independent model. The accepted optimizer groups, R18 learned retrieval
moments and R21/R22 gate moments were retained. A local CPU checkpoint
recovered and verified 123 optimizer states, 12 original book-stream
cursors, and the immutable R18 ancestor.

Source training was run locally from the exact original R19 full release
and verified NaNROUTER books. GitHub Actions source checks do not
represent this local training execution. The cumulative local recovery
ZIP contains the precise executed source and one latest checkpoint;
the separate original R19 full release provides the tokenizer, books
and accepted ancestor, and is not duplicated.

### Training stages

| Continuation | Method | Outcome |
|---|---|---|
| Step 50 → 180 | All trainable groups, twelve books, 0.4× resumed LR, longer windows | **Rejected**: development BPB at step 180 worsened to 3.66302 |
| Step 50 → 130 | Same model/optimizer; R23-only phase, twelve contiguous book streams advanced beyond 9K | **Selected 130**: dev BPB 3.511531 |
| Step 130 → 220 | Same model/optimizer; R23-only phase, nine eligible training books advanced beyond 33.1K | **Selected 220**: dev BPB 3.511140 |

Three books ended before the 33.1K warmup target and were excluded
from that phase without fabricated history. Nine source-verified books
provided continuous-state forward history. The update gradient
horizon remained two adjacent twelve-word chunks; this does **not**
constitute 32K end-to-end recurrent backpropagation. The R23 memory
index itself spans up to 32,768 completed prior tokens.

R23 train-only objectives included token NLL, multi-positive
1/2/4/8-token value contrastive loss, and a margin against shuffled
values and unrelated-book donor history. The **actual next-token label
is used in the loss only**, not in retrieval candidate admission.

## Short-window evaluation (same 2 development books, nonblind historical test)

| Metric | Step 50 | Step 130 | Step 220 |
|---|---:|---:|---:|
| Development BPB (2,227 bytes) ↓ | 3.512033 | 3.511531 | **3.511140** |
| Development nonspace NLL ↓ | 4.649751 | 4.649696 | **4.649636** |
| Nonblind historical test BPB (2,249 bytes) ↓ | 3.129366 | 3.129275 | **3.129195** |
| Unrelated-book control advantage, dev (nats/token) ↑ | 0.000151 | 0.000814 | **0.001336** |
| Scrambled-value control advantage, dev (nats/token) ↑ | 0.0000017 | 0.0000412 | **0.0000672** |

These are **small** improvements and must not be compared directly to
reset-window R18 scores because stream state, source bytes and
measurement contexts differ.

## Additional six-book development evaluation (not used to select the step)

The same 11,178 raw bytes (6,976 target tokens) on six additional
existing development books gave BPB:

- Step 50: **3.348646**
- Step 130: **3.348154**
- Step 220: **3.347756**

At step 220, correct-book advantage over an unrelated-book donor
was **+0.001199 nats/token**, versus +0.000145 at step 50.
The secondary nonspace NLL slightly worsened, from 5.203658
(step 50) to 5.203713 (step 220). These previously available
books are **not new blind author/book evaluation**.

## Genuine continuous 9K and 32K forward-state probe

The model's original PAL recurrent state was advanced through
**every preceding token** (without gradient through warmup), and
the complete unified prediction pipeline was applied at a future
query chunk. Distances below are **actual top-k selections**, not
merely supported address limits.

| Development book | Query offset | Step-50 BPB ↓ | Step-220 BPB ↓ | Furthest selected prior token |
|---|---:|---:|---:|---:|
| 970 | 9,073 | 2.887120 | **2.886135** | 9,360 |
| 970 | 33,173 | 2.676226 | **2.674654** | 31,554 |
| 1444 | 9,110 | 2.893143 | **2.892593** | 9,301 |
| 1444 | 33,137 | 2.830041 | **2.829501** | **32,758** |

For both 32K query chunks, correct history beat both
shuffled-value and unrelated-book control. However, at a shorter
2,454-token query on book 970, correct history **lost to the
unrelated-document control** and regressed relative to step 50.
Long-distance retrieval is operative, but history reliability is
not uniformly positive across distances. The probe chunks contain
only hundreds of tokens each.

## Integrity and decision

The step-220 checkpoint has SHA-256
`b6cef54cde36ab35a1eaa1e392de9c3a2b50ffb1324908611b768cfc724e1e4f`.
All stored floating tensors are finite, one optimizer has
**nine groups / 123 parameter states**, and the accepted R18
ancestor's digest remains exactly unchanged.

**Decision: retain step 220 as the ONE experimental R24 candidate;
do not promote it to accepted weights.** The current small-scale
improvements do not establish frontier quality, real semantics,
general reasoning, or a scalable loss law.

Next acceptance requires new author/book-disjoint blind data,
frequency-matched donor tests, multi-hop and paraphrase transfer,
confidence intervals/per-book scores, and significant matched
language-model gains. Truncated BPTT and long-range index retrieval
must be reported separately.

For active architecture/training source:
[`r24_unified_model.py`](../source/r24_unified_model.py),
[`r24_unified_cpu.py`](../source/r24_unified_cpu.py).
The GitHub workflow is source-verified by default; full training
requires separately accessible original R19 assets.
