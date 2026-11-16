# FA1 R7.48 — trainable mirror-pair value memory and causal attribution

**Status:** experimental, unpromoted continuation on `master`. Independent
TEST books are sealed. Accepted R7.13 step 14,416 remains unchanged.

## Why the architecture changed

R7.47 achieved a positive 2,048-position DEV word-NLL improvement over
R7.45, but substituting exact 256-head addresses had approximately zero
incremental impact. Its head features were sparse incidence indicators:
when the eight selected heads rarely intersect a historical word's two
heads, most examples cannot use the pair structure.

R7.48 stores learnable content directly at **all 32,640 undirected mirror
pair addresses**, rather than relying solely on exact membership in eight
active points. It also learns 256 shared head factors and a query from the
existing apex -> pair -> point descended representation. The active 28
pair addresses form a symmetric-and-antisymmetric wave over learned pair
values, and observed previous words are scored at their *accepted* pair
addresses. The copy distribution remains target-free until the final
supervised probability lookup.

Let `V[p] in R^16` be a persistent trainable value at each global pair
address `p`, and `H[h] in R^16` a shared factor at each head. For a real
candidate word with frozen accepted heads `(a,b)`, its key is built from
`V[pair(a,b)] + (H[a]+H[b])/2`. The active eight heads determine exactly 28
pair values, whose mean and signed orientation-weighted mean are combined
with the inherited descended point message. Two zero-initialized scalar
gates blend the resulting *normalized* prior-word distribution with the
existing R7.47 copy mixture.

The pair bank is **not** another word predictor, another optimizer, or an
independent checkpoint. It is a new trainable state inside the original
single FA1 network. Its 522,240 pair-value scalars are appended as group
28 of the existing hybrid optimizer, preserving all previous groups.

## Controls and evaluation requirements

1. Address-only swaps: different-book `vertex_ids` with the current
   target and all observed candidate word values held constant. Report
   the *incremental* R7.48 address effect after subtracting inherited
   R7.47 address sensitivity.
2. Same-point reorder: leave the eight head addresses unchanged but vary
   their orientation order.
3. Pair-bank removal: zero only the trainable `C(256,2)` values while
   preserving the learned shared-head factors and inherited model.
4. Pair-value permutation: deterministically relabel the pair-bank lookup,
   keeping the same number of parameters and all historical candidate IDs.
5. Signed-orientation removal: retain symmetric pair information while
   suppressing S+/S- contributions.
6. Foreign history: replace observed causal word history with eligible
   different-book candidates while keeping the current target intact.

Promotion requires better word NLL *and* positive causal geometric
attribution, checked by whole-book uncertainty and long-context evaluation;
these requirements are not waived for a better pooled DEV mean. Evaluation
uses the published 270-TRAIN/15-DEV split and never accesses TEST content.

## Recovery and the first pilots

The original R7.47 candidate SHA-256 is
`5f221aa492446744724c4bb527e3d62d0ac9ef8f52009f3255b91bcf8ab7c96a`.
The original R7.47 recovery ZIP was verified and reconstructed without
retraining the inherited backbone. The NaNROUTER source release checksum
matched `d6ce3a23757e85fe1b9487ef984a5ba81db9c2dc6e962ac7c5df2960480e0054`.
TRAIN contains 6,897,379 word targets from 270 books; DEV has 1,260,785
word targets from 15 books. No TEST text was read or generated.

The first 24-update pilot, with 384 selected DEV positions, saved a
recoverable R7.48 candidate SHA-256
`ea01c50169ad40542fa501543b0021b8569622d77fc2d8b3440606754e87518f`.
Its selected next-word NLL gain was +0.001688 nats/word; the descriptive
whole-book 95% interval was [-0.007398,+0.009834], crossing zero.
Address-only and pair-bank ablations were negative, so promotion failed.

Integrity of this first candidate was independently checked: all 454
inherited model-state tensors, 27 previous optimizer groups and 432
inherited AdamW states were bit-identical to R7.47. One optimizer,
28 groups, six newly trained tensors. The accepted parent is untouched.

An additional 96 updates resumed the **same** R7.48 network and optimizer
moments. The final training probe reported +0.001871 nats/word on its
512-position DEV sample, +0.003659 nats/word from retaining pair values,
-0.0000741 address-specificity and -0.000287 signed-orientation effect.
These are exploratory probes and fail the strict gate. The process was
memory-terminated when attempting a second full model load after saving;
the successor checkpoint was subsequently read separately and its native
metadata verified. It has 28 optimizer groups, 460 model-state tensors,
is unpromoted, and has SHA-256
`7d0636843708d4e2984eb862d29a8f49e0682e4a7c0848259c80a531e83f1e1b`.

## What remains unproven

- A positive address-only, signed-rail and pair-bank effect *jointly* on
  larger held-out DEV books.
- Independent TEST generalization, which remains reserved for a final
  prespecified promotion gate.
- Genuine uninterrupted UTF-8 bits/byte, full mixture word top-1/top-5,
  and +2/+3 prediction likelihood. The inherited pipeline's auxiliary
  first-byte predictor is not a substitute for these metrics.

## Independent reload and larger DEV audit

An independent R7.47 checkpoint reload and exact zero-gate comparison
completed on a separate 512-position sample from all 15 DEV books (seed
913); a monolithic 2,048-position evaluation exceeded this container's
memory allowance, so 2,048-position evidence is **not** claimed for R7.48.

| DEV metric | R7.47 parent | R7.48 continuation |
|---|---:|---:|
| Full next-word NLL (nats/word) | 5.433400 | **5.431809** |
| Candidate gain | — | +0.001591 |
| Whole-book 95% interval | — | [−0.008101,+0.012924] |

| Matched R7.48 contribution | NLL advantage (nats/word) | Whole-book 95% interval |
|---|---:|---:|
| Remove learned mirror-pair values | **+0.002976** | **[+0.000898,+0.004963]** |
| Permute pair-value addresses | +0.001843 | [−0.000486,+0.004330] |
| Swap only eight active head addresses | +0.000045 | [−0.000311,+0.000372] |
| Remove signed-orientation channel | **−0.000345** | **[−0.000615,−0.000088]** |
| Use correct rather than foreign causal word history | +0.047756 | [+0.013454,+0.093911] |

Pair-value information is useful on this limited DEV diagnostic, but the
full-model improvement is inconclusive and the S+/S− signed correction
is currently detrimental. Six of fifteen DEV books improved in word NLL.
The original 27-group trained ancestor remains unmodified; the successor
uses one optimizer with 28 groups. The new copy path must not be promoted.

The inherited first-byte auxiliary predictor on this sample gave NLL
3.211958 nats/first-byte, top-1 10.16%, and top-5 38.87%. This is *not*
raw-stream BPB, nor full-model next-word top-1/top-5. A single batched
512-word forward scored about 2,350 sampled positions/s (excluding corpus
preparation and ablation runs); do not mistake this for training throughput.

Research significance: there is now a positive *pair-value necessity*
measurement on DEV under an otherwise fixed inherited model, with direct
bank-ablation uncertainty excluding zero. That does **not** yet isolate the
exact canonical pair address or establish a generalizable geometric wave.
The next experiment should gate signed emissions using causal reliability,
preserve the trained pair values and optimizer moments, and require the
correct-address advantage before promotion.
