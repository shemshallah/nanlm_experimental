# FA1 R7.44–R7.45 — causal word-copy memory and genuine geometric value read

**Lineage:** one FA1 R7 model, one unified optimizer; experimental rounds only.
The accepted R7.13 step-14,416 parent remains immutable. R7.40 was recovered
from its complete checksum-verified archive with 24 optimizer groups. The
unavailable R7.42 26-group checkpoint was **not** reconstructed. The R7.43
source described in the preceding work was absent from GitHub, so R7.44 is an
explicitly new, source-controlled implementation of the stated causal-backoff
principle rather than a claim of byte-identical R7.43 reconstruction.

## R7.44: target-free empirical word-copy distribution

The actual pre-target 32 recent words and 12 eligible near/far modal tokens
construct a normalized discrete word distribution `Q`. Each token receives a
fixed recency/count prior. A bounded gate, learned only from pre-target
coverage, diversity, retrieval agreement and geometric-trust statistics,
mixes it into the complete inherited R7.40 normalized word distribution:

`P44(y) = (1 - a) * P40(y) + a * Q(y)` with `0 <= a <= 0.125`.

Only after this normalized mixture is formed do training/evaluation code
gather probability of the ground-truth next word. The target is **never** an
input to the gate or copy distribution. The gate initializes to exactly zero,
restoring all R7.40 scores bit-for-bit. An all-masked history also forces zero
mixture weight. `diamond_backoff_*` donor overrides affect only `Q`, not the
inherited decoder or target-conditioned original R7 retrieval likelihood.

The R7.40 weights, scalar gates and all 24 optimizer parameter groups and
moments survive. R7.44 appends optimizer group 25. No independent model is
created; the entire hierarchy, tokenizer, apex/pair/point descent and byte
head remain the same network.

## Checkpoint-backed NaNROUTER development results

The recorded 615-book short-story release is SHA-256
`d6ce3a23757e85fe1b9487ef984a5ba81db9c2dc6e962ac7c5df2960480e0054`.
The deterministic split contains **270 TRAIN, 15 DEV and 15 reserved TEST whole
books**. Its split manifest and all three generated document manifests
match the recovered R7.40 release *byte-for-byte*. This verifies the split
reconstruction, not disjointness from unknown older R7 ancestor training books.

R7.44 trained its added group for 48 CPU updates (240 sampled TRAIN positions,
192 selected DEV positions, minibatch 8, two CPU threads). The source R7.40
checkpoint SHA-256 is
`6f2cddd15dd7867bf52c0ddf7fc9f886e205a912e41ad3e1f9c45dd1bf8eefa2`.
The selected experimental R7.44 checkpoint SHA-256 is
`c00f95b6021a99335bf14581c50fc9917446bdb2ddbd78fe591aa647110a20ab`.
The checkpoint reload passed exact model tensor and unified optimizer checks.

On a **separately sampled 2,048-position DEV evaluation across all 15 books**:

| Statistic | R7.40 parent | R7.44 candidate |
|---|---:|---:|
| Full normalized word NLL | 5.417024 | **5.405410** |
| Word gain (parent minus candidate) | — | **+0.011614** nats/word |
| Whole-book bootstrap 95% interval | — | **[+0.004515, +0.020455]** |
| Correct vs foreign-word-history advantage | — | **+0.013111** nats/word |
| Whole-book bootstrap 95% interval | — | **[+0.006606, +0.023254]** |

The geometric ablations remain negative: matched foreign active addresses
give −0.000014 nats/word of incremental specificity, and disabling the apex
gives −0.000205 nats/word. Therefore the R7.44 predictive benefit is real on
this DEV protocol but **cannot be attributed to useful geometric computation**.
Neither R7.44 nor R7.40 passes the complete strict development gate.

R7.44's 192-position selected sample has conditional UTF-8 word-spelling
5.339906 bits/byte, top-1 6.93% and top-5 39.18%. Its inherited 32K-word
tree top-1 is 15.625% and top-5 is 37.5%. These are *different metrics*:
full-mixture word top-k and raw uninterrupted text BPB are not yet available;
conditional spelling BPB excludes EOW and is not raw-text BPB.

## R7.45: geometry selects observed values, not just confidence

R7.45 uses the actual **final descended point message** after the existing
triangle → apex → pair → point feedback. A new 32→96 projection compares that
point query with the unchanged 96-dimensional embeddings of real, pre-target
word candidates. Masked sparse attention reweights the same normalized `Q`;
another exact-zero gate mixes this with inherited R7.44 `Q`:

`Q45(y) = (1 - b) * Q44(y) + b * Qgeometry(y)`, `0 <= b <= 0.6`.

Then the unchanged R7.44 probability gate mixes `Q45` into inherited R7.40.
R7.45 is still one model and one optimizer: it appends **group 26**, preserving
groups 1–25, their trained weights and their Adam moments. It neither inserts
future words into attention keys nor directly uses the label to route memory.

The training objective includes the complete normalized word NLL and a small
correct-versus-foreign-geometric-context contrast. The latter subtracts the
same ablation with the geometric read disabled, preventing a change in the
inherited R7 decoder from masquerading as a benefit of the new value read.
Both target-free geometric donor controls and apex-off controls are required
in development. No source checkpoint is overwritten; TEST stays sealed.

The first 128-update pilot from R7.44 restored R7.44 exactly at attachment,
added 26th optimizer group, and passed model/optimizer round-trip. On its
384-position DEV sample, incremental geometric word gain was −0.000057
nats/word at the final step, while address and apex specificity were
approximately +0.000115 and +0.000237 nats/word respectively. The full strict
gate failed, so this is a **research candidate, not an accepted checkpoint**.

A second CPU continuation resumed those **same 26 optimizer groups and moments**
for 512 more updates, using 2,048 TRAIN / 512 DEV prediction positions. The
development selection reverted to update 1: later snapshots lost word
likelihood and address specificity. Hence 512 updates were *attempted*, but
only the development-selected update-1 continuation is in its saved checkpoint.

On a separate matched **2,048-position / 15-book DEV evaluation** (the same
positions used for R7.44's larger check), both R7.45 recoverable checkpoints
showed essentially the same small gain:

| Statistic | R7.44 parent | R7.45 continued-selected |
|---|---:|---:|
| Full word NLL | 5.405410 | **5.404824** |
| Incremental word gain | — | **+0.000587** nats/word |
| Book-bootstrap 95% interval | — | **[+0.000110, +0.001098]** |
| Apex-specific incremental gain | — | **+0.000225** nats/word |
| Apex bootstrap interval | — | **[+0.000017, +0.000465]** |
| Exact-address incremental specificity | — | **+0.000009** nats/word |
| Exact-address bootstrap interval | — | **[−0.000048, +0.000065]** |

The incremental likelihood and apex signals are positive under this
descriptive whole-book resampling. The exact-address result remains
statistically inconclusive; no full strict gate has passed. The two matched
R7.45 candidates were compared on the same books, so this is **DEV evidence,
not an independent TEST replication**. R7.44's large word-memory gain should
not be conflated with R7.45's much smaller geometric contribution.

## Remaining frontier gap

The system still does not establish general open-domain modeling, normalized
full-mixture word top-k, raw byte/BPE likelihood, genuine long-range credit
assignment, or independent ancestor-book generalization. A frontier claim
requires reproducible sustained gains on fresh books, more causal geometric
value advantage, independent TEST after final DEV selection, and longer
context/horizon evaluation without leakage. We keep the original accepted
step-14,416 checkpoint protected throughout.
