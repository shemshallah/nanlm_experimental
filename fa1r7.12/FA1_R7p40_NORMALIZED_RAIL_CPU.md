# FA1 R7.40 — normalized signed-rail continuation and negative result

**Status:** checkpoint-backed, complete 24-update CPU research continuation;
**unpromoted**. The parent is the exact DEV-selected R7.39 checkpoint,
SHA-256 `28d450550317a716eadaf513f600a59360b145ea2281f79affd1418255b7a007`.
This remains one FA1 model and one unified optimizer with inherited moments.

## Causal change

R7.39's selected eight S+/S− codes have RMS 0.01796/0.01715, so the
antisymmetric product almost vanishes. R7.40 normalizes these signed face
coefficients by their selected-address RMS (with a lower denominator of
0.005), normalizes the observed far-history projections by width, and forms
the same antisymmetric quarter-turn cross. It adds a bounded correction to
the existing R7.39 cross response through a new zero-initialized scalar.

The correction uses only observed, pre-target far modes and already-selected
triangle addresses. Completely masked history still yields zero. The
topology, inherited word/byte trunk, sentence/subword loops, sparse writer,
apex/pair/point descent, and every previously trained Parameter are retained.
The extra scalar is appended as optimizer group 24; no inherited Adam
moment is reinitialized. Zero-gate migration reproduces the full R7.39
parent predictions exactly before the candidate-only gate is opened.

## Reproducible CPU comparison

NaNROUTER short stories batch 0004, SHA-256
`d6ce3a23757e85fe1b9487ef984a5ba81db9c2dc6e962ac7c5df2960480e0054`;
270 TRAIN / 15 DEV / 15 reserved TEST whole books, SHA-disjoint within this
round. From the protected R7.39 candidate: 240 TRAIN positions,
192 DEV positions, minibatch 8, 24 updates, batch 8, two CPU threads,
learning rate 0.00012, seed 739, evaluations at 1/8/16/24.

| DEV metric | R7.39 parent | R7.40 selected (update 1) | R7.40 update 24, rejected |
|---|---:|---:|---:|
| Full word NLL ↓ | **4.968767** | 4.968996 | 4.969779 |
| Conditional spelling BPB ↓ | 5.314221 | 5.299330 | **4.972960** |
| Conditional spelling top-1 ↑ | 7.6246% | 7.0381% | **9.5308%** |
| Conditional spelling top-5 ↑ | 36.8035% | 37.0968% | **37.5367%** |
| First-byte top-1 ↑ | 13.5417% | **14.0625%** | — |
| First-byte top-5 ↑ | 38.0208% | **38.5417%** | — |

Selection: DEV update 1; strict gate **false**, selection tier 0. At that
step the calibration-only word contribution is `-3.60e-8` nats/word and
correct-versus-foreign specificity is `-5.82e-8` nats/word. The
calibration-only conditional literal-byte contribution is `-1.22e-8`
nats/byte, with specificity `-4.93e-8` nats/byte. Book-bootstrap 95% CIs
include zero. All four measured training snapshots fail the strict gate.

The checkpoint was written separately, SHA-256
`6f2cddd15dd7867bf52c0ddf7fc9f886e205a912e41ad3e1f9c45dd1bf8eefa2`,
266,059,520 bytes, one optimizer with 24 groups. The parent was hashed before
and after training and was unchanged. The candidate is **unpromoted**;
no sealed TEST positions were consulted to choose the update.

## Failure interpretation and next architectural requirement

R7.39 had an amplitude failure; R7.40 corrects the local signed-rail scale
without achieving a useful downstream causal history effect. This is
evidence *against scale alone as the bottleneck*. The correct-vs-foreign
writer-content difference remains weak or negative. The next experiment
should constrain value content and retrieval relevance directly, with a
predeclared whole-book correct-history discrimination gate, before another
wave or attention amplification path is added.

Conditional spelling BPB excludes EOW and is not continuous raw-text BPB.
Inherited-trunk word top-k is not normalized full-mixture word top-k.
The previous checkpoint's possible overlap with unknown older corpora
remains unverified. No stronger generalization claim is made.
