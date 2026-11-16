# FA1 R7.49 — canonical pair-address and signed-reliability research

Author: Justin Howard-Stanley, Independent Researcher, Santa Cruz, California, USA.

**Status:** CPU-trained, independently checkpoint-reloaded and unpromoted.
No independent TEST was accessed. This experiment belongs to FA1 only;
FA2, M3 and Terminal-RSI remain separate.

## Starting point and migration

- Exact R7.48 parent: 28-group single-optimizer checkpoint SHA-256
  `7d0636843708d4e2984eb862d29a8f49e0682e4a7c0848259c80a531e83f1e1b`.
- Recovered R7.48 included 32,640 trainable pair values, 256 head factors,
  trainable copy mixture and an antisymmetric S+/S− rail.
- R7.49 appends six scalar parameters as optimizer group 29, retaining the
  complete learned network and all prior AdamW moments. All new effects are
  exactly zero at migration; bitwise prediction parity was verified against
  the *actual* R7.48 checkpoint.
- Group 28 is continued and updated; original groups 1–27 remain frozen.
  Independent saved-checkpoint integrity check verified 454 original backbone
  tensors, the first 27 optimizer groups and 432 AdamW states bit-identical.
- The first 64-step training continuation was extended by 128 executed
  additional updates with all 29 optimizer groups and their moments resumed.
  The selected checkpoint is the 96th step in the second session, for 160
  selected training updates total. The actual 128th update was not selected.

The two additional causal read paths are:

1. **Canonical pair address read.** For each observed historical candidate
   word, use its accepted immutable head pair to detect exact equality with
   the 28 active `C(256,2)` addresses, one-endpoint overlap and exact-match
   orientation. All coefficients are signed and trainable; a negative exact
   address coefficient is a legitimate inhibitory rather than copy bias.
2. **Signed-rail reliability gate.** Decompose the inherited `R7.48` pair
   score into symmetric and signed terms using identical causal features.
   A target-independent trust score, based on pre-target active-head overlap,
   attenuates the *signed* term without changing the symmetric pair wave.

Both route only among actually observed historical words. Target labels enter
solely in the normalized final word-probability lookup and TRAIN objective,
never in address choice, signed trust or copy attention.

## Training and checkpoint

- NaNROUTER short-story release identity SHA-256:
  `d6ce3a23757e85fe1b9487ef984a5ba81db9c2dc6e962ac7c5df2960480e0054`.
- Verified TRAIN 270 books, 6,897,379 word targets; DEV 15 distinct books,
  1,260,785 targets. TEST books sealed.
- 64 + 128 executed CPU training updates, best selected at 64 + 96.
- Selected unpromoted R7.49 checkpoint SHA-256:
  `b8dabe54e2417a75defd01c776606367a3f3b0f18f479b032f8390b6fc87ddb5`.
- One model, one optimizer object, 29 groups; no training from scratch.

## Independently reloaded 512-position DEV, 15 books

| Measurement | R7.48 parent | R7.49 selected | Interpretation |
| --- | ---: | ---: | --- |
| Full next-word NLL (nats/word) | 5.431809 | 5.431276 | +0.000533 nats/word |
| Whole-book 95% interval for gain | — | [+0.000185,+0.000904] | Descriptive repeated-DEV interval |
| DEV books with improved NLL | — | 11/15 | Improvement not universal |
| Correct-history advantage (nats/word) | — | +0.047568 | Whole-book interval positive |
| Exact address-only attribution | — | +0.00000364 | 95% CI [-0.000032,+0.000039], inconclusive |
| Permuting pair-bank values: NLL penalty | — | +0.000210 | 95% CI includes zero |
| Removing remaining signed rail: NLL penalty | — | −0.0000370 | Signed rail still harmful |
| R7.49 signed-reliability gate contribution | — | +0.000376 | Main source of incremental improvement |
| First UTF-8 byte auxiliary NLL | — | 3.211958 | Not raw-text BPB |
| First UTF-8 byte top-1 / top-5 | — | 10.16% / 38.87% | Auxiliary logits, not full word top-k |
| Word scoring throughput | — | 2,803 positions/s | CPU 2 threads, fixed DEV |

The signed attenuation strength is tanh(2.3984)=0.9836 and mean causal
reliability is 0.9242, giving around 91% attenuation of the signed signal
on this development sample. Despite that repair, signed contribution remains
slightly negative: whole-book 95% interval [-0.0000685,-0.0000132].

The full system cannot report a genuine raw byte-stream BPB, complete word
top-1/top-5 probabilities, or +2/+3 prediction losses from this protocol.
Any claim of those metrics from the first-byte auxiliary would be false.

### Where the gain occurs

Only 199/512 (38.9%) of sampled targets appeared in eligible prior-word
candidate history; they improved by +0.001243 nats/word. Of these, 23 targets
had a pair exactly present among the active 28; that subgroup improved by
+0.004260. **This subgroup gain cannot be attributed solely to exact
addresses**: the signed-gate repair and updated pair bank also changed.

### Inhibitory-address hypothesis (exploratory)

Independent read-only sweeps on two different 512-position DEV selections
showed that *positive* exact-pair copy bias worsens NLL, while a moderate
negative match coefficient improves or preserves it and creates positive
true-versus-donor-address contrast. On the original seed, setting its
coefficient from approximately −0.0155 to −0.3 changed NLL 5.431276 to
5.431244 and canonical specificity to +0.000036. On the second seed,
−0.1 gave an NLL of 5.169736 with +0.000031 canonical specificity.
The magnitude is very small, and tuning on DEV makes these observations
**hypothesis-generating, not confirmatory**. They are not applied to the
saved R7.49 checkpoint.

## Promotion decision and next experiment

**REJECT promotion.** This is a positive but small matched development
comparison; the desired exact-address contribution has not cleared the
book-bootstrap gate, and suppressing the harmful signed rail explains most
of the gain. The sealed independent TEST and long-context gates remain
unopened. We therefore retain the accepted step-14,416 FA1 ancestor.

The highest-value next R7.50 experiment is TRAIN-only learning of a bounded,
context-conditioned *inhibitory* exact-pair likelihood term, with separate
train/DEV calibrations for copied vs absent targets and explicit negative
controls for exact-word pair hit versus shuffled and related-head pair hit.
Keep the same R7.49 29-group checkpoint as parent. Require positive gain and
address-attribution intervals over held-out books, not merely an increased
coefficient, before changing the accepted status.

Artifacts:

- `source/fa1_r749_address_reliability.py` — additive module and controls.
- `source/fa1_r749_cpu_pilot.py` — TRAIN-only one-optimizer continuation.
- `source/fa1_r749_large_dev.py` — independent-reload DEV audit.
- `source/fa1_r749_address_sweep.py` — read-only DEV stress test.
- `source/fa1_r749_integrity.py` — inherited state/moment audit.
- `results/FA1_R749_CPU_CONTINUATION_STATUS.json` — machine-readable status.
- One cumulative recovery ZIP: only current R7.49 checkpoint, source,
  research notes, provenance, corpus split identifiers, and SHA-256 manifest.
