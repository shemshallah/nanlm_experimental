# FA1 R7.50 — exact-address inhibitory memory: tested, not promoted

Author: Justin Howard-Stanley, Independent Researcher, Santa Cruz, California, USA.

**Result:** The new sparse inhibitory address value bank is causal and
trainable, but the DEV gain is inconclusive and exact-address attribution
does not pass the required gate. Keep R7.49 as the stronger experimental
checkpoint; R7.50 remains an immutable, unpromoted research candidate.

## Starting checkpoint and preservation

- Full R7.49 unpromoted checkpoint SHA-256:
  `b8dabe54e2417a75defd01c776606367a3f3b0f18f479b032f8390b6fc87ddb5`.
- The ONE R7.49 model has 29 optimizer groups with all prior Adam moments.
- R7.50 adds only optimizer group 30: 32,640 scalar inhibitory pair values,
  a causal apex-descended context gate, a global inhibitory bias, and a
  bounded output gain. No independent LM replaces the inherited decoder.
- The migration starts with gain exactly zero. A real checkpoint-backed
  test confirmed **bitwise identical** R7.49 word predictions.
- Only the five new parameter tensors in group 30 received gradients.
  A saved-checkpoint integrity audit verified **466 inherited model tensors,
  all 29 inherited optimizer groups and all 444 inherited AdamW states
  bitwise unchanged**. The R7.49 checkpoint hash did not change.

The causal exact-address read uses only accepted candidate-word head pairs
and eight pre-target active points. A candidate is inhibited only if its
immutable pair occurs in the 28 active edges. A learned scalar stored at
the exact `C(256,2)` address and pre-target descended-point reliability
decide the magnitude. The model's normalized copy distribution is preserved.
Gold labels are used only in TRAIN loss, never in inference-time routing.

## Why this experiment was attempted

R7.49's exploratory DEV sweeps found that negative exact-match coefficients
could outperform positive copy bonuses. R7.50 therefore examined whether
this negative evidence could be genuinely learned, not hard-coded.

TRAIN-only diagnostic samples (512 positions each) confirmed that a stronger
negative global coefficient lowered TRAIN next-word NLL. Importantly,
they also showed an apparent contradiction: exact candidate-pair matches
were **1.50x–1.66x as common among gold labels** as other candidates.
The likely explanation is *calibration*: raw association need not make a
further positive logit bias beneficial when an inherited attention system
already overweights correlated candidates. This is a working hypothesis;
the diagnostics do not prove its mechanism.

## Training protocol

- NaNROUTER TRAIN 270 books; DEV 15 disjoint books, all verified by
  identity manifests. The sealed TEST split was not read.
- 96 CPU updates executed against the *same inherited model*; 1,024
  sampled TRAIN positions and 256 sampled DEV positions for early selection.
- Training loss combined actual full next-word NLL, a small copied-history
  auxiliary, correct-vs-substituted address contrast, and learned row-permute
  contrast. No future-token information enters the model features.
- DEV word NLL progressively worsened as the new inhibitory gate trained.
  Its best exploratory checkpoint was the **first** update. No late
  checkpoint was promoted or silently substituted for the selected one.

Selected R7.50 checkpoint SHA-256:
`4742c3e6d91f0f629cbd594586f3d4caa570b3d64c882232ad42dc611e8752dc`.

## Independent checkpoint-reloaded DEV, 512 positions, all 15 books

| Measurement | R7.49 parent | R7.50 candidate | Assessment |
| --- | ---: | ---: | --- |
| Next-word NLL ↓ | 5.431276 | 5.431263 | +0.000013 nats/word |
| 95% whole-book interval for gain | — | [−0.0000296,+0.0000593] | Includes zero |
| DEV books improved | — | 9/15 | Mixed |
| Correct-versus-wrong exact addresses | — | −0.00000294 | CI includes zero |
| Inhibitory-row permutation penalty | — | −0.000000040 | No positive row attribution |
| Correct-history advantage | — | +0.047615 | Historical signal retained |
| First UTF-8 byte top-1/top-5 auxiliary | — | 10.16% / 38.87% | Unchanged |
| First UTF-8 byte NLL | — | 3.211958 | Auxiliary only |
| Word scoring throughput | — | ~3,025 positions/s | Two-thread CPU |

Exact-address whole-book attribution 95% interval:
[−0.0000588,+0.0000445] nats/word. Inhibitory row-permutation interval:
[−0.0000000839,+0.00000000047] nats/word. Both fail the strict gate.
Full raw-text BPB, full-system normalized word top-1/top-5 and next/+2/+3
metrics are **not** exposed by this checkpoint-backed FA1 protocol.

The selected checkpoint's learned inhibition strength has tanh-normalized
gain 0.4526 and a mean per-candidate logit penalty 0.0100. It therefore
does perform nonzero exact-address computation, but that fact alone is not
proof that address values make language predictions better.

## Decision and next bottleneck

**R7.50 rejected.** At most it provides a source-verified, recoverable
implementation of an exact-address inhibitory memory. The experiment
does not establish its *predictive necessity*. R7.49 remains the latest
better-supported experimental model, not an accepted frontier checkpoint.
The accepted FA1 R7.13 step-14,416 ancestor remains protected.

A future round should learn **calibrated bidirectional evidence** on a much
larger TRAIN-only sample, conditional on copy confidence and context
reliability. It must compare actual independent-book word NLL against a
matched score-normalization-only control, with per-address and swapped-
address controls and long-context safety. No increase in parameter count
or model scale alone should count as geometric attribution.

Evidence: `source/fa1_r750_calibrated_inhibition.py`,
`source/fa1_r750_inhibitory_diagnostic.py`,
`source/fa1_r750_cpu_pilot.py`,
`source/fa1_r750_large_dev.py`,
`source/fa1_r750_integrity.py`,
`results/FA1_R750_CPU_CONTINUATION_STATUS.json`, and the cumulative
research recovery archive (one current unpromoted candidate checkpoint).
