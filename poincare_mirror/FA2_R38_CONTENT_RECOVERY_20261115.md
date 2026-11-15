# FA2 R38 — non-space regression recovery gate

**Status: source-level candidate only; not a newly trained R38 checkpoint.**

The reported R38 1,000-update continuation reached step 4,057 from 3,057,
retaining optimizer state. On twelve DEV books at context 128 it reduced
overall +1 CE 2.379655 to 2.370394 nats/byte, but increased **non-space**
CE 2.650729 to 2.653181. Correct-book history outperformed corrupted
history for overall CE by 0.014465 nats/byte but not for non-space CE.
The +2 and +3 CE values remained unchanged. These are **reported prior
results**, not remeasured by this source change. The step-4,057 candidate
remains unpromoted.

## Diagnosis and bounded next experiment

Space is one ASCII byte (32). A causal successor store may be accurate on
high-frequency whitespace and local repetitions yet produce weak or misleading
retrievals on content bytes. The sign reversal of the non-space history control
is a specific warning that memory trust should not increase uniformly.

This round adds one opt-in training knob to the existing `MirrorLM`:

`--nonspace-loss-weight 1.5` increases the contribution of **observed target
bytes other than ASCII space**. It is applied only to the training objective,
not to inference or ordinary validation CE. Weights are normalized by their
sum, so the gradient scale stays approximately comparable across batches.
The default `1.0` uses the original, bitwise-identical cross-entropy call.
No new model parameter, optimizer, or independent backbone is created.

The knob does **not** constitute a fix for the separate R38 memory-control
regression. It tests whether existing trainable paths can allocate gradient to
content rather than relying disproportionately on whitespace. Its effect
on non-space DEV books and correct-vs-corrupt history must be measured.

## Source comparison now prevents space-only promotion

`scripts/compare_geometry_books.py` continues to compare the **same book-safe
windows** under before, after, and disabled reciprocal-feedback control.
It now reports additional +1 metrics for space and non-space separately:

* CE in nats/byte, top-1, top-5 for each subset.
* Whole-book paired bootstrap 95% intervals for each metric.
* A stricter development gate requiring positive non-space CE confidence
  lower bound, nondecreasing non-space top-1/top-5, and positive non-space
  benefit from the reciprocal geometry ablation.

The old overall gate remains in effect. One book is a resampling unit; targets
are never independently bootstrapped as though they came from independent
books. No test split can pass the development selection gate.

**Important:** This geometry comparison only ablates the reciprocal pathway
present in this repository. It is **not** a substitute for the R38-specific
correct-history versus corrupted-history counterfactual. The latter must be
replayed on matched causal positions using the original R38 source.

## Required continuation protocol

1. Recover the exact R38 step-4,057 checkpoint with model weights, full Adam
   states, RNG, config, and data hashes, **and its matching R38 successor-memory
   source**. No checkpoint recovery was possible in this environment; the
   GitHub `master` source currently does not contain that R38 successor module.
2. Hash the artifact, verify all optimizer moments and compare R38 config and
   state keys before any load or update. If this fails, stop; do not restart
   from R30 or replace the candidate with a fresh model.
3. Preserve an untouched step-4,057 snapshot. In a recoverable candidate copy
   run a short (e.g. 100–250 update) continuation with training weight 1.5,
   or an equivalent integrated R38 training option. Require constant book
   splits, seed, train sampler settings, optimizer history, and horizon heads.
4. Score the same DEV positions at context 128, then independent aligned
   512/2,048/4,096-byte contexts where supported. Report all, non-space, and
   space CE/BPB/top-1/top-5 plus +2/+3 loss and accuracy.
5. Re-run correct-book, wrong-book and corrupted-history controls at identical
   positions with every write strictly after its observation. Record a
   whole-book bootstrap CI for the **non-space** correct-vs-corrupt gain.
6. Do not promote unless all +1 and non-space DEV CE gates pass, +2/+3 and
   long-context results do not regress, and correct-history non-space gain is
   positive with book-bootstrap lower CI greater than zero. Use sealed test
   books only after selecting the candidate.

## Reproducible source checks

```bash
cd poincare_mirror
python -m unittest tests.test_r38_nonspace_gate tests.test_compare_geometry_books -v
python -m unittest discover -s tests -q
```

The tests verify that default objective values are unchanged, target weighting
is correctly normalized and differentiable, validation CE remains ordinary
unweighted CE, and an intentionally space-only improvement is **rejected** by
the non-space book-level gate. They are synthetic correctness tests, not a
new language-model benchmark.
