# FA2 R39 — causal, content-stratified byte diagnostics

**Status: verified source integration only. The real R38 step-4,057 checkpoint
has not been recovered or continued, and there are no new independent-book
language-model results in this round.**

## Why this is needed

The R38 DEV results improved overall next-byte CE while regressing slightly on
non-space CE. The existing trainer logs overall cross-entropy but lacks
top-1/top-5, bits/byte and content-stratified accuracy at each training
evaluation. A separate whole-book geometry comparison includes these metrics,
but that does not give the optimizer continuation an immediate diagnostic.

The new opt-in `--report-byte-metrics` flag logs `byte_metrics` alongside the
inherited fields (`validation_ce`, `validation_h2_ce`, etc.). It uses precisely
the same fixed causal validation batches as those fields. No training loss,
parameters, gradients, checkpoint schema, or Adam history are altered by the
new evaluator.

Each prediction horizon **next (+1), plus2 (+2), and plus3 (+3)** reports:

* Unweighted cross-entropy (nats/byte), bits/byte, top-1 and top-5 accuracy.
* Target stratification into `all`, `space` (ASCII 32), and `nonspace`
  (every other target, including EOS if present), with exact position counts.
* A `null` subset when it has no positions, never a fabricated score.

The horizon labels use the same causal model input and offset target positions
as `MirrorLM.loss_with_future`: only `batch[:, :-3]` enters the model; +1, +2
and +3 targets are exclusively used in scoring. Future horizon scores are
computed even if their training weights are zero, so diagnostics remain
comparable across old and newer configurations.

## Invocation

After verifying and separately copying an authenticated full checkpoint to an
**unpromoted candidate** run directory, run ordinary training with the flag:

```bash
python poincare_frontier.py train \
  --data-dir data --run-dir candidate_run \
  --steps 100 --train-context 128 --eval-context 128 \
  --nonspace-loss-weight 1.5 --report-byte-metrics \
  --cpu --cpu-threads 2
```

The invocation is illustrative; do not run it from an unverified ancestor or
without the exact matching R38 source. **`train --steps 0` still writes the
run's latest checkpoint. Never point a trial at the untouched original.**

## Interpreting results and promotion gate

The validation batches are sampled windows, not a census of whole development
books. Their top-1/top-5 and BPB are diagnostic, **not** confidence intervals
or a substitute for `scripts/compare_geometry_books.py`. For promotion require
the R38 whole-book non-space CE confidence gate and a separate matched correct
versus corrupted-history counterfactual on the *actual* successor-memory
implementation. Retain +2/+3 and independently aligned longer-context gates.
Do not touch sealed TEST books during candidate selection.

No model checkpoint, source module implementing R38 successor memory, or
reproducible R38 training corpus was fabricated or substituted in this round.

## Verification

```bash
cd poincare_mirror
python -m unittest tests.test_r39_byte_metrics tests.test_r38_nonspace_gate \
  tests.test_compare_geometry_books -v
python -m unittest discover -s tests -q
```

The tests include all three horizon alignments, correct target-space splits,
bits/byte conversion, exact agreement with the inherited unweighted CE,
gradient/parameter invariance, and opt-in CLI logging through a two-stage
Adam continuation of a disposable tiny CPU model.
