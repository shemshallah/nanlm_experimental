# FA1 R7.22 — unified 1M associative memory, CPU validation

**2026-11-12 | one R7 model on `master` | identity migration only | not promoted.**

R7.22 retains the inherited 96D R7 trunk, rank-64 temporal mirror and directed writer, inactive calibration, frozen R7.16 near (8K/32K) and frozen R7.18 packed far (32K–128K) readers. Two embedding-associative readers are added **inside the same model**: 32K–128K (stride 8) and 128K–1,048,576 token distances (stride 32). They use normalized learned 96D R7 word embeddings, a weighted four-token key, fixed 10-bit locality-sensitive hashes with one-bit neighbors, cosine-ranked past candidates and age decay. Candidate selection and gate features do not use the predicted target. Distinct document boundaries, temporal exclusion bands, and target-only scoring are enforced.

Both new `SemanticGate` modules have nonnegative, bounded probability-mixture weights capped at 12.5% per tier. Their gates are exactly zero at initialization; ordinary inference skips both indexes until a training-selected gate is positive. Explicit `probe_memory=True` measures candidate access without altering LM outputs. **Embedding similarity is not itself demonstrated semantic understanding, nor is a 1M address window equivalent to 1M attention/reasoning.**

The one 94.8 MB R7.22 checkpoint carries **105 parameter tensors and five inherited optimizer histories**. A seventh group was appended to the *fifth* original AdamW optimizer for both semantic heads. Every older parameter and Adam moment was retained bitwise; no sixth optimizer and no new Git branch. Accepted R7.13 step 14,416 remains untouched.

## Measured results

| Test | Result |
|---|---|
| Synthetic causal/age, document-boundary, future-safe, paraphrase-style and 1M incremental checks | **15/15 passed**; synthetic capability only |
| Actual historical test prediction parity with R7.21 | **128/128 positions; max NLL difference 0.0** |
| 128K associative candidate coverage in those positions | **103/128**; new gate remains off |
| Real historical train positions at 1,350,000–1,350,063 | **64/64** with 1M-tier candidates; original NLL exact |
| Standalone reload without R7.13/R7.21 checkpoint files | **105/105 tensors and 5/5 optimizer histories bitwise preserved**; metadata still required |
| 48 semantic-gate CPU updates; 96 train, 64 historical development examples | **Rejected**; selected updates 0; correct dev NLL **5.230860** unchanged |
| Short associative target probability better than inherited predictor | **7/64** dev examples, insufficient to open gate |
| Long 1M associative target probability better than inherited predictor | **11/64** sampled historical train positions, not a generalization result |

CPU benchmark (64 contiguous old training positions, both execution orders): inherited R7.21 **0.52–0.60 s**, R7.22 with closed gates **0.46–0.53 s**, and explicit 1M candidate probing **0.69–0.83 s**. Small closed-gate timing variations may reflect caching, not a proven throughput improvement. Probing memory adds cost and is optional.

Causal negative controls in the gate pilot included original old-past word-frequency probabilities and separate training-split donor histories (no future evaluation token access). The trained gate failed to demonstrate both a development gain and a correct-history advantage, so *all 48 updates were rejected*. The zero-gate checkpoint was retained.

## Frontier trajectory: required evidence, not a claim of achievement

R7 remains a small-capacity model. Independent NaNROUTER whole-book evaluation with byte-accurate BPB, document-boundary and contamination checks, correct-vs-shuffled/donor-history controls and book-cluster confidence intervals is **not yet complete**: the verified full checkpoint is not available to the GitHub Actions runner. A 1M source sequence on the historical training split is not independent long-document QA evidence. Longer NaNROUTER documents, rather than short stories that lack a million tokens each, are required to evaluate the farthest tier.

The next experiments should continue this one R7 checkpoint: learn the contextual embedding projection and calibrated utility under hard lexical and semantic distractors; perform identity-preserving 96→192→384 hidden and rank64→128 memory widening, carrying optimizer moments; train the enlarged model on diverse legally usable whole documents; then measure held-out BPB, correct-history advantage, generalization and multi-hop transfer. Old tests must not become training-selection tests. No automatic promotion.

## Reproducibility

```bash
export PYTHONPATH="$PWD/source:$R7_LEGACY_SOURCE" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
python source/test_fa1_r722.py
python source/test_fa1_r722_real.py --root "$R7_ROOT" --ancestor "$R721_CHECKPOINT" --candidate "$R722_CHECKPOINT"
python source/test_fa1_r722_long_real.py --root "$R7_ROOT" --ancestor "$R721_CHECKPOINT" --checkpoint "$R722_CHECKPOINT"
python source/test_fa1_r722_standalone_recovery.py --root "$R7_ROOT" --checkpoint "$R722_CHECKPOINT"
python source/fa1_r722_train.py --root "$R7_ROOT" --candidate "$R722_CHECKPOINT" --train 96 --dev 64 --steps 48
```

Source-only GitHub CI runs inherited and new synthetic tests; real checkpoint tests above ran locally. R7 source ZIP contains no learned weights, binary database, corpus archive, or nested release archive.