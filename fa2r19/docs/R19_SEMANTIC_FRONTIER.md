# PAL R19 — adversarial semantic retrieval, CPU continuation research

## Status — FAILED SEMANTIC PROMOTION; accepted R18 retained

No independent language model was started. We read the *accepted* R18 checkpoint,
SHA-256 `dd040106568786e7fbad2a880f4010635b626b51c3697e2ab08712905a2de91f`,
and preserved it byte-for-byte. All previous weights, optimizer histories, and
R10/R11/R12 counters remain in the same single checkpoint. R19 candidate
architectures were trained in RAM, selected on synthetic development splits,
then **not saved** because they do not meet semantic or decoder gates.

### Baseline failure isolated

`source/r19_native_audit.py` constructs random-token streams with one remote
matching prefix. The R18 matcher uses exact two-subword candidate admission.
Consequently, its learned ranker can retrieve the exact key at distances
2,048, 8,192, 32,768, and 65,536, if memory is enlarged; replacing one or
two query-prefix tokens blocks admission entirely. Those replacements are
**synthetic changed-token controls**, not natural-language synonym pairs.
The test demonstrates a structural inability to retrieve reworded keys, not
performance on 65,536-token language understanding. The default trained memory
is 2,048 tokens; the longer lengths are an explicit experimental override.

### Synthetic semantic retrieval — harder, strictly labeled

`source/r19_semantic_frontier.py` has 8 annotated relation families with 4
surface forms each. Train covers 6 families; the remaining two are blind
zero-shot transfer. Synthetic names and values are independent across train,
development, and test splits. Each query retrieves one fact from **eight
shuffled candidates**, with same-entity/wrong-relation and
same-relation/wrong-entity adversaries. Query text never contains gold value.
Four types of probes are reported: (a) held-out paraphrase, (b) held-out names
with familiar phrasing, (c) blind relation families, and (d) oracle-decomposed
two-hop query chains whose second query uses the *actual first retrieved value*.

Two trained candidates shared PAL's existing frozen piece-embedding table and
initialized their Q/K projection from R18. The second uses token-level
late interaction to reduce the information bottleneck from sentence pooling.
The benchmark is **controlled, synthetic, and template-dependent**: it is not
held-out Gutenberg QA or an external general semantic benchmark.

| Metric (320 or 120 frozen examples) | Mean + attention pooling | Token-level late interaction | Untrained overlap control |
|---|---:|---:|---:|
| Dev unseen paraphrases (320) | 36.56% | 49.06% | 53.44% |
| Test unseen paraphrases (320) | 33.13% | **48.13%** | **52.50%** |
| Test paraphrased facts (320) | 28.44% | **49.38%** | see JSON |
| Test unseen names / familiar forms (320) | 51.25% | **69.38%** | 44.38% |
| Test entirely unseen relation families (320) | 29.06% | **37.19%** | 58.13% |
| Two-hop BOTH correct with semantic distractors (120) | 12.50% | **5.00%** | not computed |

The late-interaction candidate reached test 48.13% while a simple token-overlap
rule achieved 52.50%; **no semantic frontier gain established**. The 5.00%
two-hop result is from the *corrected hard-distractor task*. A preliminary
2-hop pilot without same-entity wrong-relation distractors scored 88.33%:
that pilot was **too easy** and should not be reported as reasoning evidence.
The two subqueries are given to the model; it does not autonomously decompose
the original natural-language question or generate the final answer.

A second, 960-update curriculum intentionally exposed a third alias per
relation during training, then evaluated both **unseen query wording against
familiar facts**, and **unseen fact wording against familiar queries**. Its
held-out test scores were 31.25% and 25.00%, respectively, despite improving
on simple token overlap in the former set. That is not a semantic solution.
All three training trials selected their best steps using **development only**.
Test was run only after checkpoint selection inside each run. The three
synthetic benchmarks have common template families and are not sealed against
iterative researcher adaptation.

### Large distance: control for dataset changes

On the **same 256 sentence-level examples**, sparse fact positions were
relabeled to 8,192, 32,768, and 65,536 tokens in the past. With no dense
passage processing, accuracy was exactly 50.0% at every distance for the
late-interaction model. This shows that sparse address labeling is not a
function of distance; it does not show that model computation trained or
remembers 65K preceding tokens. The native R18 sparse-index test also includes
an explicit negative control showing default 2K memory cannot reach beyond 2K.

### Training and counterfactual safety

- The accepted checkpoint SHA-256 remained unchanged before/after each run.
- No R19 optimizer or weights were added to the active checkpoint.
- The 12 structural tests in `reports/r19_structure_tests.json` passed,
  including causal suffix invariance, changed-prefix admission failure,
  train/test nonce separation, gradients, and distance-matched comparison.
- The 11 inherited tensor/optimizer fields were verified by
  `source/r18_verify_lineage.py` against the R16-to-R18 reference hashes.
- R18's historically observed short/long BPB advantage is **retained**, not
  newly improved by these exploratory R19 experiments.

### Minimal CPU commands

From the extracted release root, with PyTorch 2.x, NumPy, and SentencePiece:

```sh
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python source/r18_verify_lineage.py --root .
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python source/test_r19.py --root .
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python source/r19_native_audit.py --root .
# 640-update mean-pool controlled experiment
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python source/r19_semantic_frontier.py \
  --root . --architecture pooled --steps 640 --train-examples 1800 \
  --dev-examples 320 --test-examples 320 --hops 120 --batch 48 --eval-every 80
# 960-update token-level alignment controlled experiment
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python source/r19_semantic_frontier.py \
  --root . --architecture late-interaction --steps 960 --train-examples 1800 \
  --dev-examples 320 --test-examples 320 --hops 120 --batch 48 --eval-every 120
# 960-update richer synthetic curriculum; newly held-out surface on query/fact
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python source/r19_semantic_frontier.py \
  --root . --architecture late-interaction --curriculum --steps 960 \
  --train-examples 2400 --dev-examples 400 --test-examples 400 \
  --hops 200 --batch 48 --eval-every 120
```

The reported run seeds are deterministic but small PyTorch/CPU floating-point
variation can affect tie-breaking and early-step selection.

### Frontier trajectory and hard promotion requirements

1. **Semantic bridge:** train from genuinely diverse *paired meanings* or a
   high-quality frozen external pretrained semantic representation; then
   distill that representation into PAL's **shared** mirror-recurrent state.
   No external model's weights may silently become a competing second PAL.
2. **Ground facts:** extract non-templated long-document QA and multi-hop
   supervision from new author-disjoint NaNROUTER Gutenberg or domain releases;
   verify source provenance, duplicates, and lexical overlap, and freeze test
   data before candidate training.
3. **Unified decoder:** route verified semantic fact values into the existing
   autoregressive PAL decoder through a zero-initialized, learned gate. Prove
   exact old-logit identity at zero gate, no future leakage, and improved
   downstream BPB / answer accuracy versus equal-compute controls.
4. **Hierarchical memory:** chunk/landmark/semantic memories should be built
   causally as documents stream through; test 8K/32K/65K from **actual text
   conditioning** (not relabeled sparse fact slots), including contradiction
   handling, stale facts, and recency-vs-relevance.
5. **Evaluation gates:** semantic paraphrase, new relation/format, semantic
   distractors, two-hop, exact vs neural retrieval, real-book cross-entropy,
   word/future regression, latency/memory. Run matching controls and 3 seeds
   before promoting weights. Do not use historical test books to tune gates.
6. **Compute curve:** only after the semantics/decoder bottleneck is fixed,
   benchmark 3 sizes by 3 token budgets on comparable fresh corpora against
   matched transformer, SSM and sparse-retrieval baselines.

**Decision:** keep R18 as the only accepted trained model, retain hard-negative
training, and move the R19 semantic adapter and audit harness into the research
source tree. Do not claim frontier capability, generalized semantic retrieval,
or long-document reasoning yet.
