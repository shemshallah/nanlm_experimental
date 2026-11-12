# FA1 R7.15 — query-conditioned mirror writer, real-text and synthetic CPU controls

**Checkpoint decision: NO PROMOTION.** The accepted FA1 rank-64 temporal/hierarchical model remains at **14,416 joint updates**, SHA-256 `54e835b00642013d4f1bf486a8150eb1ff06c547565771bd21933e143d050f2d`. It contains five original optimizer histories. FA2/FA2.2 is a separate lineage and was not touched.

## Hypothesis
R7.14 showed that better lexical retrieval is insufficient for better next-word prediction. R7.15 tests a single causal rank-64 key/value mirror constructed from selected older history transitions, with symmetric (S+) and antisymmetric (S-) read rails and a query-conditioned LM trust gate. New key/value projections, query projection, mirror scoring and trust gate receive **direct next-word LM NLL gradients**, but the accepted original model and its optimizer moments remain frozen; an experimental fifth-optimizer extra group records candidate updates. This is *not* a full-trunk joint update.

## Local CPU experiments

### A. Historical natural-text LM — 128 candidate updates

- 128 training, 64 development and 64 historical test samples; target-free top-96 historical transitions selected from the causal 6,144-token older region; two CPU threads.
- Historical development **original model** NLL **4.618318**. Initial experimental mixture NLL 4.618478. No dev-selected candidate satisfied lower NLL and positive correct-vs-unrelated-history advantage; the experimental model collapsed to a fallback gate-off baseline.
- At step 128 (UNSELECTED), candidate development NLL 4.845736; wrong-vs-correct memory advantage +0.048118 is insufficient, because the candidate badly worsens the LM.
- Accepted checkpoint remains unchanged; final gate-off historical test NLL **5.244240** matches original.

### B. Context-only causal cache opportunity — 256 train and 256 dev examples

- On development examples, a valid 1-gram match in selected older history was present 82.81% of the time, and 2-gram match was present 35.94%.
- Across 20 predeclared constant-gate combinations (1/2-gram, minimum match count 1/2, five gate strengths), none improved development LM loss or passed correct-vs-unrelated-history control. The least-harmful combination still had dev gain **-0.0000325 NLL** and memory advantage **-0.0109962 NLL**.
- An unrelated donor's entire 6,144-token older history is shifted and **top-k is recomputed using the original query**, not just shuffled preselected arrays.

### C. Synthetic long-range key/value copy — 128 updates

- Synthetic random token-ID queries and values, 256 training/128 development/128 test. Disjoint query-ID ranges; one planted key/value occurrence 2,049–8,192 tokens before the target; ten embedding-nearest hard-negative keys; fresh test query IDs.
- A valid 32,000-way uniform-token baseline has **10.373490 NLL**, but this is **not FA1's trained natural-language baseline**.
- Best selected step 112; held-out synthetic test NLL **1.078168**, vs 10.373490 uniform. Correct-vs-unrelated-history NLL difference **+9.722410**. This is a retrieval sanity check, NOT evidence that FA1 improved on natural language.
- Removing both S+/S- read scores increased synthetic test NLL to **1.167359**; removing the explicit two-token exact-context score increased it to **1.210039**. The exact-match preselector itself contributes substantially; this experiment does not establish semantic or open-domain generalization.

### D. Synthetic-trained writer applied to historical real-text development

- 256 historical dev examples with no real-text adaptation: original FA1 NLL **5.068875**; synthetic-trained reader NLL **5.156073** (worse by 0.087198).
- Correct memory beat unrelated memory by 0.020677 NLL, but **both were worse than the original**. Thus memory discrimination is not enough: the trust gate is insufficiently calibrated for actual language.

## Checks and limits

- Five new structural/causality tests passed: finite likelihood, nonzero writer/reader/gate gradients, unchanged inherited baseline under wrong-memory control, retrieval selection invariant under target change, exact gate-off fallback.
- Old full checkpoint SHA-256 checked before and after training; no inherited optimizer state modified, and no FA2 source or weights modified.
- The historical train/dev/test corpora are **not certified as book-disjoint independent NaNROUTER texts** and were repeatedly consulted in earlier releases. None of these results can be claimed as independent new-book generalization.
- The NaNROUTER book corpora are published as **GitHub Release assets** (`corpora-short-stories-0001` etc.), not as files under `corpora/`. No new-book corpus asset was incorporated in this local test.

## Promotion requirements for next round

Maintain a single integrated FA1 checkpoint and all five optimizers. To promote a future R7.16 candidate: (1) improve historical dev NLL (and preferably non-space NLL) beyond the unmodified accepted baseline; (2) beat an independently rerouted unrelated-history control; (3) confirm on sealed, deduplicated, independently imported NaNROUTER books; (4) prove invariant gate-off baseline and optimizer continuity. Synthetic-only gains never count toward model promotion.

## Reproduction

Start from `FA1_R7p13_HIERARCHICAL_SINGLE_CHECKPOINT_FULL.zip` (accepted base) and overlay this round's `source/` and `reports/` into its directory. Install the inherited Python dependencies (PyTorch, NumPy, sentencepiece). From the directory containing `checkpoint/`, run:

```sh
python source/fa1_r715_mirror.py --root . --train 128 --dev 64 --test 64 --updates 128 --eval-every 16 --topk 96 --threads 2
python source/fa1_r715_opportunity.py --root . --train 256 --dev 256
python source/fa1_r715_synthetic.py --root . --train 256 --dev 128 --test 128
python source/fa1_r715_transfer.py --root . --dev 256
python source/test_fa1_r715.py
```

The source/reports release contains SHA-256 checksums. The 90-MB accepted model is intentionally not duplicated in the small round ZIP.