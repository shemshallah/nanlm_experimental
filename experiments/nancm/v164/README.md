# NaNCM v164 — Signed Cloud + Hierarchical Distributional Worldlines

Author: **Justin Howard-Stanley**, Independent Researcher, Santa Cruz, California, USA.

## Status and lineage

**Native C++ experimental successor, not production-certified.** v158 remains the accepted development baseline and v148 remains the protected fallback. The original 128+128 signed-cloud model, 32D upper/lower/S+/S− causal extractor, neural word/spelling heads and their complete optimizers are retained without reinitialization or modification. v162's reliable single-word pointer and v164's trainable distributional word memory now compose probabilistically in a **single native inference path**. The small v163 signed-cloud confidence gate is archived as a separate ablation and is OFF in v164's accepted diagnostic configuration.

## What is new

1. **500 independent training books**: NaNROUTER GitHub Release `corpora-short-stories-0004` SHA-256-verified as `d6ce3a23757e85fe1b9487ef984a5ba81db9c2dc6e962ac7c5df2960480e0054`. Deterministic book partition: 500 TRAIN / 51 DEV / 64 TEST. Near duplicates and overlap with the much older inherited model's entire pretraining lineage have not been completely ruled out.
2. **199,956 conditional address keys and 733,069 candidate entries**, from 1-, 2- and 3-word causal contexts (at most eight words per key); votes from multiple context orders are calibrated by support and context order. The final predictive distribution has at most 16 complete-word candidates.
3. **Two normalized whole-word mixtures**: `P164(w | h) = (1 - a164) P162(w | h) + a164 Q164(w | h)`, where P162 already includes its original single-word causal pointer. The 27-symbol native prefix/EOS API returns the conditional derivative of this exact word mixture, preventing repeated per-letter penalties on long words and novel vocabulary.
4. **Causal domain arbitration** using only completed preceding words. When the score is below 1.5, the new book specialist is disabled. The entire historical documentation DEV+reused TEST fixture (4,545 checked prefixes) yields byte-for-byte identical probability outputs to v158.
5. **Checkpoint fingerprint locking** in `checkpoints/v164_native_distribution.tsv`. Its header is bound to the inherited native trained model/geometry/cloud fingerprint. The remainder is the original GitHub-trained TSV byte-for-byte; removing the first header line reproduces the GitHub artifact SHA-256 `29258157c99d5bdfd5a4fd2622101d2e143f4d893053606612346b6cc8fccd79`.

## Build and native tests

```sh
make -B v164
OPENBLAS_NUM_THREADS=1 make test-v164
```

The tests check exact opt-out parity, complete-word mixture identity, normalized EOS/letter distributions, no future target access, fail-closed index and settings, 4,545 exact historical documentation fallback vectors, and six inherited trained model/optimizer hash commitments. On this session the C++ build and these checks passed. Android/Termux cross-compilation itself remains pending.

## Recommended research-only native configuration

```sh
export V155_HEAD_STRENGTH=.05 V155_PROTECT_TOP1=1
export V155_VOCAB_HEAD="$PWD/parent/checkpoints/native_cloud155/v155_vocab_head.bin"
export V156_OPEN_SHARE=.35 V156_SPELLER="$PWD/parent/checkpoints/speller156.bin"
export V157_JOINT_HEAD="$PWD/checkpoints/v157_joint_native.bin"
export V157_WORD_SHARE=.04 V157_CHAR_SHARE=.10
export V158_POINTER_THRESHOLD=.8 V158_POINTER_BUDGET=.35 V158_TOPIC_BUDGET=.015
export V159_BYTE_BUDGET=0 V159_CHAR_BUDGET=0
export V160_ASSOC_BUDGET=.4 V161_MIN_ORDER=1
export V160_ASSOC_INDEX="$PWD/checkpoints/v161_unified.tsv"
export V162_DOMAIN_GATE=1 V162_DOMAIN_THRESHOLD=1.5
export V162_BOOK_FREQ="$PWD/external_book_gate/book_unigrams.tsv"
export V162_DOC_FREQ="$PWD/checkpoints/docs_unigrams162.tsv"
export V164_DISTRIBUTION_BUDGET=.6
export V164_DISTRIBUTION_INDEX="$PWD/checkpoints/v164_native_distribution.tsv"
export V164_DOMAIN_GATE=1 V164_DOMAIN_THRESHOLD=1.5
```

Use `bin/libwordfield164.so`, the unchanged `wf148_open/begin/next/close` ABI. To return to the **exact** v158 development baseline, set both `V164_DISTRIBUTION_BUDGET=0` and `V160_ASSOC_BUDGET=0`, with v159 byte/local-spelling features off. Do not reinitialize checkpoint or optimizer state.

## Paired benchmarks

`results/v164_DEV_FROZEN.json` and `results/v164_TEST_FROZEN.json` contain 480 deterministic scored words each from 24 distinct DEV and TEST books (20 words/book), native 96-beam, up to 20 letters, strict complete-word top-1/top-5/top-32. v164 parameters were selected on DEV, then tested on the reserved TEST **sample**. These TEST windows had been used for previous architecture research, so this is not pristine preregistered certification.

The new v164 candidate improves sampled book DEV top-1 **6.25%→9.375%**, top-5 **13.125%→17.50%**, top-32 **23.125%→30.417%** versus protected v158. On book TEST, top-1 **5.833%→9.375%**, top-5 **15.625%→20.625%**, top-32 **24.375%→32.292%**. Teacher-forced spelling/EOS bits/symbol improved from **3.01121 to 2.88021** DEV and **3.11283 to 2.92464** reused TEST. These are **not full 256-byte BPB**.

Full 256-byte BPB and +2/+3 tests, book-level near-duplicate and inherited-checkpoint provenance screening, full-volume independent-book neural training, Android native testing and the 80% strict word top-1 objective are all **still open**.

See `RESULTS.md`, `ARCHITECTURE.md`, `RESEARCH_LOG.md`, `CLAIM_LEDGER.md`, `PROVENANCE.md`, `PROMOTION.md`, `MANIFEST.json`, and `SHA256SUMS`.
