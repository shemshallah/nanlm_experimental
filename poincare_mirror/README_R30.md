# Round 30 — Poincare Mirror LM on NaNROUTER Gutenberg books (CPU)

**Canonical checkpoint:** `run/latest.pt`, step **2801**, SHA-256 `9ad6a368a9e68f4c1ccff366bee88047f7475f93ccb4d1da1c999db7bddee902`.
Download it separately; model weights and optimizer moments are deliberately excluded from the source ZIP.

## Provenance and recovery caveat

The Round 29 research records cite a step-2,280 weight state, but the **actual latest recoverable model checkpoint file at the start of this turn was step 1,761**. No recoverable step-2,280 weight file was found. We therefore resumed the existing step-1,761 checkpoint and completed 1040 more optimizer updates in place; these are not claimed as an exact replay of historical R29 training. The checkpoint is the same one-model architecture, with its learned model weights, Adam state, RNG streams, and dataset hash retained. No replacement model was initialized.

## NaNROUTER books

Corpus: GitHub Actions `pal-r11-nanrouter-raw-prose-books`, source ZIP SHA-256 `29ee56e59f00d90e47a0597490ef7af81dc1bb4a7f3e980af6dd7c047c4b8e24`. 38 verified Gutenberg-book texts: 12 training, 12 development and 14 test, split at book boundaries. No model gradient updates include development or test books. The full 1,000-book release asset was not transferred into this container; this experiment uses only the verified 38-book subset.

Training: 1,998,359 token positions, development 1,982,895, test 1,988,223. All byte symbols plus EOS remain mapped by the original fixed 257-token vocabulary.

## Model and CPU strategy

21,304,992 parameters; width 192, six layers, 16 attention heads, 32 mirrored pairs (64 signed experts), expert hidden size 256, top-k two mirror pairs per token, S+/S- weights exact. Context grew from 2,048 to 4,096 bytes at step 2,081 without a weight reset. Sparse pair-grouped CPU dispatch, AdamW foreach=False, four CPU threads, with mixed training contexts. All additions/continuations run through a **single** `latest.pt` checkpoint and original Adam history.

1. Steps 1,762–2,081: mixed 256/512/1024/2048-byte inputs (320 updates); gradually opened additional mirror pairs.
2. Steps 2,082–2,401: mixed 256/512/1024/2048/4096 contexts (320 updates).
3. Steps 2,402–2,801: CPU-throughput-oriented context mix emphasizing 256/512/1024 with occasional 2048/4096 contexts (400 updates).

Total new updates: **1,040**. New training target positions: **752,128**. Cumulative counter: **1,128,496**.

## Fixed 256-byte held-out development evaluation (4,096 prediction positions)

| Step | CE, nats/byte | BPB | Top-1 next-byte accuracy |
|---|---:|---:|---:|
| 1,761 | 2.933254 | 4.231790 | 27.4658% |
| 2,081 | 2.391618 | 3.450375 | 33.9844% |
| 2,401 | 2.295426 | 3.311599 | 36.1084% |
| 2,801 | 2.231998 | 3.220093 | 37.9639% |

Fixed-position next-byte CE reduction: 23.91%.

## Independent test split (first evaluation of this continuation)

- 256 context: CE **2.224048**, BPB **3.208623**, top-1 accuracy **38.4644%** on 8,192 byte positions.
- 4096 context: CE **2.389650**, BPB **3.447536** on 8,192 separately sampled positions; not directly comparable by target identity.
- Unigram test reference at 256 context: CE 3.128174 nats/byte.

## Same-target long-context comparison — important negative result

Using the EXACT SAME 4,096 held-out development target bytes from the ends of 4,096-byte windows:

- 256 preceding bytes: **3.516644 BPB**
- 4,096 preceding bytes: **3.829048 BPB**
- Extra context worsened BPB by **+0.312404** on this aligned comparison. Do not claim longer context improves quality yet. Possible causes include insufficient 4K updates or a context integration weakness; the test does not isolate causality.

## Run another CPU continuation

Requires Python 3.10+, PyTorch CPU, NumPy, the separately supplied canonical checkpoint at `run/latest.pt`, and the 38-book tokenized NaNROUTER dataset at `nanrouter_data/`.

```bash
STEPS=200 THREADS=4 RUN_DIR=/path/to/run DATA_DIR=/path/to/nanrouter_data bash scripts/run_cpu_r30.sh
python scripts/eval_aligned_context_r30.py --data-dir /path/to/nanrouter_data --run-dir /path/to/run --windows 16
```

**Regression checks:** `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m unittest discover -s tests -q` — 20 passing tests, reported in `logs/R30_ALL_TESTS.log`.

The model is learning byte regularities on book-level held-out text but is not yet fluent, competitive with modern frontier models, or shown to outperform matched parameter/compute dense or MoE baselines. More books, explicit data overlap auditing, and same-target long-context improvements are priorities before larger scale-up.
