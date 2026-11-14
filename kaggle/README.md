# Kaggle: frontier4-cache on WikiText-103 + FineWeb-Edu

This branch is set up for a **2-hour, one-GPU Kaggle run**.

## Notebook

`kaggle/frontier4_fineweb.ipynb`

Open it in Kaggle, or push it with the Kaggle API. The notebook:

1. Checks the GPU and installs `datasets`.
2. Clones this branch.
3. Downloads WikiText-103 raw and splits it into article documents.
4. Streams a FineWeb-Edu subset (`HuggingFaceFW/fineweb-edu`, `sample-10BT`).
5. Runs a 5-minute large-model smoke test.
6. Trains for 120 minutes with the frontier4-cache engine.
7. Prints best/final bits and top-1/top-5.

Kaggle settings: **Accelerator = GPU T4 x1**, **Internet = On**.

## Model

Default is `MODEL_SIZE=large` (~30M parameters), chosen to fit a single T4.
The `frontier` fallback is the 3.8M model. If the large model OOMs, use
`MODEL_SIZE=frontier` or `BATCH=16`.

## Corpus mix

- `wikitext`: WikiText-103 raw, 4,000 article files, up to 80M chars.
- `fineweb`: FineWeb-Edu streamed subset, 4,000 docs, up to 40M chars.

The engine gives the two categories equal quota weight; with
`CORPUS_CHARS=50000000` that is about 25M characters each. Whole-document
holdout is 10%; the test set is 4,096 examples.

## State and resume

State is written to `/kaggle/working/diamond_state` after every epoch. Within
the same session, rerun the training cell. In a new session, upload
`diamond_state.npz` and `diamond_history.jsonl` to the same directory first.

Keep `CORPUS_CHARS`, `TEST_SIZE`, model dimensions, and the corpus split
unchanged between resume sessions.

## Kaggle API

From a machine with Kaggle credentials:

```bash
kaggle kernels push -p kaggle_kernel
kaggle kernels status shem42/nanlm-frontier4-fineweb
kaggle kernels output shem42/nanlm-frontier4-fineweb -p ./output
```

The `kaggle_kernel` directory is generated locally and contains the notebook
plus `kernel-metadata.json`; it is not part of the corpus.
