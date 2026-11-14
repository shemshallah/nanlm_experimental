# Colab: frontier4-cache on WikiText-103

Open the notebook from GitHub:

`https://colab.research.google.com/github/shemshallah/nanlm_experimental/blob/colab-wikitext/colab/frontier4_wikitext.ipynb`

## Why WikiText-103

NaNROUTER publishes WikiText-103 raw, but **not FineWeb**. The only FineWeb
artifact in `nanlm_experimental` is a checkpoint release, not a text corpus.
The notebook therefore uses the SHA-pinned WikiText-103 raw train split and
falls back cleanly if you provide another `.txt`/`.jsonl` corpus directory.

## What it trains

- Engine: `frontier4-cache` (frontier4 sentence tower + causal first-word cache)
- Corpus: WikiText-103 raw, split into article-level documents
- Split: whole-document holdout, 10%
- Context: 2048 bytes / 256 words / 128 sentences x 16 tokens
- Defaults: 100 epochs, 15,000 examples/epoch, batch 32, LR 1e-3, cosine,
  dropout 0.05, weight decay 0.005, label smoothing 0.05
- Session budget: 90 minutes; state is saved every epoch to Google Drive

The checkpoint signature includes the corpus hash, test size, and model
dimensions. Keep those fixed between resume sessions. `EPOCHS` is the total
target, not the number of epochs in one Colab session.

## Files

- `prepare_wikitext.py`: stream-splits `wiki.train.raw` into one file per
  article under `wiki_corpus/wikitext103/`.
- `train_wikitext.sh`: the exact engine command; all settings are environment
  variables.
- `report_metrics.py`: prints best/final bits and byte/word/sentence top-1 and
  top-5.
- `frontier4_wikitext.ipynb`: the Colab notebook.

## FineWeb option

If you have a FineWeb text directory, point `CORPUS_DIR` at it and skip the
download/split cells. The engine reads `.txt`, `.jsonl`, and `.json` files.
For a JSONL FineWeb shard, the loader looks for fields such as `text` or
`content`; parquet is not read directly.

## Reading the metrics

- `bits`: average negative log2 probability per target; lower is better.
- `top1`: exact argmax accuracy.
- `top5`: whether the correct target is in the five highest-probability words.

Top-1 is coarse for a 2,001-way next-first-word task; top-5 and bits are the
more sensitive indicators. Use the 4,096-example test set for stable numbers.
