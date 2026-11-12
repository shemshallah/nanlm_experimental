# r31 datasets

The corpus binaries are not committed. They are Project Gutenberg text and are
large; this repo keeps only the code and the small metadata/held-out prompt.

## `r30-poincare-nanrouter-books` (Kaggle data dataset)

Private Kaggle dataset used by the kernels:

- https://www.kaggle.com/datasets/shem42/r30-poincare-nanrouter-books

Contents (flat, read-only mount):

```
train.bin          # uint16 byte+EOS tokens, 11,079,654 positions, 43 books
valid.bin          # uint16 byte+EOS tokens, 1,416,273 positions, 5 books
meta.json          # token counts, hashes, EOS id, token dtype
split_manifest.json# book-level train/valid file paths and hashes
```

Source: `github.com/shemshallah/NaNROUTER` release `corpora-sentences-0001`
(Project Gutenberg, one sentence per line). Books were split before
tokenization with:

```sh
python poincare_frontier.py prepare-books \
  --source-dir books --out-dir data --valid-fraction 0.10 --seed 2026
```

## `r31-code` (Kaggle code dataset)

Private Kaggle dataset containing the r31 code and prompt:

- https://www.kaggle.com/datasets/shem42/r31-code

`datasets/r31-code/dataset-metadata.json` is the Kaggle dataset metadata. The
canonical code lives in `../code/`; the Kaggle dataset is a copy used for
kernel mounting.

## Held-out prompt

`datasets/r31-code/heldout_prompt.txt` is the first held-out book passage used
for the 200-byte generations (Wharton, *The Touchstone*):

> It was not that he cared much for the club, but that the remote contingency of having to give it up stood to him, just then, perhaps by
