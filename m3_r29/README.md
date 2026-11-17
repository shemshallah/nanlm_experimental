# M3 R29 — causal lexical successor memory (experimental)

R29 continues the **single M3 predictor lineage**, not FA1, FA2 or
Terminal-RSI. The canonical predictor in `m3_default` remains **R26**.

## New path

The R26 distribution, learned 14 readout scalars, and all inherited wave,
signed-mirror/triangle, R24 prefix-copy, temporal-successor and byte-context
memories are frozen. A new book-local memory conditions byte successors on:

1. last completed word + separator + currently typed word prefix (32K horizon);
2. previous two completed words + separator + prefix (32K horizon);
3. last word + prefix in a recent 1K window;
4. currently typed word prefix independent of prior words (32K horizon).

All word keys are bounded ASCII-casefolded representations; observed successor
bytes retain exact original casing and raw-byte identity. History advances
**only after** the target byte has been scored. Every memory has explicit
expiry and a document-local `reset`. Four residuals and four nonspace-masked
copies add eight learned scalars. At all-zero initialization the entire new
predictor is bitwise identical to the promoted R26 implementation.

## Full optimizer continuation

`FrozenParentAdam` verifies SHA-256
`deb112d4d9b25cb57bd68f689660550a315032bea42c5e9376dbcdeb4a5c97a9`
from the independently tested R26 training artifact; it also checks the
14 exact coefficients, each group's optimizer step, and the original TRAIN
prior. It **copies** the full inherited 14 Adam groups into the R29 checkpoint
and trains only eight newly initialized groups. R26's weights and moments are
never replaced by a reconstructed approximation.

## Locked evaluation discipline

All books come from NaNROUTER release `corpora-short-stories-0004`, with the
original deterministic shuffled index and the original TRAIN prior.

| Split | Indices | Intended use |
|---|---|---|
| Original R16 prior | 0–7 | unchanged prior |
| Prior R26 fitted/gate/dev | 244–275 | never retrained by R29 |
| Prior R26 independent TEST | 276–299 | never used for R29 selection |
| **R29 TRAIN** | **300–315** | fit eight new readout weights |
| **R29 GATE** | **316–319** | prespecified acceptance gate |
| **R29 DEV** | **320–331** | new held-out development evaluation |
| **R29 TEST** | **332–355** | **sealed: do not load until candidate frozen** |

R29 reports BPB/BPC, nonspace BPB, top-1/top-5, per-book differences,
book-bootstrap uncertainty, lexical hit rates and wrong-successor and
unrelated-history controls. GATE requires improvement over R26 and both
nulls, no nonspace/top-1/top-5 regression, and at least 3/4 improved books.
DEV additionally requires at least 9/12 improved books and a positive
book-bootstrap lower interval. This is development evidence only; promotion
requires a separately frozen candidate and untouched TEST certification.

## Run on CPU

```bash
python -m unittest discover -s m3_r29 -p 'test_*.py' -v
python -m m3_r29.train_dev \
  --corpus-root /path/to/extracted/NaNROUTER-batch-0004 \
  --parent-checkpoint /path/to/R26_full_adam_checkpoint.npz \
  --output m3_r29/results/R29_DEV.json \
  --checkpoint m3_r29/results/R29_full_adam_checkpoint.npz
```

The GitHub workflow `.github/workflows/m3-r29-lexical-train-dev.yml` performs
the hash-verified download, tests and training on `master` automatically.
