# FA1 R7.58: compositional temporal geometry

R7.57's signed address was a deterministic function of a *single* previous
word. It could not encode a new temporal relationship; its DEV gain against a
matched lexical control was −0.00000915 nats/word.

R7.58 pairs the second 256-head endpoint of word `t−2` with the first endpoint
of word `t−1`, producing `address = 256*older_head + recent_head`.
The complete 65,536-cell directed square contains 32,640 mirror pairs and
256 fixed points; transpose is an explicit counterfactual.

From TRAIN-only, document-contained trigrams the model stores two independent
normalized successor distributions:

1. `Q_geom(next | temporal mirror address)` — cross-word geometric pooling.
2. `Q_exact(next | two exact preceding words)` — strong matched lexical control.

The existing normalized R7.57 prediction and all 31 inherited optimizer groups
remain unchanged. One new 32nd AdamW group contains three gate tensors.
The target-free 32D apex summary and support counts choose mixtures up to 18%:

`P_R758 = (1-e-g)*P_R757 + e*Q_exact + g*Q_geom`.

Attachment with zero gate is bitwise equivalent to the original predictor.
Supervised TRAIN events are removed from both the R7.57 and R7.58 count banks
before scoring. DEV controls include exact-bigram-matched weight,
geometry-matched weight, wrong prior word, mirror swap, wrong successor value,
no apex, and disabled R7.58. The strict gate requires positive book-bootstrap
95% lower bounds versus the parent, exact lexical control, wrong-older history
and wrong geometric values. TEST remains sealed and promotion is disabled.

The real starting point is the complete unpromoted R7.57 candidate with
31 optimizer groups from the R7.57 cumulative research archive. It has a
291 MB checkpoint and intact tokenizer metadata. The source for the original
270 TRAIN and 15 DEV books is NaNROUTER Release `corpora-short-stories-0004`
(`short_stories-0004.tar.zst`, verified SHA-256
`d6ce3a23757e85fe1b9487ef984a5ba81db9c2dc6e962ac7c5df2960480e0054`).

Recover TRAIN/DEV without extracting or tokenizing TEST:

```
python fa1r7.12/source/fa1_r758_restore_train_dev.py \
  --archive short_stories-0004.tar.zst \
  --root recovered/root \
  --metadata recovered/split_metadata \
  --destination recovered/processed
```

Then train and evaluate a new, protected candidate:

```
python fa1r7.12/source/fa1_r758_cpu_pilot.py \
  --root recovered/root \
  --corpus-dir recovered/processed \
  --split-metadata recovered/split_metadata \
  --checkpoint recovered/checkpoint/FA1_R757_1000_UNPROMOTED.pt \
  --output-checkpoint out/FA1_R758_UNPROMOTED.pt \
  --report out/FA1_R758_DEV.json --threads 4 \
  --train 4096 --dev 2048 --context 512 \
  --steps 1000 --eval-every 250
```

Do not claim raw-byte BPB or full word top-5 for this inherited word NLL API.
No automatic +9,000 training extension until the strict geometric DEV gate.
