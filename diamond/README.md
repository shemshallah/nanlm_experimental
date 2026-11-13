# Diamond engine - C(256,2) / C(256,3) on NaNROUTER corpora

Single-file CPU engine: three 256-expert MoE towers (byte / word / sentence),
staged bottom-up then two-way cross-conditioning, with a 1-dim introspection
head. Originally written for a 6 GB GitHub runner and `/usr/share/doc`; this
version ingests the NaNROUTER corpora tree and is built for long, resumable
runs.

## Corpus ingestion

`diamond_engine.py` discovers roots from, in order:

1. repeated `--corpus-dir` flags,
2. `NANROUTER_CORPORA` (colon separated),
3. `NaNROUTER/corpora`, `nanrouter/corpora`, `corpora` relative to the
   workspace / home / script directory.

It reads released `.txt` books and sentence/paragraph files, optional `.jsonl`
records, and the wordlists. Prose categories get weighted character budgets so
one large batch cannot crowd out the others:

| Category | Weight |
|---|---:|
| `short_stories/` | 0.40 |
| `sentences/` | 0.25 |
| `paragraphs/` | 0.25 |
| `domains/` | 0.10 |
| `other/` | 0.10 |

Metadata (`MANIFEST.json`, `SOURCES.json`, `BATCHES.json`, checksums, READMEs)
is skipped. The final corpus SHA-256 is stored in metrics and in the checkpoint
signature.

## Model presets

The three towers are stacks of microtransformer blocks with 256-expert MoE
layers. `--model_size` in the workflow selects:

| Preset | Params | Dims | FF | Depth | Context (bytes/words/sentences) | Chunk stem |
|---|---:|---|---|---:|---|---|
| `base` | ~0.98M | 24/32/32 | 48/64/64 | 2 | 1024 / 128 / 32 | mean-pool |
| `wide` | 2.02M | 32/48/48 | 64/96/96 | 2 | 1024 / 128 / 32 | mean-pool |
| `frontier` | 3.66M | 48/64/64 | 96/128/128 | 3 | 2048 / 256 / 64 | strided conv |

The frontier preset adds an order-preserving strided convolution inside every
byte and word chunk (`--chunk-conv`). The old mean-pooling treated the eight
tokens in a chunk as an unordered bag; the convolution learns a separate weight
for each position, so word order and byte order survive into the transformer.

## Quick smoke test

```sh
python diamond_engine.py --quick --corpus-dir ../NaNROUTER/corpora \
  --checkpoint /tmp/diamond_quick.npz --history /tmp/diamond_quick.jsonl \
  --out /tmp/diamond_quick.json --memory-limit-gb 0
```

## Held-out split and overfitting controls

The current default is a whole-document holdout (`--split-mode document`,
`document-holdout-v1`): every position from a held-out document is excluded
from training, and the held-out examples are sampled only from those documents.
That is stricter than holding out a contiguous block of positions. The older
`--split-mode position` path remains available for reproducing the 50/100-epoch
runs; it holds out one contiguous block with a gap larger than both the byte and
word context windows. The original random split was removed because it trained
on the immediate neighbors of every test position.

Training positions are resampled from the full remaining pool at every epoch
(`--fixed-train` disables this). The engine also:

- evaluates every epoch and tracks a combined held-out loss
  (`byte + 0.5 * word + 0.3 * sent`),
- writes `--best-checkpoint` / `--best-weights` whenever that loss improves,
- optionally early-stops after `--patience` evaluations without improvement,
- applies label smoothing, dropout and weight decay to slow memorization.

## 50-epoch run with best-checkpoint selection

```sh
python diamond_engine.py \
  --corpus-dir ../NaNROUTER/corpora \
  --require-real-corpus \
  --epochs 50 --examples 8000 --batch 64 \
  --byte-win 1024 --word-win 128 --sent-win 32 \
  --vocab 2000 --corpus-chars 4000000 \
  --test-size 1024 --eval-every 1 \
  --lr 0.0008 --lr-schedule cosine --warmup-epochs 5 \
  --grad-clip 1.0 --weight-decay 0.05 --dropout 0.2 \
  --label-smoothing 0.05 --patience 12 \
  --save-every 10 --time-budget-minutes 330 \
  --checkpoint state/diamond_state.npz \
  --best-checkpoint state/diamond_best.npz \
  --best-weights state/diamond_best.weights.h5 \
  --history state/diamond_history.jsonl \
  --out state/diamond_metrics.json \
  --save-weights \
  --self-loop-every 100 --self-streams 32 --self-steps 8 \
  --threads 4 --memory-limit-gb 0
```

The engine prints one line per epoch and flushes a JSONL record per epoch. If
the time budget is reached, it writes a checkpoint and exits cleanly. Re-run
the same command to resume; the checkpoint signature prevents restoring
weights into a different corpus/model shape. The latest and best checkpoints
are separate: the latest one is for resuming, while the best one is for
inference.

## Continue a run with a warm restart

To extend a finished 50-epoch run to 100 epochs, keep the same corpus, test
block and label smoothing so the checkpoint signature matches, then use a new
lower cosine cycle:

```sh
python diamond_engine.py \
  --corpus-dir ../NaNROUTER/corpora --require-real-corpus \
  --epochs 100 --examples 8000 --batch 64 \
  --byte-win 1024 --word-win 128 --sent-win 32 \
  --vocab 2000 --corpus-chars 4000000 \
  --test-size 1024 --eval-every 1 \
  --lr 0.0002 --lr-schedule cosine --warmup-epochs 5 \
  --grad-clip 1.0 --weight-decay 0.05 --dropout 0.2 \
  --label-smoothing 0.05 --patience 0 \
  --save-every 10 --checkpoint state/diamond_state.npz \
  --best-checkpoint state/diamond_best.npz \
  --best-weights state/diamond_best.weights.h5 \
  --history state/diamond_history.jsonl --out state/diamond_metrics.json \
  --save-weights --threads 4 --memory-limit-gb 0
```

The restored `diamond_state.npz` must be in place; `--epochs` is the total
target, so resuming at epoch 50 with `--epochs 100` adds 50 epochs. `--lr` is
the peak of the new cosine cycle, and `--patience 0` runs the full extension.

## Results

### 50/100 epochs, position holdout (4M)

The fixed 50-epoch run (`diamond-50-epochs-best`) reached its best combined
held-out loss at epoch 43: **6.6391** (byte 4.2082, word 6.9894, sent 6.2511
bits). The 50-epoch extension to epoch 100 (`diamond-100-epochs`) resumed from
the epoch-50 checkpoint with a second, lower cosine cycle and improved the best
at epoch 86: **6.6074** (byte 4.1851, word 6.9625, sent 6.2205 bits). The final
epoch 100 was 6.6093, so the curve had plateaued; the best checkpoint is the
one published for inference. The best byte-only epoch was 79 at 4.1841 bits,
but the release selects on the combined loss.

- 100-epoch weights: <https://github.com/shemshallah/nanlm_experimental/releases/tag/diamond-100-epochs>
- 50-epoch weights: <https://github.com/shemshallah/nanlm_experimental/releases/tag/diamond-50-epochs-best>

### Wide 8M, whole-document holdout (30 epochs)

The `wide` preset was trained fresh on 8M characters (411 documents) with 41
documents and 138,728 positions held out; the 1,000 test examples came only
from those documents. It used 15,000 examples/epoch, batch 32, 30 epochs, and
2,024,355 parameters. Best (epoch 30): combined **6.7291**, byte **4.1804**,
word **6.9642**, sent **6.8187** bits. Byte top-1 was 21.1%.

This is not directly comparable to the 100-epoch row: it is a stricter
document holdout, a different 8M corpus, 30 epochs instead of 100, and a
different model shape. The useful signal is that byte accuracy improved only
slightly (4.1804 vs 4.1851) despite fresh data and 2× parameters. That is the
evidence that the chunk mean-pooling, not data volume, was the byte bottleneck.

- wide weights: <https://github.com/shemshallah/nanlm_experimental/releases/tag/diamond-wide-8m-30e>

## Runner

`.github/workflows/diamond-1000.yml` downloads the SHA-256 pinned
`corpora-short-stories-0001`, `corpora-sentences-0001` and
`corpora-paragraphs-0001` releases from `shemshallah/NaNROUTER`, extracts them
under a cloned NaNROUTER `corpora/` tree, restores the latest `diamond-state`
artifact when requested, trains for up to the requested number of epochs,
uploads the latest and best checkpoints, publishes the weights as a GitHub
Release, and commits the per-epoch report to `diamond/reports/`.

The workflow defaults now target the frontier preset on the 8M document
holdout:

```sh
gh workflow run diamond-1000.yml -R shemshallah/nanlm_experimental \
  -f model_size=frontier -f epochs=40 -f examples=15000 \
  -f corpus_chars=8000000 -f split_mode=document \
  -f release_tag=diamond-frontier-8m-40e \
  -f report_name=frontier-8m-40e
```

The older wide comparison run:

```sh
gh workflow run diamond-1000.yml -R shemshallah/nanlm_experimental \
  -f model_size=wide -f epochs=30 -f examples=15000 \
  -f corpus_chars=8000000 -f split_mode=document \
  -f release_tag=diamond-wide-8m-30e \
  -f report_name=wide-8m-30e
```

For a historical position-holdout run, set `-f split_mode=position` and use
`-f corpus_chars=4000000`. Keep the corpus, split, model shape, vocab and label
smoothing identical when resuming; the checkpoint signature rejects a shape
change.

The workflow publishes the best held-out checkpoint and inference weights to
the release tag; `diamond_state.npz` is the latest resumable checkpoint.
Per-run metrics are committed under `diamond/reports/<report_name>/`; the old
top-level 50/100-epoch reports are preserved.

## Outputs

- `diamond/reports/<report_name>/diamond_history.jsonl` - append-only per-epoch metrics
- `diamond/reports/<report_name>/diamond_metrics.json` - full history, corpus stats, final and best metrics
- `diamond/reports/diamond_history.jsonl` and `diamond/reports/diamond_metrics.json` - preserved 50/100-epoch position-holdout reports
- `diamond/state/diamond_state.npz` - latest weights + optimizer slots + epoch counter (artifact, not committed)
- `diamond/state/diamond_best.npz` - best held-out checkpoint (artifact, not committed)
- `diamond/state/diamond_best.weights.h5` - best held-out inference weights (artifact, not committed)
- GitHub Release assets - `diamond_best.npz`, `diamond_best.weights.h5`, latest checkpoint, metrics and history
