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

## Quick smoke test

```sh
python diamond_engine.py --quick --corpus-dir ../NaNROUTER/corpora \
  --checkpoint /tmp/diamond_quick.npz --history /tmp/diamond_quick.jsonl \
  --out /tmp/diamond_quick.json --memory-limit-gb 0
```

## Held-out split and overfitting controls

The engine holds out one contiguous block of positions, not a random shuffle of
positions. The old random split trained on the immediate neighbors of every
test position, which leaked the test distribution and made the early curve look
better than it was. A gap larger than both the byte and word context windows
sits on each side of the held-out block.

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

## Runner

`.github/workflows/diamond-1000.yml` downloads the SHA-256 pinned
`corpora-short-stories-0001`, `corpora-sentences-0001` and
`corpora-paragraphs-0001` releases from `shemshallah/NaNROUTER`, extracts them
under a cloned NaNROUTER `corpora/` tree, restores the latest `diamond-state`
artifact when requested, trains for up to the requested number of epochs,
uploads the latest and best checkpoints, publishes the weights as a GitHub
Release, and commits the per-epoch report to `diamond/reports/`.

A 50-epoch run with the fixed overfitting controls:

```sh
gh workflow run diamond-1000.yml -R shemshallah/nanlm_experimental \
  -f epochs=50 -f examples=8000 -f corpus_chars=4000000 \
  -f time_budget_minutes=0 -F resume=false \
  -f release_tag=diamond-50-epochs-best
gh run watch -R shemshallah/nanlm_experimental
```

To extend that run to 100 epochs, keep the same corpus and label smoothing,
restore the latest state, and use a second lower cosine cycle:

```sh
gh workflow run diamond-1000.yml -R shemshallah/nanlm_experimental \
  -f epochs=100 -f examples=8000 -f corpus_chars=4000000 \
  -f time_budget_minutes=0 -f lr=0.0002 -f warmup_epochs=5 \
  -f patience=0 -F resume=true \
  -f release_tag=diamond-100-epochs
gh run watch -R shemshallah/nanlm_experimental
```

The workflow publishes the best held-out checkpoint and inference weights to
the release tag; `diamond_state.npz` is the latest resumable checkpoint.

## Outputs

- `diamond/reports/diamond_history.jsonl` - append-only per-epoch metrics
- `diamond/reports/diamond_metrics.json` - full history, corpus stats, final and best metrics
- `diamond/state/diamond_state.npz` - latest weights + optimizer slots + epoch counter (artifact, not committed)
- `diamond/state/diamond_best.npz` - best held-out checkpoint (artifact, not committed)
- `diamond/state/diamond_best.weights.h5` - best held-out inference weights (artifact, not committed)
- GitHub Release assets - `diamond_best.npz`, `diamond_best.weights.h5`, latest checkpoint, metrics and history
