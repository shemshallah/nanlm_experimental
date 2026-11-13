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

The workflow publishes the best held-out checkpoint and inference weights to
the release tag; `diamond_state.npz` is the latest resumable checkpoint.

## Outputs

- `diamond/reports/diamond_history.jsonl` - append-only per-epoch metrics
- `diamond/reports/diamond_metrics.json` - full history, corpus stats, final and best metrics
- `diamond/state/diamond_state.npz` - latest weights + optimizer slots + epoch counter (artifact, not committed)
- `diamond/state/diamond_best.npz` - best held-out checkpoint (artifact, not committed)
- `diamond/state/diamond_best.weights.h5` - best held-out inference weights (artifact, not committed)
- GitHub Release assets - `diamond_best.npz`, `diamond_best.weights.h5`, latest checkpoint, metrics and history
