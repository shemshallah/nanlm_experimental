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

## Long run (1000 epochs)

```sh
python diamond_engine.py \
  --corpus-dir ../NaNROUTER/corpora \
  --require-real-corpus \
  --epochs 1000 --examples 2500 --batch 64 \
  --byte-win 1024 --word-win 128 --sent-win 32 \
  --vocab 2000 --corpus-chars 2000000 \
  --test-size 512 --eval-every 20 \
  --lr 0.0025 --lr-schedule cosine --warmup-epochs 20 --grad-clip 1.0 \
  --save-every 25 --time-budget-minutes 330 \
  --checkpoint state/diamond_state.npz \
  --history state/diamond_history.jsonl \
  --out state/diamond_metrics.json \
  --self-loop-every 100 --self-streams 32 --self-steps 8 \
  --threads 4 --memory-limit-gb 0
```

The engine prints one line per epoch and flushes a JSONL record per epoch. If
the time budget is reached, it writes a checkpoint and exits cleanly. Re-run
the same command to resume; the checkpoint signature prevents restoring
weights into a different corpus/model shape.

## Runner

`.github/workflows/diamond-1000.yml` downloads the SHA-256 pinned
`corpora-short-stories-0001` and `corpora-sentences-0001` releases from
`shemshallah/NaNROUTER`, extracts them under a cloned NaNROUTER `corpora/`
tree, restores the latest `diamond-state` artifact, trains for up to 1000
epochs, uploads a resumable checkpoint, and commits the per-epoch report to
`diamond/reports/`.

Trigger it with:

```sh
gh workflow run diamond-1000.yml -R shemshallah/nanlm_experimental
gh run watch -R shemshallah/nanlm_experimental
```

## Outputs

- `diamond/reports/diamond_history.jsonl` - append-only per-epoch metrics
- `diamond/reports/diamond_metrics.json` - full history, corpus stats, final metrics
- `diamond/state/diamond_state.npz` - weights + optimizer slots + epoch counter (artifact, not committed)
