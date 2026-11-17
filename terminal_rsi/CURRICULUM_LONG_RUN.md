# Diverse curriculum + 6-hour Molab run

This guide covers the long-run pipeline added on top of the v4 controller:

- `terminal_rsi/prepare_curriculum.py` — builds a diverse, difficulty-ordered,
  book-disjoint TRAIN/DEV/RSI-holdout corpus.
- `terminal_rsi/run_long_molab.py` — supervised 6-hour run with rolling
  checkpoints and automatic resume.

## Why a curriculum

The first CPU pilot concatenated hash-sorted books. It was reproducible but not
structured: genre/difficulty distribution was accidental, and the RSI probe
sampled positions from books the model was also training on. The new builder:

1. strips Project Gutenberg headers/footers and obvious OCR noise,
2. rejects Russian/non-English and low-quality books,
3. deduplicates near-identical texts,
4. tags genre from Gutenberg bookshelves/subjects,
5. scores difficulty from sentence length, word length, vocabulary diversity,
   dialogue density and archaic-word density,
6. assigns books to four difficulty stages,
7. orders TRAIN chunks easy -> hard with 25% replay from earlier stages,
8. interleaves genres and caps author dominance,
9. reserves 4 **book-disjoint** RSI holdout books and 64 **book-disjoint**
   DEV books.

The RSI v4 evaluator uses only the holdout books. The model never trains on
them, so v4 decisions are not made on an overfit TRAIN probe.

## Build the curriculum

```bash
python -m terminal_rsi.prepare_curriculum \
  --corpus-dir /path/to/nanrouter_batch0004 \
  --output-dir /path/to/long_run \
  --train-bytes 64000000 \
  --dev-books 64 \
  --rsi-holdout-books 4 \
  --chunk-bytes 65536 \
  --replay-fraction 0.25
```

Outputs:

- `train_curriculum.bin` — 64 MB curriculum stream
- `train_curriculum_book_ids.json` — all book IDs reserved for TRAIN
- `rsi_holdout.bin` — cleaned, book-disjoint RSI validation text
- `rsi_holdout_book_ids.json`
- `win_dev.bin`, `win_dev.manifest.json` — 64-book DEV split
- `curriculum_manifest.json` — stages, genres, dedup, hashes, disjointness

The builder verifies TRAIN/DEV/holdout Gutenberg identities are pairwise
disjoint.

## Run for six hours

```bash
python -m terminal_rsi.run_long_molab \
  --corpus-dir /path/to/nanrouter_batch0004 \
  --output-dir /path/to/long_run \
  --mode cpu \
  --seconds 21600 \
  --cpu-threads 8 \
  --train-bytes 64000000 \
  --rsi-v4-cycle-updates 1000 \
  --rsi-v4-horizon 10 \
  --rsi-v4-candidates 2 \
  --rsi-v4-shards 4 \
  --rsi-v4-shard-positions 128 \
  --rsi-v4-eval-chunk 8 \
  --rsi-v4-min-gain 0.01 \
  --rsi-v4-z 2.0 \
  --rsi-v4-critic-steps 16 \
  --rsi-v4-policy-steps 8 \
  --save-every 500
```

`--mode` selects `cpu`, `t4` or `blackwell` for the **same 6,495,162-parameter
v4 graph**. The six-hour example above is the CPU profile. On the 95 GiB
Blackwell card use `--mode blackwell --batch 8192`; the upgrade branch uses
BF16 autocast, the v103 path stays FP32, and the GPU window cache avoids
per-step CPU window construction. The long supervisor always uses the v4
controller; v3 remains available in `train_verbatim` as the parity baseline.

The supervisor:

- builds the curriculum if it is missing,
- runs `train_verbatim --seconds <remaining>` in segments,
- writes a rolling checkpoint every `--save-every` updates,
- resumes from the rolling checkpoint after a failure,
- copies the final rolling checkpoint to
  `terminal_rsi_long_candidate.pt`,
- writes `long_run_summary.json`, `RUN_SUMMARY.md`, and a SHA-256-indexed
  `terminal_rsi_long_bundle.zip`.

If the Molab session restarts, run the same command again with the same
`--output-dir`; it reuses the curriculum and resumes from the rolling
checkpoint. The rolling checkpoint contains model, all three AdamW groups,
sampler/RNG and RSI-v4 policy/critic/world-model/replay state.

## CPU settings

- `--cpu-threads 8` is the recommended long-run setting on Molab (20 vCPUs).
- `--rsi-v4-eval-chunk 8` keeps probe evaluation in the efficient CPU batch
  range; large probe batches were ~2.6x slower.
- `--rsi-v4-cycle-updates 1000` amortizes branch search over a long run.
- `--rsi-v4-critic-steps 16 --rsi-v4-policy-steps 8` reduces controller
  overhead while preserving replay learning.
- `--save-every 500` bounds lost work after a restart.

## Safety

- DEV is used only for the final before/after benchmark.
- Sealed TEST is untouched.
- RSI decisions use only book-disjoint holdout text.
- The candidate remains unpromoted; champion promotion follows
  `champion_guard`.
- The v4 controller is conservative. If an action does not beat no-op by
  `max(min_gain, z * paired_SE)`, the no-op trajectory is kept.
