#!/usr/bin/env bash
# One checkpoint; no forks. CPU continuation for an already-adopted NaNROUTER corpus.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUN_DIR="${RUN_DIR:-$ROOT/../run}"
DATA_DIR="${DATA_DIR:-$ROOT/../nanrouter_data}"
STEPS="${STEPS:-40}"
THREADS="${THREADS:-4}"
BATCH="${BATCH:-1}"
ACCUM="${ACCUM:-1}"
MIX="${MIX:-256:2,512:3,1024:3,2048:2,4096:1}"
EVAL_CONTEXT="${EVAL_CONTEXT:-256}"
SAVE_EVERY="${SAVE_EVERY:-8}"
EVAL_EVERY="${EVAL_EVERY:-20}"
test -f "$RUN_DIR/latest.pt" || { echo 'Missing existing canonical checkpoint' >&2; exit 2; }
test -f "$DATA_DIR/meta.json" || { echo 'Missing adopted book-corpus manifest' >&2; exit 2; }
export OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS=1
python "$ROOT/poincare_frontier.py" train \
  --data-dir "$DATA_DIR" --run-dir "$RUN_DIR" \
  --steps "$STEPS" --context-mix "$MIX" --eval-context "$EVAL_CONTEXT" \
  --batch "$BATCH" --grad-accum "$ACCUM" --cpu --cpu-threads "$THREADS" \
  --eval-every "$EVAL_EVERY" --eval-batches 4 --save-every "$SAVE_EVERY"
# Validation split only; do not repeatedly tune on the untouched test split.
python "$ROOT/poincare_frontier.py" evaluate \
  --data-dir "$DATA_DIR" --run-dir "$RUN_DIR" --split valid \
  --context-override "$EVAL_CONTEXT" --batch 1 --eval-batches 16 --cpu
