#!/usr/bin/env bash
# Continue the ONE existing Round-30 checkpoint on NaNROUTER book-level data.
# Never initializes a second model; does not re-tokenize or change book splits.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUN_DIR="${RUN_DIR:-$ROOT/../run}"
DATA_DIR="${DATA_DIR:-$ROOT/../nanrouter_data}"
STEPS="${STEPS:-200}"
THREADS="${THREADS:-4}"
MIX="${MIX:-256:7,512:5,1024:2,2048:1,4096:0.25}"
test -s "$RUN_DIR/latest.pt" || { echo "Missing canonical checkpoint: $RUN_DIR/latest.pt" >&2; exit 1; }
test -s "$DATA_DIR/meta.json" || { echo "Missing NaNROUTER dataset" >&2; exit 1; }
export OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS=1
python "$ROOT/poincare_frontier.py" train \
  --data-dir "$DATA_DIR" --run-dir "$RUN_DIR" \
  --steps "$STEPS" --context-mix "$MIX" --eval-context 256 \
  --batch 1 --cpu --cpu-threads "$THREADS" \
  --save-every 20 --eval-every 40 --eval-batches 4
python "$ROOT/poincare_frontier.py" evaluate \
  --data-dir "$DATA_DIR" --run-dir "$RUN_DIR" \
  --split valid --context-override 256 --batch 1 \
  --eval-batches 16 --cpu
