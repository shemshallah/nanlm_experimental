#!/usr/bin/env bash
# Run from a machine with sufficient GPU memory. Keep ONE continuously
# evolving RUN_DIR/latest.pt; checkpoint includes Adam history and corpus hash.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC="$(cd "$SCRIPT_DIR/.." && pwd)"
RUN_DIR="${RUN_DIR:-$SRC/../run}"
DATA_DIR="${DATA_DIR:-$SRC/../gutenberg_data}"
CACHE_DIR="${CACHE_DIR:-$SRC/../nanrouter_release_cache}"
TAG="${NANROUTER_TAG:-corpora-short-stories-0001}"
MICROBATCH="${MICROBATCH:-1}"
ACCUM="${ACCUM:-4}"
EXTRA=()
if [[ "${BF16:-0}" == "1" ]]; then EXTRA+=(--bf16); fi
[[ -f "$RUN_DIR/latest.pt" ]] || { echo "Missing the SINGLE expanded Round-28 checkpoint: $RUN_DIR/latest.pt" >&2; exit 2; }
if [[ ! -f "$DATA_DIR/meta.json" ]]; then
  "$SCRIPT_DIR/prepare_nanrouter_short_stories.sh" "$TAG" "$DATA_DIR" "$CACHE_DIR"
fi
# Establish a book-level held-out baseline BEFORE any Gutenberg update.
# The transition updates only dataset fingerprint and configuration metadata.
python "$SRC/poincare_frontier.py" train --data-dir "$DATA_DIR" \
  --run-dir "$RUN_DIR" --adopt-data --steps 0 --train-context 256 \
  --batch "$MICROBATCH" --eval-batches 4

train_stage() {
  local context="$1" updates="$2"
  if (( updates == 0 )); then return; fi
  python "$SRC/poincare_frontier.py" train --data-dir "$DATA_DIR" \
    --run-dir "$RUN_DIR" --steps "$updates" --train-context "$context" \
    --batch "$MICROBATCH" --grad-accum "$ACCUM" \
    --eval-every 200 --eval-batches 4 --save-every 200 \
    "${EXTRA[@]}"
}
train_stage 256  "${STEPS_256:-2000}"
train_stage 512  "${STEPS_512:-4000}"
train_stage 1024 "${STEPS_1024:-8000}"
train_stage 2048 "${STEPS_2048:-16000}"
python "$SRC/poincare_frontier.py" evaluate --data-dir "$DATA_DIR" \
  --run-dir "$RUN_DIR" --context-override 2048 --batch 1 \
  --exhaustive --max-eval-batches 64 --ablations --unigram-baseline
