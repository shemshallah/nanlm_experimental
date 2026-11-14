#!/usr/bin/env bash
# Frontier4-cache training on WikiText-103 for Google Colab.
# Every argument is overridable from the environment.
set -euo pipefail

STATE_DIR="${STATE_DIR:-/content/drive/MyDrive/diamond_wiki/state}"
CORPUS_DIR="${CORPUS_DIR:-/content/wiki_corpus}"
EPOCHS="${EPOCHS:-100}"
EXAMPLES="${EXAMPLES:-15000}"
CORPUS_CHARS="${CORPUS_CHARS:-50000000}"
TIME_BUDGET="${TIME_BUDGET:-90}"
TEST_SIZE="${TEST_SIZE:-4096}"
THREADS="${THREADS:-4}"
BATCH="${BATCH:-32}"
SENT_WEIGHT="${SENT_WEIGHT:-0.8}"

mkdir -p "$STATE_DIR"
echo "state: $STATE_DIR"
echo "corpus: $CORPUS_DIR"
echo "epochs=$EPOCHS examples=$EXAMPLES corpus_chars=$CORPUS_CHARS time_budget=${TIME_BUDGET}m test=$TEST_SIZE batch=$BATCH"

python diamond/diamond_engine.py \
  --corpus-dir "$CORPUS_DIR" \
  --require-real-corpus \
  --epochs "$EPOCHS" \
  --examples "$EXAMPLES" \
  --batch "$BATCH" \
  --byte-win 2048 --word-win 256 --sent-win 128 \
  --sent-len 16 --sent-weight "$SENT_WEIGHT" --sent-depth 4 \
  --dim-b 48 --dim-w 64 --dim-s 64 \
  --ff-b 96 --ff-w 128 --ff-s 192 \
  --depth 3 --chunk-conv --byte-chunk 8 \
  --vocab 2000 --corpus-chars "$CORPUS_CHARS" \
  --file-read-chars 20000 --max-files 4000 \
  --test-size "$TEST_SIZE" --split-mode document --doc-holdout-frac 0.10 \
  --eval-every 1 --eval-batch 32 \
  --lr 0.001 --lr-schedule cosine --warmup-epochs 1 \
  --grad-clip 1.0 --weight-decay 0.005 --dropout 0.05 --label-smoothing 0.05 \
  --patience 0 --save-every 1 --report-every-steps 50 \
  --time-budget-minutes "$TIME_BUDGET" \
  --checkpoint "$STATE_DIR/diamond_state.npz" \
  --best-checkpoint "$STATE_DIR/diamond_best.npz" \
  --best-weights "$STATE_DIR/diamond_best.weights.h5" \
  --history "$STATE_DIR/diamond_history.jsonl" \
  --out "$STATE_DIR/diamond_metrics.json" \
  --save-weights \
  --self-loop-every 0 --self-loop-start-epoch 11 \
  --threads "$THREADS" --memory-limit-gb 0 --seed 11
