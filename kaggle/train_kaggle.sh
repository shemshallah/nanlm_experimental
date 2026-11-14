#!/usr/bin/env bash
# 2-hour Kaggle GPU training on WikiText-103 + FineWeb-Edu.
# Defaults to the largest practical model (~30M params); use MODEL_SIZE=frontier
# for the 3.8M model if the larger one does not fit.
set -euo pipefail

STATE_DIR="${STATE_DIR:-/kaggle/working/diamond_state}"
CORPUS_DIR="${CORPUS_DIR:-/kaggle/working/corpus}"
EPOCHS="${EPOCHS:-50}"
EXAMPLES="${EXAMPLES:-15000}"
CORPUS_CHARS="${CORPUS_CHARS:-50000000}"
TIME_BUDGET="${TIME_BUDGET:-120}"
TEST_SIZE="${TEST_SIZE:-4096}"
BATCH="${BATCH:-32}"
SENT_WEIGHT="${SENT_WEIGHT:-0.8}"
MODEL_SIZE="${MODEL_SIZE:-large}"

if [ "$MODEL_SIZE" = "large" ]; then
  DIM_B=256; DIM_W=384; DIM_S=384
  FF_B=1024; FF_W=1536; FF_S=1536
  DEPTH=6; SENT_DEPTH=6
else
  MODEL_SIZE=frontier
  DIM_B=48; DIM_W=64; DIM_S=64
  FF_B=96; FF_W=128; FF_S=192
  DEPTH=3; SENT_DEPTH=4
fi

mkdir -p "$STATE_DIR"
echo "state: $STATE_DIR"
echo "corpus: $CORPUS_DIR"
echo "model=$MODEL_SIZE dims=$DIM_B/$DIM_W/$DIM_S depth=$DEPTH sent_depth=$SENT_DEPTH"
echo "epochs=$EPOCHS examples=$EXAMPLES corpus_chars=$CORPUS_CHARS time_budget=${TIME_BUDGET}m test=$TEST_SIZE batch=$BATCH"

python diamond/diamond_engine.py \
  --corpus-dir "$CORPUS_DIR" \
  --require-real-corpus \
  --epochs "$EPOCHS" \
  --examples "$EXAMPLES" \
  --batch "$BATCH" \
  --byte-win 2048 --word-win 256 --sent-win 128 \
  --sent-len 16 --sent-weight "$SENT_WEIGHT" --sent-depth "$SENT_DEPTH" \
  --dim-b "$DIM_B" --dim-w "$DIM_W" --dim-s "$DIM_S" \
  --ff-b "$FF_B" --ff-w "$FF_W" --ff-s "$FF_S" \
  --depth "$DEPTH" --chunk-conv --byte-chunk 8 \
  --vocab 2000 --corpus-chars "$CORPUS_CHARS" \
  --file-read-chars 20000 --max-files 8000 \
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
  --threads 4 --memory-limit-gb 0 --seed 11
