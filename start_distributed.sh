#!/bin/bash
# ============================================================
# FROM + PartialFC Distributed Training Launch Script
#
# Usage:
#   bash start_distributed.sh [stage]
#
# Stages:
#   1 = Clean baseline (no occlusion)
#   2 = Occlusion training (no mask)
#   3 = Mask training (full FROM method)
#
# Adjust NUM_GPUS, BATCH_SIZE, and config paths below.
# ============================================================

NUM_GPUS=3
STAGE=${1:-1}

# Common settings
PATTERN=5
RATIO=3
WORKERS=8
BATCH_SIZE=128  # per-GPU batch size (A100 80GB can handle 128-256)
SAVE_OCC_SAMPLES=${2:-0}  # Pass number of samples as 2nd arg (e.g. bash start_distributed.sh 2 50)

case $STAGE in
  1)
    echo "=== Stage 1: Clean baseline ==="
    CONFIG="experiments/custom-112x112-Clean.yaml"
    LR=0.1
    ;;
  2)
    echo "=== Stage 2: Occlusion training ==="
    CONFIG="experiments/custom-112x112-Occ.yaml"
    LR=0.01
    ;;
  3)
    echo "=== Stage 3: Mask training ==="
    CONFIG="experiments/custom-112x112-Mask.yaml"
    LR=0.001
    ;;
  *)
    echo "Usage: bash start_distributed.sh [1|2|3]"
    exit 1
    ;;
esac

torchrun \
    --nproc_per_node=$NUM_GPUS \
    train_distributed.py \
    --cfg $CONFIG \
    --pattern $PATTERN \
    --ratio $RATIO \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --workers $WORKERS \
    --weight_pred 1.0 \
    --save_occ_samples $SAVE_OCC_SAMPLES
