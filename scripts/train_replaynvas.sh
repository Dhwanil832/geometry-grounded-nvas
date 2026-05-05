#!/bin/bash
# ============================================================
# GGAD — Training & Evaluation on Replay-NVAS Dataset
# Single model trained on all scenes jointly
#
# Usage:
#   ./scripts/train_replaynvas.sh <GPU_ID>
#
# Example:
#   ./scripts/train_replaynvas.sh 0
# ============================================================

GPU_ID=$1

if [ -z "$GPU_ID" ]; then
    echo "Usage: ./scripts/train_replaynvas.sh <GPU_ID>"
    echo "Example: ./scripts/train_replaynvas.sh 0"
    exit 1
fi

EXP_NAME="replaynvas_experiment"
LOG_DIR="logs/${EXP_NAME}"

echo "Running Replay-NVAS training on GPU $GPU_ID"

# Step 1: Train
echo "[1/2] Training..."
CUDA_VISIBLE_DEVICES=$GPU_ID python -W ignore tools/train.py \
    --cfg configs/replaynvas.yaml \
    output_dir ${EXP_NAME}

# Step 2: Evaluate on test split
echo "[2/2] Evaluating on test set..."
CUDA_VISIBLE_DEVICES=$GPU_ID python -W ignore tools/test_replaynvas.py \
    --cfg configs/replaynvas.yaml \
    --split test \
    output_dir ${EXP_NAME} \
    model.resume_path ${LOG_DIR}/${EXP_NAME}/100.pth

echo ""
echo "Replay-NVAS evaluation complete."
echo "Results saved to: av_results/${EXP_NAME}_test.pkl"
