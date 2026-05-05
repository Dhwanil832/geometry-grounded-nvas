#!/bin/bash
# ============================================================
# GGAD — Training & Evaluation on RWAVS Dataset
# Trains one scene-specific model per scene (13 scenes total)
#
# Usage:
#   ./scripts/train_rwavs.sh <GPU_ID> <SCENE_START> <SCENE_END>
#
# Example — train all 13 scenes on GPU 0:
#   ./scripts/train_rwavs.sh 0 1 13
#
# Example — train scenes 1-5 on GPU 0, scenes 6-13 on GPU 1 (parallel):
#   ./scripts/train_rwavs.sh 0 1 5  &
#   ./scripts/train_rwavs.sh 1 6 13
# ============================================================

GPU_ID=$1
SCENE_START=$2
SCENE_END=$3

if [ -z "$GPU_ID" ] || [ -z "$SCENE_START" ] || [ -z "$SCENE_END" ]; then
    echo "Usage: ./scripts/train_rwavs.sh <GPU_ID> <SCENE_START> <SCENE_END>"
    echo "Example: ./scripts/train_rwavs.sh 0 1 13"
    exit 1
fi

echo "Running RWAVS training on GPU $GPU_ID for scenes $SCENE_START to $SCENE_END"

for i in $(seq $SCENE_START $SCENE_END)
do
    echo "============================================================"
    echo "SCENE $i — Training on GPU $GPU_ID"
    echo "============================================================"

    EXP_NAME="rwavs_scene_${i}"
    LOG_DIR="logs/${EXP_NAME}"

    # Step 1: Train
    echo "[1/2] Training..."
    CUDA_VISIBLE_DEVICES=$GPU_ID python -W ignore tools/train.py \
        --cfg configs/rwavs.yaml \
        output_dir ${EXP_NAME} \
        dataset.video _${i}

    # Step 2: Evaluate on validation split
    echo "[2/2] Evaluating on validation set..."
    CUDA_VISIBLE_DEVICES=$GPU_ID python -W ignore tools/test_rwavs.py \
        --cfg configs/rwavs.yaml \
        --split val \
        output_dir ${EXP_NAME} \
        dataset.video _${i} \
        model.resume_path ${LOG_DIR}/${EXP_NAME}/100.pth

done

# Aggregate results across all scenes
echo ""
echo "============================================================"
echo "Aggregating results across all scenes..."
echo "============================================================"
python tools/aggregate_vggt_results.py --prefix rwavs_scene --split val
