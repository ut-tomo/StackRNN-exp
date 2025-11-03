#!/bin/bash

# 特定のモデルで全タスクを学習するスクリプト
# Usage: scripts/run_model.sh lstm
#        scripts/run_model.sh transformer
#        scripts/run_model.sh mamba

set -e

# 引数チェック
if [ $# -eq 0 ]; then
    echo "Error: Please specify model type"
    echo "Usage: bash scripts/run_model.sh [lstm|transformer|mamba]"
    exit 1
fi

MODEL=$1
cd "$(dirname "$0")/.."

echo "========================================"
echo "Running all tasks for model: $MODEL"
echo "========================================"
echo ""

TASKS=(1 2 3 4 5 6 7)
SEED=42
DEVICE="cpu"

for TASK in "${TASKS[@]}"; do
    echo "----------------------------------------"
    echo "Training $MODEL on Task $TASK"
    echo "----------------------------------------"
    
    python3 -m src.exp.baseline_test \
        --model "$MODEL" \
        --task "$TASK" \
        --seed "$SEED" \
        --device "$DEVICE"
    
    if [ $? -eq 0 ]; then
        echo "✓ Successfully completed: Task $TASK"
    else
        echo "✗ Failed: Task $TASK"
    fi
    echo ""
done

echo "========================================"
echo "All tasks completed for $MODEL"
echo "========================================"
