#!/bin/bash

# タスクidを指定し, 全モデルを学習するスクリプト
# Usage: scripts/run_task.sh 1
set -e

# task 1-7
if [ $# -eq 0 ]; then
    echo "Error: Please specify task ID"
    echo "Usage: bash scripts/run_task.sh [1-7]"
    exit 1
fi

TASK=$1

cd "$(dirname "$0")/.."

echo "========================================"
echo "Running all models for Task $TASK"
echo "========================================"
echo ""

MODELS=("lstm" "transformer" "mamba")
SEED=42
DEVICE="cpu"

for MODEL in "${MODELS[@]}"; do
    echo "----------------------------------------"
    echo "Training $MODEL on Task $TASK"
    echo "----------------------------------------"
    
    python3 -m src.exp.baseline_test \
        --model "$MODEL" \
        --task "$TASK" \
        --seed "$SEED" \
        --device "$DEVICE"
    
    if [ $? -eq 0 ]; then
        echo "✓ Successfully completed: $MODEL"
    else
        echo "✗ Failed: $MODEL"
    fi
    echo ""
done

echo "========================================"
echo "All models completed for Task $TASK"
echo "========================================"
