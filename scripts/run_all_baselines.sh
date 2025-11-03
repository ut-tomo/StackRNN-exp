#!/bin/bash

# ベースラインモデル（LSTM, Transformer, Mamba）で全タスク（1-7）を学習するスクリプト

set -e 

cd "$(dirname "$0")/.."

echo "=================================="
echo "Starting all baseline experiments"
echo "=================================="
echo ""

START_TIME=$(date "+%Y-%m-%d %H:%M:%S")
echo "Start time: $START_TIME"
echo ""

MODELS=("lstm" "transformer" "mamba")

TASKS=(1 2 3 4 5 6 7)

SEED=42
DEVICE="cpu"

for MODEL in "${MODELS[@]}"; do
    echo "========================================"
    echo "Running experiments for model: $MODEL"
    echo "========================================"
    echo ""
    
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
            echo "✓ Successfully completed: $MODEL - Task $TASK"
        else
            echo "✗ Failed: $MODEL - Task $TASK"
        fi
        echo ""
    done
    
    echo ""
done

END_TIME=$(date "+%Y-%m-%d %H:%M:%S")
echo "=================================="
echo "All experiments completed!"
echo "Start time: $START_TIME"
echo "End time:   $END_TIME"
echo "=================================="
