#!/bin/bash

# StackRNNで全タスク（1-9）を学習するスクリプト

set -e 

cd "$(dirname "$0")/.."

echo "=================================="
echo "Starting StackRNN experiments"
echo "=================================="
echo ""

START_TIME=$(date "+%Y-%m-%d %H:%M:%S")
echo "Start time: $START_TIME"
echo ""

TASKS=(1 2 3 4 5 6 7 8 9)

SEED=1
DEVICE="cpu"

for TASK in "${TASKS[@]}"; do
    echo "========================================"
    echo "Training StackRNN on Task $TASK"
    echo "========================================"
    
    python3 -m src.exp.stackrnn_test \
        --task "$TASK" \
        --seed "$SEED" \
        --device "$DEVICE"
    
    if [ $? -eq 0 ]; then
        echo "✓ Successfully completed: StackRNN - Task $TASK"
    else
        echo "✗ Failed: StackRNN - Task $TASK"
    fi
    echo ""
done

END_TIME=$(date "+%Y-%m-%d %H:%M:%S")
echo "=================================="
echo "All StackRNN experiments completed!"
echo "Start time: $START_TIME"
echo "End time:   $END_TIME"
echo "=================================="
