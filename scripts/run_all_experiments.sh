#!/bin/bash

# Run all tasks for all models (StackRNN and baselines)
# This script runs quick experiments with default settings

echo "=========================================="
echo "Running All Tasks - All Models"
echo "=========================================="
echo ""

# Settings
SEED=1
DEVICE="cpu"

# Tasks to run (all 9 tasks)
TASKS=(1 2 3 4 5 6 7 8 9)

# Models to run
MODELS=("stackrnn" "lstm" "transformer" "mamba")

# Create results directory
mkdir -p results/quick_test

# Log file
LOG_FILE="results/quick_test/experiment_log_$(date +%Y%m%d_%H%M%S).txt"

echo "Experiment started at $(date)" | tee -a "$LOG_FILE"
echo "Settings: seed=$SEED, device=$DEVICE" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Run experiments
for task in "${TASKS[@]}"; do
    echo "========================================" | tee -a "$LOG_FILE"
    echo "Task $task" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"
    
    for model in "${MODELS[@]}"; do
        echo "" | tee -a "$LOG_FILE"
        echo "Running $model on Task $task..." | tee -a "$LOG_FILE"
        echo "------------------------------------------" | tee -a "$LOG_FILE"
        
        if [ "$model" = "stackrnn" ]; then
            # StackRNN
            python3 -m src.exp.stackrnn_test \
                --task $task \
                --device $DEVICE \
                --seed $SEED \
                2>&1 | tee -a "$LOG_FILE"
        else
            # Baseline models
            python3 -m src.exp.baseline_test \
                --model $model \
                --task $task \
                --device $DEVICE \
                --seed $SEED \
                2>&1 | tee -a "$LOG_FILE"
        fi
        
        status=$?
        if [ $status -eq 0 ]; then
            echo "✓ $model Task $task completed successfully" | tee -a "$LOG_FILE"
        else
            echo "✗ $model Task $task failed with exit code $status" | tee -a "$LOG_FILE"
        fi
        echo "" | tee -a "$LOG_FILE"
    done
done

echo "========================================" | tee -a "$LOG_FILE"
echo "All experiments completed at $(date)" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Log saved to: $LOG_FILE"
