#!/bin/bash

# Array of models to use for tasks
TASK_MODELS=(
    "gpt-4o-2024-08-06"
)

# Only use o3 for context generation
CONTEXT_MODEL="o3-2025-04-16"

# Array of benchmarks to run
BENCHMARKS=(
    # "tomi"
    # "ori_tomi"
    # "fantom"
    # "confaide"
    # "hitom"
    "mmtom"
)

# Function to run a single experiment
run_experiment() {
    local benchmark=$1
    local model=$2
    local mode=$3

    # Set batch size based on model type
    local batch_size=100
    if [[ "$model" == *"together_ai"* ]]; then
        batch_size=50
    fi

    echo "Running $benchmark benchmark with model=$model, context_model=$CONTEXT_MODEL, mode=$mode, batch_size=$batch_size"

    uv run python run_benchmarks.py $benchmark \
        --model-name "$model" \
        --context-model "$CONTEXT_MODEL" \
        --mode "$mode" \
        --continue-mode "continue" \
        --batch-size $batch_size
}

# Run each benchmark
for benchmark in "${BENCHMARKS[@]}"; do
    echo "Processing benchmark: $benchmark"
    # Then run experiments with all task models
    for model in "${TASK_MODELS[@]}"; do
        # Run both vanilla and socialized context modes
        run_experiment "$benchmark" "$model" "vanilla_no_reasoning"
        # run_experiment "$benchmark" "$model" "socialized_context_no_json"
        # run_experiment "$benchmark" "$model" "few_shot_context"
    done
done
