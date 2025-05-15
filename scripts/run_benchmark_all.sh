#!/bin/bash

# Array of models to use for tasks
TASK_MODELS=(
    "gpt-4o-2024-08-06"
    "gpt-4.1-2025-04-14"
    "o1-2024-12-17"
    "o3-2025-04-16"
    "o3-mini-2025-01-31"
    "together_ai/deepseek-ai/DeepSeek-R1"
    "together_ai/meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"
)

# Only use o3 for context generation
CONTEXT_MODEL="o3-2025-04-16"

# Array of benchmarks to run
BENCHMARKS=(
    "paratomi"
    "tomi"
    "fantom"
    "confaide"
    "hitom"
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

    # First, generate socialized contexts
    echo "Generating socialized contexts for $benchmark with context model $CONTEXT_MODEL"

    # Set batch size for context generation based on model type
    local context_batch_size=100
    if [[ "$CONTEXT_MODEL" == *"together_ai"* ]]; then
        context_batch_size=50
    fi

    uv run python run_benchmarks.py $benchmark \
        --model-name "$CONTEXT_MODEL" \
        --context-model "$CONTEXT_MODEL" \
        --mode "generate_socialized_context" \
        --continue-mode "continue" \
        --batch-size $context_batch_size \
        --example-analysis-file "data/social_contexts_example/${benchmark}.json"

    # Then run experiments with all task models
    for model in "${TASK_MODELS[@]}"; do
        # Run both vanilla and socialized context modes
        run_experiment "$benchmark" "$model" "vanilla"
        run_experiment "$benchmark" "$model" "socialized_context"
    done
done
