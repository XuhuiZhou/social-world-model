#!/bin/bash

# Array of models to use for tasks
TASK_MODELS=(
    "gpt-4.1-2025-04-14"
    "o1-2024-12-17"
    "o3-2025-04-16"
    "o3-mini-2025-01-31"
    "together_ai/deepseek-ai/DeepSeek-R1"
    "together_ai/meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"
    "gpt-4o-2024-08-06"
)

# Array of models to use for context generation
CONTEXT_MODELS=(
    "gpt-4.1-2025-04-14"
    "o1-2024-12-17"
    "o3-2025-04-16"
    "o3-mini-2025-01-31"
    "together_ai/deepseek-ai/DeepSeek-R1"
    "together_ai/meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"
    "gpt-4o-2024-08-06"
)

# Only run ToMi benchmark
BENCHMARK="tomi"

# Function to run a single experiment
run_experiment() {
    local model=$1
    local context_model=$2
    local mode=$3

    echo "Running $BENCHMARK benchmark with model=$model, context_model=$context_model, mode=$mode"

    python run_benchmarks.py $BENCHMARK \
        --model-name "$model" \
        --context-model "$context_model" \
        --mode "$mode" \
        --continue-mode "continue" \
        --batch-size 4
}

# First, generate socialized contexts for all context models
for context_model in "${CONTEXT_MODELS[@]}"; do
    echo "Generating socialized contexts for $BENCHMARK with context model $context_model"
    python run_benchmarks.py $BENCHMARK \
        --model-name "$context_model" \
        --context-model "$context_model" \
        --mode "generate_socialized_context" \
        --continue-mode "continue" \
        --batch-size 100 \
        --example_analysis_file "data/social_contexts_example/tomi.json"
done

# Then run the full matrix of experiments
for model in "${TASK_MODELS[@]}"; do
    for context_model in "${CONTEXT_MODELS[@]}"; do
        # Run socialized context mode
        run_experiment "$model" "$context_model" "socialized_context"
    done
done
