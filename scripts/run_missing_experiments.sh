#!/bin/bash

# Define the model indices from the original arrays
declare -a AGENT_INDICES=(
    "gpt-4.1-2025-04-14:0"
    "o1-2024-12-17:1"
    "o1-mini-2024-09-12:2"
    "o3-2025-04-16:3"
    "o3-mini-2025-01-31:4"
    "together_ai/meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8:5"
    "together_ai/meta-llama/Llama-4-Scout-17B-16E-Instruct:6"
)

declare -a SOCIAL_WORLD_INDICES=(
    "gpt-4.1-2025-04-14:0"
    "o1-2024-12-17:1"
    "o1-mini-2024-09-12:2"
    "o3-2025-04-16:3"
    "o3-mini-2025-01-31:4"
    "together_ai/meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8:5"
    "together_ai/meta-llama/Llama-4-Scout-17B-16E-Instruct:6"
)

# Define the missing model combinations to test with their original experiment numbers
declare -a MISSING_EXPS=(
    "gpt-4.1-2025-04-14 o1-2024-12-17 2"
    "gpt-4.1-2025-04-14 o1-mini-2024-09-12 3"
    "o1-2024-12-17 together_ai/meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8 13"
    "o1-2024-12-17 together_ai/meta-llama/Llama-4-Scout-17B-16E-Instruct 14"
    "o1-2024-12-17 o1-mini-2024-09-12 10"
    "o1-mini-2024-09-12 o1-mini-2024-09-12 17"
    "o1-mini-2024-09-12 together_ai/meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8 19"
    "o1-mini-2024-09-12 together_ai/meta-llama/Llama-4-Scout-17B-16E-Instruct 20"
    "o3-2025-04-16 o1-mini-2024-09-12 24"
    "o3-2025-04-16 together_ai/meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8 26"
    "o3-2025-04-16 together_ai/meta-llama/Llama-4-Scout-17B-16E-Instruct 27"
    "o3-mini-2025-01-31 o1-mini-2024-09-12 31"
    "o3-mini-2025-01-31 together_ai/meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8 33"
    "together_ai/meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8 o1-2024-12-17 36"
    "together_ai/meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8 o1-mini-2024-09-12 37"
    "together_ai/meta-llama/Llama-4-Scout-17B-16E-Instruct o1-mini-2024-09-12 44"
)

# Fixed parameters
PARTNER_MODEL="gpt-4o-2024-08-06"
EVALUATOR_MODEL="o3-2025-04-16"
BATCH_SIZE=100
TASK="hard"

# Create a timestamp for the experiment batch
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EXPERIMENT_BATCH="experiment_batch_${TIMESTAMP}"

# Create a directory for logs
mkdir -p "logs/${EXPERIMENT_BATCH}"

# Function to run a single experiment
run_experiment() {
    local agent_model=$1
    local social_world_model=$2
    local experiment_num=$3

    # Create a unique tag for this experiment
    local tag="social_world_model_trial_${experiment_num}_${agent_model//[^a-zA-Z0-9]/_}_${social_world_model//[^a-zA-Z0-9]/_}"

    # Log file for this experiment
    local log_file="logs/${EXPERIMENT_BATCH}/${tag}.log"

    echo "Running experiment ${experiment_num}:"
    echo "Agent Model: ${agent_model}"
    echo "Social World Model: ${social_world_model}"
    echo "Log file: ${log_file}"

    # Run the experiment and log the output
    uv run python run_dynamic.py \
        --models "${agent_model}" \
        --partner-model "${PARTNER_MODEL}" \
        --agent-type "social_world_model" \
        --social-world-model-name "${social_world_model}" \
        --experiment-tag "${tag}" \
        --batch-size ${BATCH_SIZE} \
        --push-to-db \
        --evaluator-model "${EVALUATOR_MODEL}" \
        --task "${TASK}" \
        2>&1 | tee "${log_file}"

    # Append social world model name to the log file
    echo "Social World Model: ${social_world_model}" >> "${log_file}"

    # Add a separator between experiments
    echo "----------------------------------------"
}

# Run missing experiments
echo "Running missing experiments to complete the matrix..."
for exp in "${MISSING_EXPS[@]}"; do
    read -r agent_model social_world_model experiment_num <<< "$exp"
    run_experiment "${agent_model}" "${social_world_model}" "${experiment_num}"
done

echo "All experiments completed. Results are in logs/${EXPERIMENT_BATCH}/"
