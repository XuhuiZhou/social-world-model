#!/bin/bash

# Define the models to test
AGENT_MODELS=(
    "gpt-4.1-2025-04-14"
    "o1-2024-12-17"
    "o1-mini-2024-09-12"
    "o3-2025-04-16"
    "o3-mini-2025-01-31"
    "together_ai/deepseek-ai/DeepSeek-R1"
    "together_ai/meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"
    "together_ai/meta-llama/Llama-4-Scout-17B-16E-Instruct"
)

# Fixed parameters
PARTNER_MODEL="gpt-4o-2024-08-06"
EVALUATOR_MODEL="o3-2025-04-16"
BATCH_SIZE=100
TASK="hard"

# Create a timestamp for the experiment batch
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EXPERIMENT_BATCH="vanilla_experiment_batch_${TIMESTAMP}"

# Create a directory for logs
mkdir -p "logs/${EXPERIMENT_BATCH}"

# Function to run a single experiment
run_experiment() {
    local agent_model=$1
    local experiment_num=$2

    # Create a unique tag for this experiment
    local tag="vanilla_agent_trial_${experiment_num}_${agent_model//[^a-zA-Z0-9]/_}"

    # Log file for this experiment
    local log_file="logs/${EXPERIMENT_BATCH}/${tag}.log"

    echo "Running experiment ${experiment_num}:"
    echo "Agent Model: ${agent_model}"
    echo "Log file: ${log_file}"

    # Run the experiment and log the output
    uv run python run_dynamic.py \
        --models "${agent_model}" \
        --partner-model "${PARTNER_MODEL}" \
        --agent-type "vanilla" \
        --experiment-tag "${tag}" \
        --batch-size ${BATCH_SIZE} \
        --push-to-db \
        --evaluator-model "${EVALUATOR_MODEL}" \
        --task "${TASK}" \
        2>&1 | tee "${log_file}"

    # Add a separator between experiments
    echo "----------------------------------------"
}

# Main experiment loop
experiment_num=1
for agent_model in "${AGENT_MODELS[@]}"; do
    run_experiment "${agent_model}" "${experiment_num}"
    experiment_num=$((experiment_num + 1))
done

echo "All experiments completed. Results are in logs/${EXPERIMENT_BATCH}/"
