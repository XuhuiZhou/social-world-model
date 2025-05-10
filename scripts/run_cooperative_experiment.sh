#!/bin/bash

# Define the models to test
AGENT_MODEL="o1-2024-12-17"
SOCIAL_WORLD_MODEL="o3-2025-04-16"

# Fixed parameters
PARTNER_MODEL="gpt-4o-2024-08-06"
EVALUATOR_MODEL="o3-2025-04-16"
BATCH_SIZE=100
TASK="cooperative"

# Create a timestamp for the experiment batch
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EXPERIMENT_BATCH="cooperative_experiment_batch_${TIMESTAMP}"

# Create a directory for logs
mkdir -p "logs/${EXPERIMENT_BATCH}"

# Function to run a single experiment
run_experiment() {
    local agent_type=$1
    local tag_prefix=$2

    # Create a unique tag for this experiment
    local tag="${tag_prefix}_${AGENT_MODEL//[^a-zA-Z0-9]/_}"
    if [ "$agent_type" = "social_world_model" ]; then
        tag="${tag}_${SOCIAL_WORLD_MODEL//[^a-zA-Z0-9]/_}"
    fi

    # Log file for this experiment
    local log_file="logs/${EXPERIMENT_BATCH}/${tag}.log"

    echo "Running ${agent_type} experiment:"
    echo "Agent Model: ${AGENT_MODEL}"
    if [ "$agent_type" = "social_world_model" ]; then
        echo "Social World Model: ${SOCIAL_WORLD_MODEL}"
    fi
    echo "Log file: ${log_file}"

    # Run the experiment and log the output
    uv run python run_dynamic.py \
        --models "${AGENT_MODEL}" \
        --partner-model "${PARTNER_MODEL}" \
        --agent-type "${agent_type}" \
        --social-world-model-name "${SOCIAL_WORLD_MODEL}" \
        --experiment-tag "${tag}" \
        --batch-size ${BATCH_SIZE} \
        --push-to-db \
        --evaluator-model "${EVALUATOR_MODEL}" \
        --task "${TASK}" \
        2>&1 | tee "${log_file}"

    echo "----------------------------------------"
}

# Run social world model experiment
run_experiment "social_world_model" "cooperative_social_world_model_trial"

# Run vanilla agent experiment
run_experiment "vanilla" "cooperative_vanilla_trial"

echo "All experiments completed. Results are in logs/${EXPERIMENT_BATCH}/"
