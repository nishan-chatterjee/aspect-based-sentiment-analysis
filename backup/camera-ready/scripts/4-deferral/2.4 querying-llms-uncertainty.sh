#!/bin/bash

# This script runs the 4 specified experiments for the 'uncertainty' prompt strategy.
# It includes a function to automatically retry a command if it fails.
set -e

# --- Common Settings ---
PYTHON_SCRIPT="2.4 querying-llms-uncertainty.py"
MODEL_NAME="gemma-3-27b"
TEACHER_MODEL="qwen-2.5-72b"
MAX_TOKENS=1024
PROMPT_STRATEGY="uncertainty"

# --- Retry Logic Configuration ---
MAX_RETRIES=3
RETRY_DELAY_SECONDS=60 # 1 minute

# --- Helper Function for Retries ---
run_with_retries() {
    local description="$1"
    shift # Remove the description from the arguments list
    local cmd=("$@")

    for i in $(seq 1 $MAX_RETRIES); do
        echo "------------------------------------------------------------------------"
        echo "Attempt $i of $MAX_RETRIES for: $description"
        echo "Running command: ${cmd[*]}"
        echo "------------------------------------------------------------------------"
        
        # Execute the command
        "${cmd[@]}"

        # Check the exit code of the command
        local exit_code=$?
        if [ $exit_code -eq 0 ]; then
            echo "Command successful for: $description"
            return 0 # Success
        fi

        echo "------------------------------------------------------------------------"
        echo "Warning: Attempt $i failed with exit code $exit_code for: $description"
        
        if [ $i -lt $MAX_RETRIES ]; then
            echo "Waiting ${RETRY_DELAY_SECONDS} seconds before retrying..."
            sleep $RETRY_DELAY_SECONDS
        fi
        echo "------------------------------------------------------------------------"
    done

    echo "Error: Command failed after $MAX_RETRIES attempts for: $description"
    return 1 # Failure
}


# --- Experiment 1: Slovenian, No Mask, Autorun Heavy ---
# run_with_retries "Experiment 1: Slovenian | Unmasked | Autorun Heavy" \
#     python3 "${PYTHON_SCRIPT}" \
#     --model "${MODEL_NAME}" \
#     --split "slovenian" \
#     --name "dspy-plm-augmented-cot-teacher-qwen-1024-heavy-uncertainty-slovenian-unmasked" \
#     --teacher-model-short-name "${TEACHER_MODEL}" \
#     --dspy-max-tokens "${MAX_TOKENS}" \
#     --dspy-autorun "heavy" \
#     --prompt-strategy "${PROMPT_STRATEGY}"
# 
# 
# # --- Experiment 2: Serbian, No Mask, Autorun Medium ---
# run_with_retries "Experiment 2: Serbian | Unmasked | Autorun Medium" \
#     python3 "${PYTHON_SCRIPT}" \
#     --model "${MODEL_NAME}" \
#     --split "serbian" \
#     --name "dspy-plm-augmented-cot-teacher-qwen-1024-medium-uncertainty-serbian-unmasked" \
#     --teacher-model-short-name "${TEACHER_MODEL}" \
#     --dspy-max-tokens "${MAX_TOKENS}" \
#     --dspy-autorun "medium" \
#     --prompt-strategy "${PROMPT_STRATEGY}"
# 
# 
# # --- Experiment 3: Slovenian, Mask, Autorun Heavy ---
# run_with_retries "Experiment 3: Slovenian | Masked | Autorun Heavy" \
#     python3 "${PYTHON_SCRIPT}" \
#     --model "${MODEL_NAME}" \
#     --split "slovenian" \
#     --mask \
#     --name "dspy-plm-augmented-cot-teacher-qwen-1024-heavy-uncertainty-slovenian-masked" \
#     --teacher-model-short-name "${TEACHER_MODEL}" \
#     --dspy-max-tokens "${MAX_TOKENS}" \
#     --dspy-autorun "heavy" \
#     --prompt-strategy "${PROMPT_STRATEGY}"


# --- Experiment 4: Serbian, Mask, Autorun Medium ---
run_with_retries "Experiment 4: Serbian | Masked | Autorun Medium" \
    python3 "${PYTHON_SCRIPT}" \
    --model "${MODEL_NAME}" \
    --split "serbian" \
    --mask \
    --name "dspy-plm-augmented-cot-teacher-qwen-1024-medium-uncertainty-serbian-masked" \
    --teacher-model-short-name "${TEACHER_MODEL}" \
    --dspy-max-tokens "${MAX_TOKENS}" \
    --dspy-autorun "medium" \
    --prompt-strategy "${PROMPT_STRATEGY}"


echo "========================================================================"
echo "All experiments completed successfully."
echo "========================================================================"