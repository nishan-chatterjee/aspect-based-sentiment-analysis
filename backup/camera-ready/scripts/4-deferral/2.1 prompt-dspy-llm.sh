#!/bin/bash

# Ensure the script exits if any command fails
set -e

# Base directory for models, relative to where the script is run from
# Assuming this script is in the same directory as "2.1 prompt-dspy-llm.py"
# or that "2.1 prompt-dspy-llm.py" is in the PATH or accessible.
# The output paths in your Python script are like ../models/ollama/...
# So this script should likely be run from the directory containing "2.1 prompt-dspy-llm.py"
# and the data/models directories are peers to the script's parent dir.
# For simplicity, I'll construct paths based on the Python script's output structure.

PYTHON_SCRIPT_NAME="2.1 prompt-dspy-llm.py"
BASE_OUTPUT_DIR="../models/ollama" # Matches the Python script's output structure

# Function to run a single experiment
run_experiment() {
    # All arguments to this function are part of the python command
    COMMAND_ARGS=("$@")

    # Extract model, method, split, and name for path construction
    MODEL=""
    METHOD=""
    SPLIT=""
    NAME=""

    # Parse command arguments to find the required values for path
    # This is a bit brittle if arg order changes, but works for your specific list
    i=0
    while [ $i -lt ${#COMMAND_ARGS[@]} ]; do
        case "${COMMAND_ARGS[$i]}" in
            --model) MODEL="${COMMAND_ARGS[$((i+1))]}"; ;;
            --method) METHOD="${COMMAND_ARGS[$((i+1))]}"; ;;
            --split) SPLIT="${COMMAND_ARGS[$((i+1))]}"; ;;
            --name) NAME="${COMMAND_ARGS[$((i+1))]}"; ;;
        esac
        i=$((i+1))
    done

    if [ -z "$MODEL" ] || [ -z "$METHOD" ] || [ -z "$SPLIT" ] || [ -z "$NAME" ]; then
        echo "Error: Could not extract --model, --method, --split, or --name from command: $@"
        exit 1
    fi

    LOG_DIR_PATH="${BASE_OUTPUT_DIR}/${MODEL}/${METHOD}/${SPLIT}"
    LOG_FILE_PATH="${LOG_DIR_PATH}/terminal_logs_${NAME}.txt"

    # Create directory for log file if it doesn't exist
    mkdir -p "$LOG_DIR_PATH"

    echo "----------------------------------------------------------------------"
    echo "Running Experiment: ${NAME}"
    echo "Model: ${MODEL}, Method: ${METHOD}, Split: ${SPLIT}"
    echo "Full Command: python3 ${PYTHON_SCRIPT_NAME} ${COMMAND_ARGS[*]}"
    echo "Output will be logged to: ${LOG_FILE_PATH}"
    echo "And also displayed on terminal."
    echo "----------------------------------------------------------------------"

    # Execute the command, tee output to both terminal and log file
    # Using unbuffer to ensure tqdm progress bars update correctly when piped
    # If unbuffer is not available, you might lose live progress bar updates in the file.
    # On most systems, `script -q -c "command" /dev/null` can also force pseudo-tty.
    # Simpler approach for now:
    if command -v stdbuf &> /dev/null; then
        stdbuf -oL -eL python3 "${PYTHON_SCRIPT_NAME}" "${COMMAND_ARGS[@]}" 2>&1 | tee "${LOG_FILE_PATH}"
    else
        echo "Warning: stdbuf not found. Progress bar updates might not be ideal in the log file."
        python3 "${PYTHON_SCRIPT_NAME}" "${COMMAND_ARGS[@]}" 2>&1 | tee "${LOG_FILE_PATH}"
    fi
    
    # Check exit status of the python script
    # `tee` will return 0 even if python script fails, so we need PIPESTATUS
    # This requires `set -o pipefail` at the beginning of the script, or check manually.
    # With `set -e`, if python3 fails, the script should exit.
    # If PIPESTATUS is available and you didn't use set -e, you could do:
    # if [ "${PIPESTATUS[0]}" -ne 0 ]; then
    # echo "Error: Experiment ${NAME} failed. Check log: ${LOG_FILE_PATH}"
    # exit 1
    # fi

    echo "Finished Experiment: ${NAME}"
    echo "----------------------------------------------------------------------"
    echo ""
}

# --- Define and Run Experiments ---

echo "Starting All Experiments..."
echo ""

# testing some basic performances between the two methods:
run_experiment --model gemma-3-27b --split slovenian --method direct --num-queries 3 --debug --name direct-debug
run_experiment --model gemma-3-27b --split slovenian --method plm-augmented-direct --num-queries 3 --debug --name plm-augmented-direct-debug

# checking the PLM signature change impacts (dspy-plm-augmented):
run_experiment --model gemma-3-27b --split slovenian --method dspy-plm-augmented --num-queries 3 --debug --dspy-autorun light --name dspy-plm-augmented-light-debug
run_experiment --model gemma-3-27b --split slovenian --method dspy-plm-augmented --use-plm-reliability-signature --num-queries 3 --debug --dspy-autorun light --name dspy-plm-augmented-light-debug-with-plm-sig

# checking the PLM signature change impacts (dspy-plm-augmented-cot):
run_experiment --model gemma-3-27b --split slovenian --method dspy-plm-augmented-cot --num-queries 3 --debug --dspy-autorun light --dspy-max-tokens 512 --name dspy-plm-augmented-cot-light-debug-512
run_experiment --model gemma-3-27b --split slovenian --method dspy-plm-augmented-cot --use-plm-reliability-signature --num-queries 3 --debug --dspy-autorun light --dspy-max-tokens 512 --name dspy-plm-augmented-cot-light-debug-with-plm-sig-512

# testing how miprov2-temp affects the optimized prompt:
run_experiment --model gemma-3-27b --split slovenian --method dspy-predict --num-queries 3 --debug --miprov2-temp 1.4 --name dspy-predict-miprov-1.4-debug

# testing if mentioning the aspect tags explicitly makes a difference:
run_experiment --model gemma-3-27b --split slovenian --method dspy-predict --num-queries 3 --debug --use-aspect-marker-signature --name dspy-predict-aspect-marker-sig-debug

# checking the basic performance distribution of the cot approaches:
run_experiment --model gemma-3-27b --split slovenian --method dspy-cot --num-queries 3 --debug --dspy-autorun light --dspy-max-tokens 512 --name dspy-cot-light-debug-512
# Note: dspy-plm-augmented-cot-light-debug-512 is already covered above. If you want a specific run here without PLM reliability, ensure the command differs.
# The one above (dspy-plm-augmented-cot-light-debug-512) is WITHOUT the reliability signature.

# testing how the light vs medium vs heavy configs converge:
run_experiment --model gemma-3-27b --split slovenian --method dspy-predict --num-queries 3 --debug --dspy-autorun light --name dspy-predict-light-debug
run_experiment --model gemma-3-27b --split slovenian --method dspy-predict --num-queries 3 --debug --dspy-autorun medium --name dspy-predict-medium-debug
run_experiment --model gemma-3-27b --split slovenian --method dspy-predict --num-queries 3 --debug --dspy-autorun heavy --name dspy-predict-heavy-debug

# testing whether having a bigger model affects the optmimized prompts:
run_experiment --model gemma-3-27b --split slovenian --method dspy-predict --num-queries 3 --debug --teacher-model-short-name qwen-2.5-72b --dspy-autorun light --name dspy-predict-teacher-qwen-light-debug
run_experiment --model gemma-3-27b --split slovenian --method dspy-plm-augmented --num-queries 3 --debug --dspy-autorun light --teacher-model-short-name qwen-2.5-72b --name dspy-plm-augmented-teacher-qwen-light-debug

echo ""
echo "All experiments finished."