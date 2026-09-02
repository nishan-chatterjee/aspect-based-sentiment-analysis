#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
PYTHON_EXECUTABLE="python3"
SCRIPT_NAME="3.1 bge-m3-mlp.py" # <--- CHANGE THIS TO YOUR ACTUAL SCRIPT FILENAME
TARGET_GPU=4
TOP_LEVEL_MODEL_DIR="bge-m3_mlp"

# Default training parameters - MATCHING YOUR "whole" MODEL'S ARGS
EPOCHS=15
LR=0.0001
BATCH_SIZE=64
WEIGHT_DECAY=0.01
HIDDEN_DIM1=512
HIDDEN_DIM2=256
DROPOUT_RATE=0.3
# bge_model_name, bge_embedding_dim are handled by the Python script.
# use_filtered_sentences is handled by the method_name logic and specific flags.

# Ensure the script name is correct
if [ ! -f "$SCRIPT_NAME" ]; then
    echo "Error: Python script '$SCRIPT_NAME' not found. Please update SCRIPT_NAME variable."
    exit 1
fi

# --- Set GPU ---
export CUDA_VISIBLE_DEVICES=$TARGET_GPU
echo "Using GPU: $TARGET_GPU"
echo ""

# --- Function to run a task (either training or testing) ---
run_task() {
    local task_description="$1"
    local split_lang="$2"
    local method_name="$3"
    local use_filtered_flag="$4" # Pass "--use_filtered_sentences" or ""
    local test_only_flag="$5"    # Pass "--test" or ""
    local extra_args="${@:6}"    # Any additional arguments (not used in this specific setup yet)

    echo "------------------------------------------------------"
    echo "Starting Task: $task_description"
    echo "Language: $split_lang, Method: $method_name"
    if [ -n "$use_filtered_flag" ]; then
        echo "Sentence Filtering Flag: ON (via $use_filtered_flag)"
    elif [ "$method_name" == "filtered" ]; then
        echo "Sentence Filtering: ON (implicit for method 'filtered')"
    else
        echo "Sentence Filtering: OFF"
    fi
    if [ -n "$test_only_flag" ]; then
        echo "Mode: TEST ONLY"
    else
        echo "Mode: TRAINING & VALIDATION"
        echo "Training Params: epochs=$EPOCHS, lr=$LR, batch=$BATCH_SIZE, wd=$WEIGHT_DECAY, h1=$HIDDEN_DIM1, h2=$HIDDEN_DIM2, dr=$DROPOUT_RATE"
    fi
    echo "Timestamp: $(date)"
    echo ""

    # Construct the command
    COMMAND=("$PYTHON_EXECUTABLE" "$SCRIPT_NAME" \
             "--split" "$split_lang" \
             "--method_name" "$method_name")

    # Add filtering flag only if provided (primarily for 'masked' method)
    if [ -n "$use_filtered_flag" ]; then
        COMMAND+=("$use_filtered_flag")
    fi

    # Add test flag if provided
    if [ -n "$test_only_flag" ]; then
        COMMAND+=("$test_only_flag")
        # For testing, ensure MLP params match what the model was trained with.
        # The Python script now tries to load these from the checkpoint if available.
        # If not in checkpoint, it uses the command-line args. So, for consistency
        # when testing, it's good if these command-line args for MLP match the training.
        COMMAND+=("--hidden_dim1" "$HIDDEN_DIM1" \
                  "--hidden_dim2" "$HIDDEN_DIM2" \
                  "--dropout_rate" "$DROPOUT_RATE")
    else
        # Add training-specific args if not in test-only mode
        COMMAND+=("--epochs" "$EPOCHS" \
                  "--batch_size" "$BATCH_SIZE" \
                  "--lr" "$LR" \
                  "--weight_decay" "$WEIGHT_DECAY" \
                  "--hidden_dim1" "$HIDDEN_DIM1" \
                  "--hidden_dim2" "$HIDDEN_DIM2" \
                  "--dropout_rate" "$DROPOUT_RATE")
    fi

    # Add any extra arguments passed to the function
    if [ -n "$extra_args" ]; then
        COMMAND+=($extra_args)
    fi

    # Print the command to be executed
    echo "Executing: ${COMMAND[*]}"
    echo ""

    # Execute the command
    "${COMMAND[@]}"

    echo ""
    echo "Finished Task: $task_description"
    echo "Timestamp: $(date)"
    echo "------------------------------------------------------"
    echo ""
}

# ==============================================================================
# --- DEFINE AND RUN YOUR TASKS HERE ---
# ==============================================================================

# --- Scenario 1: Run TRAINING for the specified variants ---
# These will now use the MLP and training parameters defined at the top of the script.

echo "===== STARTING TRAINING RUNS ====="
echo "Using common training parameters: epochs=$EPOCHS, lr=$LR, batch=$BATCH_SIZE, wd=$WEIGHT_DECAY, h1=$HIDDEN_DIM1, h2=$HIDDEN_DIM2, dr=$DROPOUT_RATE"
echo ""

# 1. Slovenian, Masked on Whole Document (no --use_filtered_sentences flag)
# run_task "Train: Slovenian, Masked (Whole Doc)" \
#          "slovenian" \
#          "masked" \
#          "" \
#          "" # No test_only_flag, so it will train

# 2. Serbian, Masked on Whole Document (no --use_filtered_sentences flag)
# run_task "Train: Serbian, Masked (Whole Doc)" \
#          "serbian" \
#          "masked" \
#          "" \
#          "" # No test_only_flag

# 3. Slovenian, Filtered
# For method_name "filtered", --use_filtered_sentences is implicit.
run_task "Train: Slovenian, Filtered" \
         "slovenian" \
         "filtered" \
         "" \
         "" # No test_only_flag

# 4. Serbian, Filtered
run_task "Train: Serbian, Filtered" \
         "serbian" \
         "filtered" \
         "" \
         "" # No test_only_flag

echo "===== FINISHED ALL TRAINING RUNS ====="
echo ""


# --- Scenario 2: Run TEST-ONLY for the specified variants ---
# Comment out the TRAINING section above and uncomment this section if you only want to test.
# Ensure models are pre-trained with the parameters defined at the top (or that the Python script
# can correctly load parameters from the checkpoint).

# echo "===== STARTING TEST-ONLY RUNS ====="
# echo "Ensure models are pre-trained and exist in their respective directories under ../models/$TOP_LEVEL_MODEL_DIR/"
# echo "MLP parameters for loading models (if not in checkpoint): h1=$HIDDEN_DIM1, h2=$HIDDEN_DIM2, dr=$DROPOUT_RATE"
# echo ""

# # 1. Test: Slovenian, Masked (Whole Doc)
# run_task "Test: Slovenian, Masked (Whole Doc)" \
#          "slovenian" \
#          "masked" \
#          "" \
#          "--test"

# # 2. Test: Serbian, Masked (Whole Doc)
# run_task "Test: Serbian, Masked (Whole Doc)" \
#          "serbian" \
#          "masked" \
#          "" \
#          "--test"

# # 3. Test: Slovenian, Filtered
# run_task "Test: Slovenian, Filtered" \
#          "slovenian" \
#          "filtered" \
#          "" \
#          "--test"

# # 4. Test: Serbian, Filtered
# run_task "Test: Serbian, Filtered" \
#          "serbian" \
#          "filtered" \
#          "" \
#          "--test"


echo "======================================================"
echo "--- ALL SCRIPTED TASKS COMPLETED ---"
echo "Timestamp: $(date)"
echo "======================================================"