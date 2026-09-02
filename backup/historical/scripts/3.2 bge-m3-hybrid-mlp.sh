#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
PYTHON_EXECUTABLE="python3"
# --- IMPORTANT: Point to your new HYBRID script ---
SCRIPT_NAME="3.2 bge-m3-hybrid-mlp.py"
TARGET_GPU=0
# --- IMPORTANT: Match the new top-level directory ---
TOP_LEVEL_MODEL_DIR="bge-m3_hybrid"

# Default training parameters from your hybrid script's args
EPOCHS=15
LR=0.0001
BATCH_SIZE=32   # Using the smaller default for hybrid
WEIGHT_DECAY=0.01
HIDDEN_DIM1=512
HIDDEN_DIM2=256
DROPOUT_RATE=0.5 # Using the higher default for hybrid

# Ensure the script name is correct
if [ ! -f "$SCRIPT_NAME" ]; then
    echo "Error: Python script '$SCRIPT_NAME' not found. Please update SCRIPT_NAME variable."
    exit 1
fi

# --- Set GPU ---
export CUDA_VISIBLE_DEVICES=$TARGET_GPU
echo "Using GPU: $TARGET_GPU"
echo ""

# --- Function to run a task (training or testing) ---
run_task() {
    local task_description="$1"
    local split_lang="$2"
    local method_name="$3"
    local use_filtered_flag="$4" # Pass "--use_filtered_sentences" or ""
    local test_only_flag="$5"    # Pass "--test" or ""

    echo "------------------------------------------------------"
    echo "Starting Task: $task_description"
    echo "Language: $split_lang, Method: $method_name"
    if [ -n "$use_filtered_flag" ]; then
        echo "Sentence Filtering Flag: ON"
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

    # Add filtering flag only if provided
    if [ -n "$use_filtered_flag" ]; then
        COMMAND+=("$use_filtered_flag")
    fi

    # Add test flag if provided
    if [ -n "$test_only_flag" ]; then
        COMMAND+=("$test_only_flag")
        # Add MLP params for consistency during testing
        COMMAND+=("--hidden_dim1" "$HIDDEN_DIM1" \
                  "--hidden_dim2" "$HIDDEN_DIM2" \
                  "--dropout_rate" "$DROPOUT_RATE" \
                  "--batch_size" "$BATCH_SIZE")
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
# --- DEFINE AND RUN YOUR HYBRID TASKS HERE ---
# ==============================================================================

echo "===== STARTING BGE-M3 HYBRID MLP TRAINING RUNS ====="
echo "Using common training parameters: epochs=$EPOCHS, lr=$LR, batch=$BATCH_SIZE, h1=$HIDDEN_DIM1, h2=$HIDDEN_DIM2, dr=$DROPOUT_RATE"
echo ""

# 1. Slovenian - Unmasked, on the complete document
#    method_name='whole' handles unmasking and using the whole document.
run_task "Train Hybrid: Slovenian, Whole (Unmasked)" \
         "slovenian" \
         "whole" \
         "" \
         "" # No test_only_flag, so it will train

# 2. Serbian - Unmasked, on the complete document
run_task "Train Hybrid: Serbian, Whole (Unmasked)" \
         "serbian" \
         "whole" \
         "" \
         "" # No test_only_flag

# 3. Slovenian - Masked, on the complete document
#    method_name='masked' with NO --use_filtered_sentences flag runs on the whole document.
run_task "Train Hybrid: Slovenian, Masked (Whole Doc)" \
         "slovenian" \
         "masked" \
         "" \
         "" # No test_only_flag

# 4. Serbian - Masked, on the complete document
run_task "Train Hybrid: Serbian, Masked (Whole Doc)" \
         "serbian" \
         "masked" \
         "" \
         "" # No test_only_flag


echo "======================================================"
echo "--- ALL SCRIPTED TASKS COMPLETED ---"
echo "Timestamp: $(date)"
echo "======================================================"