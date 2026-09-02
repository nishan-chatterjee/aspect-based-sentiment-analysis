#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Set the target GPU for all tasks (you can change this)
TARGET_GPU=0 # Example: Use GPU 0. Change as needed.
export CUDA_VISIBLE_DEVICES=$TARGET_GPU

# Common training arguments based on your example
# (epochs, batch_size, lr, max_len will be used from script defaults or your new common values)
EPOCHS=10
BATCH_SIZE=32
LR=2e-05
MAX_LEN=512
# TRIPLET_LOSS will be false by default (no --triplet_loss flag)
# TRIPLET_MARGIN default is 1.0 (used if --triplet_loss is true)

echo "--- Starting XLM-R Training with Pre-generated Summaries on GPU $TARGET_GPU ---"
echo "Timestamp: $(date)"
echo "Common Args: --epochs $EPOCHS --batch_size $BATCH_SIZE --lr $LR --max_len $MAX_LEN"
echo "=============================================================================="

# --- Experiment 1: Slovenian, TextRank Summary ---
SPLIT_LANG="slovenian"
METHOD="textrank-summary"
echo ""
echo "Starting Experiment 1: $SPLIT_LANG, $METHOD on GPU $TARGET_GPU..."
python3 "4.3 llm-textrank-summarization-xlmr.py" \
    --split "$SPLIT_LANG" \
    --method_name "$METHOD" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --max_len "$MAX_LEN"
echo "Finished Experiment 1: $SPLIT_LANG, $METHOD on GPU $TARGET_GPU."
echo "------------------------------------------------------------------------------"

# --- Experiment 2: Serbian, TextRank Summary ---
SPLIT_LANG="serbian"
METHOD="textrank-summary"
echo ""
echo "Starting Experiment 2: $SPLIT_LANG, $METHOD on GPU $TARGET_GPU..."
python3 "4.3 llm-textrank-summarization-xlmr.py" \
    --split "$SPLIT_LANG" \
    --method_name "$METHOD" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --max_len "$MAX_LEN"
echo "Finished Experiment 2: $SPLIT_LANG, $METHOD on GPU $TARGET_GPU."
echo "------------------------------------------------------------------------------"

# --- Experiment 3: Slovenian, GAMS-9B Summary ---
SPLIT_LANG="slovenian"
METHOD="gams-9b-summary"
echo ""
echo "Starting Experiment 3: $SPLIT_LANG, $METHOD on GPU $TARGET_GPU..."
python3 "4.3 llm-textrank-summarization-xlmr.py" \
    --split "$SPLIT_LANG" \
    --method_name "$METHOD" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --max_len "$MAX_LEN"
echo "Finished Experiment 3: $SPLIT_LANG, $METHOD on GPU $TARGET_GPU."
echo "------------------------------------------------------------------------------"

# --- Experiment 4: Serbian, GAMS-9B Summary ---
SPLIT_LANG="serbian"
METHOD="gams-9b-summary"
echo ""
echo "Starting Experiment 4: $SPLIT_LANG, $METHOD on GPU $TARGET_GPU..."
python3 "4.3 llm-textrank-summarization-xlmr.py" \
    --split "$SPLIT_LANG" \
    --method_name "$METHOD" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --max_len "$MAX_LEN"
echo "Finished Experiment 4: $SPLIT_LANG, $METHOD on GPU $TARGET_GPU."
echo "------------------------------------------------------------------------------"

# --- Experiment 5: Slovenian, Gemma-3-27B Summary ---
SPLIT_LANG="slovenian"
METHOD="gemma-3-27b-summary"
echo ""
echo "Starting Experiment 5: $SPLIT_LANG, $METHOD on GPU $TARGET_GPU..."
python3 "4.3 llm-textrank-summarization-xlmr.py" \
    --split "$SPLIT_LANG" \
    --method_name "$METHOD" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --max_len "$MAX_LEN"
echo "Finished Experiment 5: $SPLIT_LANG, $METHOD on GPU $TARGET_GPU."
echo "------------------------------------------------------------------------------"

# --- Experiment 6: Serbian, Gemma-3-27B Summary ---
SPLIT_LANG="serbian"
METHOD="gemma-3-27b-summary"
echo ""
echo "Starting Experiment 6: $SPLIT_LANG, $METHOD on GPU $TARGET_GPU..."
python3 "4.3 llm-textrank-summarization-xlmr.py" \
    --split "$SPLIT_LANG" \
    --method_name "$METHOD" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --max_len "$MAX_LEN"
echo "Finished Experiment 6: $SPLIT_LANG, $METHOD on GPU $TARGET_GPU."
echo "------------------------------------------------------------------------------"
echo ""
echo "--- All training experiments finished ---"
echo "Timestamp: $(date)"


      
# --- Experiment 7: Slovenian, Gemma-3-27B Summary (with Masked Aspect) ---
SPLIT_LANG="slovenian"
METHOD="gemma-3-27b-summary"
echo ""
echo "Starting Experiment 7: $SPLIT_LANG, $METHOD with MASKED ASPECT on GPU $TARGET_GPU..."
python3 "4.3 llm-textrank-summarization-xlmr.py" \
    --split "$SPLIT_LANG" \
    --method_name "$METHOD" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --max_len "$MAX_LEN" \
    --mask_aspect  # Add this flag
echo "Finished Experiment: $SPLIT_LANG, $METHOD with MASKED ASPECT on GPU $TARGET_GPU."
echo "------------------------------------------------------------------------------"

    
# --- Experiment 8: Serbian, Gemma-3-27B Summary (with Masked Aspect) ---
SPLIT_LANG="serbian"
METHOD="gemma-3-27b-summary"
echo ""
echo "Starting Experiment 8: $SPLIT_LANG, $METHOD with MASKED ASPECT on GPU $TARGET_GPU..."
python3 "4.3 llm-textrank-summarization-xlmr.py" \
    --split "$SPLIT_LANG" \
    --method_name "$METHOD" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --max_len "$MAX_LEN" \
    --mask_aspect  # Add this flag
echo "Finished Experiment: $SPLIT_LANG, $METHOD with MASKED ASPECT on GPU $TARGET_GPU."
echo "------------------------------------------------------------------------------"