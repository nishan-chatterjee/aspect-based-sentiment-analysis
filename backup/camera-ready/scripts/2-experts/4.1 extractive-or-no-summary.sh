#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Set the target GPU for all tasks
TARGET_GPU=7
export CUDA_VISIBLE_DEVICES=$TARGET_GPU

echo "--- Running ALL Test-Only Tasks on GPU $TARGET_GPU ---"
echo "Timestamp: $(date)"
echo "======================================================"

# Task 1: Slovenian Extractive Summary
echo ""
echo "Starting Task 1: Slovenian Extractive Summary on GPU $TARGET_GPU..."
python3 "4.1 extractive-or-no-summary-xlmr.py" --split slovenian --method_name extractive_summary --test_only
echo "Finished Task 1: Slovenian Extractive Summary on GPU $TARGET_GPU."
echo "--------------------------------------"

# Task 2: Serbian No Summary
echo ""
echo "Starting Task 2: Serbian No Summary on GPU $TARGET_GPU..."
python3 "4.1 extractive-or-no-summary-xlmr.py" --split serbian --method_name no_summary --test_only
echo "Finished Task 2: Serbian No Summary on GPU $TARGET_GPU."
echo "--------------------------------------"

# Task 3: Serbian Extractive Summary
echo ""
echo "Starting Task 3: Serbian Extractive Summary on GPU $TARGET_GPU..."
python3 "4.1 extractive-or-no-summary-xlmr.py" --split serbian --method_name extractive_summary --test_only
echo "Finished Task 3: Serbian Extractive Summary on GPU $TARGET_GPU."
echo "--------------------------------------"

# Task 4: Slovenian No Summary
echo ""
echo "Starting Task 4: Slovenian No Summary on GPU $TARGET_GPU..."
python3 "4.1 extractive-or-no-summary-xlmr.py" --split slovenian --method_name no_summary --test_only
echo "Finished Task 4: Slovenian No Summary on GPU $TARGET_GPU."
echo "--------------------------------------"


echo ""
echo "--- All tasks assigned to GPU $TARGET_GPU finished ---"
echo "Timestamp: $(date)"
echo "======================================================"