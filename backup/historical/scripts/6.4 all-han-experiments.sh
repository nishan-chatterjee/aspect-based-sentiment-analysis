#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.
# set -x # Uncomment for debugging: Print commands and their arguments as they are executed.

# --- Configuration ---
PYTHON_SCRIPT_NAME="6.4 hierarchical-attention-networks.py"
CUDA_DEVICE="0" # Set your desired CUDA device ID

# Common Hyperparameters
MODEL_NAME="xlm-roberta-base"
EPOCHS=10
BATCH_SIZE=4
EFF_BATCH_SIZE_TARGET=32
LR=1e-5
MAX_SENTENCES=128
MAX_SEQ_LENGTH=96
INTERACTION_LAYERS=2
INTERACTION_HEADS=8
AGGREGATION_HEADS=4
DROPOUT_RATE=0.2
FINAL_MLP_HIDDEN_DIM=256

# Base output directory prefix for the --method_name argument
# The Python script will create ../models/{BASE_METHOD_PREFIX}/{METHOD_SUB_NAME}/{LANGUAGE}
BASE_METHOD_PREFIX="han"

# Languages to iterate over
LANGUAGES=("slovenian" "serbian")

# Method configurations:
# Each entry is a string: "METHOD_SUB_NAME USE_ASPECT_MARKER_FLAG MASK_ASPECTS_FLAG"
# Flags are the actual flag string (e.g., "--use_aspect_marker") or an empty string if not used.
METHODS_CONFIG=(
    "no-aspect-markers '' ''"
    "with-aspect-markers '--use_aspect_marker' ''"
    "masked-no-aspect-markers '' '--mask_aspects'"
    "masked-with-aspect-markers '--use_aspect_marker' '--mask_aspects'"
)

# --- Main Loop ---
echo "Starting all HAN experiments..."
START_TIME=$(date +%s)

for lang in "${LANGUAGES[@]}"; do
    echo "======================================================================"
    echo "Processing Language: $lang"
    echo "======================================================================"

    for config_str in "${METHODS_CONFIG[@]}"; do
        # Read the config string into an array to separate its parts
        read -r -a config_parts <<< "$config_str"
        METHOD_SUB_NAME="${config_parts[0]}"
        USE_ASPECT_MARKER_FLAG="${config_parts[1]}" # Might be empty
        MASK_ASPECTS_FLAG="${config_parts[2]}"    # Might be empty

        # Construct the full method name to be passed to the Python script
        # This will be used by the script to create directories like: ../models/han/no-aspect-markers/slovenian
        FULL_METHOD_NAME_ARG="${BASE_METHOD_PREFIX}/${METHOD_SUB_NAME}"

        echo ""
        echo "----------------------------------------------------"
        echo "  Starting Experiment:"
        echo "    Language         : $lang"
        echo "    Method Sub-Name  : $METHOD_SUB_NAME"
        echo "    --method_name arg: $FULL_METHOD_NAME_ARG"
        echo "    Use Aspect Marker: ${USE_ASPECT_MARKER_FLAG:-Not Set}"
        echo "    Mask Aspects     : ${MASK_ASPECTS_FLAG:-Not Set}"
        echo "----------------------------------------------------"
        echo ""

        # Construct the command
        # Empty flag variables will be omitted by bash when the variable is expanded,
        # correctly not passing the flag to the python script.
        COMMAND="export CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} && \\
        python3 \"${PYTHON_SCRIPT_NAME}\" \\
            --split \"${lang}\" \\
            --method_name \"${FULL_METHOD_NAME_ARG}\" \\
            --model_name \"${MODEL_NAME}\" \\
            --epochs ${EPOCHS} \\
            --batch_size ${BATCH_SIZE} \\
            --eff_batch_size_target ${EFF_BATCH_SIZE_TARGET} \\
            --lr ${LR} \\
            --max_sentences ${MAX_SENTENCES} \\
            --max_seq_length ${MAX_SEQ_LENGTH} \\
            --interaction_layers ${INTERACTION_LAYERS} \\
            --interaction_heads ${INTERACTION_HEADS} \\
            --aggregation_heads ${AGGREGATION_HEADS} \\
            --dropout_rate ${DROPOUT_RATE} \\
            --final_mlp_hidden_dim ${FINAL_MLP_HIDDEN_DIM} \\
            ${USE_ASPECT_MARKER_FLAG} \\
            ${MASK_ASPECTS_FLAG}"
        
        # Print and execute the command
        echo "Executing command:"
        echo "$COMMAND"
        echo ""
        
        # Using eval to correctly handle the CUDA_VISIBLE_DEVICES export and the '&&' chain
        eval "$COMMAND"

        echo ""
        echo "  Finished experiment for $lang with $METHOD_SUB_NAME."
        
    done
    echo "======================================================================"
    echo ""
done

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo "All HAN experiments completed."
echo "Total execution time: $((DURATION / 3600))h $(((DURATION % 3600) / 60))m $((DURATION % 60))s"