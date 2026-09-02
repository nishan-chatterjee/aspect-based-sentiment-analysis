#!/bin/bash

# Treat unset variables as an error.
set -u
# Prevent errors in pipelines from being masked.
set -o pipefail
# We won't use set -e globally as we handle exit codes manually

# --- Configuration ---
PYTHON_EXECUTABLE="python3" # Or just "python" if that's your environment
SCRIPT_PATH="5.1 llm-summarization.py" # Relative path to your Python script
BASE_OUTPUT_DIR="../data/final/summaries"
MAX_RETRIES=10 # Set a limit for retries per job (0 for infinite)
RETRY_DELAY_SECONDS=10 # Seconds to wait before retrying

# --- Define the jobs ---
# Each element is "split;model"
JOBS=(
    "slovenian;gams-9b"
    "serbian;gams-9b"
    "slovenian;gemma-3-27b"
    "serbian;gemma-3-27b"
)

# --- Function to run a single job until completion or max retries ---
run_job() {
    local split="$1"
    local model="$2"
    local model_dir="${BASE_OUTPUT_DIR}/${model}"
    local log_dir="${model_dir}/summary-log-${split}"
    local final_output_file="${model_dir}/${split}.json" # Path to the expected final file
    local python_command="${PYTHON_EXECUTABLE} \"${SCRIPT_PATH}\" --split \"${split}\" --model \"${model}\""
    local retry_count=0

    echo "-----------------------------------------------------"
    echo "Starting job: Split=${split}, Model=${model}"
    echo "Expecting final output: ${final_output_file}"
    echo "Temporary log directory: ${log_dir}"
    echo "Command: ${python_command}"
    echo "Retry limit: ${MAX_RETRIES} (0 means infinite)"
    echo "-----------------------------------------------------"

    # Loop while the log directory exists OR the final output file doesn't exist
    # AND we haven't exceeded max retries (if set)
    while [ -d "$log_dir" ] || [ ! -f "$final_output_file" ]; do

        # Check retry limit
        if [ "$MAX_RETRIES" -gt 0 ] && [ "$retry_count" -ge "$MAX_RETRIES" ]; then
             echo ""
             echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
             echo "ERROR: Maximum retries ($MAX_RETRIES) reached for Split=${split}, Model=${model}."
             echo "The job did not complete successfully."
             echo "Log directory status: $( [ -d "$log_dir" ] && echo "Exists" || echo "Gone" )"
             echo "Final file status: $( [ -f "$final_output_file" ] && echo "Exists" || echo "Missing" )"
             echo "Manual intervention required. Stopping script."
             echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
             exit 1 # Exit the entire script on max retries failure
        fi

        if [ "$retry_count" -gt 0 ]; then
            echo "$(date '+%Y-%m-%d %H:%M:%S') - Job incomplete (Log: $( [ -d "$log_dir" ] && echo "Exists" || echo "Gone" ), Final File: $( [ -f "$final_output_file" ] && echo "Exists" || echo "Missing" )). Retrying (Attempt $((retry_count + 1)) / ${MAX_RETRIES:-Infinite})..."
            echo "Waiting ${RETRY_DELAY_SECONDS} seconds before retry..."
            sleep ${RETRY_DELAY_SECONDS}
        else
             echo "$(date '+%Y-%m-%d %H:%M:%S') - Executing job command (Attempt 1)..."
        fi

        # Execute the command and capture its exit code
        echo "Running command: ${PYTHON_EXECUTABLE} ${SCRIPT_PATH} --split ${split} --model ${model}"
        ${PYTHON_EXECUTABLE} "${SCRIPT_PATH}" --split "${split}" --model "${model}"
        local exit_code=$?

        # Increment retry counter
        retry_count=$((retry_count + 1))

        # Log the exit code, but don't exit the loop based on it
        if [ "$exit_code" -ne 0 ]; then
             echo "$(date '+%Y-%m-%d %H:%M:%S') - Warning: Python script for Split=${split}, Model=${model} finished attempt with non-zero exit status (${exit_code}). Will check completion criteria and retry if needed."
        else
             echo "$(date '+%Y-%m-%d %H:%M:%S') - Python script for Split=${split}, Model=${model} finished attempt with exit status 0."
        fi

        # The loop condition `[ -d "$log_dir" ] || [ ! -f "$final_output_file" ]`
        # will determine if another iteration (retry) is needed based on the
        # presence of the log dir or absence of the final file.

    done

    echo ""
    echo "-----------------------------------------------------"
    echo "SUCCESS: Job completed for Split=${split}, Model=${model} after ${retry_count} attempt(s)."
    echo "Log directory '${log_dir}' was successfully removed."
    echo "Final output file '${final_output_file}' exists."
    echo "-----------------------------------------------------"
    echo ""
    sleep 2 # Small pause before next job
}

# --- Main Execution ---

# Check if the script path exists
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "ERROR: Python script not found at '${SCRIPT_PATH}'"
    exit 1
fi

# Iterate through the defined jobs
job_count=${#JOBS[@]}
current_job_num=1
for job in "${JOBS[@]}"; do
    echo ""
    echo "====================================================="
    echo "Processing Job ${current_job_num} of ${job_count}"
    echo "====================================================="
    IFS=';' read -r split model <<< "$job" # Split the job string
    run_job "$split" "$model"
    current_job_num=$((current_job_num + 1))
done

echo "====================================================="
echo "All defined jobs completed successfully."
echo "====================================================="

exit 0