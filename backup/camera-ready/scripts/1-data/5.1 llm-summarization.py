# --- START OF FINAL llm-summarization.py ---

import json
import asyncio
import aiohttp
import time
import os
import sys
import argparse
import shutil
import re # Added for is_garbled
from collections import Counter # Added for is_garbled
from tqdm.asyncio import tqdm_asyncio
from datetime import datetime
from tqdm import tqdm

# Attempt to import lexicalrichness, handle if not installed
try:
    from lexicalrichness import LexicalRichness
    LEXICALRICHNESS_AVAILABLE = True
except ImportError:
    print("Warning: 'lexicalrichness' library not found. TTR check will be skipped.")
    print("Install it using: pip install lexicalrichness")
    LEXICALRICHNESS_AVAILABLE = False

# --- Configuration ---
OLLAMA_URL = "http://127.0.0.1:11435/api/generate"
MAX_PARALLEL_REQUESTS = 4
BATCH_SIZE = 50
TIMEOUT = 120 # Timeout per *single* API request attempt

MODEL_MAPPING = {
    "gemma-3-27b": "gemma3:27b-it-qat",
    "gams-9b": "hf.co/tknez/GaMS-9B-Instruct-GGUF:latest",
}

LANGUAGE_NAME_MAP = {
    'sl': 'Slovenian',
    'sr': 'Serbo-Croatian',
    'hr': 'Serbo-Croatian',
    'bs': 'Serbo-Croatian',
    'sh': 'Serbo-Croatian'
}

# Basic stop words (expand for better accuracy if needed)
STOP_WORDS_SL = {'v', 'in', 'na', 's', 'z', 'je', 'so', 'se', 'pa', 'ter', 'ki', 'ko', 'kot', 'do', 'ob', 'pri', 'za', 'd', 'dd'}
STOP_WORDS_SH = {'u', 'i', 'na', 'sa', 'je', 'su', 'se', 'pa', 'te', 'koji', 'koja', 'koje', 'kao', 'do', 'kod', 'za'}

semaphore = asyncio.Semaphore(MAX_PARALLEL_REQUESTS)

# --- Prompt Formatting ---
def format_prompt_simplified(article, aspect, language_code):
    """Format the simplified prompt for the LLM, using specific language code."""
    language_name = LANGUAGE_NAME_MAP.get(language_code, language_code)

    return f"""Task: Create a concise summary focusing on the aspect "{aspect}" from the following newspaper article. The aspect is marked with <aspect></aspect> tags in the input. The summary should help understand the aspect's role and the sentiment towards it in the article.

    Key Instructions:
    1.  **Focus:** Concentrate on the parts of the article discussing or closely related to the aspect "{aspect}" (marked with <aspect> tags). Include immediate context if it's important for understanding the situation or sentiment.
    2.  **Content & Sentiment:** Summarize the main points *about the aspect* and accurately reflect the sentiment (positive, negative, or neutral) conveyed towards it in the article.
    3.  **Accuracy:** Base your summary strictly on the facts presented in the provided article text. Do not add information not present in the text or misattribute details.
    4.  **Language:** Your response MUST be in {language_name}. Do NOT translate to English.
    5.  **Conciseness:** Keep the summary reasonably brief (e.g., around 3-6 sentences), capturing the essential points without excessive detail.

    Article (in {language_name}):
    {article}

    Aspect:
    {aspect}

    Answer (in {language_name}):"""


# --- Garbled Text Detection ---
def is_garbled(text, language_code='sl', min_words=10, ttr_threshold=0.5, freq_word_threshold=0.30):
    """
    Checks if a text string appears garbled due to excessive repetition or low diversity.
    """
    if not text or not isinstance(text, str):
        return True # Treat empty or non-string as garbled

    # Simple tokenization
    words = re.findall(r'\b\w+\b', text.lower())

    if len(words) < min_words:
        return False # Too short to reliably analyze

    # 1. Check Lexical Diversity (TTR) if library is available
    if LEXICALRICHNESS_AVAILABLE:
        try:
            lex = LexicalRichness(text)
            # Handle potential division by zero if text becomes empty after cleaning by LexicalRichness
            if lex.words == 0:
                 ttr = 0.0
            else:
                 ttr = lex.ttr

            if ttr < ttr_threshold:
                # print(f"DEBUG: TTR check failed: {ttr:.2f} < {ttr_threshold}") # Optional debug
                return True
        except Exception as e:
            print(f"Warning: LexicalRichness calculation failed: {e}")
            # Continue to other checks

    # 2. Check High-Frequency Content Word
    stop_words = STOP_WORDS_SL if language_code == 'sl' else STOP_WORDS_SH
    content_words = [word for word in words if word not in stop_words and len(word) >= 3]

    if not content_words:
         return False # No content words to analyze frequency

    word_counts = Counter(content_words)
    # Check if most_common returns anything
    if not word_counts:
        return False

    most_common_word, most_common_count = word_counts.most_common(1)[0]
    frequency_percentage = most_common_count / len(content_words)

    if frequency_percentage > freq_word_threshold:
        # print(f"DEBUG: Freq check failed: Word '{most_common_word}' has freq {frequency_percentage:.2f} > {freq_word_threshold}") # Optional debug
        return True

    # 3. Check for highly repetitive short non-alphanumeric sequences (like 'a. a. a.')
    # Find sequences of 1-3 chars possibly separated by space/dot that repeat often
    short_patterns = re.findall(r'(?:^|\s)((?:\S{1,3}[.\s]?){3,})(?=\s|$)', text) # Find sequences of 3+ short tokens
    if short_patterns:
        pattern_counts = Counter(short_patterns)
        for pattern, count in pattern_counts.items():
            # If a short repetitive pattern appears multiple times
            if count > 3 and len(pattern) < 30: # Adjust thresholds
                 # Check if this pattern dominates the text
                 if text.count(pattern) * len(pattern) > 0.4 * len(text):
                      # print(f"DEBUG: Short pattern check failed: '{pattern}'")
                      return True

    return False # Passed checks


# --- Asynchronous Request Function with Retry Logic ---
async def fetch_response_with_retry(session, ollama_model_string, article_data, item_language_code, global_index):
    """Sends requests to Ollama, retrying with stricter parameters if output is garbled."""
    article = article_data.get("article", "")
    aspect = article_data.get("aspect", "")
    uuid = article_data.get("uuid", f"index_{global_index}")

    base_payload = {
        "model": ollama_model_string,
        "prompt": format_prompt_simplified(article, aspect, item_language_code),
        "stream": False
    }

    # Define parameter sets for retry attempts
    parameter_sets = [
        {"repeat_penalty": 1.15, "temperature": 0.75}, # Attempt 1: Balanced
        {"repeat_penalty": 1.3, "temperature": 0.6, "top_k": 40}, # Attempt 2: Stricter repetition penalty
        {"repeat_penalty": 1.5, "temperature": 0.5, "top_k": 30, "top_p": 0.85} # Attempt 3: Very strict
    ]

    attempts_results = [] # Store results: {"text": str, "ttr": float, "error": str/None}
    final_summary = f"Error: All {len(parameter_sets)} attempts failed validation or API calls."
    final_success = False
    final_error_reason = "Initialization failure" # Default error reason
    elapsed_total = 0.0

    for i, params in enumerate(parameter_sets):
        current_payload = base_payload.copy()
        current_payload["options"] = params
        attempt_success = False
        attempt_error = None
        result_text = None

        start_time = time.time()
        try:
            async with semaphore: # Apply semaphore to each attempt
                async with session.post(OLLAMA_URL, json=current_payload, timeout=TIMEOUT) as response:
                    elapsed = time.time() - start_time
                    elapsed_total += elapsed

                    if response.status == 200:
                        data = await response.json()
                        result_text = data.get("response", "").strip()
                        if not result_text:
                            attempt_error = f"Attempt {i+1}: Received empty response"
                            print(f"Warning: {attempt_error} for index {global_index}.")
                        else:
                            # Validate the result
                            is_bad = is_garbled(result_text, language_code=item_language_code)
                            if not is_bad:
                                # *** Success! Use this result ***
                                final_summary = result_text
                                final_success = True
                                final_error_reason = None # No error
                                print(f"Info: Index {global_index} succeeded on attempt {i+1}.")
                                attempt_success = True # Mark this attempt as successful
                                # Store this good result in case needed later (though we break)
                                attempts_results.append({"text": result_text, "ttr": 1.0, "error": None}) # Assign high TTR
                                break # Exit the retry loop
                            else:
                                attempt_error = f"Attempt {i+1}: Flagged as garbled"
                                print(f"Warning: {attempt_error} for index {global_index}.")
                    else:
                        error_text_api = await response.text()
                        attempt_error = f"Attempt {i+1}: API Error {response.status}"
                        print(f"Warning: {attempt_error} for index {global_index}. Details: {error_text_api[:200]}") # Log snippet
                        result_text = f"Error during attempt {i+1}: API Error {response.status}" # Store error in text

        except asyncio.TimeoutError:
            elapsed = time.time() - start_time # Use actual time if possible, else TIMEOUT
            elapsed_total += elapsed if elapsed < TIMEOUT else TIMEOUT
            attempt_error = f"Attempt {i+1}: Timed out after {TIMEOUT}s"
            print(f"Warning: {attempt_error} for index {global_index}.")
            result_text = f"Error during attempt {i+1}: Timeout"
        except aiohttp.ClientConnectorError as e:
             elapsed = time.time() - start_time
             elapsed_total += elapsed
             attempt_error = f"Attempt {i+1}: Connection Error - {e}"
             print(f"Warning: {attempt_error} for index {global_index}.")
             result_text = f"Error during attempt {i+1}: Connection Error"
        except Exception as e:
            elapsed = time.time() - start_time
            elapsed_total += elapsed
            attempt_error = f"Attempt {i+1}: Exception - {type(e).__name__}: {e}"
            print(f"Warning: {attempt_error} for index {global_index}.")
            result_text = f"Error during attempt {i+1}: Exception"

        # Store result of this attempt (even if failed validation or API error)
        if not attempt_success:
            current_ttr = 0.0
            if result_text and not attempt_error.startswith("Attempt"): # Only calc TTR if we got some text back and didn't fail API/timeout
                 if LEXICALRICHNESS_AVAILABLE:
                     try:
                         lex = LexicalRichness(result_text)
                         current_ttr = lex.ttr if lex.words > 0 else 0.0
                     except Exception:
                         current_ttr = 0.0 # Assign low TTR if calculation fails
            attempts_results.append({"text": result_text or attempt_error, "ttr": current_ttr, "error": attempt_error})

        # If we found a successful one, we break out early
        if final_success:
            break

    # --- Handle Fallback if all attempts failed validation/API ---
    if not final_success and attempts_results:
        print(f"Info: All {len(parameter_sets)} attempts failed validation/API for index {global_index}. Selecting best failure.")
        # Sort attempts by TTR (higher is better), prioritizing non-error texts
        attempts_results.sort(key=lambda x: (x["error"] is None, x["ttr"]), reverse=True)
        best_failure = attempts_results[0]
        final_summary = best_failure["text"] # Choose the one with highest TTR among failures
        final_success = False # Still mark as failure overall
        final_error_reason = best_failure.get("error", "All attempts failed; returning best effort.")
        if final_error_reason is None: # Ensure there's an error reason if success is False
             final_error_reason = "All attempts failed validation; returning best effort (highest TTR)."


    # --- Return the final result ---
    return {
        "global_index": global_index,
        "uuid": uuid,
        "summary": final_summary, # Best success or best failure text
        "time": elapsed_total, # Sum of time across attempts
        "success": final_success, # True only if a non-garbled summary was found
        "error": final_error_reason # Describes the final state or None if success
    }


# --- File and Progress Management ---
# get_progress_filename, load_progress, save_progress,
# get_checkpoint_filename, load_results_from_checkpoints remain the same as before
def get_progress_filename(temp_log_dir):
    return os.path.join(temp_log_dir, "processing_progress.json")

def load_progress(filename):
    progress = {"completed_indices": [], "failed_indices": []}
    if os.path.exists(filename):
        try:
            with open(filename, "r", encoding="utf-8") as f:
                progress_data = json.load(f)
                progress["completed_indices"] = progress_data.get("completed_indices", [])
                progress["failed_indices"] = progress_data.get("failed_indices", [])
        except (json.JSONDecodeError, TypeError) as e:
             print(f"Warning: Progress file {filename} corrupted or invalid format ({e}). Starting fresh.")
             try:
                 corrupt_backup_name = f"{filename}.corrupt_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                 shutil.copy(filename, corrupt_backup_name)
                 print(f"Backed up corrupted progress file to {corrupt_backup_name}")
             except Exception as backup_e:
                 print(f"Could not back up corrupted progress file: {backup_e}")
        except Exception as e:
            print(f"Warning: Could not load progress file {filename}: {e}. Starting fresh.")
    progress["completed_indices"] = [int(i) for i in progress.get("completed_indices", [])]
    progress["failed_indices"] = [int(i) for i in progress.get("failed_indices", [])]
    return progress

def save_progress(filename, progress_data):
    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        progress_data["completed_indices"] = sorted(list(set(map(int, progress_data.get("completed_indices", [])))))
        progress_data["failed_indices"] = sorted(list(set(map(int, progress_data.get("failed_indices", [])))))
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(progress_data, f, indent=2)
    except Exception as e:
        print(f"Error saving progress to {filename}: {e}")

def get_checkpoint_filename(temp_log_dir, batch_start_index):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return os.path.join(temp_log_dir, f"checkpoint_{batch_start_index:06d}_{timestamp}.json")

def load_results_from_checkpoints(temp_log_dir):
    """Loads successful items from checkpoints, ensuring global_index exists."""
    successful_items_map = {}
    if not os.path.isdir(temp_log_dir):
        return {}
    for filename in os.listdir(temp_log_dir):
        if filename.startswith("checkpoint_") and filename.endswith(".json"):
            filepath = os.path.join(temp_log_dir, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    batch_results = json.load(f)
                    for item in batch_results:
                        # Checkpoints should only contain successfully summarized items
                        if 'global_index' in item and item.get('summary') and not item.get('summary','').startswith("Error:"):
                            successful_items_map[item['global_index']] = item
                        # else:
                        #      print(f"Debug: Skipping item in checkpoint {filepath} due to missing index or error summary.")
            except json.JSONDecodeError:
                 print(f"Warning: Checkpoint file {filepath} is corrupted. Skipping.")
            except Exception as e:
                print(f"Warning: Could not load checkpoint file {filepath}: {e}")
    return successful_items_map


# --- Main Processing Logic ---
async def process_dataset(ollama_model_string, language, dataset, debug_limit, temp_log_dir):
    """Processes the dataset, focusing on incomplete/failed items based on progress."""
    total_items_in_dataset = len(dataset)
    limit = debug_limit if debug_limit else total_items_in_dataset
    target_indices_set = set(range(limit))

    print(f"\nProcessing {language} dataset with {ollama_model_string}")
    print(f"Target items: {limit} (Total in dataset: {total_items_in_dataset})")

    os.makedirs(temp_log_dir, exist_ok=True)
    progress_file = get_progress_filename(temp_log_dir)

    progress = load_progress(progress_file)
    successful_items_map = load_results_from_checkpoints(temp_log_dir) # Load only successful from checkpoints
    print(f"Loaded {len(successful_items_map)} confirmed successful items from checkpoints.")
    print(f"Progress file indicates: {len(progress['completed_indices'])} completed attempts, {len(progress['failed_indices'])} failed attempts.")

    # Indices to run: target indices that are NOT in the successful checkpoint map
    processed_successful_indices = set(successful_items_map.keys())
    indices_to_run_set = target_indices_set - processed_successful_indices
    indices_to_run = sorted(list(indices_to_run_set))

    if not indices_to_run:
        print("All target items seem to be successfully processed based on checkpoints.")
        final_items = [item for idx, item in successful_items_map.items() if idx in target_indices_set]
        # Ensure the progress file reflects this if it was lagging
        progress['completed_indices'] = sorted(list(target_indices_set))
        progress['failed_indices'] = [idx for idx in progress['failed_indices'] if idx not in target_indices_set]
        save_progress(progress_file, progress)
        return final_items, []

    print(f"Identified {len(indices_to_run)} items to process/retry within the target range.")

    newly_successful_items_this_run = {}
    failed_indices_this_run = [] # Track final failures *from this run*

    async with aiohttp.ClientSession() as session:
        overall_progress = tqdm(
            total=len(indices_to_run),
            desc=f"Overall {language} progress ({ollama_model_string.split(':')[0]})",
            unit="articles",
            position=0,
            leave=True
        )

        for i in range(0, len(indices_to_run), BATCH_SIZE):
            batch_indices = indices_to_run[i : i + BATCH_SIZE]
            tasks = []
            for idx in batch_indices:
                if 0 <= idx < len(dataset):
                    item_data = dataset[idx]
                    item_lang_code = 'sl' if language.lower() == 'slovenian' else item_data.get('language_Fasttext', 'sr')
                    # *** Use the new retry function ***
                    tasks.append(fetch_response_with_retry(session, ollama_model_string, item_data, item_lang_code, idx))
                else:
                     print(f"Warning: Index {idx} out of bounds. Skipping.")

            if not tasks: continue

            batch_desc = f"Batch {i//BATCH_SIZE + 1} ({len(tasks)} articles)"
            responses = await tqdm_asyncio.gather(*tasks, desc=batch_desc, position=1, leave=False)

            successful_in_batch_for_checkpoint = []

            for response in responses:
                idx = response["global_index"]
                if response["success"]: # Success means a non-garbled summary was found
                    original_item = dataset[idx].copy()
                    original_item["summary"] = response["summary"]
                    original_item["global_index"] = idx
                    newly_successful_items_this_run[idx] = original_item
                    successful_in_batch_for_checkpoint.append(original_item)
                    # Update progress: Add to completed, remove from failed
                    if idx not in progress['completed_indices']: progress['completed_indices'].append(idx)
                    if idx in progress['failed_indices']: progress['failed_indices'].remove(idx)
                else: # Failure means all attempts failed or resulted in garbled output
                    print(f"Error: Item {idx} ultimately failed after retries. Reason: {response.get('error', 'Unknown')}")
                    failed_indices_this_run.append(idx)
                    # Update progress: Add to failed, remove from completed
                    if idx not in progress['failed_indices']: progress['failed_indices'].append(idx)
                    if idx in progress['completed_indices']: progress['completed_indices'].remove(idx)

            # Save checkpoint ONLY for successfully generated items in this batch
            if successful_in_batch_for_checkpoint:
                checkpoint_file = get_checkpoint_filename(temp_log_dir, i)
                try:
                    with open(checkpoint_file, "w", encoding="utf-8") as f:
                        json.dump(successful_in_batch_for_checkpoint, f, ensure_ascii=False, indent=2)
                except Exception as e:
                    print(f"Error saving checkpoint {checkpoint_file}: {e}")

            # Save overall progress after each batch
            save_progress(progress_file, progress)
            overall_progress.update(len(batch_indices))

        overall_progress.close()

    # Consolidate Results
    final_successful_map = {**successful_items_map, **newly_successful_items_this_run}
    final_successful_items_in_range = [item for idx, item in final_successful_map.items() if idx in target_indices_set]

    # Get the final list of failed indices *within the target range* from the progress file
    final_failed_indices_in_range = [idx for idx in progress['failed_indices'] if idx in target_indices_set]

    return final_successful_items_in_range, final_failed_indices_in_range


# --- Finalization and Cleanup ---
# finalize_and_cleanup remains the same as before
def finalize_and_cleanup(successful_items, final_failed_indices, total_expected_count, final_output_file, temp_log_dir):
    """Performs sanity checks, saves final output, and cleans up temp files."""
    print("\n--- Finalization ---")
    passed_sanity_check = True
    error_messages = []

    # 1. Check count of successful items against expected
    if len(successful_items) != total_expected_count:
        passed_sanity_check = False
        error_messages.append(f"Error: Expected {total_expected_count} successful items, but collected {len(successful_items)}.")

    # 2. Check if there are any indices still marked as failed *within the target range*
    if final_failed_indices:
        passed_sanity_check = False
        error_messages.append(f"Error: {len(final_failed_indices)} items ultimately failed processing within the target range.")
        print(f"Failed indices: {final_failed_indices}")

    # 3. Check if all successful items have a non-empty summary that doesn't start with "Error:"
    missing_or_error_summary_indices = []
    for item in successful_items:
        item_index = item.get('global_index', -1)
        summary = item.get("summary")
        if not summary or (isinstance(summary, str) and summary.strip().startswith("Error:")):
            missing_or_error_summary_indices.append(item_index)

    if missing_or_error_summary_indices:
        passed_sanity_check = False
        error_messages.append(f"Error: {len(missing_or_error_summary_indices)} successful items have missing or error summaries.")
        print(f"Indices with missing/error summaries: {missing_or_error_summary_indices}")

    # --- Action based on sanity check ---
    if passed_sanity_check:
        print("Sanity check passed.")
        try:
            os.makedirs(os.path.dirname(final_output_file), exist_ok=True)
            successful_items.sort(key=lambda x: x.get('global_index', float('inf')))
            with open(final_output_file, "w", encoding="utf-8") as f:
                json.dump(successful_items, f, ensure_ascii=False, indent=2)
            print(f"Successfully saved {len(successful_items)} items to: {final_output_file}")
            try:
                shutil.rmtree(temp_log_dir)
                print(f"Successfully removed temporary directory: {temp_log_dir}")
            except Exception as e:
                print(f"Warning: Failed to remove temporary directory {temp_log_dir}: {e}")
        except Exception as e:
            print(f"Error saving final output file {final_output_file}: {e}")
            print("Temporary directory was NOT removed due to final save error.")
            passed_sanity_check = False
    else:
        print("Sanity check FAILED.")
        for msg in error_messages:
            print(f"- {msg}")
        print(f"Final output was NOT saved to {final_output_file}.")
        print(f"Temporary directory with logs and checkpoints kept for inspection: {temp_log_dir}")

    return passed_sanity_check


# --- Main Execution ---
# main function remains largely the same, calling the updated process_dataset
async def main():
    parser = argparse.ArgumentParser(description="Summarize articles using Ollama LLMs with retry logic.")
    parser.add_argument("--split", required=True, choices=["slovenian", "serbian"], help="Dataset split.")
    parser.add_argument("--model", required=True, choices=["gemma-3-27b", "gams-9b"], help="LLM model.")
    parser.add_argument("--debug", action="store_true", help="Process only first 100 items.")
    args = parser.parse_args()

    ollama_model_string = MODEL_MAPPING.get(args.model)
    if not ollama_model_string:
        print(f"Error: Invalid model name '{args.model}'."); sys.exit(1)

    input_file = f"../data/subsets/{args.split}.json"
    base_output_dir = "../data/final/summaries"
    model_dir = os.path.join(base_output_dir, args.model)
    final_output_file = os.path.join(model_dir, f"{args.split}.json")
    temp_log_dir = os.path.join(model_dir, f"summary-log-{args.split}")

    try:
        with open(input_file, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        print(f"Loaded {len(dataset)} articles from {input_file}")
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_file}"); sys.exit(1)
    except Exception as e:
        print(f"Error loading dataset from {input_file}: {e}"); sys.exit(1)

    debug_limit = 100 if args.debug else None
    successful_items, final_failed_indices = await process_dataset(
        ollama_model_string, args.split, dataset, debug_limit, temp_log_dir
    )

    if successful_items is None:
        print("Processing failed during setup."); sys.exit(1)

    total_expected = debug_limit if debug_limit else len(dataset)
    success = finalize_and_cleanup(
        successful_items, final_failed_indices, total_expected, final_output_file, temp_log_dir
    )

    if success:
        print("\nProcessing completed successfully.")
    else:
        print("\nProcessing finished with errors. Please check logs and the temporary directory.")
        sys.exit(1)

if __name__ == "__main__":
    # Ensure lexicalrichness is installed or handled
    if not LEXICALRICHNESS_AVAILABLE:
        print("Note: Lexical richness checks are disabled as the library is not installed.")
    asyncio.run(main())

# --- END OF FINAL llm-summarization.py ---
