# --- START OF FINAL llm-summarization.py ---

import json
import asyncio
import aiohttp
import time
import os
import sys
import argparse
import shutil
import re
from collections import Counter
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
OLLAMA_URL = "http://llm.ijs.si:11435/api/generate"
MAX_PARALLEL_REQUESTS = 4
BATCH_SIZE = 50
TIMEOUT = 120 # Timeout per *single* API request attempt

MODEL_MAPPING = {
    "gemma-3-27b": "gemma3:27b-it-qat",
}

LANGUAGE_NAME_MAP = {
    'sl': 'Slovenian',
    'sr': 'Serbian',
    'hr': 'Croatian',
    'bs': 'Bosnian',
    'sh': 'Serbo-Croatian',
    'en': 'English'
}

# Basic stop words for garbled text detection
STOP_WORDS_SL = {'v', 'in', 'na', 's', 'z', 'je', 'so', 'se', 'pa', 'ter', 'ki', 'ko', 'kot', 'do', 'ob', 'pri', 'za', 'd', 'dd'}
STOP_WORDS_SH = {'u', 'i', 'na', 'sa', 'je', 'su', 'se', 'pa', 'te', 'koji', 'koja', 'koje', 'kao', 'do', 'kod', 'za'}
STOP_WORDS_EN = {'a', 'an', 'the', 'in', 'on', 'of', 'for', 'to', 'is', 'are', 'was', 'were', 'it', 'and', 'with', 'as'}

semaphore = asyncio.Semaphore(MAX_PARALLEL_REQUESTS)

# --- Prompt Formatting ---
def format_prompt(article, aspect, target_language_name):
    """
    Formats the prompt for the LLM with enhanced instructions for evidence-based summarization.
    """
    return f"""Task: Create a concise, analytical summary focusing on the aspect "{aspect}" from the following newspaper article. The summary must provide the key facts and context needed to understand the sentiment towards the aspect.

    Key Instructions:
    1.  **Primary Focus:** Concentrate strictly on the parts of the article discussing or closely related to the aspect "{aspect}".
    2.  **Evidence-Based Content:** Your summary must include:
        - **The Specific Context of the Mention:** How is the aspect mentioned? Is it the main subject, a historical example, a competitor, a finalist for an award, or just listed as a partner? Be precise.
        - **Sentiment-Bearing Facts:** Capture the key information that reveals sentiment. If the article praises a product, mention the specific praise (e.g., "won a gold medal," "praised for its fluffy texture"). If it's negative, state the specific criticism (e.g., "accused of dangerous driving," "stock price sank by 4.7%").
        - **Narrative Framing:** If the article presents objective data (like financial numbers), also capture how the author frames that data (e.g., "...which the author called the 'main culprit' for the market's decline.").
        - **Associated Entities & Relationships:** Clarify the relationship between the aspect and other entities. For example, "The article compares {aspect} to its competitor, XYZ, noting that...".
    3.  **Accuracy:** Base your summary strictly on the facts and tone presented in the provided article. Do not add external information or your own final judgment. Present the evidence, don't just state the conclusion.
    4.  **Language:** Your response MUST be in {target_language_name}.
    5.  **Conciseness:** Keep the summary to 3-6 powerful sentences.

    Article:
    {article}

    Aspect:
    {aspect}

    Answer (in {target_language_name}):"""


# --- Garbled Text Detection ---
def is_garbled(text, language_code='sl', min_words=10, ttr_threshold=0.5, freq_word_threshold=0.30):
    """
    Checks if a text string appears garbled due to excessive repetition or low diversity.
    """
    if not text or not isinstance(text, str):
        return True # Treat empty or non-string as garbled

    words = re.findall(r'\b\w+\b', text.lower())

    if len(words) < min_words:
        return False

    if LEXICALRICHNESS_AVAILABLE:
        try:
            lex = LexicalRichness(text)
            if lex.words == 0: ttr = 0.0
            else: ttr = lex.ttr
            if ttr < ttr_threshold: return True
        except Exception as e:
            print(f"Warning: LexicalRichness calculation failed: {e}")

    stop_words = STOP_WORDS_SL
    if language_code in ['sr', 'hr', 'bs', 'sh']:
        stop_words = STOP_WORDS_SH
    elif language_code == 'en':
        stop_words = STOP_WORDS_EN

    content_words = [word for word in words if word not in stop_words and len(word) >= 3]

    if not content_words: return False

    word_counts = Counter(content_words)
    if not word_counts: return False

    most_common_word, most_common_count = word_counts.most_common(1)[0]
    frequency_percentage = most_common_count / len(content_words)

    if frequency_percentage > freq_word_threshold: return True

    short_patterns = re.findall(r'(?:^|\s)((?:\S{1,3}[.\s]?){3,})(?=\s|$)', text)
    if short_patterns:
        pattern_counts = Counter(short_patterns)
        for pattern, count in pattern_counts.items():
            if count > 3 and len(pattern) < 30:
                 if text.count(pattern) * len(pattern) > 0.4 * len(text):
                      return True

    return False


# --- Asynchronous Request Function with Retry Logic ---
async def fetch_response_with_retry(session, ollama_model_string, article_data, target_language_name, target_language_code, global_index):
    """Sends requests to Ollama, retrying with stricter parameters if output is garbled."""
    article = article_data.get("article", "")
    aspect = article_data.get("aspect", "")
    uuid = article_data.get("uuid", f"index_{global_index}")

    base_payload = {
        "model": ollama_model_string,
        "prompt": format_prompt(article, aspect, target_language_name),
        "stream": False
    }

    parameter_sets = [
        {"repeat_penalty": 1.15, "temperature": 0.75},
        {"repeat_penalty": 1.3, "temperature": 0.6, "top_k": 40},
        {"repeat_penalty": 1.5, "temperature": 0.5, "top_k": 30, "top_p": 0.85}
    ]

    attempts_results = []
    final_summary = f"Error: All {len(parameter_sets)} attempts failed validation or API calls."
    final_success = False
    final_error_reason = "Initialization failure"
    elapsed_total = 0.0

    for i, params in enumerate(parameter_sets):
        current_payload = base_payload.copy()
        current_payload["options"] = params
        attempt_success = False
        attempt_error = None
        result_text = None
        start_time = time.time()

        try:
            async with semaphore:
                async with session.post(OLLAMA_URL, json=current_payload, timeout=TIMEOUT) as response:
                    elapsed = time.time() - start_time
                    elapsed_total += elapsed

                    if response.status == 200:
                        data = await response.json()
                        result_text = data.get("response", "").strip()
                        if not result_text:
                            attempt_error = f"Attempt {i+1}: Received empty response"
                        else:
                            is_bad = is_garbled(result_text, language_code=target_language_code)
                            if not is_bad:
                                final_summary = result_text
                                final_success = True
                                final_error_reason = None
                                attempt_success = True
                                attempts_results.append({"text": result_text, "ttr": 1.0, "error": None})
                                break
                            else:
                                attempt_error = f"Attempt {i+1}: Flagged as garbled"
                    else:
                        error_text_api = await response.text()
                        attempt_error = f"Attempt {i+1}: API Error {response.status}"
                        print(f"Warning: {attempt_error} for index {global_index}. Details: {error_text_api[:200]}")
                        result_text = f"Error during attempt {i+1}: API Error {response.status}"

        except asyncio.TimeoutError:
            elapsed = time.time() - start_time
            elapsed_total += elapsed if elapsed < TIMEOUT else TIMEOUT
            attempt_error = f"Attempt {i+1}: Timed out after {TIMEOUT}s"
            result_text = f"Error during attempt {i+1}: Timeout"
        except aiohttp.ClientConnectorError as e:
             elapsed = time.time() - start_time
             elapsed_total += elapsed
             attempt_error = f"Attempt {i+1}: Connection Error - {e}"
             result_text = f"Error during attempt {i+1}: Connection Error"
        except Exception as e:
            elapsed = time.time() - start_time
            elapsed_total += elapsed
            attempt_error = f"Attempt {i+1}: Exception - {type(e).__name__}: {e}"
            result_text = f"Error during attempt {i+1}: Exception"

        if attempt_error:
            print(f"Warning: {attempt_error} for index {global_index}.")

        if not attempt_success:
            current_ttr = 0.0
            if result_text and LEXICALRICHNESS_AVAILABLE:
                 try:
                     lex = LexicalRichness(result_text)
                     current_ttr = lex.ttr if lex.words > 0 else 0.0
                 except Exception: pass
            attempts_results.append({"text": result_text or attempt_error, "ttr": current_ttr, "error": attempt_error})

        if final_success: break

    if not final_success and attempts_results:
        print(f"Info: All {len(parameter_sets)} attempts failed validation/API for index {global_index}. Selecting best failure.")
        attempts_results.sort(key=lambda x: (x["error"] is None, x["ttr"]), reverse=True)
        best_failure = attempts_results[0]
        final_summary = best_failure["text"]
        final_error_reason = best_failure.get("error", "All attempts failed; returning best effort.")
        if final_error_reason is None:
             final_error_reason = "All attempts failed validation; returning best effort (highest TTR)."

    return {
        "global_index": global_index, "uuid": uuid, "summary": final_summary,
        "time": elapsed_total, "success": final_success, "error": final_error_reason
    }


# --- File and Progress Management ---
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
             print(f"Warning: Progress file {filename} corrupted ({e}). Starting fresh.")
             try:
                 shutil.copy(filename, f"{filename}.corrupt_{datetime.now().strftime('%Y%m%d%H%M%S')}")
             except Exception: pass
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
                        if 'global_index' in item and item.get('summary') and not item.get('summary','').startswith("Error:"):
                            successful_items_map[item['global_index']] = item
            except Exception as e:
                print(f"Warning: Could not load checkpoint file {filepath}: {e}")
    return successful_items_map


# --- Main Processing Logic ---
async def process_dataset(ollama_model_string, split_name, target_language_name, target_language_code, dataset, debug_limit, temp_log_dir):
    total_items_in_dataset = len(dataset)
    limit = debug_limit if debug_limit else total_items_in_dataset
    target_indices_set = set(range(limit))

    print(f"\nProcessing {split_name} dataset with {ollama_model_string}")
    print(f"Target language: {target_language_name} ({target_language_code})")
    print(f"Target items: {limit} (Total in dataset: {total_items_in_dataset})")

    os.makedirs(temp_log_dir, exist_ok=True)
    progress_file = get_progress_filename(temp_log_dir)

    progress = load_progress(progress_file)
    successful_items_map = load_results_from_checkpoints(temp_log_dir)
    print(f"Loaded {len(successful_items_map)} confirmed successful items from checkpoints.")
    print(f"Progress file indicates: {len(progress['completed_indices'])} completed, {len(progress['failed_indices'])} failed attempts.")

    processed_successful_indices = set(successful_items_map.keys())
    indices_to_run = sorted(list(target_indices_set - processed_successful_indices))

    if not indices_to_run:
        print("All target items seem to be successfully processed based on checkpoints.")
        final_items = [item for idx, item in successful_items_map.items() if idx in target_indices_set]
        progress['completed_indices'] = sorted(list(target_indices_set))
        progress['failed_indices'] = [idx for idx in progress['failed_indices'] if idx not in target_indices_set]
        save_progress(progress_file, progress)
        return final_items, []

    print(f"Identified {len(indices_to_run)} items to process/retry within the target range.")
    newly_successful_items_this_run = {}
    failed_indices_this_run = []

    async with aiohttp.ClientSession() as session:
        pbar_desc = f"Overall {split_name} -> {target_language_code} ({ollama_model_string.split(':')[0]})"
        overall_progress = tqdm(total=len(indices_to_run), desc=pbar_desc, unit="articles", position=0, leave=True)

        for i in range(0, len(indices_to_run), BATCH_SIZE):
            batch_indices = indices_to_run[i : i + BATCH_SIZE]
            tasks = [
                fetch_response_with_retry(session, ollama_model_string, dataset[idx], target_language_name, target_language_code, idx)
                for idx in batch_indices if 0 <= idx < len(dataset)
            ]

            if not tasks: continue

            batch_desc = f"Batch {i//BATCH_SIZE + 1} ({len(tasks)} articles)"
            responses = await tqdm_asyncio.gather(*tasks, desc=batch_desc, position=1, leave=False)

            successful_in_batch_for_checkpoint = []
            for response in responses:
                idx = response["global_index"]
                if response["success"]:
                    original_item = dataset[idx].copy()
                    original_item["summary"] = response["summary"]
                    original_item["global_index"] = idx
                    newly_successful_items_this_run[idx] = original_item
                    successful_in_batch_for_checkpoint.append(original_item)
                    if idx not in progress['completed_indices']: progress['completed_indices'].append(idx)
                    if idx in progress['failed_indices']: progress['failed_indices'].remove(idx)
                else:
                    print(f"Error: Item {idx} ultimately failed after retries. Reason: {response.get('error', 'Unknown')}")
                    failed_indices_this_run.append(idx)
                    if idx not in progress['failed_indices']: progress['failed_indices'].append(idx)
                    if idx in progress['completed_indices']: progress['completed_indices'].remove(idx)

            if successful_in_batch_for_checkpoint:
                checkpoint_file = get_checkpoint_filename(temp_log_dir, i)
                try:
                    with open(checkpoint_file, "w", encoding="utf-8") as f:
                        json.dump(successful_in_batch_for_checkpoint, f, ensure_ascii=False, indent=2)
                except Exception as e:
                    print(f"Error saving checkpoint {checkpoint_file}: {e}")

            save_progress(progress_file, progress)
            overall_progress.update(len(batch_indices))

        overall_progress.close()

    final_successful_map = {**successful_items_map, **newly_successful_items_this_run}
    final_successful_items_in_range = [item for idx, item in final_successful_map.items() if idx in target_indices_set]
    final_failed_indices_in_range = [idx for idx in progress['failed_indices'] if idx in target_indices_set]
    return final_successful_items_in_range, final_failed_indices_in_range


# --- Finalization and Cleanup ---
def finalize_and_cleanup(successful_items, final_failed_indices, total_expected_count, final_output_file, temp_log_dir):
    print("\n--- Finalization ---")
    passed_sanity_check = True
    error_messages = []

    if len(successful_items) != total_expected_count:
        passed_sanity_check = False
        error_messages.append(f"Error: Expected {total_expected_count} successful items, but collected {len(successful_items)}.")

    if final_failed_indices:
        passed_sanity_check = False
        error_messages.append(f"Error: {len(final_failed_indices)} items ultimately failed processing within the target range.")
        print(f"Failed indices: {final_failed_indices}")

    missing_or_error_summary_indices = [
        item.get('global_index', -1) for item in successful_items
        if not item.get("summary") or (isinstance(item.get("summary"), str) and item.get("summary").strip().startswith("Error:"))
    ]
    if missing_or_error_summary_indices:
        passed_sanity_check = False
        error_messages.append(f"Error: {len(missing_or_error_summary_indices)} successful items have missing or error summaries.")
        print(f"Indices with missing/error summaries: {missing_or_error_summary_indices}")

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
        for msg in error_messages: print(f"- {msg}")
        print(f"Final output was NOT saved to {final_output_file}.")
        print(f"Temporary directory with logs and checkpoints kept for inspection: {temp_log_dir}")

    return passed_sanity_check


# --- Main Execution ---
async def main():
    parser = argparse.ArgumentParser(description="Summarize articles using Ollama LLMs with retry logic.")
    parser.add_argument("--split", required=True, choices=["slovenian", "serbian"], help="Dataset split to process.")
    parser.add_argument("--model", required=True, choices=["gemma-3-27b"], help="LLM model to use.")
    parser.add_argument("--target-language", required=True, choices=["sl", "sr", "en"], help="Target language for the summary.")
    parser.add_argument("--debug", action="store_true", help="Process only first 100 items for debugging.")
    args = parser.parse_args()

    ollama_model_string = MODEL_MAPPING.get(args.model)
    if not ollama_model_string:
        print(f"Error: Invalid model name '{args.model}'."); sys.exit(1)

    target_language_name = LANGUAGE_NAME_MAP.get(args.target_language)
    if not target_language_name:
        print(f"Error: Invalid target language '{args.target_language}'."); sys.exit(1)

    input_file = f"../data/subsets/{args.split}.json"
    base_output_dir = "../data/final/summaries"
    model_dir = os.path.join(base_output_dir, args.model)
    final_output_file = os.path.join(model_dir, f"{args.split}_{args.target_language}.json")
    temp_log_dir = os.path.join(model_dir, f"summary-log-{args.split}_{args.target_language}")

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
        ollama_model_string, args.split, target_language_name, args.target_language, dataset, debug_limit, temp_log_dir
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
    if not LEXICALRICHNESS_AVAILABLE:
        print("Note: Lexical richness checks are disabled as the library is not installed.")
    asyncio.run(main())

# --- END OF FINAL llm-summarization.py ---