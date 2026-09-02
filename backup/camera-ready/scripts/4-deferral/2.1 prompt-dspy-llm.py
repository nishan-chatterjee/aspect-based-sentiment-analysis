# -*- coding: utf-8 -*-
import json
import argparse
import os
import time
import re
import logging
import random
from tqdm import tqdm
from collections import Counter
from typing import List, Dict, Any, Optional, Tuple, Literal as TypingLiteral
import shutil # For deleting logs directory

# --- Dependency Imports ---
try:
    import ollama
except ImportError:
    print("Error: The 'ollama' library is not installed. Please install it: pip install ollama"); exit(1)

try:
    import dspy
    from dspy.teleprompt import MIPROv2
    from dspy.evaluate import Evaluate # Though not directly used for selecting among MIPROv2 outputs here
    DSPY_AVAILABLE = True
except ImportError:
    print("Warning: The 'dspy-ai' library is not installed. DSPy methods will not be available. Install it: pip install dspy-ai"); DSPY_AVAILABLE = False

try:
    from sklearn.metrics import (classification_report, accuracy_score, precision_recall_fscore_support, cohen_kappa_score)
    from sklearn.model_selection import train_test_split
    import numpy as np
except ImportError:
    print("Error: scikit-learn / numpy is not installed. Please install: pip install scikit-learn numpy"); exit(1)

import pandas as pd
import datetime

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MODEL_MAPPING = {
    "gemma-3-27b": "gemma3:27b-it-qat",
    "gams-9b": "hf.co/tknez/GaMS-9B-Instruct-GGUF:latest",
    "qwen-2.5-72b": "qwen2.5:72b" # Example teacher model
}
DEFAULT_MODEL_SHORT_NAME = "gemma-3-27b"

DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11435"
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 5
DEFAULT_TIMEOUT = 180 # Increased default timeout
DEBUG_FRACTION = 0.01
DEFAULT_NUM_QUERIES = 3
DEFAULT_DSPY_MAX_TOKENS = 256 # For CoT methods, might need more
DEFAULT_MIPROV2_TEMP = 1.0
DEFAULT_DSPY_AUTORUN = "heavy"


PLM_PREDICTION_KEY = "global-context-modelling/simplified-dart-xlmr"

INT_TO_STR_LABEL = {-1: "negative", 0: "neutral", 1: "positive"}
STR_TO_INT_LABEL = {v: k for k, v in INT_TO_STR_LABEL.items()}
LABELS_INT = list(INT_TO_STR_LABEL.keys())
LABELS_STR = list(STR_TO_INT_LABEL.keys())
NEUTRAL_STR = "neutral"
NEUTRAL_INT = 0

# PLM F1 scores for 'plm-augmented-direct' method and 'PLMReliabilitySignature'
PLM_F1_SCORES = {
    "slovenian": {"negative": 46.05, "neutral": 94.14, "positive": 74.51},
    "serbian": {"negative": 69.84, "neutral": 84.36, "positive": 87.39}
}
      
# --- Data Loading ---
def load_data(language: str, split_type: str, split_index: Optional[int] = None) -> List[Dict[str, Any]]:
    base_data_root = "../data/final/balanced-predicted/"
    data_folder_name = "simplified-dart-xlmr"
    filename = ""
    if split_type == 'test':
        filename = f"{language}_test_balanced.json"
    elif split_type in ['train', 'val']:
        if split_index is None: raise ValueError("split_index must be provided for train/val splits.")
        filename = f"{language}_train_val_balanced_{split_index}.json"
    else: raise ValueError(f"Invalid split_type: {split_type}. Choose 'train', 'val', or 'test'.")
    data_key = 'test' if split_type == 'test' else split_type
    file_path = os.path.join(base_data_root, data_folder_name, filename)
    logging.info(f"Attempting to load {data_key} data from: {file_path}")
    try:
        with open(file_path, "r", encoding="utf-8") as f: content = json.load(f)
        data = content.get(data_key, [])
        if not data and isinstance(content, list):
            data = content
            logging.info(f"No '{data_key}' key found, but content is a list. Assuming direct list of items.")
        if not data:
            logging.warning(f"Loaded data is empty. Check if '{data_key}' key exists in {file_path} or if the file is truly empty.")
            return []
        else: logging.info(f"Loaded {len(data)} items for {language} {data_key} (from {filename}).")
        expected_keys = ['uuid', 'article', 'aspect', 'sentiment', PLM_PREDICTION_KEY]
        # Check if first item has all keys, but allow PLM_PREDICTION_KEY to be missing for datasets not using it
        # For this script, simplified-dart-xlmr *should* have it.
        if data and not all(k in data[0] for k in expected_keys):
             missing_keys = [k for k in expected_keys if k not in data[0]]
             logging.warning(f"First item in {file_path} may be malformed. Missing expected keys: {missing_keys}.")
        return data
    except FileNotFoundError: logging.error(f"File not found: {file_path}."); raise
    except json.JSONDecodeError as jde: logging.error(f"Error decoding JSON from {file_path}: {jde}"); raise
    except Exception as e: logging.error(f"An unexpected error occurred loading data from {file_path}: {e}"); raise

# --- Stratified Sampling ---
def sample_debug_data(data: List[Dict[str, Any]], fraction: float = DEBUG_FRACTION) -> List[Dict[str, Any]]:
    if not data: return []
    logging.info(f"Performing stratified sampling (fraction={fraction})...")
    try:
        df = pd.DataFrame(data)
        if 'sentiment' not in df.columns: logging.error("Cannot stratify: 'sentiment' column missing."); return data
        df['sentiment'] = pd.to_numeric(df['sentiment'], errors='coerce')
        df.dropna(subset=['sentiment'], inplace=True)
        df['sentiment'] = df['sentiment'].astype(int)
        if df.empty: logging.warning("DataFrame empty after sentiment cleaning for stratification."); return []
        num_samples_to_select = max(len(df['sentiment'].unique()), int(round(len(df) * fraction)))
        if num_samples_to_select >= len(df): sampled_df = df
        elif df['sentiment'].value_counts().min() < 2 and len(df['sentiment'].unique()) > 1 :
            logging.warning(f"Min class size < 2 for stratification. Falling back to random sampling for debug.")
            sampled_df = df.sample(n=num_samples_to_select, random_state=42, replace=False)
        else:
            try:
                actual_fraction_for_split = num_samples_to_select / len(df) if len(df) > 0 else 0
                if actual_fraction_for_split >= 1.0: sampled_df = df
                else: _, sampled_df = train_test_split(df, test_size=actual_fraction_for_split, stratify=df['sentiment'], random_state=42)
            except ValueError as ve:
                logging.warning(f"Stratified split failed: {ve}. Falling back to random sample of size {num_samples_to_select}.")
                sampled_df = df.sample(n=num_samples_to_select, random_state=42, replace=False)
        sampled_data = sampled_df.to_dict('records')
        logging.info(f"Sampled {len(sampled_data)} items for debug.")
        return sampled_data
    except Exception as e: logging.error(f"Error in stratified sampling: {e}"); return data

# --- Ollama Interaction ---
def ollama_request_sync(prompt: str, model_name: str, host_url: str, temperature: float, top_k: int, top_p: float, max_retries: int, retry_delay: int, timeout: int, expected_max_tokens: int) -> Optional[str]:
    attempt = 0
    try: client = ollama.Client(host=host_url, timeout=timeout)
    except Exception as e: logging.error(f"Ollama client init failed for {host_url}: {e}"); return None
    options = {"temperature": temperature, "top_k": top_k, "top_p": top_p, "num_predict": expected_max_tokens}
    while attempt < max_retries:
        attempt += 1
        try:
            response = client.generate(model=model_name, prompt=prompt, stream=False, options=options)
            return response.get('response', '').strip()
        except Exception as e:
            logging.warning(f"Ollama attempt {attempt}/{max_retries} failed for {model_name}: {e}. Retrying in {retry_delay}s...")
            time.sleep(retry_delay)
    logging.error(f"Ollama failed after {max_retries} attempts for {model_name}.")
    return None

# --- Mode Calculation ---
def calculate_mode(predictions: List[Any], neutral_value: Any) -> Any:
    valid_predictions = [p for p in predictions if p is not None]
    if not valid_predictions: return neutral_value
    counts = Counter(valid_predictions)
    max_count = max(counts.values())
    modes = [item for item, count in counts.items() if count == max_count]
    if len(modes) == 1: return modes[0]
    return neutral_value if neutral_value in modes else NEUTRAL_STR

# --- Result Processing & Resumability ---
def load_processed_item_logs(logs_dir: str) -> Dict[Any, Dict]:
    processed = {}
    if os.path.exists(logs_dir):
        for filename in os.listdir(logs_dir):
            if filename.startswith("item_") and filename.endswith(".json"):
                try:
                    with open(os.path.join(logs_dir, filename), 'r', encoding="utf-8") as f: data = json.load(f)
                    if data.get("status") == "success" and "uuid" in data and data.get("prediction_int") is not None:
                        processed[data["uuid"]] = data
                except Exception as e: logging.warning(f"Error loading log file {filename}: {e}")
    logging.info(f"Loaded {len(processed)} successfully processed items from logs: {logs_dir}")
    return processed

def save_item_log(logs_dir: str, item_uuid: Any, data_to_save: Dict):
    os.makedirs(logs_dir, exist_ok=True)
    log_file_path = os.path.join(logs_dir, f"item_{item_uuid}.json")
    try:
        with open(log_file_path, 'w', encoding="utf-8") as f: json.dump(data_to_save, f, indent=4)
    except Exception as e: logging.error(f"Failed to save log for item {item_uuid} to {log_file_path}: {e}")

def get_items_to_process(all_target_items: List[Dict[str,Any]], logs_dir: str, force_reprocess: bool) -> Tuple[List[Dict[str,Any]], Dict[Any, Dict]]:
    items_needing_processing = []
    processed_item_data_from_logs = {}
    if not force_reprocess: processed_item_data_from_logs = load_processed_item_logs(logs_dir)
    for i, item in enumerate(all_target_items):
        item_uuid = item.get('uuid')
        if item_uuid is None: item_uuid = f"gen_uuid_{i}"; item['uuid'] = item_uuid; logging.warning(f"Item {i} missing UUID, assigned temp: {item_uuid}")
        if force_reprocess or item_uuid not in processed_item_data_from_logs:
            items_needing_processing.append(item)
    logging.info(f"Identified {len(items_needing_processing)} items needing processing out of {len(all_target_items)} targeted for this run.")
    return items_needing_processing, processed_item_data_from_logs

# --- Direct Prompting & PLM-Augmented Direct Prompting ---
def create_enhanced_direct_prompt(article: str, aspect: str) -> str:
    return f"""Task: Analyze the sentiment expressed towards a specific 'aspect' within the provided 'article'.
The 'aspect' in question is: "{aspect}"
The 'article' may contain mentions of this aspect, potentially highlighted with <aspect>...</aspect> tags, or as plain text. Your goal is to determine if the sentiment expressed in the 'article' *specifically concerning* "{aspect}" is negative, neutral, or positive.

Instructions:
1.  Read the entire 'article' to understand the overall context.
2.  Locate all mentions of the specific 'aspect': "{aspect}". Pay attention to how it's discussed.
3.  Evaluate the sentiment conveyed about this 'aspect' based on these mentions and their surrounding context.
4.  Your answer MUST be a single word: 'negative', 'neutral', or 'positive'.
5.  Do NOT provide any explanation, reasoning, or any other text besides the single sentiment word.
6.  Your response must be in English.

Article:
{article}

Aspect: {aspect}

Sentiment (negative, neutral, or positive):"""

def create_plm_augmented_direct_prompt(article: str, aspect: str, plm_suggestion_str: str, language: str) -> str:
    f1_scores_lang = PLM_F1_SCORES.get(language, {})
    f1_neg = f1_scores_lang.get("negative", "N/A")
    f1_neu = f1_scores_lang.get("neutral", "N/A")
    f1_pos = f1_scores_lang.get("positive", "N/A")
    plm_reliability_statement = "N/A"
    if plm_suggestion_str in f1_scores_lang and f1_scores_lang[plm_suggestion_str] != "N/A":
        plm_reliability_statement = f"{f1_scores_lang[plm_suggestion_str]}%"
    plm_perf_summary = f"For '{language}' language: 'negative' (F1 ~{f1_neg}%), 'neutral' (F1 ~{f1_neu}%), 'positive' (F1 ~{f1_pos}%)."
    return f"""Task: Re-evaluate and determine the sentiment towards '{aspect}' in the 'article'.
You are provided with a suggestion from a Prior Language Model (PLM).

Article:
{article}

Aspect: {aspect}

PLM's Sentiment Suggestion: {plm_suggestion_str}

Contextual Information on PLM Reliability ({language}):
- The PLM that made this suggestion has the following general F1 performance: {plm_perf_summary}
- For the specific suggestion of '{plm_suggestion_str}', this PLM's F1 score is approximately {plm_reliability_statement}.
- Consider this reliability: a high F1 score means the PLM is often correct for this class; a lower F1 score suggests more caution is needed.

Your Task:
Critically review the 'article' concerning the '{aspect}'. Taking into account the PLM's suggestion and its typical reliability for that class in '{language}', provide your final sentiment classification.

Instructions:
1.  Thoroughly analyze the 'article' for all mentions and context related to '{aspect}'.
2.  Weigh the PLM's suggestion ('{plm_suggestion_str}') against your own analysis and the provided PLM reliability data.
3.  Determine the most accurate sentiment: 'negative', 'neutral', or 'positive'.
4.  Respond with ONLY that single word.
5.  Your response must be in English.

Final Sentiment (negative, neutral, or positive):"""

def parse_direct_response_str(response_text: Optional[str]) -> Optional[str]:
    if not response_text: return None
    cleaned = response_text.strip().lower()
    if cleaned in LABELS_STR: return cleaned
    for label in LABELS_STR:
        if label in cleaned: return label
    for label in LABELS_STR:
        if cleaned.startswith(label): return label
    return None

def run_single_item_direct_style(
    item: Dict[str, Any], model_name: str, ollama_url: str, num_queries: int,
    temperature: float, top_k: int, top_p: float, max_retries: int, retry_delay: int, timeout: int,
    logs_dir: str, prompt_creation_func: callable, language_for_prompt: Optional[str] = None
) -> Dict[str, Any]:
    item_uuid = item['uuid']
    article, aspect = item.get('article', ''), item.get('aspect', '')
    log_data_template = {"uuid": item_uuid, "status": "failed", "prediction_int": None, "raw_responses_agg": "", "ground_truth_int": STR_TO_INT_LABEL.get(item.get('sentiment')) if isinstance(item.get('sentiment'), str) else item.get('sentiment')}
    if not article or not aspect:
        log_data_template["reason"] = "Missing article/aspect"; log_data_template["raw_responses_agg"] = "Skipped"
        save_item_log(logs_dir, item_uuid, log_data_template); return log_data_template
    prompt_args = [article, aspect]
    if prompt_creation_func == create_plm_augmented_direct_prompt:
        plm_pred_int = item.get(PLM_PREDICTION_KEY)
        plm_pred_str = NEUTRAL_STR
        if plm_pred_int is not None:
            parsed_plm = INT_TO_STR_LABEL.get(int(plm_pred_int))
            if parsed_plm: plm_pred_str = parsed_plm
        prompt_args.append(plm_pred_str)
        if language_for_prompt: prompt_args.append(language_for_prompt)
        else: logging.error(f"Language must be provided for PLM augmented direct prompt for item {item_uuid}."); log_data_template["reason"] = "Missing language for PLM prompt"; save_item_log(logs_dir, item_uuid, log_data_template); return log_data_template
    prompt = prompt_creation_func(*prompt_args)
    item_predictions_str, item_raw_responses, query_success_flag = [], [], False
    for _ in range(num_queries):
        raw_response = ollama_request_sync(prompt, model_name, ollama_url, temperature, top_k, top_p, max_retries, retry_delay, timeout, 20)
        item_raw_responses.append(raw_response if raw_response else "Request Failed")
        parsed_str = parse_direct_response_str(raw_response)
        if parsed_str: query_success_flag = True
        item_predictions_str.append(parsed_str)
    final_prediction_str = calculate_mode(item_predictions_str, NEUTRAL_STR)
    final_prediction_int = STR_TO_INT_LABEL.get(final_prediction_str)
    current_item_log = {"uuid": item_uuid, "status": "success" if query_success_flag and final_prediction_int is not None else "failed", "prediction_int": final_prediction_int, "raw_responses_agg": " | ".join(item_raw_responses), "ground_truth_int": int(item.get('sentiment')) if item.get('sentiment') is not None else None, "reason": "" if query_success_flag and final_prediction_int is not None else "Parsing/query failed"}
    save_item_log(logs_dir, item_uuid, current_item_log)
    return current_item_log

def run_direct_prompting(items_to_process: List[Dict[str, Any]], model_name: str, ollama_url: str, num_queries: int, temperature: float, top_k: int, top_p: float, max_retries: int, retry_delay: int, timeout: int, logs_dir: str) -> List[Dict[str, Any]]:
    newly_processed_item_logs = []
    for item in tqdm(items_to_process, desc=f"Direct Prompting ({model_name})"):
        log = run_single_item_direct_style(item, model_name, ollama_url, num_queries, temperature, top_k, top_p, max_retries, retry_delay, timeout, logs_dir, create_enhanced_direct_prompt)
        newly_processed_item_logs.append(log)
    return newly_processed_item_logs

def run_plm_augmented_direct_prompting(items_to_process: List[Dict[str, Any]], model_name: str, ollama_url: str, num_queries: int, temperature: float, top_k: int, top_p: float, max_retries: int, retry_delay: int, timeout: int, logs_dir: str, language: str) -> List[Dict[str, Any]]:
    newly_processed_item_logs = []
    for item in tqdm(items_to_process, desc=f"PLM-Augmented Direct Prompting ({model_name}, {language})"):
        log = run_single_item_direct_style(item, model_name, ollama_url, num_queries, temperature, top_k, top_p, max_retries, retry_delay, timeout, logs_dir, create_plm_augmented_direct_prompt, language_for_prompt=language)
        newly_processed_item_logs.append(log)
    return newly_processed_item_logs

# --- DSPy Method Signatures ---
if DSPY_AVAILABLE:
    # --- Default Signatures ---
    class SentimentClassificationSignature(dspy.Signature):
        """Analyze the sentiment expressed towards the 'aspect' within the 'article'.
The aspect is a specific phrase or entity. Determine if the sentiment towards this
aspect is 'negative', 'neutral', or 'positive' based on its context in the article.
Respond with only one of these three sentiment labels."""
        article: str = dspy.InputField(desc="The full text of the article.")
        aspect: str = dspy.InputField(desc="The specific aspect.")
        sentiment: TypingLiteral["negative", "neutral", "positive"] = dspy.OutputField(desc="Sentiment.")

    class ReasoningSentimentSignature(dspy.Signature):
        """Given an article and an aspect, first provide step-by-step reasoning
about the sentiment towards the aspect, considering its context.
Then, conclude with the final sentiment: 'negative', 'neutral', or 'positive'.
Ensure the reasoning clearly and logically leads to the final sentiment label."""
        article: str = dspy.InputField(desc="The full text of the article.")
        aspect: str = dspy.InputField(desc="The specific aspect.")
        reasoning: str = dspy.OutputField(desc="Step-by-step reasoning.")
        sentiment: TypingLiteral["negative", "neutral", "positive"] = dspy.OutputField(desc="Final sentiment.")

    class PLMAugmentedSentimentSignature(dspy.Signature):
        """Given an article, an aspect, and a prior sentiment suggestion from a PLM,
classify the final sentiment as 'negative', 'neutral', or 'positive'.
Carefully consider all inputs to make the most accurate determination."""
        article: str = dspy.InputField(desc="The full text of the article.")
        aspect: str = dspy.InputField(desc="The specific aspect.")
        plm_suggestion: TypingLiteral["negative", "neutral", "positive"] = dspy.InputField(desc="Sentiment suggestion from a prior model.")
        sentiment: TypingLiteral["negative", "neutral", "positive"] = dspy.OutputField(desc="Final sentiment.")

    class ReasoningPLMAugmentedSentimentSignature(dspy.Signature):
        """Given an article, an aspect, and a PLM sentiment suggestion,
first provide step-by-step reasoning considering all inputs,
then conclude with the final sentiment: 'negative', 'neutral', or 'positive'.
The reasoning should explain how the PLM suggestion was considered."""
        article: str = dspy.InputField(desc="The full text of the article.")
        aspect: str = dspy.InputField(desc="The specific aspect.")
        plm_suggestion: TypingLiteral["negative", "neutral", "positive"] = dspy.InputField(desc="Sentiment suggestion from a prior model.")
        reasoning: str = dspy.OutputField(desc="Step-by-step reasoning considering all inputs.")
        sentiment: TypingLiteral["negative", "neutral", "positive"] = dspy.OutputField(desc="Final sentiment.")

    # --- Optional Signatures with Aspect Marker Information ---
    class SentimentClassificationWithAspectMarkerSignature(dspy.Signature):
        """Analyze the sentiment expressed towards the 'aspect' within the 'article'.
    The 'aspect' in question is explicitly marked with <aspect>...</aspect> tags in the article text.
    Determine if the sentiment towards this tagged aspect is 'negative', 'neutral', or 'positive'.
    Respond with only one of these three sentiment labels."""
        article: str = dspy.InputField(desc="The full text of the article, with aspects tagged as <aspect>...</aspect>.")
        aspect: str = dspy.InputField(desc="The specific aspect phrase (matches text within tags).")
        sentiment: TypingLiteral["negative", "neutral", "positive"] = dspy.OutputField(desc="Sentiment.")

    class ReasoningSentimentWithAspectMarkerSignature(dspy.Signature):
        """Given an article and an aspect, provide step-by-step reasoning about sentiment.
    The 'aspect' is marked with <aspect>...</aspect> tags in the article.
    Your reasoning should focus on the context of these tagged mentions.
    Then, conclude with the final sentiment: 'negative', 'neutral', 'positive'."""
        article: str = dspy.InputField(desc="The full text of the article, with aspects tagged as <aspect>...</aspect>.")
        aspect: str = dspy.InputField(desc="The specific aspect phrase (matches text within tags).")
        reasoning: str = dspy.OutputField(desc="Step-by-step reasoning based on tagged aspect mentions.")
        sentiment: TypingLiteral["negative", "neutral", "positive"] = dspy.OutputField(desc="Final sentiment.")

    # --- Optional Signatures with PLM Reliability Information ---
    class PLMAugmentedWithReliabilitySignature(dspy.Signature):
        """Given an article, aspect, a PLM sentiment suggestion, and PLM reliability info,
    classify the final sentiment ('negative', 'neutral', 'positive').
    Critically evaluate the PLM suggestion using its provided reliability context."""
        article: str = dspy.InputField(desc="The full text of the article.")
        aspect: str = dspy.InputField(desc="The specific aspect.")
        plm_suggestion: TypingLiteral["negative", "neutral", "positive"] = dspy.InputField(desc="Sentiment suggestion from a prior model.")
        plm_reliability_info: str = dspy.InputField(desc="Information about the PLM's typical F1 scores for the suggested class and language.")
        sentiment: TypingLiteral["negative", "neutral", "positive"] = dspy.OutputField(desc="Final sentiment after considering PLM suggestion and its reliability.")

    class ReasoningPLMAugmentedWithReliabilitySignature(dspy.Signature):
        """Given an article, aspect, a PLM suggestion, and PLM reliability info:
    1. Provide step-by-step reasoning, explicitly considering the PLM suggestion and its stated reliability.
    2. Conclude with the final sentiment ('negative', 'neutral', 'positive')."""
        article: str = dspy.InputField(desc="The full text of the article.")
        aspect: str = dspy.InputField(desc="The specific aspect.")
        plm_suggestion: TypingLiteral["negative", "neutral", "positive"] = dspy.InputField(desc="Sentiment suggestion from a prior model.")
        plm_reliability_info: str = dspy.InputField(desc="Information about the PLM's typical F1 scores for the suggested class and language.")
        reasoning: str = dspy.OutputField(desc="Step-by-step reasoning, incorporating PLM suggestion and its reliability.")
        sentiment: TypingLiteral["negative", "neutral", "positive"] = dspy.OutputField(desc="Final sentiment.")


# --- DSPy Shared Functions ---
if DSPY_AVAILABLE:
    def configure_dspy_lm(model_name_str: str, host_url: str, temperature: float, top_k: int, top_p: float, dspy_max_tokens: int, role: str = "student") -> Optional[dspy.LM]:
        try:
            lm = dspy.LM(model=f"ollama_chat/{model_name_str}", api_base=host_url, model_type='chat', temperature=temperature, top_p=top_p, top_k=top_k, max_tokens=dspy_max_tokens)
            logging.info(f"DSPy {role} LM configured with max_tokens={dspy_max_tokens}: {lm}")
            return lm
        except Exception as e: logging.error(f"DSPy {role} LM config for {model_name_str} failed: {e}"); return None

    def prepare_dspy_dataset(data: List[Dict[str, Any]], method_name: str, language: str, use_plm_reliability_sig: bool, purpose: str = "training") -> List[dspy.Example]:
        dspy_dataset = []
        needs_plm_input_field = "plm-augmented" in method_name
        for item in data:
            article, aspect, sentiment_int = item.get('article'), item.get('aspect'), item.get('sentiment')
            if article and aspect and sentiment_int is not None:
                try:
                    sentiment_str = INT_TO_STR_LABEL.get(int(sentiment_int))
                    if not sentiment_str: continue
                    example_args = {"article": article, "aspect": aspect, "sentiment": sentiment_str}
                    input_keys = ["article", "aspect"]
                    if needs_plm_input_field:
                        plm_pred_int = item.get(PLM_PREDICTION_KEY)
                        plm_pred_str = NEUTRAL_STR
                        if plm_pred_int is not None:
                            parsed_plm = INT_TO_STR_LABEL.get(int(plm_pred_int))
                            if parsed_plm: plm_pred_str = parsed_plm
                        example_args["plm_suggestion"] = plm_pred_str
                        input_keys.append("plm_suggestion")
                        if use_plm_reliability_sig:
                            f1_scores_lang = PLM_F1_SCORES.get(language, {})
                            f1_neg = f1_scores_lang.get("negative", "N/A")
                            f1_neu = f1_scores_lang.get("neutral", "N/A")
                            f1_pos = f1_scores_lang.get("positive", "N/A")
                            reliability_detail = "N/A"
                            if plm_pred_str in f1_scores_lang and f1_scores_lang[plm_pred_str] != "N/A":
                                reliability_detail = f"{f1_scores_lang[plm_pred_str]}%"
                            plm_reliability_info_str = (f"PLM reliability for {language} (F1 scores): Negative ~{f1_neg}%, Neutral ~{f1_neu}%, Positive ~{f1_pos}%. Current PLM suggestion '{plm_pred_str}' has reliability ~{reliability_detail}.")
                            example_args["plm_reliability_info"] = plm_reliability_info_str
                            input_keys.append("plm_reliability_info")
                    dspy_dataset.append(dspy.Example(**example_args).with_inputs(*input_keys))
                except (ValueError, TypeError) as e: logging.warning(f"Skipping item due to error during example prep: {e} - Item: {item.get('uuid')}"); pass
        logging.info(f"Prepared {len(dspy_dataset)} DSPy {purpose} examples for method '{method_name}' (PLM reliability sig: {use_plm_reliability_sig}).")
        return dspy_dataset

    def validate_sentiment(example: dspy.Example, pred: dspy.Prediction, trace=None) -> bool:
        return getattr(pred, 'sentiment', '').lower() == example.sentiment.lower()

    def optimize_dspy_program_mipro(
        base_module: dspy.Module, trainset: List[dspy.Example], valset: List[dspy.Example], output_program_path: str,
        student_lm: dspy.LM, teacher_lm_short_name: Optional[str], ollama_url: str,
        dspy_max_tokens: int, temperature: float, top_k: int, top_p: float,
        autorun_setting: str, miprov2_init_temp: float
        ) -> Optional[dspy.Module]:
        logging.info(f"Starting DSPy program optimization with MIPROv2 for module: {type(base_module).__name__}...")
        prompt_teacher_lm = student_lm
        if teacher_lm_short_name:
            actual_teacher_model_string = MODEL_MAPPING.get(teacher_lm_short_name)
            if actual_teacher_model_string:
                _teacher_lm = configure_dspy_lm(actual_teacher_model_string, ollama_url, temperature, top_k, top_p, dspy_max_tokens, role="teacher for MIPROv2 prompt_model")
                if _teacher_lm: prompt_teacher_lm = _teacher_lm
                else: logging.warning(f"Failed to configure teacher LM {teacher_lm_short_name}. Using student LM for prompt generation.")
            else: logging.warning(f"Teacher model short name '{teacher_lm_short_name}' not found in MODEL_MAPPING. Using student LM.")
        optimizer = MIPROv2(metric=validate_sentiment, prompt_model=prompt_teacher_lm, task_model=student_lm, max_bootstrapped_demos=0, max_labeled_demos=0, auto=autorun_setting, max_errors=5, init_temperature=miprov2_init_temp, verbose=True)
        try:
            optimized_program = optimizer.compile(student=base_module, trainset=trainset, valset=valset, requires_permission_to_run=False)
            logging.info(f"DSPy MIPROv2 ('{autorun_setting}' auto mode) optimization complete.")
            optimized_program.save(output_program_path)
            logging.info(f"Optimized DSPy program (MIPROv2) saved to: {output_program_path}")
            return optimized_program
        except Exception as e: logging.error(f"DSPy MIPROv2 optimization error: {e}", exc_info=True); return None

    def run_dspy_prediction(items_to_process: List[Dict[str, Any]], optimized_program: dspy.Module, num_queries: int, logs_dir: str, method_name: str, language: str, use_plm_reliability_sig: bool) -> List[Dict[str, Any]]:
        newly_processed_item_logs = []
        for item in tqdm(items_to_process, desc=f"DSPy Prediction ({method_name})"):
            item_uuid = item['uuid']
            article, aspect = item.get('article', ''), item.get('aspect', '')
            log_data_template = {"uuid": item_uuid, "status": "failed", "prediction_int": None, "raw_responses_agg": "", "ground_truth_int": STR_TO_INT_LABEL.get(item.get('sentiment')) if isinstance(item.get('sentiment'), str) else item.get('sentiment')}
            if not article or not aspect:
                log_data_template["reason"] = "Missing article/aspect"; log_data_template["raw_responses_agg"] = "Skipped"
                save_item_log(logs_dir, item_uuid, log_data_template); newly_processed_item_logs.append(log_data_template); continue
            
            call_args = {"article": article, "aspect": aspect}
            if "plm-augmented" in method_name:
                plm_pred_int = item.get(PLM_PREDICTION_KEY)
                plm_pred_str = NEUTRAL_STR
                if plm_pred_int is not None:
                    parsed_plm = INT_TO_STR_LABEL.get(int(plm_pred_int))
                    if parsed_plm: plm_pred_str = parsed_plm
                call_args["plm_suggestion"] = plm_pred_str
                if use_plm_reliability_sig: # If the loaded program expects reliability info
                    f1_scores_lang = PLM_F1_SCORES.get(language, {})
                    f1_neg = f1_scores_lang.get("negative", "N/A")
                    f1_neu = f1_scores_lang.get("neutral", "N/A")
                    f1_pos = f1_scores_lang.get("positive", "N/A")
                    reliability_detail = "N/A"
                    if plm_pred_str in f1_scores_lang and f1_scores_lang[plm_pred_str] != "N/A":
                        reliability_detail = f"{f1_scores_lang[plm_pred_str]}%"
                    plm_reliability_info_str = (f"PLM reliability for {language} (F1 scores): Negative ~{f1_neg}%, Neutral ~{f1_neu}%, Positive ~{f1_pos}%. Current PLM suggestion '{plm_pred_str}' has reliability ~{reliability_detail}.")
                    call_args["plm_reliability_info"] = plm_reliability_info_str

            item_predictions_str, item_raw_outputs, query_success_flag = [], [], False
            for _ in range(num_queries):
                try:
                    prediction_obj = optimized_program(**call_args)
                    pred_str = getattr(prediction_obj, 'sentiment', None)
                    raw_output_detail = f"PredObj: {str(prediction_obj)[:100]}"
                    if hasattr(prediction_obj, 'reasoning'): raw_output_detail += f" | Reasoning: {str(getattr(prediction_obj, 'reasoning', ''))[:100]}"
                    if pred_str and pred_str.lower() in LABELS_STR:
                        item_predictions_str.append(pred_str.lower()); query_success_flag = True
                        item_raw_outputs.append(f"Success: Sentiment='{pred_str}'. {raw_output_detail}")
                    else: item_raw_outputs.append(f"Fail: Invalid Label '{pred_str}'. {raw_output_detail}")
                except Exception as e: item_raw_outputs.append(f"Fail: Exception {e}")
            final_prediction_str = calculate_mode(item_predictions_str, NEUTRAL_STR)
            final_prediction_int = STR_TO_INT_LABEL.get(final_prediction_str)
            current_item_log = {"uuid": item_uuid, "status": "success" if query_success_flag and final_prediction_int is not None else "failed", "prediction_int": final_prediction_int, "raw_responses_agg": " | ".join(item_raw_outputs), "ground_truth_int": int(item.get('sentiment')) if item.get('sentiment') is not None else None, "reason": "" if query_success_flag and final_prediction_int is not None else "DSPy parsing/call failed"}
            save_item_log(logs_dir, item_uuid, current_item_log); newly_processed_item_logs.append(current_item_log)
        return newly_processed_item_logs

# --- Evaluation ---
def calculate_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, Any]:
    logging.info(f"Calculating metrics for {len(y_true)} samples.")
    if not y_true or not y_pred or len(y_true) != len(y_pred): return {"error": "Invalid input for metric calculation", "num_samples_evaluated": 0}
    accuracy = accuracy_score(y_true, y_pred)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', labels=LABELS_INT, zero_division=0)
    p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(y_true, y_pred, average='micro', labels=LABELS_INT, zero_division=0)
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', labels=LABELS_INT, zero_division=0)
    qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic', labels=LABELS_INT)
    report_dict = classification_report(y_true, y_pred, target_names=[INT_TO_STR_LABEL[l] for l in LABELS_INT], labels=LABELS_INT, output_dict=True, zero_division=0)
    metrics = {"accuracy": accuracy, "precision_macro": p_macro, "recall_macro": r_macro, "f1_macro": f1_macro,
               "precision_micro": p_micro, "recall_micro": r_micro, "f1_micro": f1_micro,
               "precision_weighted": p_weighted, "recall_weighted": r_weighted, "f1_weighted": f1_weighted,
               "qwk": qwk, "num_samples_evaluated": len(y_true), "per_class_report": report_dict}
    logging.info(f"Accuracy: {accuracy:.4f}, Macro F1: {f1_macro:.4f}, QWK: {qwk:.4f}")
    return metrics

# --- Saving Final Results ---
def save_final_results(target_dataset_subset: List[Dict[str, Any]], all_item_log_results_for_subset: List[Dict[str, Any]], metrics_summary: Dict[str, Any], output_dir: str, language_prefix: str, debug_run: bool, custom_run_name: Optional[str]):
    os.makedirs(output_dir, exist_ok=True)
    if custom_run_name: file_base_name = custom_run_name
    else: debug_suffix = "_debug" if debug_run else ""; file_base_name = f"{language_prefix}{debug_suffix}"
    predictions_file = os.path.join(output_dir, f"{file_base_name}_predictions.json")
    metrics_file = os.path.join(output_dir, f"{file_base_name}_metrics.json")
    results_map = {res['uuid']: res for res in all_item_log_results_for_subset}
    final_data_to_save = []
    for original_item in target_dataset_subset:
        item_uuid = original_item.get('uuid')
        if item_uuid is None: logging.warning("Original item missing UUID during final save. Skipping."); continue
        new_item_for_output = original_item.copy()
        if item_uuid in results_map:
            logged_result = results_map[item_uuid]
            new_item_for_output["prediction"] = logged_result.get("prediction_int")
            new_item_for_output["raw_llm_response_aggregated"] = logged_result.get("raw_responses_agg")
            new_item_for_output["processing_status"] = logged_result.get("status", "unknown")
            if logged_result.get("status") != "success": new_item_for_output["processing_error_reason"] = logged_result.get("reason", "N/A")
        else:
            new_item_for_output["prediction"] = None; new_item_for_output["raw_llm_response_aggregated"] = "Not found in current run's processed logs"; new_item_for_output["processing_status"] = "missing_from_run_logs"
        final_data_to_save.append(new_item_for_output)
    logging.info(f"Saving final predictions for {len(final_data_to_save)} items to: {predictions_file}")
    with open(predictions_file, "w", encoding="utf-8") as f: json.dump(final_data_to_save, f, ensure_ascii=False, indent=4)
    logging.info(f"Saving final metrics to: {metrics_file}")
    def default_serializer(obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, (datetime.date, datetime.datetime)): return obj.isoformat()
        try: return str(obj)
        except: return repr(obj)
    with open(metrics_file, "w", encoding="utf-8") as f: json.dump(metrics_summary, f, ensure_ascii=False, indent=4, default=default_serializer)

# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aspect-Based Sentiment Analysis with Ollama & DSPy")
    parser.add_argument("--model", type=str, choices=list(MODEL_MAPPING.keys()), default=DEFAULT_MODEL_SHORT_NAME, help="LLM model short name for student/task execution.")
    parser.add_argument("--split", type=str, required=True, choices=['slovenian', 'serbian'], help="Dataset language.")
    parser.add_argument("--method", type=str, required=True, choices=['direct', 'plm-augmented-direct', 'dspy-predict', 'dspy-cot', 'dspy-plm-augmented', 'dspy-plm-augmented-cot'], help="Method to use.")
    parser.add_argument("--num-queries", type=int, default=DEFAULT_NUM_QUERIES, help="Queries per item.")
    parser.add_argument("--debug", action="store_true", help=f"Use ~{DEBUG_FRACTION*100}% of test data.")
    parser.add_argument("--name", type=str, default=None, help="Custom base name for output prediction/metric files.")
    
    parser.add_argument("--ollama-url", type=str, default=DEFAULT_OLLAMA_HOST, help="Ollama host URL.")
    parser.add_argument("--retries", type=int, default=DEFAULT_MAX_RETRIES, help="Ollama request retries.")
    parser.add_argument("--retry-delay", type=int, default=DEFAULT_RETRY_DELAY, help="Delay between retries (s).")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="Ollama request timeout (s).")
    parser.add_argument("--temperature", type=float, default=1.0, help="Ollama temperature for generation.")
    parser.add_argument("--top_k", type=int, default=64, help="Ollama top_k.")
    parser.add_argument("--top_p", type=float, default=0.95, help="Ollama top_p.")
    
    parser.add_argument("--dspy-max-tokens", type=int, default=DEFAULT_DSPY_MAX_TOKENS, help="Max tokens for DSPy LM generation.")
    parser.add_argument("--teacher-model-short-name", type=str, default=None, choices=list(MODEL_MAPPING.keys()) + [None], help="Optional LLM short name for DSPy's prompt_model (teacher).")
    parser.add_argument("--dspy-autorun", type=str, default=DEFAULT_DSPY_AUTORUN, choices=["light", "medium", "heavy"], help="DSPy MIPROv2 auto mode setting.")
    parser.add_argument("--miprov2-temp", type=float, default=DEFAULT_MIPROV2_TEMP, help="Initial temperature for MIPROv2 optimization.")
    
    parser.add_argument("--use-aspect-marker-signature", action="store_true", help="Use DSPy Signatures that explicitly mention <aspect> tags (for dspy-predict/cot).")
    parser.add_argument("--use-plm-reliability-signature", action="store_true", help="Use DSPy Signatures for PLM-augmented methods that include PLM reliability info.")

    parser.add_argument("--force-reprocess", action="store_true", help="Force reprocessing all test items, ignoring logs.")
    parser.add_argument("--force-optimize", action="store_true", help="Force DSPy re-optimization.")
    args = parser.parse_args()

    is_dspy_method = args.method.startswith("dspy")
    if is_dspy_method and not DSPY_AVAILABLE: logging.error("DSPy method chosen but library not available. Exiting."); exit(1)
    if args.num_queries < 1: args.num_queries = 1; logging.warning("--num-queries set to 1.")

    if args.use_plm_reliability_signature and not ("plm-augmented" in args.method):
        logging.error("--use-plm-reliability-signature is only compatible with 'dspy-plm-augmented' or 'dspy-plm-augmented-cot' methods.")
        exit(1)
    if args.use_aspect_marker_signature and ("plm-augmented" in args.method):
        logging.warning("--use-aspect-marker-signature is typically for non-PLM-augmented methods. It will be ignored if method is PLM-augmented and --use-plm-reliability-signature is also set, as reliability signature takes precedence for PLM methods.")


    if args.debug: logging.getLogger().setLevel(logging.DEBUG); logging.info("--- DEBUG LOGGING ENABLED ---")
    else: logging.getLogger().setLevel(logging.INFO)

    start_run_time = time.time()
    actual_model_string = MODEL_MAPPING.get(args.model)
    if not actual_model_string: logging.error(f"Model short name '{args.model}' not found."); exit(1)
    
    output_dir_base = f"../models/ollama/{args.model}/{args.method}/{args.split}"
    logs_dir = os.path.join(output_dir_base, "logs") 
    os.makedirs(output_dir_base, exist_ok=True)

    logging.info(f"Run Config: Model='{actual_model_string}' (Short: '{args.model}'), Method='{args.method}', Split='{args.split}', Name='{args.name if args.name else 'Default'}'")
    logging.info(f"Output Base: {output_dir_base}, Logs Dir for items: {logs_dir}")
    if args.teacher_model_short_name: logging.info(f"DSPy Teacher Model: {args.teacher_model_short_name}")
    if args.use_aspect_marker_signature: logging.info("Using Aspect Marker Signature variant for DSPy.")
    if args.use_plm_reliability_signature: logging.info("Using PLM Reliability Signature variant for DSPy.")


    full_test_data_for_reference = load_data(args.split, 'test')
    if not full_test_data_for_reference: exit(1)
    
    target_test_data_subset = sample_debug_data(full_test_data_for_reference, DEBUG_FRACTION) if args.debug else full_test_data_for_reference
    if not target_test_data_subset: logging.error("Target test data subset empty."); exit(1)
    logging.info(f"Targeting {len(target_test_data_subset)} items for processing in this run.")

    items_to_process_this_run, previously_processed_logs = get_items_to_process(target_test_data_subset, logs_dir, args.force_reprocess)
    newly_processed_item_logs: List[Dict[str, Any]] = []
    optimized_program_path: Optional[str] = None
    current_dspy_max_tokens_student = args.dspy_max_tokens

    if args.method == 'direct':
        if items_to_process_this_run:
            newly_processed_item_logs = run_direct_prompting(items_to_process_this_run, actual_model_string, args.ollama_url, args.num_queries, args.temperature, args.top_k, args.top_p, args.retries, args.retry_delay, args.timeout, logs_dir)
    elif args.method == 'plm-augmented-direct':
        if items_to_process_this_run:
            newly_processed_item_logs = run_plm_augmented_direct_prompting(items_to_process_this_run, actual_model_string, args.ollama_url, args.num_queries, args.temperature, args.top_k, args.top_p, args.retries, args.retry_delay, args.timeout, logs_dir, args.split)
    
    elif is_dspy_method:
        if args.method not in ["dspy-cot", "dspy-plm-augmented-cot"] and args.dspy_max_tokens == DEFAULT_DSPY_MAX_TOKENS:
            current_dspy_max_tokens_student = 50 
            logging.info(f"Using shorter dspy_max_tokens ({current_dspy_max_tokens_student}) for non-CoT DSPy student method.")
        
        student_lm = configure_dspy_lm(actual_model_string, args.ollama_url, args.temperature, args.top_k, args.top_p, current_dspy_max_tokens_student, role="student")
        if not student_lm: exit(1)
        dspy.settings.configure(lm=student_lm)

        # Determine signature class based on method and flags
        signature_class_to_use = None
        if args.method == "dspy-predict":
            signature_class_to_use = SentimentClassificationWithAspectMarkerSignature if args.use_aspect_marker_signature else SentimentClassificationSignature
        elif args.method == "dspy-cot":
            signature_class_to_use = ReasoningSentimentWithAspectMarkerSignature if args.use_aspect_marker_signature else ReasoningSentimentSignature
        elif args.method == "dspy-plm-augmented":
            signature_class_to_use = PLMAugmentedWithReliabilitySignature if args.use_plm_reliability_signature else PLMAugmentedSentimentSignature
        elif args.method == "dspy-plm-augmented-cot":
            signature_class_to_use = ReasoningPLMAugmentedWithReliabilitySignature if args.use_plm_reliability_signature else ReasoningPLMAugmentedSentimentSignature
        
        if not signature_class_to_use: logging.error(f"Could not determine signature class for method: {args.method}"); exit(1)
        logging.info(f"DSPy will use Signature: {signature_class_to_use.__name__}")

        if args.method in ["dspy-predict", "dspy-plm-augmented"]:
            base_dspy_module = dspy.Predict(signature_class_to_use)
        elif args.method in ["dspy-cot", "dspy-plm-augmented-cot"]:
            base_dspy_module = dspy.ChainOfThought(signature_class_to_use)
        else: logging.error(f"Logic error in DSPy module instantiation for method: {args.method}"); exit(1)

        optimized_program_filename = f"optimized_program_{args.split}_{args.method}"
        if args.use_aspect_marker_signature and signature_class_to_use in [SentimentClassificationWithAspectMarkerSignature, ReasoningSentimentWithAspectMarkerSignature]:
            optimized_program_filename += "_aspectmarker"
        if args.use_plm_reliability_signature and signature_class_to_use in [PLMAugmentedWithReliabilitySignature, ReasoningPLMAugmentedWithReliabilitySignature]:
            optimized_program_filename += "_plmreliability"
        if args.teacher_model_short_name: optimized_program_filename += f"_teacher_{args.teacher_model_short_name}"
        optimized_program_filename += f"_autorun_{args.dspy_autorun}_temp_{args.miprov2_temp}.json"
        optimized_program_path = os.path.join(output_dir_base, optimized_program_filename)
        
        program_to_run = None
        if not args.force_optimize and os.path.exists(optimized_program_path):
            try:
                program_to_run = base_dspy_module # Already instantiated with correct signature
                program_to_run.load(optimized_program_path)
                logging.info(f"Loaded existing DSPy program with {signature_class_to_use.__name__}: {optimized_program_path}")
            except Exception as e:
                logging.error(f"Failed to load DSPy program ({e}) with {signature_class_to_use.__name__}. Re-optimizing...")
                program_to_run = None
        
        if not program_to_run:
            logging.info(f"Optimizing DSPy program for method: {args.method} using {signature_class_to_use.__name__}...")
            train_data = load_data(args.split, 'train', split_index=0)
            val_data = load_data(args.split, 'val', split_index=0)
            if not train_data or not val_data: logging.error("Missing train/val for DSPy opt."); exit(1)
            
            dspy_trainset = prepare_dspy_dataset(train_data, args.method, args.split, args.use_plm_reliability_signature, "training")
            dspy_valset = prepare_dspy_dataset(val_data, args.method, args.split, args.use_plm_reliability_signature, "validation")
            if not dspy_trainset or not dspy_valset: logging.error("Failed to prep DSPy datasets."); exit(1)
            
            program_to_run = optimize_dspy_program_mipro(
                base_dspy_module, dspy_trainset, dspy_valset, optimized_program_path,
                student_lm, args.teacher_model_short_name, args.ollama_url, 
                args.dspy_max_tokens, args.temperature, args.top_k, args.top_p,
                args.dspy_autorun, args.miprov2_temp
            )

        if program_to_run and items_to_process_this_run:
            newly_processed_item_logs = run_dspy_prediction(items_to_process_this_run, program_to_run, args.num_queries, logs_dir, args.method, args.split, args.use_plm_reliability_signature)
        elif not program_to_run: logging.error("DSPy program not available. Cannot predict.")
        elif not items_to_process_this_run: logging.info("No new items to process for DSPy in this run.")

    # --- Final Aggregation, Evaluation & Saving ---
    all_results_for_subset: Dict[Any, Dict] = {**previously_processed_logs}
    for log_entry in newly_processed_item_logs: all_results_for_subset[log_entry['uuid']] = log_entry
    final_log_results_list_for_subset = list(all_results_for_subset.values())

    y_true_eval, y_pred_eval = [], []
    successful_count_in_subset = 0
    for res_item_log in final_log_results_list_for_subset:
        if res_item_log.get("status") == "success" and res_item_log.get("prediction_int") is not None and res_item_log.get("ground_truth_int") is not None:
            y_true_eval.append(res_item_log["ground_truth_int"])
            y_pred_eval.append(res_item_log["prediction_int"])
            successful_count_in_subset +=1
    
    logging.info(f"Total items in current run's target subset: {len(target_test_data_subset)}")
    logging.info(f"Items successfully predicted (from logs or new): {successful_count_in_subset}")

    metrics_summary = {
        "run_name": args.name if args.name else "default",
        "model_name_full": actual_model_string, "model_short_name": args.model, "method": args.method, "split": args.split,
        "num_queries_per_item": args.num_queries,
        "ollama_sampling_params": {"temperature": args.temperature, "top_k": args.top_k, "top_p": args.top_p},
        "num_items_in_target_subset": len(target_test_data_subset),
        "num_items_successfully_predicted_in_subset": successful_count_in_subset,
    }
    if is_dspy_method:
        metrics_summary.update({
            "dspy_student_max_tokens_setting": current_dspy_max_tokens_student,
            "dspy_teacher_model_short_name": args.teacher_model_short_name,
            "dspy_autorun_setting": args.dspy_autorun,
            "dspy_miprov2_temp": args.miprov2_temp,
            "dspy_used_aspect_marker_signature": args.use_aspect_marker_signature and signature_class_to_use in [SentimentClassificationWithAspectMarkerSignature, ReasoningSentimentWithAspectMarkerSignature],
            "dspy_used_plm_reliability_signature": args.use_plm_reliability_signature and signature_class_to_use in [PLMAugmentedWithReliabilitySignature, ReasoningPLMAugmentedWithReliabilitySignature],
            "dspy_actual_signature_class_used": signature_class_to_use.__name__ if signature_class_to_use else "N/A",
            "optimized_dspy_program_path": optimized_program_path if optimized_program_path and os.path.exists(optimized_program_path) else None
        })
    
    if not y_true_eval:
         metrics_summary["error"] = "No samples available for evaluation."
         metrics_summary.update({k: 0.0 for k in ["accuracy", "precision_macro", "recall_macro", "f1_macro", "precision_micro", "recall_micro", "f1_micro", "precision_weighted", "recall_weighted", "f1_weighted", "qwk"]})
         metrics_summary["num_samples_evaluated"] = 0; metrics_summary["per_class_report"] = {}
    else:
         calculated_metrics = calculate_metrics(y_true_eval, y_pred_eval)
         metrics_summary.update(calculated_metrics)

    save_final_results(target_test_data_subset, final_log_results_list_for_subset, metrics_summary, output_dir_base, args.split, args.debug, args.name)

    all_targeted_items_processed_successfully = True
    if os.path.exists(logs_dir) and len(target_test_data_subset) > 0 :
        for item_in_target_subset in target_test_data_subset:
            uuid_targeted = item_in_target_subset.get('uuid')
            if uuid_targeted not in all_results_for_subset or all_results_for_subset[uuid_targeted].get("status") != "success":
                all_targeted_items_processed_successfully = False; break
        if all_targeted_items_processed_successfully and not args.debug:
            logging.info(f"All {len(target_test_data_subset)} items targeted in this run were successfully processed. Deleting logs: {logs_dir}")
            try: shutil.rmtree(logs_dir)
            except Exception as e: logging.error(f"Failed to delete logs directory {logs_dir}: {e}")
        else: logging.info(f"Not all targeted items successfully processed, logs incomplete, or debug mode. Retaining logs: {logs_dir}")
    elif not os.path.exists(logs_dir) and len(target_test_data_subset) > 0:
         logging.info("No logs directory found, implies all items were processed in this single run or force_reprocess was used without prior logs.")
    
    end_run_time = time.time()
    logging.info(f"Script finished. Total run time: {end_run_time - start_run_time:.2f} seconds.")
