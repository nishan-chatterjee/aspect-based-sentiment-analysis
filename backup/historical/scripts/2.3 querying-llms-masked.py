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
    DSPY_AVAILABLE = True
except ImportError:
    print("Error: The 'dspy-ai' library is not installed. This script requires it. Install it: pip install dspy-ai"); DSPY_AVAILABLE = False; exit(1)

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
    "qwen-2.5-72b": "qwen2.5:72b" # Teacher model
}
DEFAULT_MODEL_SHORT_NAME = "gemma-3-27b"

DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11435"
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 5
DEFAULT_TIMEOUT = 180
DEBUG_FRACTION = 0.01
DEFAULT_NUM_QUERIES = 3
DEFAULT_DSPY_MAX_TOKENS = 1024 # Updated default as per request
DEFAULT_MIPROV2_TEMP = 1.0
DEFAULT_DSPY_AUTORUN = "heavy"


PLM_PREDICTION_KEY = "global-context-modelling/simplified-dart-xlmr"
PLM_PROBABILITIES_KEY = "global-context-modelling/simplified-dart-xlmr/probabilities"

INT_TO_STR_LABEL = {-1: "negative", 0: "neutral", 1: "positive"}
STR_TO_INT_LABEL = {v: k for k, v in INT_TO_STR_LABEL.items()}
LABELS_INT = list(INT_TO_STR_LABEL.keys())
LABELS_STR = list(STR_TO_INT_LABEL.keys())
NEUTRAL_STR = "neutral"
NEUTRAL_INT = 0
MASK_TOKEN = "[MASK]"

# PLM F1 scores for reliability signatures
PLM_F1_SCORES = {
    "slovenian": {"negative": 46.05, "neutral": 94.14, "positive": 74.51},
    "serbian": {"negative": 69.84, "neutral": 84.36, "positive": 87.39}
}

# --- Aspect Masking Function ---
def mask_aspect_in_text(article_text: str) -> str:
    """
    Replaces all occurrences of '<aspect>...</aspect>' in the text with a '[MASK]' token.
    """
    if not isinstance(article_text, str): return ""
    # This regex finds the aspect tags and everything between them, non-greedily.
    return re.sub(r'<aspect>.*?</aspect>', MASK_TOKEN, article_text)

# --- Data Loading ---
def load_data(language: str, split_type: str, split_index: Optional[int] = None) -> List[Dict[str, Any]]:
    base_data_root = "../data/final/balanced-predicted-softmax/"
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
        
        expected_base_keys = ['uuid', 'article', 'aspect', 'sentiment', PLM_PREDICTION_KEY]
        if data and not all(k in data[0] for k in expected_base_keys):
             missing_keys = [k for k in expected_base_keys if k not in data[0]]
             logging.warning(f"First item in {file_path} may be malformed. Missing expected base keys: {missing_keys}.")
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

# --- DSPy Signature for PLM-Augmented CoT with Masking Awareness ---
class ReasoningPLMAugmentedWithReliabilityAndSoftmaxSignature(dspy.Signature):
    """Given an article, aspect, a PLM suggestion, PLM reliability info, and PLM softmax probabilities:
1. Provide step-by-step reasoning, explicitly considering the PLM suggestion, its stated reliability, and its softmax probabilities.
2. Conclude with the final sentiment ('negative', 'neutral', 'positive')."""
    article: str = dspy.InputField(desc="The full text of the article. The 'aspect' may be explicitly named or replaced with a generic '[MASK]' placeholder.")
    aspect: str = dspy.InputField(desc="The specific aspect. This will either be the specific aspect phrase or the generic placeholder '[MASK]' if the aspect's name has been hidden in the article.")
    plm_suggestion: TypingLiteral["negative", "neutral", "positive"] = dspy.InputField(desc="Sentiment suggestion from a prior model.")
    plm_reliability_info: str = dspy.InputField(desc="Information about the PLM's typical F1 scores for the suggested class and language.")
    plm_softmax_probabilities: str = dspy.InputField(desc="PLM's softmax probabilities for its suggestion (e.g., 'Negative: 0.1023, Neutral: 0.8054, Positive: 0.0923').")
    reasoning: str = dspy.OutputField(desc="Step-by-step reasoning, incorporating PLM suggestion, its reliability, and softmax probabilities.")
    sentiment: TypingLiteral["negative", "neutral", "positive"] = dspy.OutputField(desc="Final sentiment.")

# --- DSPy Shared Functions ---
def configure_dspy_lm(model_name_str: str, host_url: str, temperature: float, top_k: int, top_p: float, dspy_max_tokens: int, role: str = "student") -> Optional[dspy.LM]:
    try:
        lm = dspy.LM(
            model=f"ollama_chat/{model_name_str}", 
            api_base=host_url, 
            model_type='chat', 
            temperature=temperature, 
            top_p=top_p, 
            top_k=top_k, 
            max_tokens=dspy_max_tokens
        )
        logging.info(f"DSPy {role} LM configured with max_tokens={dspy_max_tokens} using dspy.LM: ollama_chat/{model_name_str}")
        return lm
    except Exception as e: 
        logging.error(f"DSPy {role} LM config for ollama_chat/{model_name_str} failed: {e}", exc_info=True)
        return None

def prepare_dspy_dataset(data: List[Dict[str, Any]], language: str, mask_aspect: bool, purpose: str = "training") -> List[dspy.Example]:
    dspy_dataset = []
    for item in data:
        article, aspect, sentiment_int = item.get('article'), item.get('aspect'), item.get('sentiment')
        if article and aspect and sentiment_int is not None:
            try:
                sentiment_str = INT_TO_STR_LABEL.get(int(sentiment_int))
                if not sentiment_str: continue

                # Apply masking if enabled
                current_article = mask_aspect_in_text(article) if mask_aspect else article
                current_aspect = MASK_TOKEN if mask_aspect else aspect

                example_args = {"article": current_article, "aspect": current_aspect, "sentiment": sentiment_str}
                input_keys = ["article", "aspect"]
                
                plm_pred_int = item.get(PLM_PREDICTION_KEY)
                plm_pred_str = NEUTRAL_STR
                if plm_pred_int is not None:
                    parsed_plm = INT_TO_STR_LABEL.get(int(plm_pred_int))
                    if parsed_plm: plm_pred_str = parsed_plm
                example_args["plm_suggestion"] = plm_pred_str
                input_keys.append("plm_suggestion")
                
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

                plm_probs_dict = item.get(PLM_PROBABILITIES_KEY)
                softmax_str_for_prompt = "PLM Softmax Probabilities: N/A."
                if plm_probs_dict and isinstance(plm_probs_dict, dict):
                    prob_neg = plm_probs_dict.get("Negative", 0.0) 
                    prob_neu = plm_probs_dict.get("Neutral", 0.0)
                    prob_pos = plm_probs_dict.get("Positive", 0.0)
                    softmax_str_for_prompt = (
                        f"PLM Prediction Softmax Probabilities: "
                        f"Negative={prob_neg:.4f}, Neutral={prob_neu:.4f}, Positive={prob_pos:.4f}."
                    )
                else:
                    logging.debug(f"Item {item.get('uuid')} missing or malformed PLM probabilities for softmax signature during dataset prep. Using N/A.")
                example_args["plm_softmax_probabilities"] = softmax_str_for_prompt
                input_keys.append("plm_softmax_probabilities")
                                
                dspy_dataset.append(dspy.Example(**example_args).with_inputs(*input_keys))
            except (ValueError, TypeError) as e: logging.warning(f"Skipping item due to error during example prep: {e} - Item: {item.get('uuid')}"); pass
    logging.info(f"Prepared {len(dspy_dataset)} DSPy {purpose} examples (Masking: {mask_aspect}).")
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
            _teacher_lm = configure_dspy_lm(actual_teacher_model_string, ollama_url, temperature, top_k, top_p, dspy_max_tokens * 2 if dspy_max_tokens < 500 else dspy_max_tokens, role="teacher for MIPROv2 prompt_model")
            if _teacher_lm: prompt_teacher_lm = _teacher_lm
            else: logging.warning(f"Failed to configure teacher LM {teacher_lm_short_name}. Using student LM for prompt generation.")
        else: logging.warning(f"Teacher model short name '{teacher_lm_short_name}' not found in MODEL_MAPPING. Using student LM.")
    
    optimizer = MIPROv2(metric=validate_sentiment, 
                        prompt_model=prompt_teacher_lm, 
                        task_model=student_lm, 
                        max_bootstrapped_demos=0,
                        max_labeled_demos=0,
                        init_temperature=miprov2_init_temp, 
                        verbose=True) 
    
    logging.info(f"MIPROv2 optimizer configured. Autorun setting '{autorun_setting}' noted (not directly used by MIPROv2 compile).")

    try:
        optimized_program = optimizer.compile(student=base_module, trainset=trainset, valset=valset, requires_permission_to_run=True)
        logging.info(f"DSPy MIPROv2 optimization complete.")
        optimized_program.save(output_program_path)
        logging.info(f"Optimized DSPy program (MIPROv2) saved to: {output_program_path}")
        return optimized_program
    except Exception as e: logging.error(f"DSPy MIPROv2 optimization error: {e}", exc_info=True); return None

def run_dspy_prediction(items_to_process: List[Dict[str, Any]], optimized_program: dspy.Module, num_queries: int, logs_dir: str, language: str, mask_aspect: bool) -> List[Dict[str, Any]]:
    newly_processed_item_logs = []
    for item in tqdm(items_to_process, desc=f"DSPy Prediction (dspy-plm-augmented-cot, Masking: {mask_aspect})"):
        item_uuid = item['uuid']
        article, aspect = item.get('article', ''), item.get('aspect', '')
        log_data_template = {"uuid": item_uuid, "status": "failed", "prediction_int": None, "dspy_query_details": [], "ground_truth_int": STR_TO_INT_LABEL.get(item.get('sentiment')) if isinstance(item.get('sentiment'), str) else item.get('sentiment')}
        
        if not article or not aspect:
            log_data_template["reason"] = "Missing article/aspect"
            save_item_log(logs_dir, item_uuid, log_data_template); newly_processed_item_logs.append(log_data_template); continue
        
        # Apply masking if enabled for the prediction call
        current_article = mask_aspect_in_text(article) if mask_aspect else article
        current_aspect = MASK_TOKEN if mask_aspect else aspect

        call_args = {"article": current_article, "aspect": current_aspect}
        
        plm_pred_int = item.get(PLM_PREDICTION_KEY)
        plm_pred_str = NEUTRAL_STR
        if plm_pred_int is not None:
            parsed_plm = INT_TO_STR_LABEL.get(int(plm_pred_int))
            if parsed_plm: plm_pred_str = parsed_plm
        call_args["plm_suggestion"] = plm_pred_str
        
        f1_scores_lang = PLM_F1_SCORES.get(language, {})
        f1_neg = f1_scores_lang.get("negative", "N/A")
        f1_neu = f1_scores_lang.get("neutral", "N/A")
        f1_pos = f1_scores_lang.get("positive", "N/A")
        reliability_detail = "N/A"
        if plm_pred_str in f1_scores_lang and f1_scores_lang[plm_pred_str] != "N/A":
            reliability_detail = f"{f1_scores_lang[plm_pred_str]}%"
        plm_reliability_info_str = (f"PLM reliability for {language} (F1 scores): Negative ~{f1_neg}%, Neutral ~{f1_neu}%, Positive ~{f1_pos}%. Current PLM suggestion '{plm_pred_str}' has reliability ~{reliability_detail}.")
        call_args["plm_reliability_info"] = plm_reliability_info_str

        plm_probs_dict = item.get(PLM_PROBABILITIES_KEY)
        softmax_str_for_call = "PLM Softmax Probabilities: N/A."
        if plm_probs_dict and isinstance(plm_probs_dict, dict):
            prob_neg = plm_probs_dict.get("Negative", 0.0)
            prob_neu = plm_probs_dict.get("Neutral", 0.0)
            prob_pos = plm_probs_dict.get("Positive", 0.0)
            softmax_str_for_call = (
                f"PLM Prediction Softmax Probabilities: "
                f"Negative={prob_neg:.4f}, Neutral={prob_neu:.4f}, Positive={prob_pos:.4f}."
            )
        else:
             logging.debug(f"Item {item_uuid} missing or malformed PLM probabilities for softmax signature during prediction. Using N/A.")
        call_args["plm_softmax_probabilities"] = softmax_str_for_call
        
        item_predictions_str, item_query_details_list, query_success_flag = [], [], False
        for query_idx in range(num_queries):
            query_detail = {"query_index": query_idx, "status": "failed", "predicted_sentiment": None, "reasoning": None, "raw_prediction_object_str": None}
            try:
                prediction_obj = optimized_program(**call_args)
                pred_str = getattr(prediction_obj, 'sentiment', None)
                query_detail["raw_prediction_object_str"] = str(prediction_obj)
                
                if hasattr(prediction_obj, 'reasoning'):
                    query_detail["reasoning"] = str(getattr(prediction_obj, 'reasoning', ''))

                if pred_str and pred_str.lower() in LABELS_STR:
                    item_predictions_str.append(pred_str.lower())
                    query_success_flag = True
                    query_detail["status"] = "success"
                    query_detail["predicted_sentiment"] = pred_str.lower()
                else:
                    query_detail["status"] = "invalid_label"
                    query_detail["predicted_sentiment"] = pred_str
                    logging.warning(f"DSPy item {item_uuid} query {query_idx}: Invalid sentiment label '{pred_str}'. Raw: {prediction_obj}")
            except Exception as e:
                logging.error(f"DSPy item {item_uuid} query {query_idx} exception: {e}", exc_info=True)
                query_detail["status"] = "exception"
                query_detail["raw_prediction_object_str"] = f"Exception: {str(e)}"
            item_query_details_list.append(query_detail)

        final_prediction_str = calculate_mode(item_predictions_str, NEUTRAL_STR)
        final_prediction_int = STR_TO_INT_LABEL.get(final_prediction_str)
        
        current_item_log = {
            "uuid": item_uuid,
            "status": "success" if query_success_flag and final_prediction_int is not None else "failed",
            "prediction_int": final_prediction_int,
            "dspy_query_details": item_query_details_list,
            "ground_truth_int": int(item.get('sentiment')) if item.get('sentiment') is not None else None,
            "reason": "" if query_success_flag and final_prediction_int is not None else "DSPy parsing/call failed or no valid labels from queries"
        }
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
            if "dspy_query_details" in logged_result:
                new_item_for_output["dspy_query_details"] = logged_result["dspy_query_details"]
            new_item_for_output["processing_status"] = logged_result.get("status", "unknown")
            if logged_result.get("status") != "success": new_item_for_output["processing_error_reason"] = logged_result.get("reason", "N/A")
        else:
            new_item_for_output["prediction"] = None
            new_item_for_output["processing_status"] = "missing_from_run_logs"
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
    parser = argparse.ArgumentParser(description="Aspect-Based Sentiment Analysis with Ollama & DSPy using PLM-Augmented Chain-of-Thought")
    parser.add_argument("--model", type=str, choices=list(MODEL_MAPPING.keys()), default=DEFAULT_MODEL_SHORT_NAME, help="LLM model short name for student/task execution.")
    parser.add_argument("--split", type=str, required=True, choices=['slovenian', 'serbian'], help="Dataset language.")
    parser.add_argument("--mask", action="store_true", help="Replace the aspect text with a '[MASK]' token in the article and prompt.")
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
    parser.add_argument("--teacher-model-short-name", type=str, default="qwen-2.5-72b", choices=list(MODEL_MAPPING.keys()) + [None], help="Optional LLM short name for DSPy's prompt_model (teacher).")
    parser.add_argument("--dspy-autorun", type=str, default=DEFAULT_DSPY_AUTORUN, choices=["light", "medium", "heavy"], help="DSPy MIPROv2 auto mode setting (Note: Not directly used by MIPROv2 compile, for config tracking).")
    parser.add_argument("--miprov2-temp", type=float, default=DEFAULT_MIPROV2_TEMP, help="Initial temperature for MIPROv2 optimization.")
    
    # These are now implicitly active for the sole method used, but kept for clarity/potential future use
    parser.add_argument("--use-plm-reliability-signature", action="store_true", help="[DEPRECATED, always on] Use DSPy Signatures that include PLM reliability info.")
    parser.add_argument("--use-plm-softmax-signature", action="store_true", help="[DEPRECATED, always on] Use DSPy Signatures that include PLM softmax probabilities.")

    parser.add_argument("--force-reprocess", action="store_true", help="Force reprocessing all test items, ignoring logs.")
    parser.add_argument("--force-optimize", action="store_true", help="Force DSPy re-optimization.")
    args = parser.parse_args()
    
    # --- Script Configuration ---
    METHOD_NAME = "dspy-plm-augmented-cot" # Hardcoded as per request
    if args.num_queries < 1: args.num_queries = 1; logging.warning("--num-queries set to 1.")
    
    if args.debug: logging.getLogger().setLevel(logging.DEBUG); logging.info("--- DEBUG LOGGING ENABLED ---")
    else: logging.getLogger().setLevel(logging.INFO)

    start_run_time = time.time()
    actual_model_string = MODEL_MAPPING.get(args.model)
    if not actual_model_string: logging.error(f"Model short name '{args.model}' not found."); exit(1)
    
    mask_suffix = "masked" if args.mask else "unmasked"
    output_dir_base = f"../models/ollama/{args.model}/{METHOD_NAME}/{args.split}/{mask_suffix}"
    logs_dir = os.path.join(output_dir_base, "logs") 
    os.makedirs(output_dir_base, exist_ok=True)

    logging.info(f"Run Config: Model='{actual_model_string}' (Short: '{args.model}'), Method='{METHOD_NAME}', Split='{args.split}', Masking='{args.mask}', Name='{args.name if args.name else 'Default'}'")
    logging.info(f"Output Base: {output_dir_base}, Logs Dir for items: {logs_dir}")
    if args.teacher_model_short_name: logging.info(f"DSPy Teacher Model: {args.teacher_model_short_name}")
    
    full_test_data_for_reference = load_data(args.split, 'test')
    if not full_test_data_for_reference: exit(1)
    
    target_test_data_subset = sample_debug_data(full_test_data_for_reference, DEBUG_FRACTION) if args.debug else full_test_data_for_reference
    if not target_test_data_subset: logging.error("Target test data subset empty."); exit(1)
    logging.info(f"Targeting {len(target_test_data_subset)} items for processing in this run.")

    items_to_process_this_run, previously_processed_logs = get_items_to_process(target_test_data_subset, logs_dir, args.force_reprocess)
    newly_processed_item_logs: List[Dict[str, Any]] = []
    
    # --- DSPy Execution Logic ---
    student_lm = configure_dspy_lm(actual_model_string, args.ollama_url, args.temperature, args.top_k, args.top_p, args.dspy_max_tokens, role="student")
    if not student_lm: exit(1)
    dspy.settings.configure(lm=student_lm)

    signature_class_to_use = ReasoningPLMAugmentedWithReliabilityAndSoftmaxSignature
    logging.info(f"DSPy will use Signature: {signature_class_to_use.__name__}")

    base_dspy_module = dspy.ChainOfThought(signature_class_to_use)

    optimized_program_filename = f"optimized_program_{args.split}_{METHOD_NAME}_{mask_suffix}"
    if args.teacher_model_short_name: optimized_program_filename += f"_teacher_{args.teacher_model_short_name}"
    optimized_program_filename += f"_autorun_{args.dspy_autorun}_temp_{args.miprov2_temp}.json"
    optimized_program_path = os.path.join(output_dir_base, optimized_program_filename)
    
    program_to_run = None
    if not args.force_optimize and os.path.exists(optimized_program_path):
        try:
            program_to_run = base_dspy_module
            program_to_run.load(optimized_program_path)
            logging.info(f"Loaded existing DSPy program from: {optimized_program_path}")
        except Exception as e:
            logging.error(f"Failed to load DSPy program ({e}). Re-optimizing...")
            program_to_run = None
    
    if not program_to_run:
        logging.info(f"Optimizing DSPy program...")
        train_data = load_data(args.split, 'train', split_index=0)
        val_data = load_data(args.split, 'val', split_index=0)
        if not train_data or not val_data: logging.error("Missing train/val data for DSPy optimization."); exit(1)
        
        # Pass the mask_aspect flag to the dataset preparation
        dspy_trainset = prepare_dspy_dataset(train_data, args.split, args.mask, "training")
        dspy_valset = prepare_dspy_dataset(val_data, args.split, args.mask, "validation")
        if not dspy_trainset or not dspy_valset: logging.error("Failed to prepare DSPy datasets."); exit(1)
        
        program_to_run = optimize_dspy_program_mipro(
            base_dspy_module, dspy_trainset, dspy_valset, optimized_program_path,
            student_lm, args.teacher_model_short_name, args.ollama_url, 
            args.dspy_max_tokens, args.temperature, args.top_k, args.top_p,
            args.dspy_autorun, args.miprov2_temp
        )

    if program_to_run and items_to_process_this_run:
        newly_processed_item_logs = run_dspy_prediction(items_to_process_this_run, program_to_run, args.num_queries, logs_dir, args.split, args.mask)
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
        "model_name_full": actual_model_string, "model_short_name": args.model, "method": METHOD_NAME, "split": args.split,
        "use_masking": args.mask,
        "num_queries_per_item": args.num_queries,
        "ollama_sampling_params": {"temperature": args.temperature, "top_k": args.top_k, "top_p": args.top_p},
        "num_items_in_target_subset": len(target_test_data_subset),
        "num_items_successfully_predicted_in_subset": successful_count_in_subset,
        "dspy_student_max_tokens_setting": args.dspy_max_tokens,
        "dspy_teacher_model_short_name": args.teacher_model_short_name,
        "dspy_autorun_setting": args.dspy_autorun,
        "dspy_miprov2_temp": args.miprov2_temp,
        "dspy_actual_signature_class_used": signature_class_to_use.__name__,
        "optimized_dspy_program_path": optimized_program_path if optimized_program_path and os.path.exists(optimized_program_path) else None
    }
    
    if not y_true_eval:
         metrics_summary["error"] = "No samples available for evaluation."
         metrics_summary.update({k: 0.0 for k in ["accuracy", "precision_macro", "recall_macro", "f1_macro", "precision_micro", "recall_micro", "f1_micro", "precision_weighted", "recall_weighted", "f1_weighted", "qwk"]})
         metrics_summary["num_samples_evaluated"] = 0; metrics_summary["per_class_report"] = {}
    else:
         calculated_metrics = calculate_metrics(y_true_eval, y_pred_eval)
         metrics_summary.update(calculated_metrics)

    save_final_results(target_test_data_subset, final_log_results_list_for_subset, metrics_summary, output_dir_base, f"{args.split}_{mask_suffix}", args.debug, args.name)

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
