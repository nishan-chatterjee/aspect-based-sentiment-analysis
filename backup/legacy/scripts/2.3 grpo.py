# Ensure unsloth is imported first as per its recommendation
import unsloth 

import argparse
import json
import os
import random
import re
from typing import List, Dict, Tuple, Any
import time 

import numpy as np
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, TextStreamer 
from trl import GRPOConfig, GRPOTrainer
from peft import PeftModel 
from unsloth import FastLanguageModel 

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    cohen_kappa_score,
    classification_report,
)

# --- Global Constants and Prompts (Identical) ---
GRPO_SYSTEM_PROMPT = """Given an article, an aspect, and a PLM sentiment suggestion, you are tasked with providing a detailed step-by-step reasoning that considers all inputs, and concluding with the final sentiment: 'negative', 'neutral', or 'positive'. The reasoning should explain how the PLM suggestion was considered.

High Stakes Scenario: You are part of a critical team analyzing media sentiment for a major regulatory body in Slovenia. Your analysis will directly inform policy decisions that could impact public health, environmental protection, and financial stability. Therefore, it is essential to provide accurate and well-reasoned sentiment analysis.

### Example:
Article: "The local government has approved the expansion of Termoelektrarna's coal-fired power plant, despite widespread opposition from environmental groups."
Aspect: Termoelektrarna
PLM Suggestion: negative

<reasoning>
1. Article Analysis: The article discusses the approval of Termoelektrarna's expansion plans for a coal-fired power plant.
2. Contextual Evaluation: Despite the local government's approval, the article highlights significant opposition from environmental groups concerned about air pollution and health risks.
3. PLM Suggestion Consideration: The PLM suggests a 'negative' sentiment, which aligns with the widespread opposition and concerns raised in the article.
4. Reasoning Construction: The negative sentiment is justified by the strong opposition from environmental groups, highlighting potential environmental and health impacts. The government's approval does not mitigate these concerns.
5. Final Sentiment Determination: Based on the analysis, the final sentiment for Termoelektrarna is 'negative'.
</reasoning>
<sentiment>
negative
</sentiment>"""

REASONING_START_TAG = "<reasoning>"
REASONING_END_TAG = "</reasoning>"
SENTIMENT_START_TAG = "<sentiment>"
SENTIMENT_END_TAG = "</sentiment>"

SENTIMENT_MAP_INT_TO_STR = {-1: "negative", 0: "neutral", 1: "positive"}
SENTIMENT_MAP_STR_TO_INT = {"negative": -1, "neutral": 0, "positive": 1, "unknown": -99} 
VALID_SENTIMENT_STRINGS = ["negative", "neutral", "positive"]

MATCH_FULL_FORMAT_RE = re.compile(
    rf"^{REASONING_START_TAG}(.*?){REASONING_END_TAG}\s*"
    rf"{SENTIMENT_START_TAG}(.*?){SENTIMENT_END_TAG}\s*$",
    re.DOTALL | re.MULTILINE,
)

MATCH_SENTIMENT_TAG_RE = re.compile(
    rf"{SENTIMENT_START_TAG}(.*?){SENTIMENT_END_TAG}",
    re.DOTALL | re.MULTILINE,
)


# --- Helper Functions (Identical) ---
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def map_sentiment_to_string(sentiment_int: int) -> str:
    return SENTIMENT_MAP_INT_TO_STR.get(sentiment_int, "unknown")

def map_string_to_sentiment(sentiment_str: str) -> int:
    return SENTIMENT_MAP_STR_TO_INT.get(sentiment_str.lower().strip(), -99) 

def load_absa_data(data_dir: str, language: str, train_split_index: int, plm_key: str, 
                       max_train_samples: int = None, max_val_samples: int = None) -> Tuple[Dataset, Dataset, Dataset]:
    train_val_file = os.path.join(data_dir, f"{language}_train_val_balanced_{train_split_index}.json")
    test_file = os.path.join(data_dir, f"{language}_test_balanced.json")

    with open(train_val_file, 'r', encoding='utf-8') as f:
        train_val_data = json.load(f)
    with open(test_file, 'r', encoding='utf-8') as f:
        test_data_json = json.load(f)

    raw_train_ds = Dataset.from_list(train_val_data['train'])
    raw_val_ds = Dataset.from_list(train_val_data['val'])
    # This is the full raw test dataset before any limiting or stratification
    raw_test_ds_full = Dataset.from_list(test_data_json['test']) 
    
    print(f"Loaded raw train samples: {len(raw_train_ds)}")
    print(f"Loaded raw val samples: {len(raw_val_ds)}")
    print(f"Loaded full raw test samples: {len(raw_test_ds_full)}")

    def _prepare_grpo_sample(example):
        article = example.get("article", "")
        aspect = example.get("aspect", "")
        plm_sentiment_int = example.get(plm_key, 0) 
        plm_sentiment_str = map_sentiment_to_string(plm_sentiment_int)
        
        true_sentiment_int = example.get("sentiment", 0)
        true_sentiment_str = map_sentiment_to_string(true_sentiment_int)

        user_prompt_content = (
            f"Article: \"{article}\"\n"
            f"Aspect: \"{aspect}\"\n"
            f"PLM Suggestion: {plm_sentiment_str}"
        )
        
        example["original_article"] = article
        example["original_aspect"] = aspect
        example["original_plm_suggestion_str"] = plm_sentiment_str
        example["original_plm_suggestion_int"] = plm_sentiment_int
        example["true_sentiment_str"] = true_sentiment_str 
        example["true_sentiment_int"] = true_sentiment_int # used by rewards

        example["prompt"] = [
            {"role": "system", "content": GRPO_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt_content},
        ]
        example["answer"] = true_sentiment_str # ground truth for rewards
        return example

    train_dataset = raw_train_ds.map(_prepare_grpo_sample, remove_columns=raw_train_ds.column_names)
    val_dataset = raw_val_ds.map(_prepare_grpo_sample, remove_columns=raw_val_ds.column_names)
    
    if max_train_samples is not None and len(train_dataset) > max_train_samples:
        train_dataset = train_dataset.select(range(max_train_samples))
        print(f"Sliced train dataset to {len(train_dataset)} samples based on max_train_samples.")
    
    if max_val_samples is not None and len(val_dataset) > max_val_samples:
        val_dataset = val_dataset.select(range(max_val_samples))
        print(f"Sliced val dataset to {len(val_dataset)} samples based on max_val_samples.")

    print(f"Processed train samples for GRPO: {len(train_dataset)}")
    if len(train_dataset) > 0: print(f"Sample train entry: {train_dataset[0]}")
    print(f"Processed val samples for GRPO: {len(val_dataset)}")
    if len(val_dataset) > 0: print(f"Sample val entry: {val_dataset[0]}")

    # Return the full raw test dataset; limiting/stratification will happen in main()
    return train_dataset, val_dataset, raw_test_ds_full


def extract_reasoning_and_sentiment(completion_text: str) -> Tuple[str, str]:
    match = MATCH_FULL_FORMAT_RE.search(completion_text)
    if match:
        reasoning = match.group(1).strip()
        sentiment = match.group(2).strip().lower()
        return reasoning, sentiment
    
    sentiment_match = MATCH_SENTIMENT_TAG_RE.search(completion_text)
    extracted_sentiment = sentiment_match.group(1).strip().lower() if sentiment_match else "unknown"
    return "parsing_error_reasoning_not_found", extracted_sentiment


# --- Reward Functions (Identical) ---
def correctness_reward_func(completions: List[List[Dict[str, str]]], answer: List[str], **kwargs) -> List[float]:
    scores = []
    true_sentiment_str = answer[0] 
    true_sentiment_int = map_string_to_sentiment(true_sentiment_str)

    for completion_list in completions: 
        completion_text = completion_list[0]["content"] 
        _, pred_sentiment_str = extract_reasoning_and_sentiment(completion_text)
        pred_sentiment_int = map_string_to_sentiment(pred_sentiment_str)
        
        score = 0.0
        if pred_sentiment_str not in VALID_SENTIMENT_STRINGS:
            score = -5.0  
        elif pred_sentiment_int == true_sentiment_int:
            score = 5.0   
        elif abs(pred_sentiment_int - true_sentiment_int) == 1: 
            score = 0.5   
        elif abs(pred_sentiment_int - true_sentiment_int) == 2: 
            score = -3.0  
        scores.append(score)
    return scores

def format_reward_func(completions: List[List[Dict[str, str]]], **kwargs) -> List[float]:
    scores = []
    for completion_list in completions:
        completion_text = completion_list[0]["content"]
        score = 0.0
        if MATCH_FULL_FORMAT_RE.search(completion_text):
            score += 3.0  
        else:
            if REASONING_START_TAG in completion_text: score += 0.5
            if REASONING_END_TAG in completion_text: score += 0.5
            if SENTIMENT_START_TAG in completion_text: score += 0.5
            if SENTIMENT_END_TAG in completion_text: score += 0.5
            if score < 2.0 : score -= 1.0 
        scores.append(score)
    return scores

def reasoning_quality_reward_func(completions: List[List[Dict[str, str]]], **kwargs) -> List[float]:
    scores = []
    for completion_list in completions:
        completion_text = completion_list[0]["content"]
        reasoning, _ = extract_reasoning_and_sentiment(completion_text)
        score = 0.0
        if reasoning != "parsing_error_reasoning_not_found" and len(reasoning) > 50: 
            score += 1.0
        if "PLM Suggestion" in reasoning or "PLM" in reasoning: 
             score += 0.5
        scores.append(score)
    return scores

def plm_agreement_reward_func(completions: List[List[Dict[str, str]]], prompts: List[List[Dict[str, str]]], **kwargs) -> List[float]:
    scores = []
    user_prompt_content = prompts[0][-1]["content"] 
    plm_suggestion_match = re.search(r"PLM Suggestion: (negative|neutral|positive)", user_prompt_content)
    
    if not plm_suggestion_match: 
        return [0.0] * len(completions)
        
    plm_suggestion_str_from_prompt = plm_suggestion_match.group(1)

    for completion_list in completions:
        completion_text = completion_list[0]["content"]
        _, pred_sentiment_str = extract_reasoning_and_sentiment(completion_text)
        score = 0.0
        if pred_sentiment_str == plm_suggestion_str_from_prompt:
            score += 0.25
        scores.append(score)
    return scores

def label_validity_reward_func(completions: List[List[Dict[str, str]]], **kwargs) -> List[float]:
    scores = []
    for completion_list in completions:
        completion_text = completion_list[0]["content"]
        _, pred_sentiment_str = extract_reasoning_and_sentiment(completion_text)
        if pred_sentiment_str in VALID_SENTIMENT_STRINGS:
            scores.append(1.0)
        else:
            scores.append(-2.0) 
    return scores

# --- Evaluation (Identical) ---
def compute_metrics_absa(predictions: List[int], labels: List[int], model_run_idx: int, model_path: str, loss: float = -1.0) -> Dict:
    accuracy = accuracy_score(labels, predictions)
    precision_macro = precision_score(labels, predictions, average="macro", zero_division=0)
    recall_macro = recall_score(labels, predictions, average="macro", zero_division=0)
    f1_macro = f1_score(labels, predictions, average="macro", zero_division=0)
    
    precision_micro = precision_score(labels, predictions, average="micro", zero_division=0)
    recall_micro = recall_score(labels, predictions, average="micro", zero_division=0)
    f1_micro = f1_score(labels, predictions, average="micro", zero_division=0)

    precision_weighted = precision_score(labels, predictions, average="weighted", zero_division=0)
    recall_weighted = recall_score(labels, predictions, average="weighted", zero_division=0)
    f1_weighted = f1_score(labels, predictions, average="weighted", zero_division=0)
    
    qwk = cohen_kappa_score(labels, predictions, weights="quadratic")
    
    filtered_labels = [l for l in labels if l != -99]
    filtered_predictions = [p for i, p in enumerate(predictions) if labels[i] != -99]
    
    report_dict = {}
    if filtered_labels and filtered_predictions :
        unique_filtered_elements = sorted(list(set(filtered_labels) | set(filtered_predictions)))
        current_target_names = [SENTIMENT_MAP_INT_TO_STR.get(val, f"Unknown({val})") for val in unique_filtered_elements]
        try:
            if not current_target_names:
                 report_dict = classification_report(filtered_labels, filtered_predictions, output_dict=True, zero_division=0)
            else:
                 report_dict = classification_report(filtered_labels, filtered_predictions, target_names=current_target_names, output_dict=True, zero_division=0)
        except ValueError as e:
            print(f"Warning: classification_report failed: {e}. Trying without target_names.")
            try: 
                report_dict = classification_report(filtered_labels, filtered_predictions, output_dict=True, zero_division=0)
            except Exception as e2:
                print(f"Fallback classification_report also failed: {e2}. Using empty report.")
                report_dict = {}

    metrics = {
        f"model_{model_run_idx}": {
            "model_run_index": model_run_idx,
            "model_path": model_path,
            "test_loss": loss, 
            "loss": loss, 
            "accuracy": accuracy,
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
            "f1_macro": f1_macro,
            "precision_micro": precision_micro,
            "recall_micro": recall_micro,
            "f1_micro": f1_micro,
            "precision_weighted": precision_weighted,
            "recall_weighted": recall_weighted,
            "f1_weighted": f1_weighted,
            "qwk": qwk,
            "per_class_report": report_dict 
        }
    }
    if "accuracy" in report_dict: metrics[f"model_{model_run_idx}"]["accuracy_from_report"] = report_dict["accuracy"]
    if "macro avg" in report_dict: metrics[f"model_{model_run_idx}"]["macro_avg_from_report"] = report_dict["macro avg"]
    if "weighted avg" in report_dict: metrics[f"model_{model_run_idx}"]["weighted_avg_from_report"] = report_dict["weighted avg"]
    
    return metrics

def main(args):
    set_seed(args.seed)

    base_model_name_for_path = args.model_name_or_path.split("/")[-1]
    experiment_output_dir = os.path.join(args.output_dir_base, base_model_name_for_path, args.dataset_language, args.experiment_name)
    lora_output_dir_for_current_experiment = os.path.join(experiment_output_dir, "lora_adapters")
    eval_output_dir = os.path.join(experiment_output_dir, "evaluation")
    logs_output_dir = os.path.join(experiment_output_dir, "training_logs") 

    os.makedirs(eval_output_dir, exist_ok=True)
    
    model_max_seq_length = args.max_seq_length 
    eval_model = None
    tokenizer = None 
    raw_test_dataset_full = None # Will hold the full raw test set

    plm_key_map = { # Define plm_key early as it's needed for loading raw_test_dataset_full
        "slovenian": "global-context-modelling/simplified-dart-xlmr",
        "serbian": "global-context-modelling/simplified-dart-xlmr" 
    }
    plm_key = plm_key_map[args.dataset_language]

    if not args.test_only:
        os.makedirs(lora_output_dir_for_current_experiment, exist_ok=True)
        os.makedirs(logs_output_dir, exist_ok=True)
        print("Starting Training Mode...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model_name_or_path,
            max_seq_length=model_max_seq_length,
            dtype=None,  
            load_in_4bit=True,
            token=args.hf_token,
        )
        
        model = FastLanguageModel.get_peft_model(
            model,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",  
            use_gradient_checkpointing="unsloth", 
            random_state=args.seed,
            target_modules=args.lora_target_modules if args.lora_target_modules != ["all-linear"] else None, 
            finetune_vision_layers=False, 
            finetune_language_layers=True,
            finetune_attention_modules=True, 
            finetune_mlp_modules=True, 
            max_seq_length=model_max_seq_length,
        )
        
        train_dataset, val_dataset, raw_test_dataset_full = load_absa_data(
            args.dataset_dir, args.dataset_language, args.train_split_index, plm_key,
            max_train_samples=args.max_train_samples,
            max_val_samples=args.max_val_samples
        )

        grpo_max_prompt_length = model_max_seq_length - args.max_completion_length
        if grpo_max_prompt_length <= 0:
            raise ValueError("max_seq_length must be greater than max_completion_length.")

        config_per_device_train_batch_size = args.per_device_train_batch_size * args.num_generations
        print(f"GRPOConfig: Setting per_device_train_batch_size to {config_per_device_train_batch_size} "
              f"({args.per_device_train_batch_size} unique prompts * {args.num_generations} generations)")

        grpo_config = GRPOConfig(
            output_dir=logs_output_dir,
            learning_rate=args.learning_rate,
            per_device_train_batch_size=config_per_device_train_batch_size, 
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            num_train_epochs=args.num_train_epochs,
            max_steps=args.max_steps if args.max_steps > 0 else -1, 
            remove_unused_columns=False, 
            logging_steps=args.logging_steps,
            save_steps=args.save_steps,
            seed=args.seed,
            report_to="tensorboard" if args.report_to_tensorboard else "none",
            num_generations=args.num_generations,
            max_prompt_length=grpo_max_prompt_length,
            max_completion_length=args.max_completion_length,
            beta=args.grpo_beta, 
            adam_beta1=0.9,
            adam_beta2=0.99,
            weight_decay=0.1,
            warmup_ratio=args.warmup_ratio,
            lr_scheduler_type=args.lr_scheduler_type,
            optim="adamw_torch_fused", 
            max_grad_norm=0.1, 
            bf16=torch.cuda.is_bf16_supported(),
            fp16=not torch.cuda.is_bf16_supported(),
        )

        reward_functions = [
            correctness_reward_func,
            format_reward_func,
            reasoning_quality_reward_func,
            plm_agreement_reward_func,
            label_validity_reward_func,
        ]

        trainer = GRPOTrainer(
            model=model,
            args=grpo_config,
            train_dataset=train_dataset,
            eval_dataset=val_dataset, 
            reward_funcs=reward_functions,
            processing_class=tokenizer, 
        )

        print("Starting GRPO training...")
        trainer.train()
        print("Training finished.")

        print(f"Saving LoRA adapters to {lora_output_dir_for_current_experiment}")
        trainer.save_model(lora_output_dir_for_current_experiment)
        if tokenizer: tokenizer.save_pretrained(lora_output_dir_for_current_experiment) 

        if args.merge_and_save_final_model:
            print("Merging and saving final model...")
            merged_model_dir = os.path.join(experiment_output_dir, "merged_model")
            os.makedirs(merged_model_dir, exist_ok=True)
            try:
                if hasattr(trainer.model, 'save_pretrained_merged'):
                     trainer.model.save_pretrained_merged(merged_model_dir, tokenizer, save_method="merged_16bit")
                     print(f"Merged model saved to {merged_model_dir} using Unsloth's save_pretrained_merged.")
                else: 
                    print("Unsloth's save_pretrained_merged not found on model. Attempting standard PEFT merge.")
                    merged_model_for_saving = trainer.model.merge_and_unload()
                    merged_model_for_saving.save_pretrained(merged_model_dir)
                    if tokenizer: tokenizer.save_pretrained(merged_model_dir)
                    print(f"Merged model saved to {merged_model_dir} using standard PEFT merge.")
            except Exception as e:
                print(f"Could not save merged model: {e}. Adapters are saved at {lora_output_dir_for_current_experiment}.")
        
        eval_model = trainer.model 
        # tokenizer is defined from training
    
    else: # --test_only mode
        print("Starting Test Only Mode...")
        
        effective_lora_model_path = args.lora_model_path_for_test
        if not effective_lora_model_path:
            print(f"--lora_model_path_for_test not specified, attempting to infer from current experiment: {lora_output_dir_for_current_experiment}")
            if os.path.exists(lora_output_dir_for_current_experiment) and \
               (os.path.exists(os.path.join(lora_output_dir_for_current_experiment, "adapter_config.json"))): # Check for adapter_config
                effective_lora_model_path = lora_output_dir_for_current_experiment
                print(f"Inferred LoRA model path: {effective_lora_model_path}")
            else:
                raise ValueError(
                    "--lora_model_path_for_test must be specified in --test_only mode, "
                    "or a trained model must exist at the inferred location for the current experiment: "
                    f"{lora_output_dir_for_current_experiment}"
                )
        
        base_model_path = args.base_model_for_test_only if args.base_model_for_test_only else args.model_name_or_path
        print(f"Loading base model for test_only: {base_model_path}")
        print(f"Loading LoRA adapters from: {effective_lora_model_path}")

        base_eval_model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model_path,
            max_seq_length=model_max_seq_length,
            dtype=None,
            load_in_4bit=True,
            token=args.hf_token,
        )
        
        try:
            print("Attempting to load adapters using Unsloth's method (passing loaded base model object)...")
            eval_model, tokenizer = FastLanguageModel.from_pretrained(
                model_name = base_eval_model, 
                model_name_lora = effective_lora_model_path,
                tokenizer_name_lora = effective_lora_model_path, 
            )
            print("Successfully loaded LoRA adapters using Unsloth's from_pretrained with model_name_lora.")
        except Exception as e_unsloth:
            print(f"Warning: Unsloth's from_pretrained with model_name_lora failed ({e_unsloth}). "
                  "Attempting standard PeftModel loading as a fallback.")
            try:
                eval_model = PeftModel.from_pretrained(base_eval_model, effective_lora_model_path)
                print("Successfully loaded adapters with PeftModel (unmerged by default for test_only).")
                if args.merge_and_save_final_model: 
                     print("Attempting to merge PeftModel for test_only as --merge_and_save_final_model is set...")
                     if hasattr(eval_model, 'merge_and_unload'):
                         eval_model = eval_model.merge_and_unload()
                         print("PeftModel merged and unloaded successfully.")
                     else:
                         print("PeftModel does not have merge_and_unload attribute. Using unmerged.")
                try:
                    tokenizer_test = AutoTokenizer.from_pretrained(effective_lora_model_path, use_fast=True)
                    if tokenizer_test: tokenizer = tokenizer_test 
                    print(f"Tokenizer (re)loaded from adapter path: {effective_lora_model_path}")
                except:
                    print(f"Could not load tokenizer from adapter path {effective_lora_model_path}. Using tokenizer from base model.")
            except Exception as e_peft:
                raise RuntimeError(f"Failed to load LoRA adapters using PeftModel fallback. Last PeftModel error: {e_peft}")
    
    # --- Evaluation (common for both modes) ---
    if eval_model is None:
        raise RuntimeError("Evaluation model was not loaded or trained.")
    if tokenizer is None: 
        tokenizer_path_fallback = effective_lora_model_path if args.test_only and effective_lora_model_path else \
                                  (lora_output_dir_for_current_experiment if not args.test_only and os.path.exists(lora_output_dir_for_current_experiment) else \
                                   args.model_name_or_path)
        print(f"CRITICAL: Tokenizer not loaded. Attempting fallback from: {tokenizer_path_fallback}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path_fallback, use_fast=True)
        
    print("Preparing test set for evaluation...")
    # Load the full raw test dataset if not already loaded (e.g. in test_only mode)
    if raw_test_dataset_full is None:
        _, _, raw_test_dataset_full = load_absa_data(
            args.dataset_dir, args.dataset_language, args.train_split_index, plm_key,
            max_train_samples=0, max_val_samples=0 
        )

    # Apply limiting and stratification to the full raw test dataset
    eval_raw_dataset_to_process = raw_test_dataset_full
    if args.limit_test_samples is not None:
        limit_str = str(args.limit_test_samples).strip().lower()
        num_total_test_samples = len(eval_raw_dataset_to_process)
        target_num_samples = 0

        is_percentage = False
        if limit_str.endswith('%') or limit_str.endswith('p'):
            is_percentage = True
            try:
                percentage_val = float(limit_str.rstrip('%p')) / 100.0
                if 0.0 < percentage_val <= 1.0:
                    target_num_samples = int(num_total_test_samples * percentage_val)
                else:
                    raise ValueError("Percentage value out of range (0-100).")
            except ValueError:
                raise ValueError(f"Invalid percentage format for --limit_test_samples: {args.limit_test_samples}")
        elif '.' in limit_str: # float, assume direct proportion if <=1.0
            try:
                val = float(limit_str)
                if 0.0 < val <= 1.0: # Direct proportion
                    target_num_samples = int(num_total_test_samples * val)
                elif val > 1.0: # Absolute number as float
                     target_num_samples = int(val)
                else:
                    raise ValueError("Float limit must be > 0.")
            except ValueError:
                raise ValueError(f"Invalid float value for --limit_test_samples: {args.limit_test_samples}")
        else: # Integer
            try:
                target_num_samples = int(limit_str)
            except ValueError:
                raise ValueError(f"Invalid integer value for --limit_test_samples: {args.limit_test_samples}")

        if target_num_samples <= 0:
            print(f"Warning: Calculated target_num_samples is {target_num_samples}. Using full test set.")
        elif target_num_samples < num_total_test_samples:
            print(f"Limiting test set: attempting to stratify sample {target_num_samples} from {num_total_test_samples} items.")
            if "sentiment" not in eval_raw_dataset_to_process.column_names:
                raise ValueError("Stratification column 'sentiment' not found in raw test dataset.")
            
            # Ensure at least 1 sample per class if possible, or a minimum number for train_test_split
            num_classes = len(set(eval_raw_dataset_to_process['sentiment']))
            min_samples_for_stratify = num_classes * 2 # Heuristic
            
            if target_num_samples < min_samples_for_stratify and target_num_samples < num_total_test_samples :
                 print(f"Warning: target_num_samples ({target_num_samples}) is small for stratification with {num_classes} classes. Stratification might be imperfect or fail. Trying anyway.")
            
            try:
                # train_test_split requires train_size or test_size. We want 'target_num_samples'.
                # If target_num_samples is very small, ensure it's at least num_classes if possible
                actual_train_size = max(target_num_samples, num_classes if num_total_test_samples >= num_classes else 1)
                if actual_train_size >= num_total_test_samples: # If actual_train_size forces using all data
                    print("Calculated train_size for stratification is >= total samples. Using all test data.")
                    eval_raw_dataset_to_process = eval_raw_dataset_to_process
                else:
                    stratified_split = eval_raw_dataset_to_process.train_test_split(
                        train_size=actual_train_size, 
                        shuffle=True, 
                        seed=args.seed,
                        stratify_by_column="sentiment"
                    )
                    eval_raw_dataset_to_process = stratified_split['train']
                    print(f"Stratified sampling complete. Selected {len(eval_raw_dataset_to_process)} test samples.")
            except Exception as e:
                print(f"Stratified sampling failed: {e}. Falling back to simple slicing of {target_num_samples} samples.")
                eval_raw_dataset_to_process = eval_raw_dataset_to_process.select(range(min(target_num_samples, num_total_test_samples)))
        else:
            print("Target sample limit is >= total or invalid. Using all test samples.")
            # eval_raw_dataset_to_process remains the full raw_test_dataset_full

    def _prepare_test_for_generation(example):
        article = example.get("article", "")
        aspect = example.get("aspect", "")
        plm_sentiment_int = example.get(plm_key, 0)
        plm_sentiment_str = map_sentiment_to_string(plm_sentiment_int)
        
        user_prompt_content = (
            f"Article: \"{article}\"\n"
            f"Aspect: \"{aspect}\"\n"
            f"PLM Suggestion: {plm_sentiment_str}"
        )
        example["original_article"] = article
        example["original_aspect"] = aspect
        example["original_plm_suggestion_str"] = plm_sentiment_str
        example["true_sentiment_int"] = example.get("sentiment", 0) 

        chat_template_input = [
            {"role": "system", "content": GRPO_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt_content},
        ]
        example["formatted_prompt_text"] = tokenizer.apply_chat_template(
            chat_template_input, tokenize=False, add_generation_prompt=True
        )
        return example

    test_dataset_for_eval = eval_raw_dataset_to_process.map(_prepare_test_for_generation)
    print(f"Final size of test_dataset_for_eval: {len(test_dataset_for_eval)}")
    
    all_predictions_data = []
    predicted_sentiments_int = []
    true_sentiments_int = []

    eval_model.eval() 
    print(f"Starting generation for {len(test_dataset_for_eval)} test items...")

    generation_kwargs_eval = {
        "max_new_tokens": args.max_completion_length,
        "temperature": 1.0, 
        "top_p": 0.95, 
        "top_k": 64,
    }
    if args.do_sample_eval:
        generation_kwargs_eval["do_sample"] = True
    else:
        generation_kwargs_eval["do_sample"] = False
    
    if tokenizer.eos_token_id is not None:
        generation_kwargs_eval["eos_token_id"] = tokenizer.eos_token_id
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id 
    if tokenizer.pad_token_id is not None:
        generation_kwargs_eval["pad_token_id"] = tokenizer.pad_token_id

    for item_idx, item in enumerate(test_dataset_for_eval):
        print(f"Processing test item {item_idx+1}/{len(test_dataset_for_eval)} - UUID: {item.get('uuid', 'N/A')}")
        if not item["formatted_prompt_text"]: # Basic check
            print(f"  Skipping item {item_idx+1} due to empty formatted_prompt_text.")
            continue
        try:
            inputs = tokenizer(item["formatted_prompt_text"], return_tensors="pt").to(eval_model.device)
            print(f"  Input token length: {inputs.input_ids.shape[1]}")
        except Exception as e:
            print(f"  Error tokenizing item {item_idx+1}: {e}. Skipping.")
            continue # Skip this item
        
        generation_start_time = time.time()
        with torch.no_grad():
            try:
                outputs = eval_model.generate(**inputs, **generation_kwargs_eval)
            except Exception as e:
                print(f"  Error during model.generate for item {item_idx+1}: {e}. Skipping.")
                # Add a placeholder or handle differently if needed for metrics
                pred_sent_int = -99 # Mark as error
                raw_model_output = f"GENERATION_ERROR: {e}"
                parsed_reasoning = "GENERATION_ERROR"
                parsed_sentiment_str = "unknown"
                # Continue to append this error record
                prediction_data_item = {
                    "uuid": item.get("uuid", f"no_uuid_{item_idx}"), 
                    "article": item["original_article"], "aspect": item["original_aspect"],
                    "plm_suggestion": item["original_plm_suggestion_str"],
                    "true_sentiment": map_sentiment_to_string(item["true_sentiment_int"]),
                    "raw_model_output": raw_model_output, "parsed_reasoning": parsed_reasoning,
                    "parsed_sentiment_str": parsed_sentiment_str, "parsed_sentiment_int": pred_sent_int,
                }
                all_predictions_data.append(prediction_data_item)
                predicted_sentiments_int.append(pred_sent_int)
                true_sentiments_int.append(item["true_sentiment_int"])
                continue # Move to next item

        generation_end_time = time.time()
        print(f"  Generated in {generation_end_time - generation_start_time:.2f} seconds.")
        
        raw_model_output = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        parsed_reasoning, parsed_sentiment_str = extract_reasoning_and_sentiment(raw_model_output)
        pred_sent_int = map_string_to_sentiment(parsed_sentiment_str)
        
        prediction_data_item = {
            "uuid": item.get("uuid", f"no_uuid_{item_idx}"), 
            "article": item["original_article"],
            "aspect": item["original_aspect"],
            "plm_suggestion": item["original_plm_suggestion_str"],
            "true_sentiment": map_sentiment_to_string(item["true_sentiment_int"]),
            "raw_model_output": raw_model_output,
            "parsed_reasoning": parsed_reasoning,
            "parsed_sentiment_str": parsed_sentiment_str,
            "parsed_sentiment_int": pred_sent_int,
        }
        all_predictions_data.append(prediction_data_item)
        predicted_sentiments_int.append(pred_sent_int)
        true_sentiments_int.append(item["true_sentiment_int"])

    metrics_model_path = effective_lora_model_path if args.test_only else lora_output_dir_for_current_experiment

    predictions_file = os.path.join(eval_output_dir, f"{args.experiment_name}_full_test_predictions_split{args.train_split_index}.json")
    with open(predictions_file, 'w', encoding='utf-8') as f:
        json.dump(all_predictions_data, f, indent=2, ensure_ascii=False)
    print(f"Saved full test predictions to {predictions_file}")

    final_metrics = compute_metrics_absa(
        predicted_sentiments_int, 
        true_sentiments_int,
        model_run_idx=args.train_split_index, 
        model_path=metrics_model_path
    )
    
    metrics_file = os.path.join(eval_output_dir, f"{args.experiment_name}_full_test_metrics_split{args.train_split_index}.json")
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(final_metrics, f, indent=2, ensure_ascii=False)
    print(f"Saved full test metrics to {metrics_file}")
    if f'model_{args.train_split_index}' in final_metrics: # Check if metrics were actually computed
        print(f"Metrics for split {args.train_split_index}: F1 Macro={final_metrics[f'model_{args.train_split_index}']['f1_macro']:.4f}, Accuracy={final_metrics[f'model_{args.train_split_index}']['accuracy']:.4f}, QWK={final_metrics[f'model_{args.train_split_index}']['qwk']:.4f}")
    else:
        print(f"Metrics could not be computed for split {args.train_split_index}. Check logs for errors or empty predictions.")


    print("Script finished successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Gemma-3 model using GRPO with Unsloth for ABSA.")
    
    parser.add_argument("--model_name_or_path", type=str, default="unsloth/gemma-3-12b-it-unsloth-bnb-4bit")
    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face token if using a gated model.")
    
    parser.add_argument("--dataset_dir", type=str, default="../data/final/balanced-predicted/simplified-dart-xlmr/", help="Base directory for dataset JSON files.")
    parser.add_argument("--dataset_language", type=str, choices=["slovenian", "serbian"], default="slovenian")
    parser.add_argument("--train_split_index", type=int, default=0, help="Index of the train/val split to use.")
    parser.add_argument("--max_train_samples", type=int, default=None, help="Max training samples for quick testing.")
    parser.add_argument("--max_val_samples", type=int, default=None, help="Max validation samples for quick testing.")
    parser.add_argument("--limit_test_samples", type=str, default=None, help="Limit test samples: int for count, float (0-1) or 'X%%'/'Xp' for percentage (stratified). Eg. 100 or 0.1 or '10%%'.")
    
    parser.add_argument("--output_dir_base", type=str, default="../models/grpo/", help="Base directory to save models/results.")
    parser.add_argument("--experiment_name", type=str, default="gemma3_12b_grpo_run1", help="Unique name for this experiment.")
    
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64) 
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", type=str, nargs="+", default=["all-linear"], help="LoRA target modules.")

    parser.add_argument("--max_seq_length", type=int, default=1024, help="Overall max sequence length.") 
    parser.add_argument("--max_completion_length", type=int, default=256, help="Max length for generated completion.") 
    parser.add_argument("--num_generations", type=int, default=4, help="Completions per prompt in GRPO.")
    parser.add_argument("--grpo_beta", type=float, default=0.1, help="GRPO KL divergence beta.")

    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="Unique prompts per device step.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2) 
    parser.add_argument("--num_train_epochs", type=int, default=1) 
    parser.add_argument("--max_steps", type=int, default=-1, help="Max training steps.")
    parser.add_argument("--learning_rate", type=float, default=5e-5) 
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    
    parser.add_argument("--logging_steps", type=int, default=1) 
    parser.add_argument("--save_steps", type=int, default=20) 
    parser.add_argument("--report_to_tensorboard", action="store_true", help="Report to TensorBoard.")
    parser.add_argument("--merge_and_save_final_model", action="store_true", help="Merge LoRA and save full model (also affects test_only merge).")
    parser.add_argument("--do_sample_eval", action="store_true", help="Use sampling for evaluation generation.")

    parser.add_argument("--test_only", action="store_true", help="Skip training, only run evaluation.")
    parser.add_argument("--lora_model_path_for_test", type=str, default=None, help="Path to LoRA adapters for --test_only mode.")
    parser.add_argument("--base_model_for_test_only", type=str, default=None, help="Base model for test_only if different.")
    
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    main(args)