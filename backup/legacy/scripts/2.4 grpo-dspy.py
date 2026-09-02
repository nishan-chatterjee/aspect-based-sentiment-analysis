import argparse
import json
import os
import random
import re
from typing import List, Dict, Tuple, Any
import time 

import numpy as np
import torch
import dspy
from dspy.teleprompt.grpo import GRPO
from dspy.clients.lm_local_arbor import ArborProvider

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    cohen_kappa_score,
    classification_report,
)

# --- Global Constants and Prompts (Identical) ---
# NOTE: The core prompt text is now embedded within the DSPy Signatures.
# We keep these constants for parsing and mapping.
REASONING_START_TAG = "<reasoning>"
REASONING_END_TAG = "</reasoning>"
SENTIMENT_START_TAG = "<sentiment>"
SENTIMENT_END_TAG = "</sentiment>"

SENTIMENT_MAP_INT_TO_STR = {-1: "negative", 0: "neutral", 1: "positive"}
SENTIMENT_MAP_STR_TO_INT = {"negative": -1, "neutral": 0, "positive": 1, "unknown": -99} 
VALID_SENTIMENT_STRINGS = ["negative", "neutral", "positive"]

# --- DSPy Components (The New Core) ---

# Setting 1: Signature that USES the PLM suggestion
class AbsaSentimentWithPLM(dspy.Signature):
    """Given an article, an aspect, and a PLM sentiment suggestion, you are tasked with providing a detailed step-by-step reasoning that considers all inputs, and concluding with the final sentiment: 'negative', 'neutral', or 'positive'. The reasoning should explain how the PLM suggestion was considered.

High Stakes Scenario: You are part of a critical team analyzing media sentiment for a major regulatory body in Slovenia. Your analysis will directly inform policy decisions that could impact public health, environmental protection, and financial stability. Therefore, it is essential to provide accurate and well-reasoned sentiment analysis."""
    
    article = dspy.InputField(desc="The full text of the article.")
    aspect = dspy.InputField(desc="The company or product name being analyzed.")
    plm_suggestion = dspy.InputField(desc="A sentiment suggestion from a baseline model (e.g., 'positive', 'negative').")
    
    reasoning = dspy.OutputField(desc="A detailed step-by-step reasoning that considers all inputs, especially the PLM suggestion.", prefix=f"{REASONING_START_TAG}\n")
    sentiment = dspy.OutputField(desc=f"The final sentiment, which must be one of: {VALID_SENTIMENT_STRINGS}", prefix=f"\n{REASONING_END_TAG}\n{SENTIMENT_START_TAG}\n", suffix=f"\n{SENTIMENT_END_TAG}")

# Setting 2: Signature that DOES NOT use the PLM suggestion
class AbsaSentimentWithoutPLM(dspy.Signature):
    """Given an article and an aspect, you are tasked with providing a detailed step-by-step reasoning and concluding with the final sentiment: 'negative', 'neutral', or 'positive'.

High Stakes Scenario: You are part of a critical team analyzing media sentiment for a major regulatory body in Slovenia. Your analysis will directly inform policy decisions that could impact public health, environmental protection, and financial stability. Therefore, it is essential to provide accurate and well-reasoned sentiment analysis."""
    
    article = dspy.InputField(desc="The full text of the article.")
    aspect = dspy.InputField(desc="The company or product name being analyzed.")
    
    reasoning = dspy.OutputField(desc="A detailed step-by-step reasoning that considers the article and aspect.", prefix=f"{REASONING_START_TAG}\n")
    sentiment = dspy.OutputField(desc=f"The final sentiment, which must be one of: {VALID_SENTIMENT_STRINGS}", prefix=f"\n{REASONING_END_TAG}\n{SENTIMENT_START_TAG}\n", suffix=f"\n{SENTIMENT_END_TAG}")

# The main DSPy program, configurable for both settings
class DSPy_ABSA_Program(dspy.Module):
    def __init__(self, use_plm_suggestion: bool = True):
        super().__init__()
        self.use_plm_suggestion = use_plm_suggestion

        if self.use_plm_suggestion:
            self.generate_sentiment = dspy.ChainOfThought(AbsaSentimentWithPLM)
        else:
            self.generate_sentiment = dspy.ChainOfThought(AbsaSentimentWithoutPLM)

    def forward(self, article, aspect, plm_suggestion=None):
        if self.use_plm_suggestion:
            if plm_suggestion is None:
                raise ValueError("plm_suggestion cannot be None when use_plm_suggestion is True")
            prediction = self.generate_sentiment(article=article, aspect=aspect, plm_suggestion=plm_suggestion)
        else:
            prediction = self.generate_sentiment(article=article, aspect=aspect)
        
        # Pass through all inputs to the prediction object for the metric function
        return dspy.Prediction(
            article=article,
            aspect=aspect,
            plm_suggestion=plm_suggestion,
            reasoning=prediction.reasoning,
            sentiment=prediction.sentiment
        )


# --- Helper Functions ---
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def map_sentiment_to_string(sentiment_int: int) -> str:
    return SENTIMENT_MAP_INT_TO_STR.get(sentiment_int, "unknown")

def map_string_to_sentiment(sentiment_str: str) -> int:
    return SENTIMENT_MAP_STR_TO_INT.get(str(sentiment_str).lower().strip(), -99)

def load_absa_data(data_dir: str, language: str, train_split_index: int, plm_key: str, 
                       max_train_samples: int = None, max_val_samples: int = None, max_test_samples: int = None) -> Tuple[List[dspy.Example], List[dspy.Example], List[dspy.Example]]:
    """Loads data and converts it into lists of dspy.Example objects."""
    train_val_file = os.path.join(data_dir, f"{language}_train_val_balanced_{train_split_index}.json")
    test_file = os.path.join(data_dir, f"{language}_test_balanced.json")

    with open(train_val_file, 'r', encoding='utf-8') as f:
        train_val_data = json.load(f)
    with open(test_file, 'r', encoding='utf-8') as f:
        test_data_json = json.load(f)

    def _create_dspy_example(sample_dict):
        true_sentiment_int = sample_dict.get("sentiment", 0)
        plm_sentiment_int = sample_dict.get(plm_key, 0)
        return dspy.Example(
            article=sample_dict.get("article", ""),
            aspect=sample_dict.get("aspect", ""),
            plm_suggestion=map_sentiment_to_string(plm_sentiment_int),
            sentiment=map_sentiment_to_string(true_sentiment_int), # Ground truth
            # Keep original ints for easier evaluation later
            true_sentiment_int=true_sentiment_int,
            plm_suggestion_int=plm_sentiment_int,
            uuid=sample_dict.get("uuid", None)
        )

    train_examples = [_create_dspy_example(s) for s in train_val_data['train']]
    val_examples = [_create_dspy_example(s) for s in train_val_data['val']]
    test_examples = [_create_dspy_example(s) for s in test_data_json['test']]

    if max_train_samples:
        train_examples = train_examples[:max_train_samples]
    if max_val_samples:
        val_examples = val_examples[:max_val_samples]
    if max_test_samples:
        # Simple slicing for test set limiting, can be improved with stratification if needed
        test_examples = test_examples[:max_test_samples]

    print(f"Loaded {len(train_examples)} training, {len(val_examples)} validation, and {len(test_examples)} test examples as dspy.Examples.")
    return train_examples, val_examples, test_examples


# --- Reward Metric for dspy.GRPO ---
def combined_reward_metric(gold: dspy.Example, pred: dspy.Prediction, trace=None) -> float:
    """
    Calculates a comprehensive reward score for a generated prediction,
    combining correctness, format, reasoning quality, and conditional PLM agreement.
    """
    # --- 1. Correctness Reward ---
    true_sentiment_str = gold.sentiment.lower().strip()
    true_sentiment_int = SENTIMENT_MAP_STR_TO_INT.get(true_sentiment_str, -99)
    
    pred_sentiment_str = str(pred.sentiment).lower().strip()
    pred_sentiment_int = map_string_to_sentiment(pred_sentiment_str)

    if pred_sentiment_str not in VALID_SENTIMENT_STRINGS:
        correctness_score = -5.0  # Harsh penalty for invalid output
    elif pred_sentiment_int == true_sentiment_int:
        correctness_score = 5.0   # High reward for correct answer
    elif abs(pred_sentiment_int - true_sentiment_int) == 1:
        correctness_score = 0.5   # Small reward for adjacent sentiment
    else: # abs difference is 2
        correctness_score = -3.0  # Penalty for major error

    # --- 2. Format Reward ---
    format_score = 0.0
    # DSPy handles parsing. If fields are None, it failed.
    if pred.reasoning is not None and pred.sentiment is not None:
        format_score = 3.0
    else:
        format_score = -2.0 # Penalize parsing failures

    # --- 3. Reasoning Quality Reward ---
    reasoning_score = 0.0
    if pred.reasoning and len(pred.reasoning) > 50:
        reasoning_score += 1.0
    # Only reward PLM mention if it was part of the input
    if hasattr(pred, 'plm_suggestion') and pred.plm_suggestion and pred.reasoning and ("PLM Suggestion" in pred.reasoning or "PLM" in pred.reasoning):
        reasoning_score += 0.5
        
    # --- 4. PLM Agreement Reward (Conditional) ---
    agreement_score = 0.0
    if hasattr(pred, 'plm_suggestion') and pred.plm_suggestion:
        plm_suggestion_from_input = pred.plm_suggestion.lower().strip()
        if pred_sentiment_str == plm_suggestion_from_input:
            agreement_score += 0.25

    # Combine scores
    total_reward = correctness_score + format_score + reasoning_score + agreement_score
    return total_reward


# --- Evaluation (Identical to original script) ---
def compute_metrics_absa(predictions: List[int], labels: List[int], model_run_idx: int, model_path: str, loss: float = -1.0) -> Dict:
    # This function is identical to the one in your grpo.py file
    accuracy = accuracy_score(labels, predictions)
    precision_macro = precision_score(labels, predictions, average="macro", zero_division=0)
    recall_macro = recall_score(labels, predictions, average="macro", zero_division=0)
    f1_macro = f1_score(labels, predictions, average="macro", zero_division=0)
    qwk = cohen_kappa_score(labels, predictions, weights="quadratic")
    
    filtered_labels = [l for l in labels if l != -99]
    filtered_predictions = [p for i, p in enumerate(predictions) if labels[i] != -99]
    
    report_dict = {}
    if filtered_labels and filtered_predictions :
        unique_filtered_elements = sorted(list(set(filtered_labels) | set(filtered_predictions)))
        current_target_names = [SENTIMENT_MAP_INT_TO_STR.get(val, f"Unknown({val})") for val in unique_filtered_elements]
        try:
            report_dict = classification_report(filtered_labels, filtered_predictions, target_names=current_target_names, output_dict=True, zero_division=0)
        except Exception:
            report_dict = classification_report(filtered_labels, filtered_predictions, output_dict=True, zero_division=0)

    metrics = {
        f"model_{model_run_idx}": {
            "model_run_index": model_run_idx,
            "model_path": model_path,
            "test_loss": loss, 
            "accuracy": accuracy,
            "f1_macro": f1_macro,
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
            "qwk": qwk,
            "per_class_report": report_dict 
        }
    }
    return metrics


def main(args):
    print("--- Starting DSPy GRPO ABSA Script ---")
    set_seed(args.seed)

    # --- Setup Output Directories ---
    base_model_name_for_path = args.model_name_or_path.split("/")[-1]
    experiment_name = f"{args.experiment_name}_{'with_plm' if args.use_plm_suggestion else 'no_plm'}"
    experiment_output_dir = os.path.join(args.output_dir_base, base_model_name_for_path, args.dataset_language, experiment_name)
    
    eval_output_dir = os.path.join(experiment_output_dir, "evaluation")
    # NOTE: dspy.GRPO manages its own output directory for checkpoints via train_kwargs
    grpo_output_dir = os.path.join(experiment_output_dir, "grpo_training_output")

    os.makedirs(eval_output_dir, exist_ok=True)
    os.makedirs(grpo_output_dir, exist_ok=True)
    
    # --- Connect to Arbor RL Server ---
    print(f"Connecting to Arbor RL server for model: {args.model_name_or_path}")
    print("Please ensure the Arbor server is running in a separate terminal.")
    
    arbor_lm = dspy.LM(
        model=f"openai/arbor:{args.model_name_or_path}",
        provider=ArborProvider(),
        api_base=f"http://localhost:{args.arbor_port}/v1/",
        api_key="arbor",
        temperature=args.temperature,
        # Other generation parameters can be added here
    )
    dspy.configure(lm=arbor_lm)

    # --- Load Data ---
    plm_key_map = {
        "slovenian": "global-context-modelling/simplified-dart-xlmr",
        "serbian": "global-context-modelling/simplified-dart-xlmr" 
    }
    plm_key = plm_key_map[args.dataset_language]

    train_examples, val_examples, test_examples = load_absa_data(
        args.dataset_dir, args.dataset_language, args.train_split_index, plm_key,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples
    )
    
    # Define which fields are inputs for DSPy based on the experiment setting
    if args.use_plm_suggestion:
        print("Setting up experiment WITH PLM suggestion.")
        input_keys = ["article", "aspect", "plm_suggestion"]
    else:
        print("Setting up experiment WITHOUT PLM suggestion.")
        input_keys = ["article", "aspect"]

    train_dataset = [ex.with_inputs(*input_keys) for ex in train_examples]
    val_dataset = [ex.with_inputs(*input_keys) for ex in val_examples]

    # --- Setup and Run GRPO Compilation (Training) ---
    if not args.test_only:
        print("Starting GRPO training...")
        
        program = DSPy_ABSA_Program(use_plm_suggestion=args.use_plm_suggestion)
        program.set_lm(arbor_lm)

        # Map our script's args to the train_kwargs expected by GRPO's trainer
        train_kwargs = {
            "output_dir": grpo_output_dir,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "learning_rate": args.learning_rate,
            "lr_scheduler_type": args.lr_scheduler_type,
            "warmup_ratio": args.warmup_ratio,
            "bf16": torch.cuda.is_bf16_supported(),
            "fp16": not torch.cuda.is_bf16_supported(),
            "logging_steps": args.logging_steps,
            "report_to": "tensorboard" if args.report_to_tensorboard else "none",
            "beta": args.grpo_beta,
            "lora": True, # Enable LoRA training
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
        }

        compiler = GRPO(
            metric=combined_reward_metric,
            num_rollouts_per_grpo_step=args.num_generations,
            num_train_steps=args.max_steps,
            num_threads=args.num_threads,
            train_kwargs=train_kwargs,
            exclude_demos=True, 
        )

        optimized_program = compiler.compile(
            student=program,
            trainset=train_dataset,
            valset=val_dataset,
        )
        print("GRPO training finished.")

    else:
        print("Skipping training (--test_only). Loading a pre-trained program is not yet supported in this script.")
        print("Please run without --test_only to train a model first.")
        # NOTE: Loading a GRPO-trained model would involve pointing the Arbor LM
        # to the final checkpoint in the grpo_output_dir. This is an advanced use case.
        # For now, we assume we have the `optimized_program` from a fresh run.
        return

    # --- Evaluation on Holdout Test Set ---
    print(f"\n--- Starting Evaluation on {len(test_examples)} Test Examples ---")
    
    all_predictions_data = []
    predicted_sentiments_int = []
    true_sentiments_int = []

    for idx, item in enumerate(test_examples):
        print(f"Processing test item {idx+1}/{len(test_examples)} - UUID: {item.uuid or 'N/A'}")
        
        try:
            # Call the optimized program with the correct set of inputs
            if args.use_plm_suggestion:
                prediction = optimized_program(article=item.article, aspect=item.aspect, plm_suggestion=item.plm_suggestion)
            else:
                prediction = optimized_program(article=item.article, aspect=item.aspect)

            parsed_sentiment_str = str(prediction.sentiment).strip()
            pred_sent_int = map_string_to_sentiment(parsed_sentiment_str)
            
            prediction_data_item = {
                "uuid": item.uuid, 
                "article": item.article, "aspect": item.aspect,
                "plm_suggestion": item.plm_suggestion,
                "true_sentiment": item.sentiment,
                "parsed_reasoning": str(prediction.reasoning),
                "parsed_sentiment_str": parsed_sentiment_str,
                "parsed_sentiment_int": pred_sent_int,
            }

        except Exception as e:
            print(f"  ERROR processing item {idx+1}: {e}")
            pred_sent_int = -99 # Mark as error
            prediction_data_item = {
                "uuid": item.uuid, "article": item.article, "aspect": item.aspect,
                "plm_suggestion": item.plm_suggestion, "true_sentiment": item.sentiment,
                "parsed_reasoning": f"GENERATION_ERROR: {e}",
                "parsed_sentiment_str": "unknown", "parsed_sentiment_int": pred_sent_int,
            }

        all_predictions_data.append(prediction_data_item)
        predicted_sentiments_int.append(pred_sent_int)
        true_sentiments_int.append(item.true_sentiment_int)

    # --- Save Results ---
    model_path_for_metrics = os.path.join(grpo_output_dir, "final_checkpoint") # Hypothetical path
    
    predictions_file = os.path.join(eval_output_dir, f"{experiment_name}_full_test_predictions_split{args.train_split_index}.json")
    with open(predictions_file, 'w', encoding='utf-8') as f:
        json.dump(all_predictions_data, f, indent=2, ensure_ascii=False)
    print(f"\nSaved full test predictions to {predictions_file}")

    final_metrics = compute_metrics_absa(
        predicted_sentiments_int, 
        true_sentiments_int,
        model_run_idx=args.train_split_index, 
        model_path=model_path_for_metrics
    )
    
    metrics_file = os.path.join(eval_output_dir, f"{experiment_name}_full_test_metrics_split{args.train_split_index}.json")
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(final_metrics, f, indent=2, ensure_ascii=False)
    print(f"Saved full test metrics to {metrics_file}")
    
    metric_key = f'model_{args.train_split_index}'
    if metric_key in final_metrics:
        print(f"\n--- Final Metrics for Split {args.train_split_index} ---")
        print(f"  F1 Macro: {final_metrics[metric_key]['f1_macro']:.4f}")
        print(f"  Accuracy: {final_metrics[metric_key]['accuracy']:.4f}")
        print(f"  QWK:      {final_metrics[metric_key]['qwk']:.4f}")
    
    print("\nScript finished successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a model using DSPy GRPO for ABSA.")
    
    # Model and Backend
    parser.add_argument("--model_name_or_path", type=str, default="google/gemma-3-12b-it")
    parser.add_argument("--arbor_port", type=int, default=7453, help="Port for the Arbor RL server.")
    
    # Data
    parser.add_argument("--dataset_dir", type=str, default="../data/final/balanced-predicted/simplified-dart-xlmr/", help="Base directory for dataset JSON files.")
    parser.add_argument("--dataset_language", type=str, choices=["slovenian", "serbian"], default="slovenian")
    parser.add_argument("--train_split_index", type=int, default=0, help="Index of the train/val split to use.")
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_val_samples", type=int, default=None)
    
    # Experiment Control
    parser.add_argument("--output_dir_base", type=str, default="../models/dspy_grpo/", help="Base directory to save results.")
    parser.add_argument("--experiment_name", type=str, default="gemma3_12b_grpo_run1", help="Unique name for this experiment.")
    parser.add_argument("--use_plm_suggestion", action="store_true", help="Include the PLM suggestion in the prompt and training.")
    
    # GRPO and Training Hyperparameters
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32) 
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--num_generations", type=int, default=4, help="Completions per prompt in GRPO (num_samples_per_input).")
    parser.add_argument("--grpo_beta", type=float, default=0.1, help="GRPO KL divergence beta.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Generation temperature for the LM.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4) 
    parser.add_argument("--max_steps", type=int, default=100, help="Max training steps for GRPO.")
    parser.add_argument("--learning_rate", type=float, default=5e-5) 
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--num_threads", type=int, default=16, help="Number of threads for data generation in GRPO.")
    
    # Logging and Execution
    parser.add_argument("--logging_steps", type=int, default=1) 
    parser.add_argument("--report_to_tensorboard", action="store_true")
    parser.add_argument("--test_only", action="store_true", help="Skip training (currently not fully supported, for structure).")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    main(args)