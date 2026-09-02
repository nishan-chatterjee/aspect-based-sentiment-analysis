# -*- coding: utf-8 -*-
import os
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import datetime
from torch.utils.data import Dataset, DataLoader
from transformers import XLMRobertaModel, XLMRobertaTokenizerFast, logging as hf_logging
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    cohen_kappa_score,
    f1_score
)
from tqdm.auto import tqdm
import time
import gc
import copy
import warnings # Import warnings
import re # For potential future regex use
import torch.multiprocessing as mp # <--- ADD THIS LINE HERE

# Suppress unnecessary warnings from transformers
hf_logging.set_verbosity_error() # This should suppress "Token indices sequence length..."

# Suppress the specific FutureWarning from torch.load for your script's calls
warnings.filterwarnings("ignore", category=FutureWarning, message=".*`torch.load` with `weights_only=False`.*")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device == torch.device("cuda"):
    print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    print(f"Available VRAM: {torch.cuda.mem_get_info()[0] / (1024**3):.2f} GB")


# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="Train XLM-RoBERTa models using pre-generated summaries.")
parser.add_argument("--split", type=str, required=True, choices=['slovenian', 'serbian'],
                    help="Which language split to use ('slovenian' or 'serbian').")
parser.add_argument("--method_name", type=str, required=True,
                    choices=['gams-9b-summary', 'textrank-summary', 'gemma-3-27b-summary'],
                    help="Which summarization method's output to use.")
parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs (ignored if --test_only is set).")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training and evaluation.")
parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate (ignored if --test_only is set).")
parser.add_argument("--max_len", type=int, default=512, help="Maximum sequence length for XLM-R tokenizer.")
parser.add_argument("--triplet_loss", action='store_true', help="Use Triplet Loss on logits (ignored if --test_only is set).")
parser.add_argument("--triplet_margin", type=float, default=1.0, help="Margin for Triplet Loss (ignored if --test_only is set).")
parser.add_argument("--mask_aspect", action='store_true', help="Replace aspect string with a special [ASPECT] token in the summary.")
parser.add_argument("--test_only", action='store_true', help="If set, skip training and only run evaluation on the test set using existing best models.")

args = parser.parse_args()

# --- Validate Arguments ---
if args.test_only and not (args.split and args.method_name):
    print("Error: --split and --method_name are required when using --test_only.")
    exit(1)

# --- Global Variables & Configuration ---
LANGUAGE = args.split
METHOD_NAME = args.method_name
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
LR = args.lr
MAX_LEN = args.max_len
USE_TRIPLET_LOSS = args.triplet_loss
TRIPLET_MARGIN = args.triplet_margin
MASK_ASPECT = args.mask_aspect
ASPECT_MASK_TOKEN = "[ASPECT]" # The special token to use for masked aspects

# --- Path Definitions ---
if METHOD_NAME == "gams-9b-summary":
    METHOD_DATA_SUBDIR = "gams-9b"
elif METHOD_NAME == "textrank-summary":
    METHOD_DATA_SUBDIR = "textrank"
elif METHOD_NAME == "gemma-3-27b-summary":
    METHOD_DATA_SUBDIR = "gemma-3-27b"
else:
    print(f"Error: Unknown method_name '{METHOD_NAME}' for data path determination.")
    exit(1)

BASE_DATA_DIR = f"../data/final/summaries-third-train-val-test/{METHOD_DATA_SUBDIR}"
# Adjust model save directory based on mask_aspect option
model_dir_suffix = "_mask_aspect" if MASK_ASPECT else ""
MODEL_SAVE_DIR = f"../models/xlmr/{METHOD_NAME}/{LANGUAGE}{model_dir_suffix}"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# --- Data Loading Function (Adjusted for specific base path) ---
def load_split_data(method_base_path, language, split_index=None):
    if split_index is not None: # Train/Val split
        file_path = os.path.join(method_base_path, f"{language}_train_val_complete_{split_index}.json")
        print(f"Loading train/val data from: {file_path}")
        try:
            with open(file_path, "r", encoding="utf-8") as f: data = json.load(f)
            train_data, val_data = data.get('train', []), data.get('val', [])
            if not train_data or not val_data: raise ValueError(f"Missing 'train' or 'val' key in {file_path}")
            print(f"Loaded {len(train_data)} training samples and {len(val_data)} validation samples for split {split_index}.")
            return train_data, val_data
        except FileNotFoundError: print(f"Error: File not found - {file_path}"); exit(1)
        except Exception as e: print(f"Error loading {file_path}: {e}"); exit(1)
    else: # Test split
        file_path = os.path.join(method_base_path, f"{language}_test_complete.json")
        print(f"Loading test data from: {file_path}")
        try:
            with open(file_path, "r", encoding="utf-8") as f: data = json.load(f)
            test_data = data.get('test', [])
            if not test_data: raise ValueError(f"Missing 'test' key in {file_path}")
            print(f"Loaded {len(test_data)} test samples.")
            return test_data
        except FileNotFoundError: print(f"Error: File not found - {file_path}"); exit(1)
        except Exception as e: print(f"Error loading {file_path}: {e}"); exit(1)

# --- Utility function dict_tree (Unchanged) ---
def dict_tree(data, indent=0, max_length=20):
    for key, value in data.items():
        if isinstance(value, dict): print(' ' * indent + f'└── {key}'); dict_tree(value, indent + 4)
        else:
            preview_value = value
            if isinstance(value, str) and len(value) > max_length: preview_value = f"{value[:max_length]}..."
            print(' ' * indent + f'└── {key}: {preview_value}')


# --- Dataset Class (Modified for Masked Aspect) ---
class SentimentDataset(Dataset):
    def __init__(self, data, tokenizer, max_length, use_mask_aspect=False, mask_aspect_token="[ASPECT]", max_summary_preview=200):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_mask_aspect = use_mask_aspect
        self.mask_aspect_token = mask_aspect_token
        self.aspect_tag = "<aspect>"
        self.aspect_end_tag = "</aspect>"
        self.max_summary_preview = max_summary_preview

        if self.use_mask_aspect:
            self.input_type_desc = f"Summary with Masked Aspect ('{self.mask_aspect_token}')"
        else:
            self.input_type_desc = "Pre-generated Summary (Cleaned) + Aspect"

    def __len__(self):
        return len(self.data)

    def _clean_text(self, text):
        """Removes aspect tags from text if present."""
        if text:
             cleaned_text = text.replace(self.aspect_tag, "").replace(self.aspect_end_tag, "").strip()
             return cleaned_text
        return ""

    def __getitem__(self, idx):
        item = self.data[idx]
        summary_text_raw = item.get('summary', '')
        aspect_str = item.get('aspect', '')

        input_text = ""
        pair_text = None # Default to None for text_pair

        if self.use_mask_aspect and aspect_str:
            # 1. Clean the summary of any pre-existing <aspect> and </aspect> tags.
            text_for_replacement = summary_text_raw.replace(self.aspect_tag, "").replace(self.aspect_end_tag, "").strip()
            # 2. Replace all occurrences of the clean aspect_str with the mask_aspect_token.
            # Using direct string replacement. For more robustness (e.g. whole word), regex could be used:
            # text_content = re.sub(r'\b' + re.escape(aspect_str) + r'\b', self.mask_aspect_token, text_for_replacement)
            text_content = text_for_replacement.replace(aspect_str, self.mask_aspect_token)
            input_text = text_content
            # pair_text remains None as aspect info is in input_text
        else:
            # Original behavior: clean summary, aspect as pair
            text_content = self._clean_text(summary_text_raw)
            input_text = text_content
            if aspect_str: # Only set pair_text if aspect_str is not empty
                pair_text = aspect_str

        # Tokenization
        encoding = self.tokenizer(
            text=input_text if input_text else "", # Ensure empty string if no text
            text_pair=pair_text, # Pass None if no pair_text
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )

        # Sentiment Label Processing
        sentiment_original = item.get('sentiment', 0)
        sentiment_mapped = sentiment_original + 1
        if sentiment_mapped not in [0, 1, 2]:
            sentiment_mapped = 1 # Default to neutral if out of expected range

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(sentiment_mapped, dtype=torch.long)
        }


# --- Model Definition (Unchanged) ---
class AspectBasedSentimentClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super(AspectBasedSentimentClassifier, self).__init__()
        self.xlmr = XLMRobertaModel.from_pretrained("xlm-roberta-base")
        self.dropout = nn.Dropout(self.xlmr.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.xlmr.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.xlmr(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :] # CLS token
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


# --- Triplet Loss (Unchanged) ---
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0, use_triplet=True):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ce_loss = nn.CrossEntropyLoss()
        self.use_triplet = use_triplet

    def forward(self, outputs, labels):
        ce = self.ce_loss(outputs, labels)
        if not self.use_triplet or outputs.size(0) <= 2: # Need at least 3 samples for a triplet
            return ce
        
        triplet_loss_val = 0.0
        count = 0
        embeddings_for_triplet = outputs # Using logits as embeddings for triplet loss

        for i in range(outputs.size(0)):
            anchor_label = labels[i]
            anchor_emb = embeddings_for_triplet[i]

            pos_indices = [j for j, lbl in enumerate(labels) if lbl == anchor_label and j != i]
            neg_indices = [j for j, lbl in enumerate(labels) if lbl != anchor_label]

            if len(pos_indices) > 0 and len(neg_indices) > 0:
                pos_idx = random.choice(pos_indices)
                neg_idx = random.choice(neg_indices)

                positive_emb = embeddings_for_triplet[pos_idx]
                negative_emb = embeddings_for_triplet[neg_idx]

                pos_dist_sq = torch.sum((anchor_emb - positive_emb).pow(2))
                neg_dist_sq = torch.sum((anchor_emb - negative_emb).pow(2))
                
                loss = torch.clamp(pos_dist_sq - neg_dist_sq + self.margin, min=0.0)
                if loss > 0: # Only accumulate if loss is positive
                    triplet_loss_val += loss
                    count += 1
        
        return ce + (triplet_loss_val / count if count > 0 else 0.0)


# --- Evaluation Function (Unchanged) ---
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels) # Criterion here is usually just CE for eval
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
    target_names_mapped = ["Negative (0)", "Neutral (1)", "Positive (2)"]
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0)
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='micro', zero_division=0)
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0)
    qwk = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
    
    report_dict = classification_report(
        all_labels, all_preds, target_names=target_names_mapped, zero_division=0, output_dict=True)
    report_str = classification_report(
        all_labels, all_preds, target_names=target_names_mapped, zero_division=0)

    metrics_results = {
        "loss": avg_loss, "accuracy": accuracy,
        "precision_macro": precision_macro, "recall_macro": recall_macro, "f1_macro": f1_macro,
        "precision_micro": precision_micro, "recall_micro": recall_micro, "f1_micro": f1_micro,
        "precision_weighted": precision_weighted, "recall_weighted": recall_weighted, "f1_weighted": f1_weighted,
        "qwk": qwk, "per_class_report": report_dict
    }
    return avg_loss, metrics_results, report_str, all_preds


# --- Gradient Norm Calculation (Unchanged) ---
def get_grad_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

# --- Training Function (Unchanged) ---
def train(model, train_dataloader, val_dataloader, criterion, optimizer, scheduler, device, epochs, best_model_save_path, run_index):
    best_val_f1_macro = -1.0
    run_metrics = {"train": [], "eval": []}
    print(f"\n--- Starting Training for Run {run_index} ---")
    print(f"Best model (based on Val Macro F1) will be saved to: {best_model_save_path}")

    eval_criterion = TripletLoss(use_triplet=False).to(device) # Use CE for validation loss reporting

    for epoch in range(epochs):
        epoch_start_time = time.time()
        model.train()
        train_loss, total_grad_norm, batch_count = 0, 0, 0
        progress_bar = tqdm(train_dataloader, desc=f"Run {run_index} Epoch {epoch+1}/{epochs} Training", leave=False)
        for batch_idx, batch in enumerate(progress_bar):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels) # This is the training criterion (can be TripletLoss)

            if torch.isnan(loss):
                print(f"Warning: NaN loss encountered at run {run_index}, epoch {epoch+1}, batch {batch_idx}. Skipping batch.")
                optimizer.zero_grad() # Crucial to zero grad again if skipping optimizer step
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Gradient clipping
            grad_norm = get_grad_norm(model)
            total_grad_norm += grad_norm
            optimizer.step()
            
            train_loss += loss.item()
            batch_count += 1
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}", 'grad_norm': f"{grad_norm:.4f}"})

        avg_train_loss = train_loss / batch_count if batch_count > 0 else 0
        avg_grad_norm = total_grad_norm / batch_count if batch_count > 0 else 0
        
        run_metrics["train"].append({
            "run_index": run_index, "epoch": epoch + 1, "loss": avg_train_loss,
            "grad_norm": avg_grad_norm, "learning_rate": optimizer.param_groups[0]['lr'],
            "timestamp": datetime.datetime.now().isoformat(), "type": "train"
        })

        val_loss, val_metrics, val_report_str, _ = evaluate(model, val_dataloader, eval_criterion, device)
        val_f1_macro = val_metrics["f1_macro"]
        
        run_metrics["eval"].append({
            "run_index": run_index, "epoch": epoch + 1, "loss": val_loss, # val_loss here is CE
            **val_metrics, "timestamp": datetime.datetime.now().isoformat(), "type": "eval"
        })
        
        epoch_duration = time.time() - epoch_start_time
        print(f"Run {run_index} Epoch {epoch+1}/{epochs} Summary ({epoch_duration:.2f}s): "
              f"Train Loss: {avg_train_loss:.4f}, Avg Grad Norm: {avg_grad_norm:.4f}, "
              f"Val Loss (CE): {val_loss:.4f}, Val Macro F1: {val_f1_macro:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")

        if val_f1_macro > best_val_f1_macro:
            best_val_f1_macro = val_f1_macro
            try:
                torch.save({
                    'epoch': epoch + 1, 'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(), 
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'val_f1_macro': best_val_f1_macro, 'val_metrics': val_metrics,
                    'run_index': run_index, 'args': vars(args) # Save args for reproducibility
                }, best_model_save_path)
                print(f"  Saved new best model (Epoch {epoch+1}, Val Macro F1: {val_f1_macro:.4f}) to {best_model_save_path}")
            except Exception as e:
                print(f"Error saving model: {e}")
        
        if scheduler:
             scheduler.step(val_loss) # Or scheduler.step(val_f1_macro) if mode='max'

        if device == torch.device("cuda"):
            torch.cuda.empty_cache()
        gc.collect()

    print(f"--- Finished Training for Run {run_index} --- Best Val Macro F1: {best_val_f1_macro:.4f}")
    return run_metrics, best_model_save_path


# --- Main Execution ---
def main():
    print("--- Experiment Setup ---")
    print(f"Method Name (Summarization Source): {METHOD_NAME}")
    print(f"Language Split: {LANGUAGE}")
    print(f"Model: XLM-RoBERTa Base")
    print(f"Input Processing: Using pre-generated summaries from '{METHOD_DATA_SUBDIR}' directory.")
    print(f"Using Masked Aspect Token: {MASK_ASPECT}")
    if MASK_ASPECT:
        print(f"  Aspect Mask Token: '{ASPECT_MASK_TOKEN}'")
    print(f"Output Directory: {MODEL_SAVE_DIR}")
    print(f"Base Data Directory for this method: {BASE_DATA_DIR}")
    print(f"Max Sequence Length: {MAX_LEN}")

    if args.test_only:
        print("\n*** RUNNING IN TEST-ONLY MODE ***")
        print("Training will be skipped. Loading existing models for evaluation.")
    else:
        print(f"Epochs per run: {EPOCHS}")
        print(f"Batch Size: {BATCH_SIZE}")
        print(f"Learning Rate: {LR}")
        print(f"Use Triplet Loss: {USE_TRIPLET_LOSS} (Margin: {TRIPLET_MARGIN if USE_TRIPLET_LOSS else 'N/A'})")
    print("------------------------")

    run_args_dict = vars(args) # Get args as a dictionary
    run_args_dict['model_base'] = 'xlm-roberta-base'
    run_args_dict['model_save_dir'] = MODEL_SAVE_DIR
    run_args_dict['data_source_subdir'] = METHOD_DATA_SUBDIR
    # MASK_ASPECT is already in args, so it will be in run_args_dict

    print("Loading XLM-RoBERTa tokenizer...")
    try:
        tokenizer = XLMRobertaTokenizerFast.from_pretrained("xlm-roberta-base")
        if MASK_ASPECT:
            num_added_toks = tokenizer.add_special_tokens({'additional_special_tokens': [ASPECT_MASK_TOKEN]})
            if num_added_toks > 0:
                print(f"Added '{ASPECT_MASK_TOKEN}' to tokenizer. New vocab size: {len(tokenizer)}")
            else:
                 print(f"Token '{ASPECT_MASK_TOKEN}' already present in tokenizer. Vocab size: {len(tokenizer)}")
    except Exception as e:
        print(f"Error loading XLM-R tokenizer or adding special token: {e}"); exit(1)

    all_test_metrics = {}
    best_model_paths = {} # Stores path for each run i

    num_workers = min(4, os.cpu_count() // 2 if os.cpu_count() else 1)
    if num_workers > 0: print(f"Using {num_workers} dataloader workers.")
    else: print(f"Using 0 dataloader workers (main process data loading).")

    if not args.test_only:
        print("\n===== Starting Training Phase =====")
        for i in range(3): # 3 runs for robustness
            print(f"\n===== Starting Run {i} =====")
            current_best_model_path = os.path.join(MODEL_SAVE_DIR, f"best_model_{i}.pt")
            metrics_file_path = os.path.join(MODEL_SAVE_DIR, f"training_metrics_{i}.json")

            train_data_raw, val_data_raw = load_split_data(BASE_DATA_DIR, LANGUAGE, split_index=i)

            train_dataset = SentimentDataset(train_data_raw, tokenizer, MAX_LEN,
                                             use_mask_aspect=MASK_ASPECT, mask_aspect_token=ASPECT_MASK_TOKEN)
            val_dataset = SentimentDataset(val_data_raw, tokenizer, MAX_LEN,
                                           use_mask_aspect=MASK_ASPECT, mask_aspect_token=ASPECT_MASK_TOKEN)

            if i == 0 and train_dataset.data: # Display sample for the first run
                print(f"\nSample data item structure (Input Type: '{train_dataset.input_type_desc}'):")
                if train_dataset.data[0]:
                    print("Raw data item (first from split 0):")
                    dict_tree(train_dataset.data[0], max_length=100)
                try:
                    sample_item = train_dataset[0]
                    print("Sample processed item (first item):")
                    print(f"  input_ids shape: {sample_item['input_ids'].shape}")
                    # Uncomment to see the tokenized input including special tokens:
                    # print(f"  decoded sample: {tokenizer.decode(sample_item['input_ids'], skip_special_tokens=False)}")
                    # print(f"  first 10 tokens: {tokenizer.convert_ids_to_tokens(sample_item['input_ids'][:10])}")
                    print(f"  attention_mask shape: {sample_item['attention_mask'].shape}")
                    print(f"  label: {sample_item['labels'].item()}")
                except Exception as e:
                    print(f"  Could not process sample item 0 for display: {e}")
                print("")

            persistent_dl = num_workers > 0 and torch.__version__ >= "1.7.0"
            try:
                train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=True, persistent_workers=persistent_dl)
                val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=persistent_dl)
            except (TypeError, NotImplementedError): # Fallback for older PyTorch or specific systems
                 print("Warning: Persistent workers not supported or caused an error. Disabling.")
                 persistent_dl = False
                 train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=True)
                 val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=True)

            model = AspectBasedSentimentClassifier().to(device)
            if MASK_ASPECT:
                model.xlmr.resize_token_embeddings(len(tokenizer))
                print(f"Run {i}: Resized model token embeddings to {len(tokenizer)} for new token.")
            
            criterion = TripletLoss(margin=TRIPLET_MARGIN, use_triplet=USE_TRIPLET_LOSS).to(device)
            optimizer = optim.AdamW(model.parameters(), lr=LR)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=2, verbose=False)

            run_metrics, saved_model_path = train(
                model, train_dataloader, val_dataloader, criterion, optimizer, scheduler,
                device, EPOCHS, current_best_model_path, run_index=i
            )

            run_metrics_to_save = {
                "arguments": run_args_dict, "run_index": i,
                "train_metrics": run_metrics["train"], "eval_metrics": run_metrics["eval"],
                "best_model_path": saved_model_path if saved_model_path and os.path.exists(saved_model_path) else None
            }
            try:
                with open(metrics_file_path, 'w', encoding='utf-8') as f:
                    json.dump(run_metrics_to_save, f, indent=4, default=str) # default=str for datetime
                print(f"Training and validation metrics for run {i} saved to: {metrics_file_path}")
            except Exception as e: print(f"Error saving metrics for run {i}: {e}")

            if saved_model_path and os.path.exists(saved_model_path):
                best_model_paths[i] = saved_model_path
            else:
                print(f"Warning: Best model for run {i} ('{saved_model_path}') was not saved or not found.")
                best_model_paths[i] = None # Mark as None if not found

            del model, optimizer, scheduler, train_dataloader, val_dataloader, train_dataset, val_dataset
            del train_data_raw, val_data_raw, run_metrics, run_metrics_to_save
            gc.collect()
            if device == torch.device("cuda"): torch.cuda.empty_cache()
            print(f"===== Finished Run {i} =====")
        print("\n===== Finished Training Phase =====")
    else: # args.test_only is True
        print("\n===== Test Only Mode: Locating existing models =====")
        found_any_model = False
        for i in range(3):
            potential_model_path = os.path.join(MODEL_SAVE_DIR, f"best_model_{i}.pt")
            if os.path.exists(potential_model_path):
                best_model_paths[i] = potential_model_path
                print(f"Found existing model for run {i}: {potential_model_path}")
                found_any_model = True
            else:
                best_model_paths[i] = None
                print(f"Warning: Model file not found for run {i} at {potential_model_path}")
        if not found_any_model:
            print(f"\nError: No 'best_model_X.pt' files found in {MODEL_SAVE_DIR}.")
            print("Cannot proceed with testing. Ensure --method_name, --split are correct, training was completed,")
            print("and --mask_aspect flag matches the training condition of the models.")
            exit(1)

    # === Testing Phase ===
    print("\n===== Starting Testing Phase =====")
    test_data_raw = load_split_data(BASE_DATA_DIR, LANGUAGE, split_index=None)
    test_dataset = SentimentDataset(test_data_raw, tokenizer, MAX_LEN,
                                    use_mask_aspect=MASK_ASPECT, mask_aspect_token=ASPECT_MASK_TOKEN)
    
    test_batch_size = BATCH_SIZE * 2 # Can often use larger batch for inference
    persistent_test_dl = num_workers > 0 and torch.__version__ >= "1.7.0"
    try:
        test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=persistent_test_dl)
    except (TypeError, NotImplementedError):
        test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    criterion_test = TripletLoss(use_triplet=False).to(device) # Use CrossEntropy for test evaluation
    test_results_summary = [] # To calculate averages

    for i in range(3): # Evaluate each of the 3 models
        print(f"\n--- Evaluating Best Model from Run {i} on Test Set ---")
        model_path = best_model_paths.get(i)
        if not model_path: # Handles if a model for a run was not saved/found
            print(f"Warning: No model path found for run {i}. Skipping.")
            all_test_metrics[f"model_{i}"] = {"error": "Model path not found or model was not saved/found."}
            continue

        model_test = AspectBasedSentimentClassifier().to(device)
        if MASK_ASPECT:
            model_test.xlmr.resize_token_embeddings(len(tokenizer))
            print(f"Test model {i}: Resized model token embeddings to {len(tokenizer)} before loading state.")

        try:
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            if 'model_state_dict' in checkpoint:
                model_test.load_state_dict(checkpoint['model_state_dict'])
                print(f"Successfully loaded model state from epoch {checkpoint.get('epoch', 'N/A')} in {model_path}")
                # Log args from checkpoint if they exist, useful for verifying consistency
                if 'args' in checkpoint:
                    loaded_args = checkpoint['args']
                    if loaded_args.get('mask_aspect', False) != MASK_ASPECT:
                        print(f"WARNING: --mask_aspect mismatch! Current: {MASK_ASPECT}, Model trained with: {loaded_args.get('mask_aspect')}")
            else: # Old format or direct state_dict save
                model_test.load_state_dict(checkpoint)
                print(f"Successfully loaded model state directly from {model_path} (assumed raw state_dict or old format).")
            model_test.eval()
        except Exception as e:
            print(f"Error loading model state from {model_path}: {e}")
            all_test_metrics[f"model_{i}"] = {"error": f"Failed loading state: {e}"}
            del model_test; gc.collect();
            if device == torch.device("cuda"): torch.cuda.empty_cache()
            continue

        test_loss, test_metrics_dict, test_report_str, test_predictions_mapped = evaluate(
            model_test, test_dataloader, criterion_test, device
        )
        print(f"Test Results Model {i}: Loss={test_loss:.4f}, "
              f"Macro F1={test_metrics_dict['f1_macro']:.4f}, Acc={test_metrics_dict['accuracy']:.4f}, "
              f"QWK={test_metrics_dict['qwk']:.4f}")
        print("Full Classification Report (Test Set):"); print(test_report_str)

        test_result_data = {
            "model_run_index": i, "model_path": model_path, 
            "test_loss": test_loss, **test_metrics_dict
        }
        all_test_metrics[f"model_{i}"] = test_result_data
        test_results_summary.append({ # For averaging
            "f1_macro": test_metrics_dict['f1_macro'],
            "accuracy": test_metrics_dict['accuracy'],
            "qwk": test_metrics_dict['qwk']
        })

        # Save predictions
        predictions_file_path = os.path.join(MODEL_SAVE_DIR, f"test_predictions_{i}.json")
        # test_predictions_mapped are 0,1,2. Original sentiment is -1,0,1. So map back.
        test_predictions_original_scale = [p - 1 for p in test_predictions_mapped]

        if len(test_predictions_original_scale) == len(test_data_raw):
            test_data_with_preds = copy.deepcopy(test_data_raw) # Avoid modifying original raw data
            for idx_pred, item_pred in enumerate(test_data_with_preds):
                item_pred['prediction'] = int(test_predictions_original_scale[idx_pred]) # Ensure it's Python int
            try:
                with open(predictions_file_path, 'w', encoding='utf-8') as f:
                    json.dump(test_data_with_preds, f, indent=4, ensure_ascii=False)
                print(f"Test predictions for model {i} saved to: {predictions_file_path}")
            except Exception as e: print(f"Error saving test predictions for model {i}: {e}")
        else:
            print(f"Error: Mismatch between #predictions ({len(test_predictions_original_scale)}) & #test items ({len(test_data_raw)}). Predictions not saved.")
            all_test_metrics[f"model_{i}"]["prediction_error"] = "Prediction count mismatch."


        del model_test, checkpoint; gc.collect()
        if device == torch.device("cuda"): torch.cuda.empty_cache()

    # === Final Reporting ===
    if test_results_summary: # If any model was successfully tested
        avg_test_f1_macro = np.mean([r['f1_macro'] for r in test_results_summary])
        avg_test_accuracy = np.mean([r['accuracy'] for r in test_results_summary])
        avg_test_qwk = np.mean([r['qwk'] for r in test_results_summary])
        print("\n--- Average Test Set Performance Across Successfully Evaluated Models ---")
        print(f"Avg Macro F1: {avg_test_f1_macro:.4f}")
        print(f"Avg Accuracy: {avg_test_accuracy:.4f}")
        print(f"Avg QWK: {avg_test_qwk:.4f}")
        all_test_metrics["average_performance"] = {
            "f1_macro": avg_test_f1_macro, "accuracy": avg_test_accuracy, "qwk": avg_test_qwk,
            "num_models_averaged": len(test_results_summary)
        }
    else:
        print("\nNo models were successfully evaluated on the test set.")
        all_test_metrics["average_performance"] = {"error": "No models available or evaluation failed for all."}

    combined_test_metrics_file = os.path.join(MODEL_SAVE_DIR, "test_metrics_summary.json")
    try:
        with open(combined_test_metrics_file, 'w', encoding='utf-8') as f:
            json.dump(all_test_metrics, f, indent=4, default=str) # default=str for np types if any
        print(f"\nCombined test metrics summary saved to: {combined_test_metrics_file}")
    except Exception as e: print(f"Error saving combined test metrics: {e}")

    print("\nExperiment Complete.")
    print(f"Outputs saved in: {MODEL_SAVE_DIR}")


if __name__ == "__main__":
    import torch.multiprocessing as mp # <--- ADD THIS LINE

    # Set multiprocessing start method for CUDA compatibility if needed
    # This is often needed when using 'spawn' or 'forkserver' with CUDA.
    # 'fork' (default on Linux) can sometimes cause issues with CUDA in subprocesses.
    if torch.cuda.is_available(): # Only if CUDA is relevant
        current_start_method = mp.get_start_method(allow_none=True)
        if current_start_method != 'spawn':
            try:
                mp.set_start_method('spawn', force=True)
                print("Multiprocessing start method set to 'spawn'.")
            except RuntimeError as e:
                if "context has already been set" in str(e):
                    print(f"Warning: Multiprocessing context already set to '{current_start_method}'. Cannot change to 'spawn'.")
                else:
                    print(f"Warning: Could not set multiprocessing start method to 'spawn': {e}")
        else:
            print("Multiprocessing start method is already 'spawn'.")


    try: import transformers
    except ImportError: print("Error: transformers not found. Run: pip install transformers"); exit(1)
    try: import sklearn
    except ImportError: print("Error: scikit-learn not found. Run: pip install scikit-learn"); exit(1)
    # pandas, numpy, tqdm are also dependencies, but usually installed with torch/transformers/sklearn

    main()