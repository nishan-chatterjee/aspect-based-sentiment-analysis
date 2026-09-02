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

# --- SpaCy Import and Setup ---
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    print("SpaCy library not found. 'extractive_summary' mode will not be available.")
    # ... (rest of SpaCy import messages)

# --- SpaCy Model Loading (FIXED) ---
# ... (load_spacy_model function - no changes from your provided version)
def load_spacy_model(language):
    """Loads the appropriate SpaCy model based on language."""
    if not SPACY_AVAILABLE:
        print("Error: SpaCy is not installed. Cannot use 'extractive_summary' mode.")
        exit(1)

    model_name = None
    if language == 'slovenian':
        model_name = "sl_core_news_sm"
    elif language == 'serbian':
        model_name = "hr_core_news_sm" # Or your preferred small model like "hr_core_news_sm"
    else:
        print(f"Error: No SpaCy model configured for language '{language}'.")
        exit(1)

    try:
        print(f"Loading SpaCy model: {model_name}...")
        nlp = spacy.load(model_name)
        if 'sentencizer' not in nlp.pipe_names and 'senter' not in nlp.pipe_names:
             print(f"Warning: Sentencizer pipe not found in '{model_name}' default pipes. Attempting to add.")
             try:
                 nlp.add_pipe('sentencizer', first=True)
             except ValueError as e:
                 if "already exists in pipeline" in str(e):
                     print("Info: Sentencizer implicitly present, proceeding.")
                 else:
                     raise e
        print(f"SpaCy model '{model_name}' loaded successfully with pipes: {nlp.pipe_names}")
        return nlp
    except OSError:
        print(f"Error: SpaCy model '{model_name}' not found."); print(f"Please download it: python -m spacy download {model_name}"); exit(1)
    except Exception as e: print(f"An unexpected error occurred loading SpaCy model '{model_name}': {e}"); exit(1)


# Suppress unnecessary warnings from transformers
hf_logging.set_verbosity_error()

# --- Suppress the specific FutureWarning from torch.load for your script's calls ---
warnings.filterwarnings("ignore", category=FutureWarning, message=".*`torch.load` with `weights_only=False`.*")
# You might also see a similar warning from SpaCy/Thinc internals if they use torch.load.
# To suppress that too (if it appears and is noisy):
# warnings.filterwarnings("ignore", category=FutureWarning, module="thinc.shims.pytorch")


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device == torch.device("cuda"):
    # ... (CUDA info print)
    print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    print(f"Available VRAM: {torch.cuda.mem_get_info()[0] / (1024**3):.2f} GB")


# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="Train XLM-RoBERTa models using different article processing methods.")
parser.add_argument("--split", type=str, required=True, choices=['slovenian', 'serbian'],
                    help="Which language split to use ('slovenian' or 'serbian').")
parser.add_argument("--method_name", type=str, required=True, choices=['no_summary', 'extractive_summary'],
                    help="Processing method: 'no_summary' (whole article) or 'extractive_summary' (SpaCy filtered sentences).")
parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs (ignored if --test_only is set).")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training and evaluation.")
parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate (ignored if --test_only is set).")
parser.add_argument("--max_len", type=int, default=512, help="Maximum sequence length for XLM-R tokenizer.")
parser.add_argument("--triplet_loss", action='store_true', help="Use Triplet Loss on logits (ignored if --test_only is set).")
parser.add_argument("--triplet_margin", type=float, default=1.0, help="Margin for Triplet Loss (ignored if --test_only is set).")
parser.add_argument("--test_only", action='store_true', help="If set, skip training and only run evaluation on the test set using existing best models.") # <-- New Argument


args = parser.parse_args()

# --- Validate Arguments ---
# ... (Validation remains the same) ...
if args.method_name == 'extractive_summary' and not SPACY_AVAILABLE:
    print("Error: --method_name 'extractive_summary' requires SpaCy, but it's not installed or models are missing.")
    exit(1)
if args.test_only and not (args.split and args.method_name):
    print("Error: --split and --method_name are required when using --test_only.")
    exit(1)


# --- Global Variables & Configuration ---
# ... (Globals remain the same) ...
LANGUAGE = args.split
METHOD_NAME = args.method_name
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
LR = args.lr
MAX_LEN = args.max_len
USE_TRIPLET_LOSS = args.triplet_loss
TRIPLET_MARGIN = args.triplet_margin

# --- Path Definitions ---
# ... (Paths remain the same) ...
BASE_DATA_DIR = "../data/final/complete"
MODEL_SAVE_DIR = f"../models/xlmr/{METHOD_NAME}/{LANGUAGE}" # Corrected path base
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# --- Data Loading Function (Unchanged) ---
# ... (load_split_data remains the same) ...
def load_split_data(base_path, language, split_index=None):
    if split_index is not None:
        file_path = os.path.join(base_path, f"{language}_train_val_complete_{split_index}.json")
        print(f"Loading train/val data from: {file_path}")
        try:
            with open(file_path, "r", encoding="utf-8") as f: data = json.load(f)
            train_data, val_data = data.get('train', []), data.get('val', [])
            if not train_data or not val_data: raise ValueError(f"Missing 'train' or 'val' key in {file_path}")
            print(f"Loaded {len(train_data)} training samples and {len(val_data)} validation samples for split {split_index}.")
            return train_data, val_data
        except FileNotFoundError: print(f"Error: File not found - {file_path}"); exit(1)
        except Exception as e: print(f"Error loading {file_path}: {e}"); exit(1)
    else:
        file_path = os.path.join(base_path, f"{language}_test_complete.json")
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
# ... (dict_tree remains the same) ...
def dict_tree(data, indent=0, max_length=20):
    for key, value in data.items():
        if isinstance(value, dict): print(' ' * indent + f'└── {key}'); dict_tree(value, indent + 4)
        else:
            preview_value = value
            if isinstance(value, str) and len(value) > max_length: preview_value = f"{value[:max_length]}..."
            print(' ' * indent + f'└── {key}: {preview_value}')

# --- Dataset Class (Unchanged) ---
# ... (SentimentDataset remains the same) ...
class SentimentDataset(Dataset):
    def __init__(self, data, tokenizer, max_length, method_name, spacy_nlp=None, max_article_preview=200):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.method_name = method_name
        self.spacy_nlp = spacy_nlp
        self.max_article_preview = max_article_preview
        self.aspect_tag = "<aspect>"
        self.aspect_end_tag = "</aspect>" # Added for completeness in cleaning

        if self.method_name == 'extractive_summary' and self.spacy_nlp is None:
            raise ValueError("SpaCy model instance ('spacy_nlp') must be provided when method_name is 'extractive_summary'.")

        self.input_type_desc = "Article (Whole Cleaned + Aspect)" if method_name == 'no_summary' else "Article (Filtered Sentences - SpaCy + Aspect)"

    def __len__(self):
        return len(self.data)

    def _clean_text(self, text):
        if text:
             return text.replace(self.aspect_tag, "").replace(self.aspect_end_tag, "").strip()
        return ""

    def __getitem__(self, idx):
        item = self.data[idx]
        article = item.get('article', '')
        aspect = item.get('aspect', '')
        text_content = ""

        if self.method_name == 'extractive_summary':
            if article and self.aspect_tag in article:
                try:
                    doc = self.spacy_nlp(article.strip())
                    filtered_sentences = [sent.text for sent in doc.sents if self.aspect_tag in sent.text]
                    if filtered_sentences:
                        joined_text = ' '.join(filtered_sentences).strip()
                        text_content = self._clean_text(joined_text)
                    else:
                        text_content = ""
                except Exception as e:
                    print(f"Error processing article with SpaCy for item {idx}: {e}")
                    print("Article snippet:", article[:self.max_article_preview])
                    text_content = self._clean_text(article)
            else:
                text_content = ""
        else:
            text_content = self._clean_text(article)

        encoding = self.tokenizer(
            text=text_content if text_content else "",
            text_pair=aspect if aspect else "",
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )

        sentiment_original = item.get('sentiment', 0)
        sentiment_mapped = sentiment_original + 1
        if sentiment_mapped not in [0, 1, 2]:
            sentiment_mapped = 1

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(sentiment_mapped, dtype=torch.long)
        }

# --- Model Definition (Unchanged) ---
# ... (AspectBasedSentimentClassifier remains the same) ...
class AspectBasedSentimentClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super(AspectBasedSentimentClassifier, self).__init__()
        self.xlmr = XLMRobertaModel.from_pretrained("xlm-roberta-base")
        self.dropout = nn.Dropout(self.xlmr.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.xlmr.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.xlmr(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

# --- Triplet Loss (Unchanged) ---
# ... (TripletLoss remains the same) ...
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0, use_triplet=True):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ce_loss = nn.CrossEntropyLoss()
        self.use_triplet = use_triplet

    def forward(self, outputs, labels):
        ce = self.ce_loss(outputs, labels)
        if not self.use_triplet or outputs.size(0) <= 2:
            return ce

        triplet_loss_val = 0.0
        count = 0
        embeddings_for_triplet = outputs # Using logits directly as embeddings for triplet
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
                if loss > 0: # Only add if loss is positive
                    triplet_loss_val += loss
                    count += 1
        return ce + (triplet_loss_val / count if count > 0 else 0.0)

# --- Evaluation Function (Unchanged) ---
# ... (evaluate remains the same) ...
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
            loss = criterion(outputs, labels)
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
# ... (get_grad_norm remains the same) ...
def get_grad_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

# --- Training Function (Unchanged) ---
# ... (train remains the same) ...
def train(model, train_dataloader, val_dataloader, criterion, optimizer, scheduler, device, epochs, best_model_save_path, run_index):
    best_val_f1_macro = -1.0
    run_metrics = {"train": [], "eval": []}
    print(f"\n--- Starting Training for Run {run_index} ---")
    print(f"Best model (based on Val Macro F1) will be saved to: {best_model_save_path}")
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
            loss = criterion(outputs, labels)
            if torch.isnan(loss):
                print(f"Warning: NaN loss encountered at run {run_index}, epoch {epoch+1}, batch {batch_idx}. Skipping batch.")
                optimizer.zero_grad()
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
        val_loss, val_metrics, val_report_str, _ = evaluate(model, val_dataloader, criterion, device)
        val_f1_macro = val_metrics["f1_macro"]
        run_metrics["eval"].append({
            "run_index": run_index, "epoch": epoch + 1, "loss": val_loss,
            **val_metrics, "timestamp": datetime.datetime.now().isoformat(), "type": "eval"
        })
        epoch_duration = time.time() - epoch_start_time
        print(f"Run {run_index} Epoch {epoch+1}/{epochs} Summary ({epoch_duration:.2f}s): "
              f"Train Loss: {avg_train_loss:.4f}, Avg Grad Norm: {avg_grad_norm:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Macro F1: {val_f1_macro:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")
        if val_f1_macro > best_val_f1_macro:
            best_val_f1_macro = val_f1_macro
            try:
                torch.save({
                    'epoch': epoch + 1, 'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(),
                    'val_f1_macro': best_val_f1_macro, 'val_metrics': val_metrics,
                    'run_index': run_index, 'args': vars(args)
                }, best_model_save_path)
                print(f"  Saved new best model (Epoch {epoch+1}, Val Macro F1: {val_f1_macro:.4f}) to {best_model_save_path}")
            except Exception as e: print(f"Error saving model: {e}")
        scheduler.step(val_loss)
        if device == torch.device("cuda"): torch.cuda.empty_cache()
        gc.collect()
    print(f"--- Finished Training for Run {run_index} --- Best Val Macro F1: {best_val_f1_macro:.4f}")
    return run_metrics, best_model_save_path


# --- Main Execution (Modified for Test-Only Mode) ---
def main():
    print("--- Experiment Setup ---")
    # ... (print setup info) ...
    print(f"Method Name: {METHOD_NAME}")
    print(f"Language Split: {LANGUAGE}")
    # ... (rest of setup prints)
    if args.test_only:
        print("\n*** RUNNING IN TEST-ONLY MODE ***")
        print("Training will be skipped. Loading existing models for evaluation.")
    else:
        print(f"Epochs per run: {EPOCHS}")
        # ... (other training-specific prints)

    run_args_dict = vars(args)
    run_args_dict['model_base'] = 'xlm-roberta-base'
    run_args_dict['model_save_dir'] = MODEL_SAVE_DIR

    print("Loading XLM-RoBERTa tokenizer...")
    try:
        tokenizer = XLMRobertaTokenizerFast.from_pretrained("xlm-roberta-base")
    except Exception as e: print(f"Error loading XLM-R tokenizer: {e}"); exit(1)

    spacy_nlp = None
    if METHOD_NAME == 'extractive_summary':
        # Load SpaCy even in test_only mode if needed for data processing
        spacy_nlp = load_spacy_model(LANGUAGE)

    all_test_metrics = {}
    best_model_paths = {} # Will store paths to best_model_0.pt, _1.pt, _2.pt

    num_workers = 4 # Force loading in main process for simplicity, can be changed
    if num_workers > 0:
        print(f"Using {num_workers} dataloader workers.")
    else:
        print(f"WARNING: Running with num_workers=0 (main process data loading).")
    persistent = False # Cannot use persistent workers with num_workers=0

    if not args.test_only:
        # === Training Phase ===
        print("\n===== Starting Training Phase =====")
        for i in range(3):
            print(f"\n===== Starting Run {i} =====")
            current_best_model_path = os.path.join(MODEL_SAVE_DIR, f"best_model_{i}.pt")
            metrics_file_path = os.path.join(MODEL_SAVE_DIR, f"training_metrics_{i}.json")
            train_data_raw, val_data_raw = load_split_data(BASE_DATA_DIR, LANGUAGE, split_index=i)
            train_dataset = SentimentDataset(train_data_raw, tokenizer, MAX_LEN, METHOD_NAME, spacy_nlp)
            val_dataset = SentimentDataset(val_data_raw, tokenizer, MAX_LEN, METHOD_NAME, spacy_nlp)

            # ... (Sample data display logic - can be kept) ...

            # Dataloader creation
            # ... (DataLoader creation logic - can be kept) ...
            persistent_dl = num_workers > 0 and torch.__version__ >= "1.7.0"
            try:
                train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=True, persistent_workers=persistent_dl)
                val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=persistent_dl)
            except (TypeError, NotImplementedError):
                 print("Warning: Persistent workers not supported or caused an error. Disabling.")
                 persistent_dl = False
                 train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=True)
                 val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=True)


            model = AspectBasedSentimentClassifier().to(device)
            criterion = TripletLoss(margin=TRIPLET_MARGIN, use_triplet=USE_TRIPLET_LOSS).to(device)
            optimizer = optim.AdamW(model.parameters(), lr=LR)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=2, verbose=True)

            run_metrics, saved_model_path = train(
                model, train_dataloader, val_dataloader, criterion, optimizer, scheduler,
                device, EPOCHS, current_best_model_path, run_index=i
            )

            # ... (Save metrics logic - can be kept) ...
            run_metrics_to_save = {
                "arguments": run_args_dict, "run_index": i,
                "train_metrics": run_metrics["train"], "eval_metrics": run_metrics["eval"],
                "best_model_path": saved_model_path if saved_model_path and os.path.exists(saved_model_path) else None
            }
            # ... (json.dump logic) ...


            if saved_model_path and os.path.exists(saved_model_path):
                best_model_paths[i] = saved_model_path
            else:
                print(f"Warning: Best model for run {i} ('{saved_model_path}') was not saved or not found.")
                best_model_paths[i] = None

            # ... (Cleanup logic - can be kept) ...
            del model, optimizer, scheduler, train_dataloader, val_dataloader, train_dataset, val_dataset
            del train_data_raw, val_data_raw, run_metrics, run_metrics_to_save
            gc.collect()
            if device == torch.device("cuda"): torch.cuda.empty_cache()
            print(f"===== Finished Run {i} =====")
        print("\n===== Finished Training Phase =====")
    else:
        # === Test Only Mode: Populate best_model_paths ===
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
            print("Cannot proceed with testing. Ensure --method_name and --split are correct and training was completed.")
            exit(1)
        # --- End of Test Only Mode Model Path Population ---


    # === Testing Phase (runs in both modes using populated best_model_paths) ===
    print("\n===== Starting Testing Phase =====")
    test_data_raw = load_split_data(BASE_DATA_DIR, LANGUAGE, split_index=None)
    test_dataset = SentimentDataset(test_data_raw, tokenizer, MAX_LEN, METHOD_NAME, spacy_nlp)
    test_batch_size = BATCH_SIZE * 2 # Can be larger for inference
    persistent_test_dl = num_workers > 0 and torch.__version__ >= "1.7.0"

    try:
        test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=persistent_test_dl)
    except (TypeError, NotImplementedError):
        test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)


    criterion_test = TripletLoss(use_triplet=False).to(device) # For test, usually just CE
    test_results_summary = []

    for i in range(3): # Iterate through the 3 model runs
        print(f"\n--- Evaluating Best Model from Run {i} on Test Set ---")
        model_path = best_model_paths.get(i) # Get path from dict

        if not model_path: # Handles None if model wasn't found/saved
            print(f"Warning: No model path found for run {i}. Skipping.")
            all_test_metrics[f"model_{i}"] = {"error": "Model path not found or model was not saved/found."}
            continue

        model_test = AspectBasedSentimentClassifier().to(device)
        try:
            # The torch.load call below will no longer show the FutureWarning due to the filter at the top
            checkpoint = torch.load(model_path, map_location=device) # weights_only=False is default & needed
            if 'model_state_dict' in checkpoint:
                model_test.load_state_dict(checkpoint['model_state_dict'])
                print(f"Successfully loaded model state from epoch {checkpoint.get('epoch', 'N/A')} in {model_path}")
            else: # Older checkpoint format
                model_test.load_state_dict(checkpoint)
                print(f"Successfully loaded model state directly from {model_path} (assumed old format).")
            model_test.eval()
        except Exception as e:
            print(f"Error loading model state from {model_path}: {e}")
            all_test_metrics[f"model_{i}"] = {"error": f"Failed loading state: {e}"}
            del model_test; gc.collect();
            if device == torch.device("cuda"): torch.cuda.empty_cache()
            continue

        # Evaluate the loaded model
        test_loss, test_metrics_dict, test_report_str, test_predictions_mapped = evaluate(
            model_test, test_dataloader, criterion_test, device
        )
        print(f"Test Results Model {i}: Loss={test_loss:.4f}, "
              f"Macro F1={test_metrics_dict['f1_macro']:.4f}, Acc={test_metrics_dict['accuracy']:.4f}, "
              f"QWK={test_metrics_dict['qwk']:.4f}")
        print("Full Classification Report (Test Set):"); print(test_report_str)

        test_result_data = {
            "model_run_index": i, "model_path": model_path, "test_loss": test_loss, **test_metrics_dict
        }
        all_test_metrics[f"model_{i}"] = test_result_data
        test_results_summary.append({
            "f1_macro": test_metrics_dict['f1_macro'],
            "accuracy": test_metrics_dict['accuracy'],
            "qwk": test_metrics_dict['qwk']
        })

        # Save Test Predictions (with int() fix)
        predictions_file_path = os.path.join(MODEL_SAVE_DIR, f"test_predictions_{i}.json")
        test_predictions_original_scale = [p - 1 for p in test_predictions_mapped] # Map 0,1,2 -> -1,0,1
        if len(test_predictions_original_scale) == len(test_data_raw):
            test_data_with_preds = copy.deepcopy(test_data_raw)
            for idx_pred, item_pred in enumerate(test_data_with_preds): # Renamed to avoid conflict
                item_pred['prediction'] = int(test_predictions_original_scale[idx_pred]) # Ensure Python int
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
    if test_results_summary:
        avg_test_f1_macro = np.mean([r['f1_macro'] for r in test_results_summary])
        avg_test_accuracy = np.mean([r['accuracy'] for r in test_results_summary])
        avg_test_qwk = np.mean([r['qwk'] for r in test_results_summary])
        print("\n--- Average Test Set Performance Across Models ---")
        print(f"Avg Macro F1: {avg_test_f1_macro:.4f}")
        print(f"Avg Accuracy: {avg_test_accuracy:.4f}")
        print(f"Avg QWK: {avg_test_qwk:.4f}")
        all_test_metrics["average_performance"] = {
            "f1_macro": avg_test_f1_macro, "accuracy": avg_test_accuracy, "qwk": avg_test_qwk
        }
    else:
        print("\nNo models were successfully evaluated on the test set.")
        all_test_metrics["average_performance"] = {"error": "No models available or evaluation failed."}

    combined_test_metrics_file = os.path.join(MODEL_SAVE_DIR, "test_metrics_summary.json")
    # ... (Save combined metrics logic - remains the same) ...
    try:
        with open(combined_test_metrics_file, 'w', encoding='utf-8') as f:
            json.dump(all_test_metrics, f, indent=4, default=str) # default=str for any missed numpy types
        print(f"\nCombined test metrics summary saved to: {combined_test_metrics_file}")
    except Exception as e: print(f"Error saving combined test metrics: {e}")


    print("\nExperiment Complete.")
    print(f"Outputs saved in: {MODEL_SAVE_DIR}")

if __name__ == "__main__":
    import torch.multiprocessing as mp
    try:
        mp.set_start_method('spawn', force=True)
        print("Multiprocessing start method set to 'spawn'.")
    except RuntimeError as e:
        if "context has already been set" in str(e): print("Warning: Multiprocessing context already set.")
        else: print(f"Warning: Could not set multiprocessing start method: {e}")

    try: import transformers
    except ImportError: print("Error: transformers not found. pip install transformers"); exit(1)
    try: import sklearn
    except ImportError: print("Error: scikit-learn not found. pip install scikit-learn"); exit(1)

    main()