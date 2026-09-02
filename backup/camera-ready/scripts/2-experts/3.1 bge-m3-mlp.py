# -*- coding: utf-8 -*-
import os
import json
import random
import re # Added for masking
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import argparse
import datetime
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
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

# --- SpaCy Import and Setup ---
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    print("SpaCy library not found. Sentence filtering (method_name='filtered' or 'masked' with --use_filtered_sentences) will not be available.")
    print("Install SpaCy and models: pip install spacy")
    print("Then download models: python -m spacy download sl_core_news_sm")
    print("                     python -m spacy download hr_core_news_sm")

# --- SpaCy Model Loading ---
def load_spacy_model(language):
    if not SPACY_AVAILABLE:
        print("Error: SpaCy is not installed. Cannot use sentence filtering.")
        exit(1)
    model_name = None
    if language == 'slovenian': model_name = "sl_core_news_sm"
    elif language == 'serbian': model_name = "hr_core_news_sm"
    else: print(f"Error: No SpaCy model configured for language '{language}'."); exit(1)
    try:
        print(f"Loading SpaCy model: {model_name}...")
        # Disable unnecessary pipes for sentence splitting only
        nlp = spacy.load(model_name, disable=["tagger", "parser", "attribute_ruler", "lemmatizer", "ner"])
        # Ensure sentencizer is active (some models might have 'senter' instead of 'sentencizer')
        if 'sentencizer' not in nlp.pipe_names and 'senter' not in nlp.pipe_names:
            nlp.add_pipe('sentencizer')
        print(f"SpaCy model '{model_name}' loaded successfully.")
        return nlp
    except OSError: print(f"Error: SpaCy model '{model_name}' not found. Please download it: python -m spacy download {model_name}"); exit(1)
    except Exception as e: print(f"An unexpected error occurred loading SpaCy model '{model_name}': {e}"); exit(1)
# ----------------------------

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device == torch.device("cuda"):
    try:
        print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
        total_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        available_mem_gb = torch.cuda.mem_get_info()[0] / (1024**3) # Corrected: mem_get_info returns (free, total)
        print(f"Total VRAM: {total_mem_gb:.2f} GB")
        print(f"Available VRAM: {available_mem_gb:.2f} GB") # This 'available' is free memory.
    except Exception as e:
        print(f"Could not get CUDA device info: {e}")


# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="Train an MLP classifier on BGE-M3 embeddings of articles.")
parser.add_argument("--split", type=str, required=True, choices=['slovenian', 'serbian'],
                    help="Which language split to use ('slovenian' or 'serbian').")
parser.add_argument("--method_name", type=str, required=True, choices=['whole', 'filtered', 'masked'],
                    help="Processing method: 'whole' (entire article), 'filtered' (SpaCy filtered sentences with aspect), 'masked' (aspect mentions and name masked).")
parser.add_argument("--use_filtered_sentences", action='store_true',
                    help="If set (primarily for 'masked' method), use SpaCy to find sentences containing aspect tags/masks and encode only those. "
                         "This flag is implicitly True for method_name='filtered' and implicitly False for method_name='whole'.")
parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs (ignored if --test is set).")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training and evaluation.")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for the MLP (ignored if --test is set).")
parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay (L2 regularization) for the optimizer (ignored if --test is set).")
parser.add_argument("--hidden_dim1", type=int, default=512, help="Size of the first hidden layer.")
parser.add_argument("--hidden_dim2", type=int, default=256, help="Size of the second hidden layer (0 for none).")
parser.add_argument("--dropout_rate", type=float, default=0.3, help="Dropout rate for MLP.")
parser.add_argument("--test", action='store_true',
                    help="If set, skip training and only run evaluation on the test set using existing best models.")

args = parser.parse_args()

# --- Determine Actual Use of Filtered Sentences and Validate ---
LANGUAGE = args.split
METHOD_NAME = args.method_name
USER_SPECIFIED_USE_FILTERED = args.use_filtered_sentences

if METHOD_NAME == "whole":
    ACTUAL_USE_FILTERED_SENTENCES = False
elif METHOD_NAME == "filtered":
    ACTUAL_USE_FILTERED_SENTENCES = True
    if not SPACY_AVAILABLE:
        print("Error: method_name 'filtered' requires SpaCy, but it's not installed or models are missing.")
        exit(1)
elif METHOD_NAME == "masked":
    ACTUAL_USE_FILTERED_SENTENCES = USER_SPECIFIED_USE_FILTERED
    if ACTUAL_USE_FILTERED_SENTENCES and not SPACY_AVAILABLE:
        print("Error: method_name 'masked' with --use_filtered_sentences requires SpaCy, but it's not installed or models are missing.")
        exit(1)
else: # Should be caught by argparse choices
    print(f"Internal Error: Undefined behavior for method_name '{METHOD_NAME}'") # Should not happen
    exit(1)

# --- Global Variables & Configuration (derived) ---
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
LR = args.lr
WEIGHT_DECAY = args.weight_decay
HIDDEN_DIM1 = args.hidden_dim1
HIDDEN_DIM2 = args.hidden_dim2
DROPOUT_RATE = args.dropout_rate
BGE_MODEL_NAME = 'BAAI/bge-m3'
BGE_EMBEDDING_DIM = 1024

# --- Path Definitions ---
BASE_DATA_DIR = "../data/final/complete"
TOP_LEVEL_MODEL_DIR_NAME = "bge-m3_mlp" # Define your desired top-level directory name
MODEL_SAVE_DIR = f"../models/{TOP_LEVEL_MODEL_DIR_NAME}/{METHOD_NAME}/{LANGUAGE}"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# --- Data Loading Function (Unchanged from your provided file) ---
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

# --- Utility function dict_tree (Unchanged from your provided file) ---
def dict_tree(data, indent=0, max_length=20):
    for key, value in data.items():
        if isinstance(value, dict): print(' ' * indent + f'└── {key}'); dict_tree(value, indent + 4)
        else:
            preview_value = value
            if isinstance(value, str) and len(value) > max_length: preview_value = f"{value[:max_length]}..."
            print(' ' * indent + f'└── {key}: {preview_value}')

# --- Dataset Class (Modified for "masked" method and clarity) ---
class SentimentDataset(Dataset):
    def __init__(self, data, bge_model, device, method_name_arg,
                 actual_use_filtered_arg, spacy_nlp_arg=None, max_article_preview=200):
        self.data = data
        self.bge_model = bge_model
        self.device = device
        self.method_name = method_name_arg # Store the method_name from args
        self.actual_use_filtered = actual_use_filtered_arg # Store the derived boolean
        self.spacy_nlp = spacy_nlp_arg
        self.embedding_dim = BGE_EMBEDDING_DIM
        self.max_article_preview = max_article_preview

        if self.actual_use_filtered and self.spacy_nlp is None:
            raise ValueError("SpaCy model ('spacy_nlp_arg') must be provided when using filtered sentences.")

        # Determine input_type_desc based on the processing method
        if self.method_name == "masked":
            filter_status = "Filtered (SpaCy)" if self.actual_use_filtered else "Whole Document"
            self.input_type_desc = f"Article (Masked Mentions, Appended Generic Aspect Token, {filter_status}) -> BGE-M3"
        elif self.method_name == "filtered": # Implies actual_use_filtered is True
            self.input_type_desc = "Article (Filtered Tagged Sentences - SpaCy -> BGE-M3)"
        elif self.method_name == "whole": # Implies actual_use_filtered is False
            self.input_type_desc = "Article (Whole Document, Cleaned Tags -> BGE-M3)"
        else:
            self.input_type_desc = "Unknown Processing -> BGE-M3" # Fallback

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        article_text_raw = item.get('article', '')

        ASPECT_MENTION_MASK_TOKEN = "[ASPECT_MENTION]"  # For <aspect>...</aspect> in article for "masked" method
        ASPECT_NAME_APPEND_TOKEN = "[ASPECT_NAME]" # Token to append if method is "masked"

        processed_article_content = article_text_raw
        token_to_append = ""
        tag_for_sentence_filtering = "<aspect>" # Default for "filtered" method

        if self.method_name == "masked":
            # 1. Mask <aspect>...</aspect> in the article content
            processed_article_content = re.sub(r"<aspect>.*?</aspect>", ASPECT_MENTION_MASK_TOKEN, article_text_raw)
            # 2. Set the token to append later
            token_to_append = ASPECT_NAME_APPEND_TOKEN
            # 3. If filtering sentences for masked content, search for the mask token
            tag_for_sentence_filtering = ASPECT_MENTION_MASK_TOKEN
        # For "whole" and "filtered", processed_article_content remains article_text_raw initially.

        # --- Apply Sentence Filtering (if actual_use_filtered is True) ---
        article_part_for_bge = ""
        if self.actual_use_filtered:
            if processed_article_content: # Ensure there's content to process
                try:
                    doc = self.spacy_nlp(processed_article_content.strip())
                    # Filter sentences based on `tag_for_sentence_filtering`
                    filtered_sentences = [sent.text for sent in doc.sents if tag_for_sentence_filtering in sent.text]
                    if filtered_sentences:
                        article_part_for_bge = ' '.join(filtered_sentences).strip()
                    # else: article_part_for_bge remains ""
                except Exception as e:
                    print(f"Error in SpaCy processing for item {idx} (method: {self.method_name}, filtering: True): {e}")
                    print("Content snippet:", processed_article_content[:self.max_article_preview])
                    article_part_for_bge = processed_article_content.strip() # Fallback to using the (potentially masked) content
            # else: article_part_for_bge remains "" if processed_article_content was initially empty
        else: # Not using filtered sentences
            article_part_for_bge = processed_article_content.strip()

        # --- Final Cleaning/Tag Removal (for "whole" and "filtered" methods) ---
        # This happens *after* potential sentence filtering for "filtered",
        # and on the whole document for "whole".
        # "masked" method does not need this as aspect tags were already replaced.
        if self.method_name == "whole" or self.method_name == "filtered":
            article_part_for_bge = article_part_for_bge.replace("<aspect>", "").replace("</aspect>", "")

        # --- Construct final text for BGE by appending the aspect name token if needed ---
        if token_to_append: # This is non-empty only for "masked" method
            final_text_to_encode = f"{article_part_for_bge} {token_to_append}".strip()
        else:
            final_text_to_encode = article_part_for_bge # .strip() already applied

        # Encode the final text
        try:
            embedding = self.bge_model.encode(final_text_to_encode, normalize_embeddings=True)
        except Exception as e:
            print(f"Error encoding text for item {idx} (method: {self.method_name}): {e}")
            print("Text snippet for BGE:", final_text_to_encode[:self.max_article_preview])
            embedding = np.zeros(self.embedding_dim, dtype=np.float32)

        # Sentiment Label Processing
        sentiment_original = item.get('sentiment', 0)
        sentiment_mapped = sentiment_original + 1
        if sentiment_mapped not in [0, 1, 2]: sentiment_mapped = 1

        return {
            'embeddings': torch.tensor(embedding, dtype=torch.float32),
            'labels': torch.tensor(sentiment_mapped, dtype=torch.long)
        }

# --- Model Definition (MLP Classifier - Unchanged from your provided file) ---
class MLPClassifier(nn.Module):
    def __init__(self, input_dim=BGE_EMBEDDING_DIM, hidden_dim1=512, hidden_dim2=256, output_dim=3, dropout_rate=0.3):
        super(MLPClassifier, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim1))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        if hidden_dim2 > 0:
            layers.append(nn.Linear(hidden_dim1, hidden_dim2))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            layers.append(nn.Linear(hidden_dim2, output_dim))
        else:
             layers.append(nn.Linear(hidden_dim1, output_dim))
        self.classifier = nn.Sequential(*layers)
    def forward(self, embeddings):
        return self.classifier(embeddings)

# --- Evaluation Function (Unchanged from your provided file) ---
def evaluate(model, dataloader, criterion, device):
    model.eval(); total_loss = 0; all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            embeddings, labels = batch['embeddings'].to(device), batch['labels'].to(device)
            outputs = model(embeddings); loss = criterion(outputs, labels)
            total_loss += loss.item(); preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds); all_labels.extend(labels.cpu().numpy())
    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
    target_names_mapped = ["Negative (0)", "Neutral (1)", "Positive (2)"]
    accuracy = accuracy_score(all_labels, all_preds)
    # Using explicit f1_score calls for macro to ensure consistency if needed, though p_r_f_s should be fine
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(all_labels, all_preds, average='micro', zero_division=0)
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)
    qwk = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
    report_dict = classification_report(all_labels, all_preds, target_names=target_names_mapped, zero_division=0, output_dict=True)
    report_str = classification_report(all_labels, all_preds, target_names=target_names_mapped, zero_division=0)
    metrics_results = {
        "loss": avg_loss, "accuracy": accuracy,
        "precision_macro": precision_macro, "recall_macro": recall_macro, "f1_macro": f1_macro,
        "precision_micro": precision_micro, "recall_micro": recall_micro, "f1_micro": f1_micro,
        "precision_weighted": precision_weighted, "recall_weighted": recall_weighted, "f1_weighted": f1_weighted,
        "qwk": qwk, "per_class_report": report_dict
    }
    return avg_loss, metrics_results, report_str, all_preds

# --- Gradient Norm Calculation (Unchanged from your provided file) ---
def get_grad_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None: total_norm += p.grad.data.norm(2).item() ** 2
    return total_norm ** 0.5

# --- Training Function (Corrected val_metrics cleaning for checkpoint) ---
def train(model, train_dataloader, val_dataloader, criterion, optimizer, scheduler, device, epochs, best_model_save_path, run_index):
    best_val_f1_macro = -1.0
    run_metrics = {"train": [], "eval": []}
    print(f"\n--- Starting Training for Run {run_index} ---")
    print(f"Best model (Val Macro F1) will be saved to: {best_model_save_path}")

    for epoch in range(epochs):
        epoch_start_time = time.time(); model.train(); train_loss, total_grad_norm, batch_count = 0, 0, 0
        progress_bar = tqdm(train_dataloader, desc=f"Run {run_index} Epoch {epoch+1}/{epochs} Training", leave=False)
        for batch_idx, batch in enumerate(progress_bar):
            optimizer.zero_grad()
            embeddings, labels = batch['embeddings'].to(device), batch['labels'].to(device)
            outputs = model(embeddings); loss = criterion(outputs, labels)
            if torch.isnan(loss): print(f"NaN loss at run {run_index}, epoch {epoch+1}, batch {batch_idx}. Skip."); optimizer.zero_grad(); continue
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0); grad_norm = get_grad_norm(model)
            total_grad_norm += grad_norm; optimizer.step(); train_loss += loss.item(); batch_count += 1
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}", 'grad_norm': f"{grad_norm:.4f}"})
        
        avg_train_loss = train_loss / batch_count if batch_count > 0 else 0
        avg_grad_norm = total_grad_norm / batch_count if batch_count > 0 else 0
        run_metrics["train"].append({"run_index": run_index, "epoch": epoch + 1, "loss": avg_train_loss, "grad_norm": avg_grad_norm, "learning_rate": optimizer.param_groups[0]['lr'], "timestamp": datetime.datetime.now().isoformat(), "type": "train"})
        
        val_loss, val_metrics_raw, val_report_str, _ = evaluate(model, val_dataloader, criterion, device)
        val_f1_macro = val_metrics_raw["f1_macro"] # Use raw for comparison
        run_metrics["eval"].append({"run_index": run_index, "epoch": epoch + 1, "loss": val_loss, **val_metrics_raw, "timestamp": datetime.datetime.now().isoformat(), "type": "eval"})
        
        epoch_duration = time.time() - epoch_start_time
        print(f"Run {run_index} Epoch {epoch+1}/{epochs} Summary ({epoch_duration:.2f}s): Train Loss: {avg_train_loss:.4f}, Avg Grad Norm: {avg_grad_norm:.4f}, Val Loss: {val_loss:.4f}, Val Macro F1: {val_f1_macro:.4f}, Val Acc: {val_metrics_raw['accuracy']:.4f}")
        
        if val_f1_macro > best_val_f1_macro:
            best_val_f1_macro = val_f1_macro
            try:
                # Clean val_metrics for saving (convert numpy types to standard Python types)
                cleaned_val_metrics = {}
                for k, v in val_metrics_raw.items():
                    if isinstance(v, np.generic):
                        cleaned_val_metrics[k] = v.item()
                    elif k == "per_class_report" and isinstance(v, dict):
                        cleaned_val_metrics[k] = {}
                        for class_label, metrics_dict_or_val in v.items():
                            if isinstance(metrics_dict_or_val, dict): # For class-specific metrics like "Negative (0)"
                                cleaned_val_metrics[k][class_label] = {}
                                for metric_name, metric_value in metrics_dict_or_val.items():
                                    cleaned_val_metrics[k][class_label][metric_name] = metric_value.item() if isinstance(metric_value, np.generic) else metric_value
                            else: # For top-level report items like 'accuracy', 'macro avg', 'weighted avg'
                                cleaned_val_metrics[k][class_label] = metrics_dict_or_val.item() if isinstance(metrics_dict_or_val, np.generic) else metrics_dict_or_val
                    else:
                        cleaned_val_metrics[k] = v # Handles other types like basic floats, ints, or already cleaned dicts

                checkpoint_data = {
                    'epoch': epoch + 1, 'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(),
                    'val_f1_macro': best_val_f1_macro, 'val_metrics': cleaned_val_metrics,
                    'run_index': run_index, 'args': vars(args)
                }
                torch.save(checkpoint_data, best_model_save_path)
                print(f"  Saved new best model (Epoch {epoch+1}, Val Macro F1: {val_f1_macro:.4f})")
            except Exception as e: print(f"Error saving model checkpoint: {e}")
        
        scheduler.step(val_loss)
        if device == torch.device("cuda"): torch.cuda.empty_cache(); gc.collect()
    print(f"--- Finished Training for Run {run_index} --- Best Val Macro F1: {best_val_f1_macro:.4f}")
    return run_metrics, best_model_save_path

# --- Main Execution ---
def main():
    # METHOD_NAME, LANGUAGE, ACTUAL_USE_FILTERED_SENTENCES are defined globally based on args.

    # --- Load SpaCy Model (if needed for the entire run, including for the description) ---
    spacy_nlp_instance = None  # Initialize
    if ACTUAL_USE_FILTERED_SENTENCES: # If this run configuration actually uses filtering
        spacy_nlp_instance = load_spacy_model(LANGUAGE) # Load it here

    print("--- Experiment Setup ---")
    print(f"Method Name: {METHOD_NAME}")
    print(f"Language Split: {LANGUAGE}")
    
    # Create a dummy dataset instance to get the input_type_desc
    # Pass the spacy_nlp_instance (which will be the loaded model if needed, or None otherwise)
    dummy_dataset_for_desc = SentimentDataset(
        data=[], bge_model=None, device=None, # Dummy values not used for description
        method_name_arg=METHOD_NAME,
        actual_use_filtered_arg=ACTUAL_USE_FILTERED_SENTENCES,
        spacy_nlp_arg=spacy_nlp_instance # <<< CRITICAL: Pass the instance here
    )
    print(f"Input Processing: {dummy_dataset_for_desc.input_type_desc.split('->')[0].strip()}")
    del dummy_dataset_for_desc
    
    print(f"Output Directory: {MODEL_SAVE_DIR}")
    print(f"Base Data Directory: {BASE_DATA_DIR}")
    print(f"MLP Hidden Dims: {HIDDEN_DIM1} / {HIDDEN_DIM2 if HIDDEN_DIM2 > 0 else 'None'}")
    print(f"MLP Dropout: {DROPOUT_RATE}")

    if args.test: print("\n*** RUNNING IN TEST-ONLY MODE ***")
    else:
        print(f"Epochs per run: {EPOCHS}"); print(f"Batch Size: {BATCH_SIZE}"); print(f"Learning Rate: {LR}"); print(f"Weight Decay: {WEIGHT_DECAY}")
        print("\n--- MLP Tuning Reminder ---"); print("Key hyperparameters: --lr, --hidden_dim1/2, --dropout_rate, --weight_decay, --batch_size\n")

    run_args_dict = vars(args).copy() # Use a copy to avoid modifying original args
    run_args_dict['actual_use_filtered_sentences'] = ACTUAL_USE_FILTERED_SENTENCES
    run_args_dict['bge_model_name'] = BGE_MODEL_NAME; run_args_dict['bge_embedding_dim'] = BGE_EMBEDDING_DIM
    run_args_dict['model_save_dir'] = MODEL_SAVE_DIR # This is already correct

    print(f"Loading BGE model: {BGE_MODEL_NAME}...")
    start_load_time = time.time()
    try:
        bge_model = SentenceTransformer(BGE_MODEL_NAME, device=device)
        print(f"BGE model loaded to {device} in {time.time() - start_load_time:.2f} seconds.")
        if device == torch.device("cuda"): print(f"Available VRAM after BGE: {torch.cuda.mem_get_info()[0] / (1024**3):.2f} GB")
    except Exception as e: print(f"Error loading BGE model: {e}"); exit(1)

    spacy_nlp = None
    if ACTUAL_USE_FILTERED_SENTENCES: spacy_nlp = load_spacy_model(LANGUAGE)

    all_test_metrics = {}; best_model_paths = {}
    num_workers = min(4, os.cpu_count() // 2 if os.cpu_count() else 1) # Default to 1 if os.cpu_count() is None or 0
    if num_workers < 1: num_workers = 1
    print(f"Using {num_workers} dataloader workers.")

    if not args.test:
        print("\n===== Starting Training Phase =====")
        for i in range(3): # 3 runs for train/val splits
            print(f"\n===== Starting Run {i} =====")
            current_best_model_path = os.path.join(MODEL_SAVE_DIR, f"best_model_{i}.pt")
            metrics_file_path = os.path.join(MODEL_SAVE_DIR, f"training_metrics_{i}.json")
            train_data_raw, val_data_raw = load_split_data(BASE_DATA_DIR, LANGUAGE, split_index=i)
            
            # Pass all necessary args to SentimentDataset
            train_dataset = SentimentDataset(train_data_raw, bge_model, device, METHOD_NAME, ACTUAL_USE_FILTERED_SENTENCES, spacy_nlp)
            val_dataset = SentimentDataset(val_data_raw, bge_model, device, METHOD_NAME, ACTUAL_USE_FILTERED_SENTENCES, spacy_nlp)

            if i == 0 and train_dataset.data: # Print sample only for the first run
                print(f"\nSample data item structure (Input Type: '{train_dataset.input_type_desc}'):")
                sample_item = train_dataset[0] # Get one processed item
                print(f"  embedding shape: {sample_item['embeddings'].shape}, label: {sample_item['labels'].item()}\n")

            persistent = num_workers > 0
            try:
                train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=True, persistent_workers=persistent)
                val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=persistent)
            except TypeError: # Older PyTorch might not support persistent_workers
                 print("Warning: Persistent workers not supported for DataLoader. Disabling.")
                 train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=True)
                 val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=True)

            model = MLPClassifier(BGE_EMBEDDING_DIM, HIDDEN_DIM1, HIDDEN_DIM2, 3, DROPOUT_RATE).to(device)
            criterion = nn.CrossEntropyLoss().to(device)
            optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=2, verbose=True)
            
            run_metrics, saved_model_path = train(model, train_dataloader, val_dataloader, criterion, optimizer, scheduler, device, EPOCHS, current_best_model_path, i)
            
            run_metrics_to_save = {"arguments": run_args_dict, "run_index": i, "train_metrics": run_metrics["train"], "eval_metrics": run_metrics["eval"], "best_model_path": saved_model_path if saved_model_path and os.path.exists(saved_model_path) else None}
            try:
                with open(metrics_file_path, 'w', encoding='utf-8') as f: json.dump(run_metrics_to_save, f, indent=4, default=str) # default=str for numpy types if any sneak in
                print(f"Training metrics for run {i} saved to: {metrics_file_path}")
            except Exception as e: print(f"Error saving metrics for run {i}: {e}")
            
            best_model_paths[i] = saved_model_path if saved_model_path and os.path.exists(saved_model_path) else None
            del model, optimizer, scheduler, train_dataloader, val_dataloader, train_dataset, val_dataset, train_data_raw, val_data_raw, run_metrics; gc.collect()
            if device == torch.device("cuda"): torch.cuda.empty_cache()
            print(f"===== Finished Run {i} =====")
        print("\n===== Finished Training Phase =====")
    else: # Test Only Mode: Populate best_model_paths
        print("\n===== Test Only Mode: Locating existing models =====")
        found_any_model = False
        for i in range(3):
            path = os.path.join(MODEL_SAVE_DIR, f"best_model_{i}.pt")
            if os.path.exists(path):
                best_model_paths[i] = path
                print(f"Found model for run {i}: {path}")
                found_any_model = True
            else:
                best_model_paths[i] = None
                print(f"Warning: Model not found for run {i}: {path}")
        if not found_any_model:
            print(f"Error: No 'best_model_X.pt' files found in {MODEL_SAVE_DIR}.")
            print("Cannot proceed with testing. Ensure --method_name and --split are correct and training was completed for this method.")
            exit(1)

    # === Testing Phase ===
    print("\n===== Starting Testing Phase =====")
    test_data_raw = load_split_data(BASE_DATA_DIR, LANGUAGE, split_index=None)
    test_dataset = SentimentDataset(test_data_raw, bge_model, device, METHOD_NAME, ACTUAL_USE_FILTERED_SENTENCES, spacy_nlp)
    test_batch_size = BATCH_SIZE * 2
    persistent_test = num_workers > 0
    try:
        test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=persistent_test)
    except TypeError: test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    criterion_test = nn.CrossEntropyLoss().to(device)
    test_results_summary = []

    for i in range(3): # Iterate through the 3 potential models
        print(f"\n--- Evaluating Best Model from Run {i} on Test Set ---")
        model_path = best_model_paths.get(i) # Get path from dict (populated either during training or --test mode)
        if not model_path:
            print(f"Warning: No model path found for run {i} (model might not have been saved or found). Skipping.")
            all_test_metrics[f"model_{i}"] = {"error": "Model path not found or model was not saved/found."}
            continue

        # Model architecture must match the saved one.
        # Assumes args.hidden_dim1, args.hidden_dim2, args.dropout_rate define this.
        model_test = MLPClassifier(BGE_EMBEDDING_DIM, HIDDEN_DIM1, HIDDEN_DIM2, 3, DROPOUT_RATE).to(device)
        try:
            print(f"Loading checkpoint from: {model_path}")
            checkpoint = torch.load(model_path, map_location=device, weights_only=False) # weights_only=False to load dicts
            
            # Check if 'args' is in checkpoint to potentially load MLP architecture
            # This makes testing more robust if models were trained with different MLP params
            ckpt_args = checkpoint.get('args')
            if ckpt_args:
                print("  Checkpoint contains args. Re-initializing MLP if different from current test args.")
                model_test = MLPClassifier(
                    input_dim=BGE_EMBEDDING_DIM,
                    hidden_dim1=ckpt_args.get('hidden_dim1', HIDDEN_DIM1), # Use checkpoint's, fallback to current
                    hidden_dim2=ckpt_args.get('hidden_dim2', HIDDEN_DIM2),
                    output_dim=3,
                    dropout_rate=ckpt_args.get('dropout_rate', DROPOUT_RATE)
                ).to(device)

            model_test.load_state_dict(checkpoint['model_state_dict'])
            print(f"  Successfully loaded model state from epoch {checkpoint.get('epoch', 'N/A')}")
            model_test.eval()
        except FileNotFoundError:
             print(f"Error: Model file not found at {model_path}")
             all_test_metrics[f"model_{i}"] = {"error": f"File not found: {model_path}"}
             del model_test; gc.collect();
             if device == torch.device("cuda"): torch.cuda.empty_cache()
             continue
        except KeyError as e:
             print(f"Error: Missing key in checkpoint file {model_path}: {e}. Checkpoint might be corrupted or saved incorrectly.")
             all_test_metrics[f"model_{i}"] = {"error": f"KeyError loading state: {e}"}
             del model_test; gc.collect();
             if device == torch.device("cuda"): torch.cuda.empty_cache()
             continue
        except Exception as e:
            print(f"Error loading model state from {model_path}: {e}")
            all_test_metrics[f"model_{i}"] = {"error": f"Failed loading state: {e}"}
            del model_test; gc.collect();
            if device == torch.device("cuda"): torch.cuda.empty_cache()
            continue

        test_loss, test_metrics_dict, test_report_str, test_preds_mapped = evaluate(model_test, test_dataloader, criterion_test, device)
        print(f"Test Results Model {i}: Loss={test_loss:.4f}, Macro F1={test_metrics_dict['f1_macro']:.4f}, Acc={test_metrics_dict['accuracy']:.4f}, QWK={test_metrics_dict['qwk']:.4f}")
        print("Full Classification Report (Test Set):\n", test_report_str)
        all_test_metrics[f"model_{i}"] = {"model_run_index": i, "model_path": model_path, "test_loss": test_loss, **test_metrics_dict}
        test_results_summary.append({"f1_macro": test_metrics_dict['f1_macro'], "accuracy": test_metrics_dict['accuracy'], "qwk": test_metrics_dict['qwk']})

        preds_file_path = os.path.join(MODEL_SAVE_DIR, f"test_predictions_{i}.json")
        preds_orig_scale = [int(p - 1) for p in test_preds_mapped] # ensure python int for JSON
        if len(preds_orig_scale) == len(test_data_raw):
            test_data_cp = copy.deepcopy(test_data_raw)
            for idx_pred, item_pred in enumerate(test_data_cp): item_pred['prediction'] = preds_orig_scale[idx_pred]
            try:
                with open(preds_file_path, 'w', encoding='utf-8') as f: json.dump(test_data_cp, f, indent=4, ensure_ascii=False)
                print(f"Test predictions for model {i} saved to: {preds_file_path}")
            except Exception as e: print(f"Error saving test predictions for model {i}: {e}")
        else:
             print(f"Error: Mismatch between #predictions ({len(preds_orig_scale)}) & #test items ({len(test_data_raw)}). Predictions not saved for model {i}.")
             all_test_metrics[f"model_{i}"]["prediction_error"] = "Prediction count mismatch."
        
        del model_test, checkpoint; gc.collect()
        if device == torch.device("cuda"): torch.cuda.empty_cache()

    # === Final Reporting ===
    if test_results_summary:
        valid_f1 = [r['f1_macro'] for r in test_results_summary if 'f1_macro' in r and r['f1_macro'] is not None]
        valid_acc = [r['accuracy'] for r in test_results_summary if 'accuracy' in r and r['accuracy'] is not None]
        valid_qwk = [r['qwk'] for r in test_results_summary if 'qwk' in r and r['qwk'] is not None]

        avg_test_f1_macro = np.mean(valid_f1) if valid_f1 else 0.0
        avg_test_accuracy = np.mean(valid_acc) if valid_acc else 0.0
        avg_test_qwk = np.mean(valid_qwk) if valid_qwk else 0.0

        print("\n--- Average Test Set Performance Across Successfully Evaluated Models ---")
        print(f"Avg Macro F1: {avg_test_f1_macro:.4f} (from {len(valid_f1)} models)")
        print(f"Avg Accuracy: {avg_test_accuracy:.4f} (from {len(valid_acc)} models)")
        print(f"Avg QWK: {avg_test_qwk:.4f} (from {len(valid_qwk)} models)")
        all_test_metrics["average_performance"] = {
            "f1_macro": avg_test_f1_macro, "accuracy": avg_test_accuracy, "qwk": avg_test_qwk,
            "num_models_averaged": len(valid_f1)
        }
    else:
        print("\nNo models were successfully evaluated on the test set.")
        all_test_metrics["average_performance"] = {"error": "No models available or evaluation failed."}

    combined_test_metrics_file = os.path.join(MODEL_SAVE_DIR, "test_metrics_summary.json")
    try:
        with open(combined_test_metrics_file, 'w', encoding='utf-8') as f:
            json.dump(all_test_metrics, f, indent=4, default=str) # default=str handles numpy types
        print(f"\nCombined test metrics summary saved to: {combined_test_metrics_file}")
    except Exception as e: print(f"Error saving combined test metrics: {e}")

    # Clean up shared resources before exiting
    del bge_model # Ensure large model is deleted
    if spacy_nlp is not None: del spacy_nlp
    if 'test_dataset' in locals(): del test_dataset # Check if defined before deleting
    if 'test_dataloader' in locals(): del test_dataloader
    gc.collect()
    if device == torch.device("cuda"): torch.cuda.empty_cache()

    print("\nExperiment Complete.")
    print(f"Outputs saved in: {MODEL_SAVE_DIR}")


if __name__ == "__main__":
    try:
        mp.set_start_method('spawn', force=True)
        print("Multiprocessing start method set to 'spawn'.")
    except RuntimeError as e:
        if "context has already been set" not in str(e).lower(): # Check if context not already set
            print(f"Warning: Could not set multiprocessing start method to 'spawn': {e}. CUDA + DataLoader workers might cause issues.")
        else:
            print("Warning: Multiprocessing context already set. Assuming 'spawn' or compatible.")

    # Check essential libraries
    try: import sentence_transformers
    except ImportError: print("Error: sentence-transformers not found. Please install: pip install sentence-transformers"); exit(1)
    try: import sklearn
    except ImportError: print("Error: scikit-learn not found. Please install: pip install scikit-learn"); exit(1)
    # SpaCy availability is checked near its usage.

    main()