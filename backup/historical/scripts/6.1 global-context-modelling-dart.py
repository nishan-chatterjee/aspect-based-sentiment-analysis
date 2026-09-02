# -*- coding: utf-8 -*-
import os
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
# from torch.cuda.amp import autocast, GradScaler # Old import
import argparse
import datetime
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, logging as hf_logging
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    cohen_kappa_score,
)
from tqdm.auto import tqdm
import time
import gc
import copy
import math
import functools
import warnings

# Suppress unnecessary Hugging Face warnings
hf_logging.set_verbosity_error()

# Suppress warnings from Thinc (used by SpaCy) and other potential FutureWarning from torch.load
warnings.filterwarnings("ignore", category=FutureWarning, module="thinc.shims.pytorch")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*`torch.load` with `weights_only=False`.*")


# --- SpaCy Global Variable for Workers ---
worker_spacy_nlp = None

# --- SpaCy Import and Setup ---
SPACY_MODELS = {}

def load_spacy_model(language):
    lang_key = language.lower()
    if lang_key in SPACY_MODELS and SPACY_MODELS[lang_key] is not None: # Check if already loaded in this process
        return SPACY_MODELS[lang_key]

    print(f"Process {os.getpid()}: Loading SpaCy model for language: {language}...")
    model_name = None
    loader_func = None

    if lang_key == "english":
        model_name = "en_core_web_sm"
        try: import en_core_web_sm; loader_func = en_core_web_sm.load
        except ImportError: print(f"Warning (Process {os.getpid()}): SpaCy model '{model_name}' not installed via package. Trying spacy.load().\nInstall with: python -m spacy download {model_name}")
    elif lang_key == "slovenian":
        model_name = "sl_core_news_sm"
        try: import sl_core_news_sm; loader_func = sl_core_news_sm.load
        except ImportError: print(f"Warning (Process {os.getpid()}): SpaCy model '{model_name}' not installed via package. Trying spacy.load().\nInstall with: python -m spacy download {model_name}")
    elif lang_key in ["croatian", "serbian"]:
        model_name = "hr_core_news_sm"
        try: import hr_core_news_sm; loader_func = hr_core_news_sm.load
        except ImportError: print(f"Warning (Process {os.getpid()}): SpaCy model '{model_name}' not installed via package. Trying spacy.load().\nInstall with: python -m spacy download {model_name}")
    else:
        print(f"Warning (Process {os.getpid()}): Unsupported language '{language}'. Defaulting to 'en_core_web_sm'.")
        model_name = "en_core_web_sm"
        try: import en_core_web_sm; loader_func = en_core_web_sm.load
        except ImportError: print(f"Warning (Process {os.getpid()}): SpaCy model '{model_name}' not installed via package. Trying spacy.load().\nInstall with: python -m spacy download {model_name}")

    nlp = None
    try:
        if loader_func:
            nlp = loader_func()
        else:
            import spacy
            nlp = spacy.load(model_name)
        
        # Ensure sentencizer is present
        if 'sentencizer' not in nlp.pipe_names and 'senter' not in nlp.pipe_names:
             print(f"Warning (Process {os.getpid()}): Sentencizer pipe not found in '{model_name}' default pipes. Attempting to add.")
             try:
                 nlp.add_pipe('sentencizer', first=True)
             except ValueError as e: # Handle if already exists implicitly
                 if "already exists in pipeline" in str(e):
                     print(f"Info (Process {os.getpid()}): Sentencizer implicitly present, proceeding.")
                 else:
                     raise e # Re-raise other ValueErrors
        
        print(f"Process {os.getpid()}: SpaCy model '{model_name}' loaded successfully with pipes: {nlp.pipe_names}.")
        SPACY_MODELS[lang_key] = nlp
        return nlp
    except ImportError:
         print(f"Error (Process {os.getpid()}): SpaCy library not found. Please install it: pip install spacy")
         raise RuntimeError("SpaCy library not found in worker process.")
    except OSError:
        print(f"Error (Process {os.getpid()}): SpaCy model '{model_name}' not found or downloadable.")
        print(f"Please ensure it's installed: python -m spacy download {model_name}")
        raise RuntimeError(f"SpaCy model '{model_name}' not found in worker process.")
    except Exception as e:
        print(f"An unexpected error occurred loading SpaCy model '{model_name}' in process {os.getpid()}: {e}")
        raise RuntimeError(f"SpaCy model loading failed in worker process: {e}")

def worker_init_spacy(worker_id, language_for_worker):
    global worker_spacy_nlp
    print(f"Initializing SpaCy for worker {worker_id} (PID: {os.getpid()}, Language: {language_for_worker})...")
    try:
        worker_spacy_nlp = load_spacy_model(language_for_worker)
        if worker_spacy_nlp is None:
            print(f"Error: Failed to load SpaCy model in worker {worker_id} (PID: {os.getpid()}). Exiting worker.")
            exit(1) 
        print(f"SpaCy initialized successfully for worker {worker_id} (PID: {os.getpid()}).")
    except Exception as e:
        print(f"CRITICAL ERROR during SpaCy initialization in worker {worker_id} (PID: {os.getpid()}): {e}")
        exit(1)

def split_document(raw_text, nlp_model):
    if nlp_model is None:
        print(f"Error (PID {os.getpid()}): SpaCy model is None in split_document. Worker likely failed initialization.")
        raise RuntimeError("SpaCy model not available in worker process.")
    if not raw_text or not isinstance(raw_text, str):
        return []
    try:
        doc = nlp_model(raw_text.strip())
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        return sentences
    except Exception as e:
        print(f"Error processing text in worker (PID: {os.getpid()}) with SpaCy: {e}")
        print(f"Problematic text snippet: {raw_text[:200]}...")
        return []


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1) 
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (seq_len, batch_size, d_model)
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

parser = argparse.ArgumentParser(description="Train a DART-style classifier on document sentences.")
parser.add_argument("--split", type=str, required=True, choices=['slovenian', 'serbian'], help="Language split.")
parser.add_argument("--method_name", type=str, required=True, help="Name for model/method directory (e.g., 'dart_xlmr').")
parser.add_argument("--model_name", type=str, default="xlm-roberta-base", help="Pre-trained transformer model name (Hugging Face).")
parser.add_argument("--max_seq_length", type=int, default=128, help="Max token sequence length for EACH sentence encoding.")
parser.add_argument("--max_sentences", type=int, default=64, help="Max number of sentences per document (truncates longer docs).")
parser.add_argument("--interaction_layers", type=int, default=2, help="Number of Transformer layers for sentence interaction.")
parser.add_argument("--interaction_heads", type=int, default=8, help="Number of attention heads in sentence interaction layers.")
parser.add_argument("--final_mlp_hidden_dim", type=int, default=256, help="Hidden dimension for the final classification MLP.")
parser.add_argument("--dropout_rate", type=float, default=0.2, help="Dropout rate for interaction layers and MLP.")
parser.add_argument("--aggregation_method", type=str, default="attention", choices=["attention", "mean", "max"], help="Method to aggregate sentence embeddings.")
parser.add_argument("--use_aspect_marker", action='store_true', help="Prepend aspect text to each sentence before encoding.")
parser.add_argument("--freeze_base_model", action='store_true', help="Freeze the weights of the base transformer model during training.")
parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
parser.add_argument("--batch_size", type=int, default=16, help="Batch size (adjust based on GPU memory).")
parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate (typical for fine-tuning transformers).")
parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay (L2 regularization).")
parser.add_argument("--test_only", action='store_true', help="Run test evaluation only using existing models.") # Renamed from --test
args = parser.parse_args()

LANGUAGE = args.split
METHOD_NAME = args.method_name
MODEL_NAME = args.model_name
MAX_SEQ_LENGTH = args.max_seq_length
MAX_SENTENCES = args.max_sentences
INTERACTION_LAYERS = args.interaction_layers
INTERACTION_HEADS = args.interaction_heads
FINAL_MLP_HIDDEN_DIM = args.final_mlp_hidden_dim
DROPOUT_RATE = args.dropout_rate
AGGREGATION_METHOD = args.aggregation_method
USE_ASPECT_MARKER = args.use_aspect_marker
FREEZE_BASE_MODEL = args.freeze_base_model
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
LR = args.lr
WEIGHT_DECAY = args.weight_decay

BASE_DATA_DIR = "../data/final/complete"
MODEL_SAVE_DIR = f"../models/{METHOD_NAME}/{LANGUAGE}"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device == torch.device("cuda"):
    print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
    try:
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        # available_vram_gb = torch.cuda.mem_get_info()[0] / (1024**3) # mem_get_info might not be available on all setups
        print(f"Total VRAM: {total_vram_gb:.2f} GB")
        # print(f"Available VRAM: {available_vram_gb:.2f} GB") # Commented out for broader compatibility
    except Exception as e:
        print(f"Could not get VRAM info: {e}")


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

class SentimentSentenceDataset(Dataset):
    def __init__(self, data, tokenizer, language, max_seq_length, max_sentences, use_aspect_marker=False):
        self.data = data
        self.tokenizer = tokenizer
        self.language = language
        self.max_seq_length = max_seq_length
        self.max_sentences = max_sentences
        self.use_aspect_marker = use_aspect_marker
        self.aspect_tag = "<aspect>"
        self.aspect_end_tag = "</aspect>"

    def __len__(self):
        return len(self.data)

    def _extract_aspect(self, text):
        start = text.find(self.aspect_tag)
        end = text.find(self.aspect_end_tag)
        if start != -1 and end != -1:
            if start + len(self.aspect_tag) <= end:
                 return text[start + len(self.aspect_tag):end].strip()
            else: 
                 print(f"Warning: Malformed aspect tags found: {text[start:end+len(self.aspect_end_tag)]}")
                 return ""
        return ""

    def __getitem__(self, idx):
        global worker_spacy_nlp
        if worker_spacy_nlp is None:
             raise RuntimeError(f"SpaCy model not initialized in worker (PID {os.getpid()}) for language {self.language}. Check worker_init_fn.")

        item = self.data[idx]
        raw_article = item.get('article', '')
        aspect_text = self._extract_aspect(raw_article)
        cleaned_article = raw_article.replace(self.aspect_tag, "").replace(self.aspect_end_tag, "").strip()

        try:
            sentences = split_document(cleaned_article, worker_spacy_nlp)
        except Exception as e:
             print(f"Error during split_document in worker (PID: {os.getpid()}) for item {idx}: {e}")
             sentences = []

        if len(sentences) > self.max_sentences:
            sentences = sentences[:self.max_sentences]
        
        all_input_ids = []
        all_attention_masks = []

        for sentence in sentences:
            text_to_encode = sentence
            if self.use_aspect_marker and aspect_text:
                text_to_encode = f"{aspect_text} {self.tokenizer.sep_token} {sentence}"
            try:
                encoding = self.tokenizer.encode_plus(
                    text_to_encode, add_special_tokens=True, max_length=self.max_seq_length,
                    padding='max_length', truncation=True, return_attention_mask=True, return_tensors='pt',
                )
                all_input_ids.append(encoding['input_ids'].squeeze(0))
                all_attention_masks.append(encoding['attention_mask'].squeeze(0))
            except Exception as e:
                 print(f"Error during tokenization in worker (PID: {os.getpid()}) for item {idx}, sentence: '{sentence[:50]}...': {e}")
                 continue
        
        num_sentences_processed = len(all_input_ids)
        sentence_padding_count = self.max_sentences - num_sentences_processed
        
        if sentence_padding_count > 0:
            pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
            pad_ids = torch.full((self.max_seq_length,), pad_token_id, dtype=torch.long)
            pad_mask = torch.zeros((self.max_seq_length,), dtype=torch.long)
            for _ in range(sentence_padding_count):
                all_input_ids.append(pad_ids)
                all_attention_masks.append(pad_mask)

        if not all_input_ids: # Should only happen if sentences list was empty AND tokenization failed for all
             pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
             all_input_ids = [torch.full((self.max_seq_length,), pad_token_id, dtype=torch.long)] * self.max_sentences
             all_attention_masks = [torch.zeros((self.max_seq_length,), dtype=torch.long)] * self.max_sentences
             num_sentences_processed = 0 # Ensure this is 0 if we fall back to full padding

        input_ids = torch.stack(all_input_ids)
        attention_mask = torch.stack(all_attention_masks)
        sentence_mask = torch.zeros(self.max_sentences, dtype=torch.long)
        if num_sentences_processed > 0:
             sentence_mask[:num_sentences_processed] = 1

        sentiment_original = item.get('sentiment', 0)
        sentiment_mapped = sentiment_original + 1 
        if sentiment_mapped not in [0, 1, 2]: sentiment_mapped = 1

        return {
            'input_ids': input_ids, 'attention_mask': attention_mask,
            'sentence_mask': sentence_mask, 'labels': torch.tensor(sentiment_mapped, dtype=torch.long)
        }

class DARTStyleModel(nn.Module):
    def __init__(self, model_name, interaction_layers, interaction_heads,
                 final_mlp_hidden_dim, dropout_rate, aggregation_method, num_classes=3, freeze_base=False):
        super().__init__()
        self.base_model = AutoModel.from_pretrained(model_name)
        self.config = self.base_model.config
        self.hidden_dim = self.config.hidden_size
        self.aggregation_method = aggregation_method
        self.dropout = nn.Dropout(dropout_rate)

        if freeze_base:
            print("Freezing base transformer model weights.")
            for param in self.base_model.parameters():
                param.requires_grad = False
        
        # Positional encoder for sentence embeddings (if used, apply before sentence_transformer)
        # This is defined but not explicitly used in the forward pass provided in the original script.
        # If it's intended to be used, it should be applied to sentence_embeddings.
        # For now, keeping it as defined, as its presence might be what `load_state_dict` expects.
        self.sentence_pos_encoder = PositionalEncoding(self.hidden_dim, dropout_rate)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim, nhead=interaction_heads,
            dim_feedforward=self.hidden_dim * 4, dropout=dropout_rate,
            activation='relu', batch_first=True
        )
        self.sentence_transformer = nn.TransformerEncoder(encoder_layer, num_layers=interaction_layers)

        if self.aggregation_method == 'attention':
            self.attention_query = nn.Parameter(torch.randn(1, self.hidden_dim))
            self.aggregation_attention = nn.MultiheadAttention(
                embed_dim=self.hidden_dim, num_heads=interaction_heads,
                dropout=dropout_rate, batch_first=True
            )

        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, final_mlp_hidden_dim), nn.ReLU(),
            nn.Dropout(dropout_rate), nn.Linear(final_mlp_hidden_dim, num_classes)
        )

    def forward(self, input_ids, attention_mask, sentence_mask):
        batch_size, num_sentences, seq_length = input_ids.shape
        input_ids_flat = input_ids.view(-1, seq_length)
        attention_mask_flat = attention_mask.view(-1, seq_length)

        outputs = self.base_model(input_ids=input_ids_flat, attention_mask=attention_mask_flat)
        sentence_embeddings_flat = outputs.last_hidden_state[:, 0, :] 
        sentence_embeddings = sentence_embeddings_flat.view(batch_size, num_sentences, self.hidden_dim)
        sentence_embeddings = self.dropout(sentence_embeddings)

        # Optional: Apply sentence_pos_encoder here if intended
        # sentence_embeddings_p = sentence_embeddings.permute(1,0,2) # num_sentences, batch_size, hidden_dim
        # sentence_embeddings_p = self.sentence_pos_encoder(sentence_embeddings_p)
        # sentence_embeddings = sentence_embeddings_p.permute(1,0,2) # batch_size, num_sentences, hidden_dim

        sentence_padding_mask = (sentence_mask == 0) 
        contextualized_sentences = self.sentence_transformer(
            sentence_embeddings, src_key_padding_mask=sentence_padding_mask
        )
        
        expanded_sentence_mask = sentence_mask.unsqueeze(-1).float()
        contextualized_sentences = contextualized_sentences * expanded_sentence_mask # Mask out padding sentences

        aggregated_representation = None
        if self.aggregation_method == 'mean':
            sum_embeddings = torch.sum(contextualized_sentences, dim=1)
            num_real_sentences = torch.sum(sentence_mask, dim=1, keepdim=True).clamp(min=1)
            aggregated_representation = sum_embeddings / num_real_sentences
        elif self.aggregation_method == 'max':
             masked_for_max = contextualized_sentences.masked_fill(expanded_sentence_mask == 0, -torch.inf)
             aggregated_representation = torch.max(masked_for_max, dim=1)[0]
             aggregated_representation = aggregated_representation.masked_fill(aggregated_representation == -torch.inf, 0.0) # Reset -inf to 0
        elif self.aggregation_method == 'attention':
            query = self.attention_query.unsqueeze(0).repeat(batch_size, 1, 1)
            attn_output, _ = self.aggregation_attention(
                query=query, key=contextualized_sentences, value=contextualized_sentences,
                key_padding_mask=sentence_padding_mask 
            )
            aggregated_representation = attn_output.squeeze(1)
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")

        aggregated_representation = self.dropout(aggregated_representation)
        logits = self.classifier(aggregated_representation)
        return logits

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    scaler_enabled = (device == torch.device("cuda"))

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            sentence_mask = batch['sentence_mask'].to(device)
            labels = batch['labels'].to(device)

            with torch.amp.autocast(device_type=str(device).split(':')[0], dtype=torch.float16, enabled=scaler_enabled):      
                logits = model(input_ids, attention_mask, sentence_mask)
                loss = criterion(logits, labels)

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
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
    try:
        report_dict = classification_report(
            all_labels, all_preds, target_names=target_names_mapped, zero_division=0, output_dict=True)
        report_str = classification_report(
            all_labels, all_preds, target_names=target_names_mapped, zero_division=0)
    except Exception as e:
        print(f"Warning: classification_report failed: {e}")
        report_dict = {"error": str(e)}
        report_str = f"Classification report failed: {e}"

    metrics_results = {
        "loss": avg_loss, "accuracy": accuracy,
        "precision_macro": precision_macro, "recall_macro": recall_macro, "f1_macro": f1_macro,
        "precision_micro": precision_micro, "recall_micro": recall_micro, "f1_micro": f1_micro,
        "precision_weighted": precision_weighted, "recall_weighted": recall_weighted, "f1_weighted": f1_weighted,
        "qwk": qwk, "per_class_report": report_dict
    }
    return avg_loss, metrics_results, report_str, all_preds

def get_grad_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

def train(model, train_dataloader, val_dataloader, criterion, optimizer, scheduler, device, epochs,
          best_model_save_path, run_index, accumulation_steps=4):
    best_val_f1_macro = -1.0
    run_metrics = {"train": [], "eval": []}
    scaler = torch.amp.GradScaler(enabled=(device == torch.device("cuda")))

    print(f"\n--- Starting Training for Run {run_index} ---")
    print(f"Gradient Accumulation Steps: {accumulation_steps}")
    print(f"Automatic Mixed Precision (AMP): {'Enabled' if scaler.is_enabled() else 'Disabled'}")
    print(f"Effective Batch Size: {args.batch_size * accumulation_steps}") # Use args.batch_size
    print(f"Best model (based on Val Macro F1) will be saved to: {best_model_save_path}")

    global_step = 0
    for epoch in range(epochs):
        epoch_start_time = time.time()
        model.train()
        train_loss, total_grad_norm, batch_count = 0, 0, 0
        accumulated_loss = 0.0
        optimizer.zero_grad() 

        progress_bar = tqdm(train_dataloader, desc=f"Run {run_index} Epoch {epoch+1}/{epochs} Training", leave=False)
        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            sentence_mask = batch['sentence_mask'].to(device)
            labels = batch['labels'].to(device)

            with torch.amp.autocast(device_type=str(device).split(':')[0], dtype=torch.float16, enabled=scaler.is_enabled()):
                logits = model(input_ids, attention_mask, sentence_mask)
                loss = criterion(logits, labels)
                loss = loss / accumulation_steps

            if torch.isnan(loss):
                print(f"Warning: NaN loss encountered at run {run_index}, epoch {epoch+1}, batch {batch_idx}. Skipping accumulation for this batch.")
                # loss.backward() might fail, so skip scaling and backward for this specific NaN loss
                # but still need to handle optimizer step at accumulation boundary
                if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_dataloader):
                    if accumulated_loss > 0 : # Only step if some valid loss was accumulated
                         scaler.unscale_(optimizer)
                         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                         scaler.step(optimizer)
                         scaler.update()
                    optimizer.zero_grad()
                    accumulated_loss = 0.0 # Reset for next cycle
                continue # Skip to next batch

            scaler.scale(loss).backward()
            accumulated_loss += loss.item() * accumulation_steps # Track original scale loss

            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_dataloader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                grad_norm = get_grad_norm(model)
                total_grad_norm += grad_norm if not np.isnan(grad_norm) else 0
                train_loss += accumulated_loss # Add the accumulated loss for this step
                batch_count += 1 # This counts optimizer steps, not raw batches
                progress_bar.set_postfix({'loss': f"{accumulated_loss:.4f}", 'grad_norm': f"{grad_norm:.4f}"})
                accumulated_loss = 0.0
                global_step += 1
        
        avg_train_loss = train_loss / batch_count if batch_count > 0 else 0
        avg_grad_norm = total_grad_norm / batch_count if batch_count > 0 else 0
        current_lr = optimizer.param_groups[0]['lr']
        run_metrics["train"].append({
            "run_index": run_index, "epoch": epoch + 1, "loss": avg_train_loss,
            "grad_norm": avg_grad_norm, "learning_rate": current_lr,
            "timestamp": datetime.datetime.now().isoformat(), "type": "train"
        })

        model.eval()
        val_loss_eval, val_metrics, val_report_str, _ = evaluate(model, val_dataloader, criterion, device)
        val_f1_macro = val_metrics["f1_macro"]
        
        # Make val_metrics JSON serializable before appending
        serializable_val_metrics = {}
        for k, v in val_metrics.items():
            if isinstance(v, dict): # Handle per_class_report
                serializable_val_metrics[k] = {}
                for k_inner, v_inner in v.items():
                    if isinstance(v_inner, dict):
                        serializable_val_metrics[k][k_inner] = {}
                        for k_innermost, v_innermost in v_inner.items():
                            if isinstance(v_innermost, np.generic):
                                serializable_val_metrics[k][k_inner][k_innermost] = v_innermost.item()
                            else:
                                serializable_val_metrics[k][k_inner][k_innermost] = v_innermost
                    elif isinstance(v_inner, np.generic):
                        serializable_val_metrics[k][k_inner] = v_inner.item()
                    else:
                        serializable_val_metrics[k][k_inner] = v_inner
            elif isinstance(v, np.generic):
                serializable_val_metrics[k] = v.item()
            else:
                serializable_val_metrics[k] = v

        run_metrics["eval"].append({
            "run_index": run_index, "epoch": epoch + 1, "loss": val_loss_eval,
            **serializable_val_metrics, "timestamp": datetime.datetime.now().isoformat(), "type": "eval"
        })

        epoch_duration = time.time() - epoch_start_time
        print(f"Run {run_index} Epoch {epoch+1}/{epochs} Summary ({epoch_duration:.2f}s): "
              f"LR: {current_lr:.2e}, Train Loss: {avg_train_loss:.4f}, Avg Grad Norm: {avg_grad_norm:.4f}, "
              f"Val Loss: {val_loss_eval:.4f}, Val Macro F1: {val_f1_macro:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")

        if val_f1_macro > best_val_f1_macro:
            best_val_f1_macro = val_f1_macro
            try:
                checkpoint_data = {
                    'epoch': epoch + 1, 'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(), 
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_f1_macro': best_val_f1_macro, 'val_metrics': serializable_val_metrics,
                    'run_index': run_index, 'args': vars(args)
                }
                torch.save(checkpoint_data, best_model_save_path)
                print(f"  Saved new best model (Epoch {epoch+1}, Val Macro F1: {val_f1_macro:.4f}) to {best_model_save_path}")
            except Exception as e:
                print(f"Error saving model checkpoint: {e}")

        scheduler.step(val_loss_eval)
        gc.collect()
        if device == torch.device("cuda"): torch.cuda.empty_cache()

    print(f"--- Finished Training for Run {run_index} --- Best Val Macro F1: {best_val_f1_macro:.4f}")
    return run_metrics, best_model_save_path

def calculate_class_weights(data):
    labels = [item.get('sentiment', 0) + 1 for item in data] 
    counts = np.bincount(labels, minlength=3)
    if counts.sum() == 0: return None # Avoid division by zero if data is empty or labels are missing
    weights = counts.sum() / (counts + 1e-6) # Add epsilon to avoid division by zero for missing classes
    weights = weights / weights.sum() # Normalize
    return torch.tensor(weights, dtype=torch.float)

def main():
    print("--- Experiment Setup ---")
    print(f"Method Name: {METHOD_NAME}")
    print(f"Language Split: {LANGUAGE}")
    print(f"Base Model: {MODEL_NAME}")
    print(f"Max Seq Length (Sentence): {MAX_SEQ_LENGTH}, Max Sentences: {MAX_SENTENCES}")
    print(f"Interaction Layers: {INTERACTION_LAYERS}, Heads: {INTERACTION_HEADS}")
    print(f"Aggregation: {AGGREGATION_METHOD}, Aspect Marker: {USE_ASPECT_MARKER}")
    print(f"Freeze Base: {FREEZE_BASE_MODEL}")

    if args.test_only:
        print("\n*** RUNNING IN TEST-ONLY MODE ***")
        print("Training will be skipped. Loading existing models for evaluation.")
    else:
        print(f"Epochs per run: {EPOCHS}")
        print(f"Batch Size: {BATCH_SIZE}, LR: {LR}, Weight Decay: {WEIGHT_DECAY}")
        print(f"Dropout: {DROPOUT_RATE}, MLP Hidden Dim: {FINAL_MLP_HIDDEN_DIM}")
    
    run_args_dict = vars(args)
    run_args_dict['model_base'] = MODEL_NAME # For consistency with 4.1 script
    run_args_dict['model_save_dir'] = MODEL_SAVE_DIR

    print(f"Loading tokenizer for '{MODEL_NAME}'...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    except Exception as e:
        print(f"Error loading tokenizer: {e}"); exit(1)

    print("Checking SpaCy model availability in main process...")
    try:
        main_spacy_nlp = load_spacy_model(LANGUAGE)
        if not main_spacy_nlp:
            print("SpaCy model failed to load in main process check. Exiting.")
            exit(1)
        del main_spacy_nlp; gc.collect()
        print("SpaCy model check successful.")
    except Exception as e:
         print(f"Failed initial SpaCy model check in main process: {e}")
         exit(1)

    all_test_metrics = {}
    best_model_paths = {}
    
    # Determine num_workers, ensuring it's not too high.
    # max_workers = os.cpu_count() // 2 if os.cpu_count() and os.cpu_count() > 1 else 1
    # num_workers = min(4, max_workers) # Cap at 4 or half of CPUs
    num_workers = 0 # Forcing to 0 for debugging SpaCy issues in workers, can be changed.
    # If num_workers > 0 is desired, ensure SpaCy models are downloaded and accessible by worker processes.
    print(f"Using {num_workers} dataloader workers.")
    if num_workers == 0:
        print("WARNING: Running with num_workers=0. SpaCy will run in the main process.")
        # Initialize SpaCy in main process if num_workers is 0
        global worker_spacy_nlp
        worker_spacy_nlp = load_spacy_model(LANGUAGE)


    init_fn = functools.partial(worker_init_spacy, language_for_worker=LANGUAGE) if num_workers > 0 else None
    
    eff_batch_size_target = 64 # Example target
    accumulation_steps = max(1, round(eff_batch_size_target / BATCH_SIZE))


    if not args.test_only:
        print("\n===== Starting Training Phase =====")
        for i in range(3): 
            print(f"\n===== Starting Run {i} =====")
            current_best_model_path = os.path.join(MODEL_SAVE_DIR, f"best_model_{i}.pt")
            metrics_file_path = os.path.join(MODEL_SAVE_DIR, f"training_metrics_{i}.json")

            train_data_raw, val_data_raw = load_split_data(BASE_DATA_DIR, LANGUAGE, split_index=i)
            class_weights = calculate_class_weights(train_data_raw)
            if class_weights is not None:
                print(f"Calculated class weights for Run {i} training: {class_weights.numpy().round(4)}")
                class_weights = class_weights.to(device)
            else:
                print("Warning: Could not calculate class weights. Using uniform weights.")

            train_dataset = SentimentSentenceDataset(
                train_data_raw, tokenizer, LANGUAGE, MAX_SEQ_LENGTH, MAX_SENTENCES, USE_ASPECT_MARKER
            )
            val_dataset = SentimentSentenceDataset(
                val_data_raw, tokenizer, LANGUAGE, MAX_SEQ_LENGTH, MAX_SENTENCES, USE_ASPECT_MARKER
            )
            
            persistent_dl = num_workers > 0 and torch.__version__ >= "1.7.0"
            try:
                 train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                               num_workers=num_workers, pin_memory=True,
                                               persistent_workers=persistent_dl if num_workers > 0 else False, # persistent_workers requires num_workers > 0
                                               worker_init_fn=init_fn)
                 val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                             num_workers=num_workers, pin_memory=True,
                                             persistent_workers=persistent_dl if num_workers > 0 else False,
                                             worker_init_fn=init_fn)
            except (TypeError, NotImplementedError) as e:
                 print(f"Warning: DataLoader persistent workers not supported or caused an error ({e}). Disabling.")
                 persistent_dl = False
                 train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                               num_workers=num_workers, pin_memory=True, worker_init_fn=init_fn)
                 val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                             num_workers=num_workers, pin_memory=True, worker_init_fn=init_fn)

            print(f"Initializing DARTStyleModel for Run {i}...")
            try:
                model = DARTStyleModel(
                    model_name=MODEL_NAME, interaction_layers=INTERACTION_LAYERS,
                    interaction_heads=INTERACTION_HEADS, final_mlp_hidden_dim=FINAL_MLP_HIDDEN_DIM,
                    dropout_rate=DROPOUT_RATE, aggregation_method=AGGREGATION_METHOD,
                    num_classes=3, freeze_base=FREEZE_BASE_MODEL
                ).to(device)
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                print(f"Model Parameters: Total={total_params/1e6:.2f}M, Trainable={trainable_params/1e6:.2f}M")
            except Exception as e:
                print(f"Error initializing model: {e}"); exit(1)

            criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)
            optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=1, verbose=True)

            run_metrics, saved_model_path = train(
                model, train_dataloader, val_dataloader, criterion, optimizer, scheduler,
                device, EPOCHS, current_best_model_path, run_index=i,
                accumulation_steps=accumulation_steps
            )
            
            run_metrics_to_save = {
                "arguments": run_args_dict, "run_index": i,
                "train_metrics": run_metrics["train"], "eval_metrics": run_metrics["eval"],
                "best_model_path": saved_model_path if saved_model_path and os.path.exists(saved_model_path) else None
            }
            try:
                with open(metrics_file_path, 'w', encoding='utf-8') as f:
                    json.dump(run_metrics_to_save, f, indent=4, default=str) # Use default=str for numpy types
                print(f"Training and validation metrics for run {i} saved to: {metrics_file_path}")
            except Exception as e:
                print(f"Error saving metrics for run {i}: {e}")

            if saved_model_path and os.path.exists(saved_model_path):
                best_model_paths[i] = saved_model_path
            else:
                print(f"Warning: Best model for run {i} ('{saved_model_path}') was not saved or not found.")
                best_model_paths[i] = None

            del model, optimizer, scheduler, train_dataloader, val_dataloader, train_dataset, val_dataset
            del train_data_raw, val_data_raw, run_metrics, run_metrics_to_save, class_weights
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
            print("Cannot proceed with testing. Ensure --method_name and --split are correct and training was completed.")
            exit(1)

    print("\n===== Starting Testing Phase =====")
    test_data_raw = load_split_data(BASE_DATA_DIR, LANGUAGE, split_index=None)
    test_dataset = SentimentSentenceDataset(
        test_data_raw, tokenizer, LANGUAGE, MAX_SEQ_LENGTH, MAX_SENTENCES, USE_ASPECT_MARKER
    )
    test_batch_size = BATCH_SIZE # Can be BATCH_SIZE * 2 for inference if memory allows
    persistent_test_dl = num_workers > 0 and torch.__version__ >= "1.7.0"
    try:
        test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False,
                                     num_workers=num_workers, pin_memory=True,
                                     persistent_workers=persistent_test_dl if num_workers > 0 else False,
                                     worker_init_fn=init_fn)
    except (TypeError, NotImplementedError):
        test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False,
                                     num_workers=num_workers, pin_memory=True, worker_init_fn=init_fn)

    criterion_test = nn.CrossEntropyLoss().to(device)
    test_results_summary = []

    for i in range(3):
        print(f"\n--- Evaluating Best Model from Run {i} on Test Set ---")
        model_path = best_model_paths.get(i)
        if not model_path:
            print(f"Warning: No model path found for run {i}. Skipping.")
            all_test_metrics[f"model_{i}"] = {"error": "Model path not found or model was not saved/found."}
            continue
        
        print(f"Re-Initializing DARTStyleModel for testing Run {i} using global config...")
        try:
            model_test = DARTStyleModel(
                model_name=MODEL_NAME, interaction_layers=INTERACTION_LAYERS,
                interaction_heads=INTERACTION_HEADS, final_mlp_hidden_dim=FINAL_MLP_HIDDEN_DIM,
                dropout_rate=DROPOUT_RATE, aggregation_method=AGGREGATION_METHOD,
                num_classes=3, freeze_base=FREEZE_BASE_MODEL # freeze_base state should match how it was trained for arch consistency
            ).to(device)
        except Exception as e:
            print(f"Error re-initializing model for test run {i}: {e}")
            all_test_metrics[f"model_{i}"] = {"error": f"Failed re-initializing model: {e}"}
            continue
        
        try:
            # weights_only=False is important if the checkpoint contains more than just weights (e.g. pickled code, which is not the case here)
            # For state_dict, it's less critical but good to be explicit if matching 4.1's implicit behavior.
            checkpoint = torch.load(model_path, map_location=device) # weights_only=False is default
            if 'model_state_dict' in checkpoint:
                # Using strict=True is a good test. If it fails, the model architectures differ.
                model_test.load_state_dict(checkpoint['model_state_dict'], strict=True)
                print(f"Successfully loaded model state from epoch {checkpoint.get('epoch', 'N/A')} in {model_path}")
            else: # Older format, unlikely for this script but good for robustness
                model_test.load_state_dict(checkpoint, strict=True)
                print(f"Successfully loaded model state directly from {model_path} (assumed old format).")
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
            "model_run_index": i, "model_path": model_path, "test_loss": test_loss, **test_metrics_dict
        }
        all_test_metrics[f"model_{i}"] = test_result_data
        test_results_summary.append({
            "f1_macro": test_metrics_dict['f1_macro'],
            "accuracy": test_metrics_dict['accuracy'],
            "qwk": test_metrics_dict['qwk']
        })

        predictions_file_path = os.path.join(MODEL_SAVE_DIR, f"test_predictions_{i}.json")
        test_predictions_original_scale = [p - 1 for p in test_predictions_mapped] 
        if len(test_predictions_original_scale) == len(test_data_raw):
            test_data_with_preds = copy.deepcopy(test_data_raw)
            for idx_pred, item_pred in enumerate(test_data_with_preds):
                item_pred['prediction'] = int(test_predictions_original_scale[idx_pred])
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

    if test_results_summary:
        avg_test_f1_macro = np.mean([r['f1_macro'] for r in test_results_summary if 'f1_macro' in r])
        avg_test_accuracy = np.mean([r['accuracy'] for r in test_results_summary if 'accuracy' in r])
        avg_test_qwk = np.mean([r['qwk'] for r in test_results_summary if 'qwk' in r])
        print("\n--- Average Test Set Performance Across Successfully Evaluated Models ---")
        print(f"Avg Macro F1: {avg_test_f1_macro:.4f}")
        print(f"Avg Accuracy: {avg_test_accuracy:.4f}")
        print(f"Avg QWK: {avg_test_qwk:.4f}")
        all_test_metrics["average_performance"] = {
            "f1_macro": avg_test_f1_macro, "accuracy": avg_test_accuracy, "qwk": avg_test_qwk
        }
    else:
        print("\nNo models were successfully evaluated on the test set or no valid metrics to average.")
        all_test_metrics["average_performance"] = {"error": "No models available or evaluation failed for all."}

    combined_test_metrics_file = os.path.join(MODEL_SAVE_DIR, "test_metrics_summary.json")
    try:
        with open(combined_test_metrics_file, 'w', encoding='utf-8') as f:
            json.dump(all_test_metrics, f, indent=4, default=str) 
        print(f"\nCombined test metrics summary saved to: {combined_test_metrics_file}")
    except Exception as e: print(f"Error saving combined test metrics: {e}")

    if num_workers == 0 and worker_spacy_nlp is not None: # Clean up main process SpaCy if used
        del worker_spacy_nlp
        SPACY_MODELS.clear()

    del tokenizer, test_dataset, test_dataloader, test_data_raw
    gc.collect()
    if device == torch.device("cuda"): torch.cuda.empty_cache()

    print("\nExperiment Complete.")
    print(f"Outputs saved in: {MODEL_SAVE_DIR}")

if __name__ == "__main__":
    try:
        mp.set_start_method('spawn', force=True)
        print("Multiprocessing start method set to 'spawn'.")
    except RuntimeError as e:
        if "context has already been set" in str(e):
             print("Warning: Multiprocessing context already set.")
        else:
            print(f"Warning: Could not set multiprocessing start method: {e}.")

    try: import transformers
    except ImportError: print("Error: transformers not found. pip install transformers"); exit(1)
    try: import spacy
    except ImportError: print("Error: spacy not found. pip install spacy"); exit(1)
    try: import sklearn
    except ImportError: print("Error: scikit-learn not found. pip install scikit-learn"); exit(1)

    main()