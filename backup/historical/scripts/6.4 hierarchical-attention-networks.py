# -*- coding: utf-8 -*-
# 6.2 simplified-dart-placeholder.py
import os
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
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
ASPECT_PLACEHOLDER = "[ASPECT_TARGET]" # Define placeholder globally

# --- SpaCy Import and Setup ---
SPACY_MODELS = {}

def load_spacy_model(language):
    lang_key = language.lower()
    if lang_key in SPACY_MODELS and SPACY_MODELS[lang_key] is not None:
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
        
        if 'sentencizer' not in nlp.pipe_names and 'senter' not in nlp.pipe_names:
             print(f"Warning (Process {os.getpid()}): Sentencizer pipe not found in '{model_name}' default pipes. Attempting to add.")
             try:
                 nlp.add_pipe('sentencizer', first=True)
             except ValueError as e:
                 if "already exists in pipeline" in str(e):
                     print(f"Info (Process {os.getpid()}): Sentencizer implicitly present, proceeding.")
                 else:
                     raise e
        
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

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="Train a Simplified DART-style classifier with Aspect Placeholder.")
# Essential
parser.add_argument("--split", type=str, required=True, choices=['slovenian', 'serbian'], help="Language split.")
parser.add_argument("--method_name", type=str, required=True, help="Name for model/method directory.")
# Model - Base
parser.add_argument("--model_name", type=str, default="xlm-roberta-base", help="Pre-trained transformer for sentence encoding.")
parser.add_argument("--max_seq_length", type=int, default=96, help="Max token sequence length for EACH sentence.") 
parser.add_argument("--max_sentences", type=int, default=32, help="Max number of sentences per document.") 
# Model - Interaction (Simplified DART uses one interaction block)
parser.add_argument("--interaction_layers", type=int, default=2, help="Number of Transformer layers for sentence CLS interaction.")
parser.add_argument("--interaction_heads", type=int, default=8, help="Number of attention heads in sentence CLS interaction.")
# Model - Aggregation & Output
parser.add_argument("--aggregation_heads", type=int, default=4, help="Number of attention heads for global aggregation.") 
parser.add_argument("--final_mlp_hidden_dim", type=int, default=256, help="Hidden dimension for the final classification MLP.")
parser.add_argument("--dropout_rate", type=float, default=0.2, help="Dropout rate.")
parser.add_argument("--use_aspect_marker", action='store_true', help=f"Prepend '{ASPECT_PLACEHOLDER}' to each sentence if True.")
parser.add_argument("--mask_aspects", action='store_true', help=f"Replace aspect tags in text with '{ASPECT_PLACEHOLDER}'. If False, tags are removed.")
parser.add_argument("--freeze_base_model", action='store_true', help="Freeze base transformer weights.")
# Training
parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
parser.add_argument("--batch_size", type=int, default=2, help="Micro-batch size (adjust based on GPU memory).") 
parser.add_argument("--eff_batch_size_target", type=int, default=32, help="Target effective batch size via gradient accumulation.")
parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate.") 
parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
parser.add_argument("--test_only", action='store_true', help="Run test evaluation only.")
args = parser.parse_args()

# --- Global Variables & Configuration ---
LANGUAGE = args.split
METHOD_NAME = args.method_name
MODEL_NAME = args.model_name
MAX_SEQ_LENGTH = args.max_seq_length
MAX_SENTENCES = args.max_sentences
INTERACTION_LAYERS = args.interaction_layers
INTERACTION_HEADS = args.interaction_heads
AGGREGATION_HEADS = args.aggregation_heads 
FINAL_MLP_HIDDEN_DIM = args.final_mlp_hidden_dim
DROPOUT_RATE = args.dropout_rate
USE_ASPECT_MARKER = args.use_aspect_marker
MASK_ASPECTS = args.mask_aspects # New global variable
FREEZE_BASE_MODEL = args.freeze_base_model
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size 
EFF_BATCH_SIZE_TARGET = args.eff_batch_size_target
LR = args.lr
WEIGHT_DECAY = args.weight_decay

# --- Path Definitions ---
BASE_DATA_DIR = "../data/final/complete"
MODEL_SAVE_DIR = f"../models/{METHOD_NAME}/{LANGUAGE}"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# --- Device Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device == torch.device("cuda"):
    print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
    try:
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"Total VRAM: {total_vram_gb:.2f} GB")
    except Exception as e:
        print(f"Could not get VRAM info: {e}")

# --- Load Split Data ---
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

# --- Dataset Class (Simplified DART Style with Aspect Placeholder) ---
class SentimentSentenceDataset(Dataset):
    def __init__(self, data, tokenizer, language, max_seq_length, max_sentences, use_aspect_marker=False, mask_aspects=False):
        self.data = data
        self.tokenizer = tokenizer
        self.language = language
        self.max_seq_length = max_seq_length
        self.max_sentences = max_sentences
        self.use_aspect_marker = use_aspect_marker
        self.mask_aspects = mask_aspects # Store the new flag
        self.aspect_tag_start = "<aspect>"
        self.aspect_tag_end = "</aspect>"

    def __len__(self):
        return len(self.data)

    def _process_article_aspects(self, text):
        if self.mask_aspects:
            # Replace aspect with placeholder
            start_idx = text.find(self.aspect_tag_start)
            end_idx = text.find(self.aspect_tag_end)
            while start_idx != -1 and end_idx != -1 and start_idx < end_idx:
                text = text[:start_idx] + ASPECT_PLACEHOLDER + text[end_idx + len(self.aspect_tag_end):]
                start_idx = text.find(self.aspect_tag_start) 
                end_idx = text.find(self.aspect_tag_end)
            # Clean any remaining unmatched tags
            text = text.replace(self.aspect_tag_start, "").replace(self.aspect_tag_end, "")
        else:
            # Just remove aspect tags without placeholder
            text = text.replace(self.aspect_tag_start, "").replace(self.aspect_tag_end, "")
        return text

    def __getitem__(self, idx):
        global worker_spacy_nlp
        if worker_spacy_nlp is None:
             raise RuntimeError(f"SpaCy model not initialized in worker (PID {os.getpid()}) for language {self.language}.")

        item = self.data[idx]
        raw_article = item.get('article', '')

        # Process aspects in the article based on the mask_aspects flag
        article_processed = self._process_article_aspects(raw_article)

        try:
            sentences = split_document(article_processed, worker_spacy_nlp)
        except Exception as e:
             print(f"Error during split_document in worker (PID: {os.getpid()}) for item {idx}: {e}")
             sentences = []

        if len(sentences) > self.max_sentences:
            sentences = sentences[:self.max_sentences]
        
        all_input_ids = []
        all_attention_masks = []
        sentence_pos_ids_list = [] 

        for sent_idx, sentence in enumerate(sentences):
            text_to_encode = sentence
            if self.use_aspect_marker: 
                text_to_encode = f"{ASPECT_PLACEHOLDER} {self.tokenizer.sep_token} {sentence}"
            
            try:
                encoding = self.tokenizer.encode_plus(
                    text_to_encode, add_special_tokens=True, max_length=self.max_seq_length,
                    padding='max_length', truncation=True, return_attention_mask=True, return_tensors='pt',
                )
                all_input_ids.append(encoding['input_ids'].squeeze(0))
                all_attention_masks.append(encoding['attention_mask'].squeeze(0))
                sentence_pos_ids_list.append(sent_idx + 1) 
            except Exception as e:
                 print(f"Error during tokenization in worker (PID: {os.getpid()}) for item {idx}, sentence: '{sentence[:50]}...': {e}")
                 continue
        
        num_sentences_processed = len(all_input_ids)
        
        sentence_padding_count = self.max_sentences - num_sentences_processed
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        
        if sentence_padding_count > 0:
            pad_ids_tensor = torch.full((self.max_seq_length,), pad_token_id, dtype=torch.long)
            pad_mask_tensor = torch.zeros((self.max_seq_length,), dtype=torch.long)
            for _ in range(sentence_padding_count):
                all_input_ids.append(pad_ids_tensor)
                all_attention_masks.append(pad_mask_tensor)
                sentence_pos_ids_list.append(0) 

        if not all_input_ids: 
             all_input_ids = [torch.full((self.max_seq_length,), pad_token_id, dtype=torch.long)] * self.max_sentences
             all_attention_masks = [torch.zeros((self.max_seq_length,), dtype=torch.long)] * self.max_sentences
             sentence_pos_ids_list = [0] * self.max_sentences
             num_sentences_processed = 0

        input_ids = torch.stack(all_input_ids)
        attention_mask = torch.stack(all_attention_masks)
        sentence_position_ids = torch.tensor(sentence_pos_ids_list, dtype=torch.long)

        sentence_mask = torch.zeros(self.max_sentences, dtype=torch.long)
        if num_sentences_processed > 0:
             sentence_mask[:num_sentences_processed] = 1

        sentiment_original = item.get('sentiment', 0)
        sentiment_mapped = sentiment_original + 1 
        if sentiment_mapped not in [0, 1, 2]: sentiment_mapped = 1

        return {
            'input_ids': input_ids, 
            'attention_mask': attention_mask, 
            'sentence_mask': sentence_mask,   
            'sentence_position_ids': sentence_position_ids, 
            'labels': torch.tensor(sentiment_mapped, dtype=torch.long)
        }

# --- Model Definition (Simplified DART with Aspect Placeholder) ---
class SimplifiedDARTModel(nn.Module):
    def __init__(self, model_name, tokenizer_len,
                 interaction_layers, interaction_heads,
                 aggregation_heads, max_sentences, 
                 final_mlp_hidden_dim, dropout_rate,
                 num_classes=3, freeze_base=False):
        super().__init__()
        self.base_model = AutoModel.from_pretrained(model_name)
        self.base_model.resize_token_embeddings(tokenizer_len) # Resize for [ASPECT_TARGET]

        self.config = self.base_model.config
        self.hidden_dim = self.config.hidden_size
        self.dropout = nn.Dropout(dropout_rate)

        if freeze_base:
            print("Freezing base transformer model weights.")
            for param in self.base_model.parameters():
                param.requires_grad = False
        
        # Sentence Positional Embeddings
        self.sentence_pos_embedding = nn.Embedding(
            max_sentences + 1, # Vocab size: max_sentences + 1 for padding_idx 0
            self.hidden_dim,
            padding_idx=0
        )

        # Sentence Interaction Transformer (operates on CLS sentence embeddings)
        interact_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim, nhead=interaction_heads,
            dim_feedforward=self.hidden_dim * 4, dropout=dropout_rate,
            activation='relu', batch_first=True
        )
        self.sentence_interact_transformer = nn.TransformerEncoder(interact_encoder_layer, num_layers=interaction_layers)
        
        # Global Aggregation (sentence-level attention, queried by ASPECT_PLACEHOLDER embedding)
        self.global_aggregation_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim, num_heads=aggregation_heads,
            dropout=dropout_rate, batch_first=True
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, final_mlp_hidden_dim), nn.ReLU(),
            nn.Dropout(dropout_rate), nn.Linear(final_mlp_hidden_dim, num_classes)
        )

    def forward(self, input_ids, attention_mask, sentence_mask, sentence_position_ids, aspect_target_token_id):
        # input_ids: (B, N_sent, N_tok)
        # attention_mask: (B, N_sent, N_tok) -> token level
        # sentence_mask: (B, N_sent) -> sentence level (1 for real, 0 for pad)
        # sentence_position_ids: (B, N_sent) -> positional IDs for sentences
        # aspect_target_token_id: scalar tensor, ID of [ASPECT_TARGET]

        batch_size, num_max_sentences, num_max_tokens = input_ids.shape

        # --- 1. Initial Sentence Encoding ---
        input_ids_flat = input_ids.view(-1, num_max_tokens)
        attention_mask_flat = attention_mask.view(-1, num_max_tokens)
        
        base_model_outputs = self.base_model(input_ids=input_ids_flat, attention_mask=attention_mask_flat)
        cls_embeddings_flat = base_model_outputs.last_hidden_state[:, 0, :] # (B*N_sent, H)
        cls_embeddings = cls_embeddings_flat.view(batch_size, num_max_sentences, self.hidden_dim)

        # Add sentence positional embeddings
        pos_embs = self.sentence_pos_embedding(sentence_position_ids) # (B, N_sent, H)
        cls_embeddings_with_pos = cls_embeddings + pos_embs
        cls_embeddings_with_pos = self.dropout(cls_embeddings_with_pos) # Dropout after adding PE

        # --- 2. Global Context Interaction (on CLS sentence embeddings) ---
        sentence_interact_padding_mask = (sentence_mask == 0) # True for padded sentences
        contextualized_sentence_summaries = self.sentence_interact_transformer(
            cls_embeddings_with_pos,
            src_key_padding_mask=sentence_interact_padding_mask
        ) # (B, N_sent, H)
        
        # Mask out summaries from padded sentences before aggregation
        expanded_sentence_mask = sentence_mask.unsqueeze(-1).float()
        contextualized_sentence_summaries = contextualized_sentence_summaries * expanded_sentence_mask

        # --- 3. Global Aggregation (queried by ASPECT_PLACEHOLDER embedding) ---
        # Get embedding for ASPECT_PLACEHOLDER (it's a single ID, need to embed it)
        # Ensure aspect_target_token_id is on the same device as model embeddings
        aspect_placeholder_emb = self.base_model.get_input_embeddings()(aspect_target_token_id.to(input_ids.device)) # (1, H)
        # If aspect_target_token_id was passed as a scalar and made a tensor [ID], squeeze might be needed if it's (1,1,H)
        if aspect_placeholder_emb.ndim > 2: aspect_placeholder_emb = aspect_placeholder_emb.squeeze(0)

        global_query = aspect_placeholder_emb.unsqueeze(0).repeat(batch_size, 1, 1) # (B, 1, H)

        global_attn_output, _ = self.global_aggregation_attention(
            query=global_query,                     # (B, 1, H)
            key=contextualized_sentence_summaries,  # (B, N_sent, H)
            value=contextualized_sentence_summaries,# (B, N_sent, H)
            key_padding_mask=sentence_interact_padding_mask  # True for padded sentences
        ) # Output: (B, 1, H)
        
        aggregated_doc_representation = global_attn_output.squeeze(1) # (B, H)
        aggregated_doc_representation = self.dropout(aggregated_doc_representation)

        # --- 4. Classification ---
        logits = self.classifier(aggregated_doc_representation)
        return logits

# --- Evaluation Function ---
def evaluate(model, dataloader, criterion, device, aspect_target_token_id_scalar):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    scaler_enabled = (device == torch.device("cuda"))

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            sentence_mask = batch['sentence_mask'].to(device)
            sentence_position_ids = batch['sentence_position_ids'].to(device)
            labels = batch['labels'].to(device)
            
            aspect_target_id_tensor = torch.tensor([aspect_target_token_id_scalar], device=device)

            with torch.amp.autocast(device_type=str(device).split(':')[0], dtype=torch.float16, enabled=scaler_enabled):      
                logits = model(input_ids, attention_mask, sentence_mask, sentence_position_ids, aspect_target_id_tensor)
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

# --- Gradient Norm ---
def get_grad_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

# --- Training Function ---
def train(model, train_dataloader, val_dataloader, criterion, optimizer, scheduler, device, epochs,
          best_model_save_path, run_index, accumulation_steps, aspect_target_token_id_scalar):
    best_val_f1_macro = -1.0
    run_metrics = {"train": [], "eval": []}
    scaler = torch.amp.GradScaler(enabled=(device == torch.device("cuda")))

    print(f"\n--- Starting Training for Run {run_index} ---")
    print(f"Gradient Accumulation Steps: {accumulation_steps}")
    print(f"Automatic Mixed Precision (AMP): {'Enabled' if scaler.is_enabled() else 'Disabled'}")
    print(f"Effective Batch Size: {args.batch_size * accumulation_steps}")
    print(f"Best model (based on Val Macro F1) will be saved to: {best_model_save_path}")

    aspect_target_id_tensor = torch.tensor([aspect_target_token_id_scalar], device=device)

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
            sentence_position_ids = batch['sentence_position_ids'].to(device)
            labels = batch['labels'].to(device)

            with torch.amp.autocast(device_type=str(device).split(':')[0], dtype=torch.float16, enabled=scaler.is_enabled()):
                logits = model(input_ids, attention_mask, sentence_mask, sentence_position_ids, aspect_target_id_tensor)
                loss = criterion(logits, labels)
                loss = loss / accumulation_steps

            if torch.isnan(loss):
                print(f"Warning: NaN loss encountered at run {run_index}, epoch {epoch+1}, batch {batch_idx}. Skipping accumulation for this batch.")
                if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_dataloader):
                    if accumulated_loss > 0 : 
                         scaler.unscale_(optimizer)
                         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                         scaler.step(optimizer)
                         scaler.update()
                    optimizer.zero_grad()
                    accumulated_loss = 0.0 
                continue 

            scaler.scale(loss).backward()
            accumulated_loss += loss.item() * accumulation_steps 

            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_dataloader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                grad_norm = get_grad_norm(model)
                total_grad_norm += grad_norm if not np.isnan(grad_norm) else 0
                train_loss += accumulated_loss 
                batch_count += 1 
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
        val_loss_eval, val_metrics, val_report_str, _ = evaluate(model, val_dataloader, criterion, device, aspect_target_token_id_scalar)
        val_f1_macro = val_metrics["f1_macro"]
        
        serializable_val_metrics = {}
        for k, v in val_metrics.items():
            if isinstance(v, dict): 
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
                    'run_index': run_index, 'args': vars(args) # Save current script args
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

# --- Calculate Class Weights ---
def calculate_class_weights(data):
    labels = [item.get('sentiment', 0) + 1 for item in data] 
    counts = np.bincount(labels, minlength=3)
    if counts.sum() == 0: return None
    weights = counts.sum() / (counts + 1e-6) 
    weights = weights / weights.sum() 
    return torch.tensor(weights, dtype=torch.float)

# --- Main Execution ---
def main():
    print("--- Experiment Setup ---")
    print(f"Method Name: {METHOD_NAME}")
    print(f"Language Split: {LANGUAGE}")
    print(f"Base Model: {MODEL_NAME}")
    print(f"Max Seq Length (Sentence): {MAX_SEQ_LENGTH}, Max Sentences: {MAX_SENTENCES}")
    print(f"Interaction Layers: {INTERACTION_LAYERS}, Heads: {INTERACTION_HEADS}")
    print(f"Aggregation Heads: {AGGREGATION_HEADS}")
    print(f"Aspect Marker ([ASPECT_TARGET] prepended): {USE_ASPECT_MARKER}")
    print(f"Mask Aspects (replace in-text aspect with placeholder): {MASK_ASPECTS}") # Print new flag
    print(f"Freeze Base: {FREEZE_BASE_MODEL}")

    if args.test_only:
        print("\n*** RUNNING IN TEST-ONLY MODE ***")
    else:
        print(f"Epochs per run: {EPOCHS}")
        print(f"Micro-Batch Size: {BATCH_SIZE}, Target Effective Batch Size: {EFF_BATCH_SIZE_TARGET}")
        print(f"LR: {LR}, Weight Decay: {WEIGHT_DECAY}")
        print(f"Dropout: {DROPOUT_RATE}, MLP Hidden Dim: {FINAL_MLP_HIDDEN_DIM}")
    
    run_args_dict = vars(args)
    run_args_dict['model_base'] = MODEL_NAME 
    run_args_dict['model_save_dir'] = MODEL_SAVE_DIR

    print(f"Loading tokenizer for '{MODEL_NAME}'...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        special_tokens_dict = {'additional_special_tokens': [ASPECT_PLACEHOLDER]}
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
        if num_added_toks > 0:
            print(f"Added {num_added_toks} special token(s): {ASPECT_PLACEHOLDER}")
        aspect_target_token_id_scalar = tokenizer.convert_tokens_to_ids(ASPECT_PLACEHOLDER)
        if aspect_target_token_id_scalar == tokenizer.unk_token_id:
            print(f"CRITICAL WARNING: {ASPECT_PLACEHOLDER} was not added to tokenizer vocab correctly and is UNK. This will fail.")
            exit(1)

    except Exception as e:
        print(f"Error loading tokenizer or adding special token: {e}"); exit(1)

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
    
    num_workers = 0 # Set to 0 for easier debugging if SpaCy/multiprocessing issues arise
    # num_workers = min(4, mp.cpu_count() // 2) if mp.cpu_count() > 1 else 0 # Example: Use a few workers if available
    print(f"Using {num_workers} dataloader workers.")
    if num_workers == 0:
        print("INFO: Running with num_workers=0. SpaCy will run in the main process.")
        global worker_spacy_nlp
        worker_spacy_nlp = load_spacy_model(LANGUAGE)

    init_fn = functools.partial(worker_init_spacy, language_for_worker=LANGUAGE) if num_workers > 0 else None
    accumulation_steps = max(1, round(EFF_BATCH_SIZE_TARGET / BATCH_SIZE))

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
                train_data_raw, tokenizer, LANGUAGE, MAX_SEQ_LENGTH, MAX_SENTENCES, 
                use_aspect_marker=USE_ASPECT_MARKER, mask_aspects=MASK_ASPECTS # Pass new flag
            )
            val_dataset = SentimentSentenceDataset(
                val_data_raw, tokenizer, LANGUAGE, MAX_SEQ_LENGTH, MAX_SENTENCES, 
                use_aspect_marker=USE_ASPECT_MARKER, mask_aspects=MASK_ASPECTS # Pass new flag
            )
            
            persistent_dl = num_workers > 0 and torch.__version__ >= "1.7.0" 
            try:
                 train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                               num_workers=num_workers, pin_memory=True,
                                               persistent_workers=persistent_dl if num_workers > 0 else False,
                                               worker_init_fn=init_fn)
                 val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                             num_workers=num_workers, pin_memory=True,
                                             persistent_workers=persistent_dl if num_workers > 0 else False,
                                             worker_init_fn=init_fn)
            except (TypeError, NotImplementedError) as e:
                 print(f"Warning: DataLoader persistent workers not supported or caused an error ({e}). Disabling.")
                 train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                               num_workers=num_workers, pin_memory=True, worker_init_fn=init_fn)
                 val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                             num_workers=num_workers, pin_memory=True, worker_init_fn=init_fn)

            print(f"Initializing SimplifiedDARTModel for Run {i}...")
            try:
                model = SimplifiedDARTModel(
                    model_name=MODEL_NAME, tokenizer_len=len(tokenizer),
                    interaction_layers=INTERACTION_LAYERS, interaction_heads=INTERACTION_HEADS,
                    aggregation_heads=AGGREGATION_HEADS, max_sentences=MAX_SENTENCES,
                    final_mlp_hidden_dim=FINAL_MLP_HIDDEN_DIM, dropout_rate=DROPOUT_RATE,
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
                accumulation_steps=accumulation_steps,
                aspect_target_token_id_scalar=aspect_target_token_id_scalar
            )
            
            run_metrics_to_save = {
                "arguments": run_args_dict, "run_index": i,
                "train_metrics": run_metrics["train"], "eval_metrics": run_metrics["eval"],
                "best_model_path": saved_model_path if saved_model_path and os.path.exists(saved_model_path) else None
            }
            try:
                with open(metrics_file_path, 'w', encoding='utf-8') as f:
                    json.dump(run_metrics_to_save, f, indent=4, default=str)
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
    else: 
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
            exit(1)

    print("\n===== Starting Testing Phase =====")
    test_data_raw = load_split_data(BASE_DATA_DIR, LANGUAGE, split_index=None)
    test_dataset = SentimentSentenceDataset(
        test_data_raw, tokenizer, LANGUAGE, MAX_SEQ_LENGTH, MAX_SENTENCES, 
        use_aspect_marker=USE_ASPECT_MARKER, mask_aspects=MASK_ASPECTS # Pass new flag
    )
    test_batch_size = BATCH_SIZE 
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
        
        try:
            checkpoint = torch.load(model_path, map_location=device)
            # Use args from checkpoint to ensure model architecture matches
            loaded_args_dict = checkpoint.get('args', vars(args)) 
            if isinstance(loaded_args_dict, dict):
                # Ensure all expected args are present, falling back to current script args if missing in checkpoint
                current_args_dict_for_fallback = vars(args)
                merged_args_dict = {**current_args_dict_for_fallback, **loaded_args_dict}
                loaded_args_ns = argparse.Namespace(**merged_args_dict)

            else: # Should not happen if args are saved as dict
                loaded_args_ns = loaded_args_dict if loaded_args_dict else args
        except Exception as e:
            print(f"Warning: Could not load args from checkpoint {model_path}, using current script args. Error: {e}")
            loaded_args_ns = args 
            checkpoint = None # Will reload later

        print(f"Re-Initializing SimplifiedDARTModel for testing Run {i} using {'checkpoint' if checkpoint and 'args' in checkpoint else 'current script'} args...")
        try:
            model_test = SimplifiedDARTModel(
                model_name=loaded_args_ns.model_name, tokenizer_len=len(tokenizer),
                interaction_layers=loaded_args_ns.interaction_layers, 
                interaction_heads=loaded_args_ns.interaction_heads,
                aggregation_heads=getattr(loaded_args_ns, 'aggregation_heads', AGGREGATION_HEADS), 
                max_sentences=loaded_args_ns.max_sentences,
                final_mlp_hidden_dim=loaded_args_ns.final_mlp_hidden_dim, 
                dropout_rate=loaded_args_ns.dropout_rate,
                num_classes=3, freeze_base=False # Freeze is a training concern, for testing it's about loaded weights
            ).to(device)
        except Exception as e:
            print(f"Error re-initializing model for test run {i}: {e}")
            all_test_metrics[f"model_{i}"] = {"error": f"Failed re-initializing model: {e}"}
            if 'checkpoint' in locals() and checkpoint: del checkpoint
            gc.collect()
            if device == torch.device("cuda"): torch.cuda.empty_cache()
            continue
        
        try:
            if checkpoint is None: # If args loading failed and checkpoint was reset
                 checkpoint = torch.load(model_path, map_location=device)

            if 'model_state_dict' in checkpoint:
                model_test.load_state_dict(checkpoint['model_state_dict'], strict=True)
                print(f"Successfully loaded model state from epoch {checkpoint.get('epoch', 'N/A')} in {model_path}")
            else: # Older checkpoint format
                model_test.load_state_dict(checkpoint, strict=True)
                print(f"Successfully loaded model state directly from {model_path} (assumed old format).")
            model_test.eval()
        except Exception as e:
             print(f"Error loading model state from {model_path}: {e}")
             all_test_metrics[f"model_{i}"] = {"error": f"Failed loading state: {e}"}
             del model_test
             if 'checkpoint' in locals() and checkpoint: del checkpoint
             gc.collect();
             if device == torch.device("cuda"): torch.cuda.empty_cache()
             continue

        test_loss, test_metrics_dict, test_report_str, test_predictions_mapped = evaluate(
            model_test, test_dataloader, criterion_test, device, aspect_target_token_id_scalar
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

        del model_test
        if 'checkpoint' in locals() and checkpoint: del checkpoint
        gc.collect()
        if device == torch.device("cuda"): torch.cuda.empty_cache()

    if test_results_summary:
        f1_scores = [r['f1_macro'] for r in test_results_summary if r.get('f1_macro') is not None]
        acc_scores = [r['accuracy'] for r in test_results_summary if r.get('accuracy') is not None]
        qwk_scores = [r['qwk'] for r in test_results_summary if r.get('qwk') is not None]

        avg_test_f1_macro = np.mean(f1_scores) if f1_scores else 0.0
        avg_test_accuracy = np.mean(acc_scores) if acc_scores else 0.0
        avg_test_qwk = np.mean(qwk_scores) if qwk_scores else 0.0
        
        std_test_f1_macro = np.std(f1_scores) if len(f1_scores) > 1 else 0.0
        std_test_accuracy = np.std(acc_scores) if len(acc_scores) > 1 else 0.0
        std_test_qwk = np.std(qwk_scores) if len(qwk_scores) > 1 else 0.0


        print("\n--- Average Test Set Performance Across Successfully Evaluated Models ---")
        print(f"Avg Macro F1: {avg_test_f1_macro:.4f} (Std: {std_test_f1_macro:.4f}) (from {len(f1_scores)} models)")
        print(f"Avg Accuracy: {avg_test_accuracy:.4f} (Std: {std_test_accuracy:.4f}) (from {len(acc_scores)} models)")
        print(f"Avg QWK: {avg_test_qwk:.4f} (Std: {std_test_qwk:.4f}) (from {len(qwk_scores)} models)")
        all_test_metrics["average_performance"] = {
            "f1_macro_mean": avg_test_f1_macro, "f1_macro_std": std_test_f1_macro, "num_models_f1": len(f1_scores),
            "accuracy_mean": avg_test_accuracy, "accuracy_std": std_test_accuracy, "num_models_accuracy": len(acc_scores),
            "qwk_mean": avg_test_qwk, "qwk_std": std_test_qwk, "num_models_qwk": len(qwk_scores),
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

    if num_workers == 0 and 'worker_spacy_nlp' in globals() and worker_spacy_nlp is not None:
        del worker_spacy_nlp
        SPACY_MODELS.clear()

    del tokenizer
    if 'test_dataset' in locals(): del test_dataset
    if 'test_dataloader' in locals(): del test_dataloader
    if 'test_data_raw' in locals(): del test_data_raw
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