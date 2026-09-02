# predict_with_best_gcm_simplified_dart.py
import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, logging as hf_logging
import argparse # For Namespace
from tqdm.auto import tqdm
import warnings
import gc
import copy

# Suppress unnecessary Hugging Face warnings
hf_logging.set_verbosity_error()

# Suppress warnings from Thinc (used by SpaCy) and other potential FutureWarning from torch.load
warnings.filterwarnings("ignore", category=FutureWarning, module="thinc.shims.pytorch")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*`torch.load` with `weights_only=False`.*")


# --- Global Variables & Configuration ---
ASPECT_PLACEHOLDER = "[ASPECT_TARGET]" # From original script
SPACY_MODELS = {} # Cache for loaded SpaCy models

FULL_METHOD_NAME = "global-context-modelling/simplified-dart-xlmr"
METHOD_SLUG = FULL_METHOD_NAME.split('/')[-1] # e.g., "simplified-dart-xlmr"
METHOD_KEY_FOR_JSON = FULL_METHOD_NAME # Key for adding prediction to JSON

BASE_MODEL_DIR_TEMPLATE = "../models/{method_name_full}/{language}"
BASE_INPUT_DATA_DIR = "../data/final/balanced"
BASE_OUTPUT_DATA_DIR_TEMPLATE = "../data/final/balanced-predicted/{method_slug}"

# --- SpaCy Loading Functions (adapted from original script) ---
def load_spacy_model(language):
    lang_key = language.lower()
    if lang_key in SPACY_MODELS and SPACY_MODELS[lang_key] is not None:
        return SPACY_MODELS[lang_key]

    print(f"Process {os.getpid()}: Loading SpaCy model for language: {language}...")
    model_name = None
    loader_func = None

    if lang_key == "english": model_name = "en_core_web_sm"
    elif lang_key == "slovenian": model_name = "sl_core_news_sm"
    elif lang_key in ["croatian", "serbian"]: model_name = "hr_core_news_sm"
    else:
        print(f"Warning (Process {os.getpid()}): Unsupported language '{language}'. Defaulting to 'en_core_web_sm'.")
        model_name = "en_core_web_sm"

    if model_name:
        try:
            if lang_key == "english": import en_core_web_sm; loader_func = en_core_web_sm.load
            elif lang_key == "slovenian": import sl_core_news_sm; loader_func = sl_core_news_sm.load
            elif lang_key in ["croatian", "serbian"]: import hr_core_news_sm; loader_func = hr_core_news_sm.load
            else: import en_core_web_sm; loader_func = en_core_web_sm.load
        except ImportError:
            print(f"Warning (Process {os.getpid()}): SpaCy model '{model_name}' not installed via package. Trying spacy.load().")
            print(f"Install with: python -m spacy download {model_name}")

    nlp = None
    try:
        if loader_func: nlp = loader_func()
        else: import spacy; nlp = spacy.load(model_name)

        if 'sentencizer' not in nlp.pipe_names and 'senter' not in nlp.pipe_names:
            try: nlp.add_pipe('sentencizer', first=True)
            except ValueError as e:
                if "already exists in pipeline" not in str(e): raise e
        
        print(f"Process {os.getpid()}: SpaCy model '{model_name}' loaded successfully.")
        SPACY_MODELS[lang_key] = nlp
        return nlp
    except Exception as e:
        print(f"Error loading SpaCy model '{model_name}' in process {os.getpid()}: {e}")
        raise RuntimeError(f"SpaCy model loading failed: {e}")

def split_document(raw_text, nlp_model):
    if nlp_model is None:
        raise RuntimeError("SpaCy model not available.")
    if not raw_text or not isinstance(raw_text, str):
        return []
    try:
        doc = nlp_model(raw_text.strip())
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        return sentences
    except Exception as e:
        print(f"Error processing text with SpaCy: {e}")
        print(f"Problematic text snippet: {raw_text[:200]}...")
        return []

# --- Model Definition (Copied from original script) ---
class SimplifiedDARTModel(nn.Module):
    def __init__(self, model_name, tokenizer_len,
                 interaction_layers, interaction_heads,
                 aggregation_heads, max_sentences,
                 final_mlp_hidden_dim, dropout_rate,
                 num_classes=3, freeze_base=False):
        super().__init__()
        self.base_model = AutoModel.from_pretrained(model_name)
        self.base_model.resize_token_embeddings(tokenizer_len)

        self.config = self.base_model.config
        self.hidden_dim = self.config.hidden_size
        self.dropout = nn.Dropout(dropout_rate)

        if freeze_base:
            print("Warning: freeze_base=True during inference model init. Ensure this is intended.")
            for param in self.base_model.parameters():
                param.requires_grad = False
        
        self.sentence_pos_embedding = nn.Embedding(
            max_sentences + 1, self.hidden_dim, padding_idx=0
        )
        interact_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim, nhead=interaction_heads,
            dim_feedforward=self.hidden_dim * 4, dropout=dropout_rate,
            activation='relu', batch_first=True
        )
        self.sentence_interact_transformer = nn.TransformerEncoder(interact_encoder_layer, num_layers=interaction_layers)
        self.global_aggregation_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim, num_heads=aggregation_heads,
            dropout=dropout_rate, batch_first=True
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, final_mlp_hidden_dim), nn.ReLU(),
            nn.Dropout(dropout_rate), nn.Linear(final_mlp_hidden_dim, num_classes)
        )

    def forward(self, input_ids, attention_mask, sentence_mask, sentence_position_ids, aspect_target_token_id):
        batch_size, num_max_sentences, num_max_tokens = input_ids.shape
        input_ids_flat = input_ids.view(-1, num_max_tokens)
        attention_mask_flat = attention_mask.view(-1, num_max_tokens)
        
        base_model_outputs = self.base_model(input_ids=input_ids_flat, attention_mask=attention_mask_flat)
        cls_embeddings_flat = base_model_outputs.last_hidden_state[:, 0, :]
        cls_embeddings = cls_embeddings_flat.view(batch_size, num_max_sentences, self.hidden_dim)

        pos_embs = self.sentence_pos_embedding(sentence_position_ids)
        cls_embeddings_with_pos = cls_embeddings + pos_embs
        cls_embeddings_with_pos = self.dropout(cls_embeddings_with_pos)

        sentence_interact_padding_mask = (sentence_mask == 0)
        contextualized_sentence_summaries = self.sentence_interact_transformer(
            cls_embeddings_with_pos, src_key_padding_mask=sentence_interact_padding_mask
        )
        expanded_sentence_mask = sentence_mask.unsqueeze(-1).float()
        contextualized_sentence_summaries = contextualized_sentence_summaries * expanded_sentence_mask

        aspect_placeholder_emb = self.base_model.get_input_embeddings()(aspect_target_token_id.to(input_ids.device))
        if aspect_placeholder_emb.ndim > 2: aspect_placeholder_emb = aspect_placeholder_emb.squeeze(0)
        global_query = aspect_placeholder_emb.unsqueeze(0).repeat(batch_size, 1, 1)

        global_attn_output, _ = self.global_aggregation_attention(
            query=global_query, key=contextualized_sentence_summaries,
            value=contextualized_sentence_summaries, key_padding_mask=sentence_interact_padding_mask
        )
        aggregated_doc_representation = global_attn_output.squeeze(1)
        aggregated_doc_representation = self.dropout(aggregated_doc_representation)
        logits = self.classifier(aggregated_doc_representation)
        return logits

# --- Data Preprocessing Logic (adapted from SentimentSentenceDataset) ---
ASPECT_TAG_START = "<aspect>"
ASPECT_TAG_END = "</aspect>"

def _replace_aspect_with_placeholder_func(text):
    start_idx = text.find(ASPECT_TAG_START)
    end_idx = text.find(ASPECT_TAG_END)
    while start_idx != -1 and end_idx != -1 and start_idx < end_idx:
        text = text[:start_idx] + ASPECT_PLACEHOLDER + text[end_idx + len(ASPECT_TAG_END):]
        start_idx = text.find(ASPECT_TAG_START)
        end_idx = text.find(ASPECT_TAG_END)
    text = text.replace(ASPECT_TAG_START, "").replace(ASPECT_TAG_END, "")
    return text

def preprocess_single_item(item_data, tokenizer, spacy_nlp, max_seq_length, max_sentences, use_aspect_marker):
    # item_data is now guaranteed to be a dictionary (the actual item)
    raw_article = item_data.get('article', '') 
    article_with_placeholder = _replace_aspect_with_placeholder_func(raw_article)
    
    try:
        sentences = split_document(article_with_placeholder, spacy_nlp)
    except Exception as e:
        print(f"Error during split_document for an item: {e}")
        sentences = []

    if len(sentences) > max_sentences:
        sentences = sentences[:max_sentences]
    
    all_input_ids, all_attention_masks, sentence_pos_ids_list = [], [], []

    for sent_idx, sentence in enumerate(sentences):
        text_to_encode = sentence
        if use_aspect_marker:
            text_to_encode = f"{ASPECT_PLACEHOLDER} {tokenizer.sep_token} {sentence}"
        
        try:
            encoding = tokenizer.encode_plus(
                text_to_encode, add_special_tokens=True, max_length=max_seq_length,
                padding='max_length', truncation=True, return_attention_mask=True, return_tensors='pt',
            )
            all_input_ids.append(encoding['input_ids'].squeeze(0))
            all_attention_masks.append(encoding['attention_mask'].squeeze(0))
            sentence_pos_ids_list.append(sent_idx + 1)
        except Exception as e:
            print(f"Error during tokenization for sentence: '{sentence[:50]}...': {e}")
            continue
    
    num_sentences_processed = len(all_input_ids)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    
    if num_sentences_processed < max_sentences:
        pad_ids_tensor = torch.full((max_seq_length,), pad_token_id, dtype=torch.long)
        pad_mask_tensor = torch.zeros((max_seq_length,), dtype=torch.long)
        for _ in range(max_sentences - num_sentences_processed):
            all_input_ids.append(pad_ids_tensor)
            all_attention_masks.append(pad_mask_tensor)
            sentence_pos_ids_list.append(0)

    if not all_input_ids: 
        all_input_ids = [torch.full((max_seq_length,), pad_token_id, dtype=torch.long)] * max_sentences
        all_attention_masks = [torch.zeros((max_seq_length,), dtype=torch.long)] * max_sentences
        sentence_pos_ids_list = [0] * max_sentences
        num_sentences_processed = 0

    input_ids_tensor = torch.stack(all_input_ids)
    attention_mask_tensor = torch.stack(all_attention_masks)
    sentence_position_ids_tensor = torch.tensor(sentence_pos_ids_list, dtype=torch.long)

    sentence_mask_tensor = torch.zeros(max_sentences, dtype=torch.long)
    if num_sentences_processed > 0:
        sentence_mask_tensor[:num_sentences_processed] = 1

    return {
        'input_ids': input_ids_tensor.unsqueeze(0),
        'attention_mask': attention_mask_tensor.unsqueeze(0),
        'sentence_mask': sentence_mask_tensor.unsqueeze(0),
        'sentence_position_ids': sentence_position_ids_tensor.unsqueeze(0),
    }

# --- Helper Functions ---
def find_best_model_for_language(method_name_full, language, model_base_dir_template):
    model_dir = model_base_dir_template.format(method_name_full=method_name_full, language=language)
    summary_file = os.path.join(model_dir, "test_metrics_summary.json")

    if not os.path.exists(summary_file):
        print(f"Error: test_metrics_summary.json not found in {model_dir}")
        return None, None

    with open(summary_file, 'r', encoding='utf-8') as f:
        summary_data = json.load(f)

    best_f1_macro = -1.0
    best_model_run_key = None

    for model_key, metrics in summary_data.items():
        if model_key.startswith("model_") and isinstance(metrics, dict):
            f1_macro = metrics.get("f1_macro")
            if f1_macro is not None and f1_macro > best_f1_macro:
                best_f1_macro = f1_macro
                best_model_run_key = model_key
    
    if best_model_run_key is None:
        print(f"Error: No valid model runs found with f1_macro in {summary_file}")
        return None, None
    
    try:
        run_index = int(best_model_run_key.split('_')[-1])
        best_model_filename_constructed = f"best_model_{run_index}.pt"
        best_model_path_constructed = os.path.join(model_dir, best_model_filename_constructed)
    except ValueError:
        print(f"Warning: Could not parse run index from {best_model_run_key}. Will rely on 'model_path' in summary.")
        best_model_path_constructed = None
        
    best_model_path_from_summary = summary_data.get(best_model_run_key, {}).get("model_path")
    
    final_model_path = None
    if best_model_path_constructed and os.path.exists(best_model_path_constructed):
        final_model_path = best_model_path_constructed
        print(f"Best model for {language} ({best_model_run_key}) found at constructed path: {final_model_path} (F1 Macro: {best_f1_macro:.4f})")
    elif best_model_path_from_summary and os.path.exists(best_model_path_from_summary):
        final_model_path = best_model_path_from_summary
        print(f"Best model for {language} ({best_model_run_key}) found via summary path: {final_model_path} (F1 Macro: {best_f1_macro:.4f})")
    else:
        print(f"Error: Best model file not found. Constructed path: '{best_model_path_constructed}'. Summary path: '{best_model_path_from_summary}'.")
        return None, None

    return final_model_path, summary_data[best_model_run_key].get("f1_macro")


def collect_target_items_recursive(node, collection_list):
    """
    Recursively traverses the JSON structure and appends references to actual data items
    (dictionaries with 'article' and 'uuid' keys) to collection_list.
    """
    if isinstance(node, dict):
        # Check if this dict is an item we want to process
        if 'article' in node and 'uuid' in node: # Assuming 'uuid' also helps identify an item
            collection_list.append(node) # Append the reference to the item
        else:
            # If not an item itself, recurse on its values
            for key in node:
                collect_target_items_recursive(node[key], collection_list)
    elif isinstance(node, list):
        # If it's a list, recurse on its elements
        for element in node:
            collect_target_items_recursive(element, collection_list)


# --- Main Prediction Script ---
def main_predict():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    output_base_dir_specific = BASE_OUTPUT_DATA_DIR_TEMPLATE.format(method_slug=METHOD_SLUG)
    os.makedirs(output_base_dir_specific, exist_ok=True)
    print(f"Output will be saved in: {output_base_dir_specific}")

    input_files_config = [
        {"name": "serbian_test_balanced.json", "language": "serbian"},
        {"name": "serbian_train_val_balanced_0.json", "language": "serbian"},
        {"name": "serbian_train_val_balanced_1.json", "language": "serbian"},
        {"name": "serbian_train_val_balanced_2.json", "language": "serbian"},
        {"name": "slovenian_test_balanced.json", "language": "slovenian"},
        {"name": "slovenian_train_val_balanced_0.json", "language": "slovenian"},
        {"name": "slovenian_train_val_balanced_1.json", "language": "slovenian"},
        {"name": "slovenian_train_val_balanced_2.json", "language": "slovenian"},
    ]

    loaded_models_cache = {} 

    for file_config in input_files_config:
        input_filename = file_config["name"]
        language = file_config["language"]
        input_filepath = os.path.join(BASE_INPUT_DATA_DIR, input_filename)
        output_filepath = os.path.join(output_base_dir_specific, input_filename)

        print(f"\n--- Processing file: {input_filepath} (Language: {language}) ---")

        if not os.path.exists(input_filepath):
            print(f"Warning: Input file {input_filepath} not found. Skipping.")
            continue

        if language not in loaded_models_cache:
            print(f"Setting up model and tokenizer for {language}...")
            best_model_path, _ = find_best_model_for_language(FULL_METHOD_NAME, language, BASE_MODEL_DIR_TEMPLATE)
            if not best_model_path:
                print(f"Could not find best model for {language}. Skipping files for this language.")
                loaded_models_cache[language] = None 
                continue
            
            try:
                checkpoint = torch.load(best_model_path, map_location=device)
            except Exception as e:
                print(f"Error loading checkpoint from {best_model_path}: {e}")
                loaded_models_cache[language] = None
                continue
                
            loaded_args_dict = checkpoint.get('args', {})
            args_ns = argparse.Namespace(**loaded_args_dict) if isinstance(loaded_args_dict, dict) else loaded_args_dict

            model_identifier = getattr(args_ns, 'model_name', "xlm-roberta-base")
            
            tokenizer = AutoTokenizer.from_pretrained(model_identifier)
            special_tokens_dict = {'additional_special_tokens': [ASPECT_PLACEHOLDER]}
            tokenizer.add_special_tokens(special_tokens_dict)
            aspect_target_token_id = tokenizer.convert_tokens_to_ids(ASPECT_PLACEHOLDER)
            if aspect_target_token_id == tokenizer.unk_token_id:
                 print(f"CRITICAL: {ASPECT_PLACEHOLDER} is UNK. Model will not work correctly for {language}.")
                 loaded_models_cache[language] = None
                 continue

            model = SimplifiedDARTModel(
                model_name=model_identifier, tokenizer_len=len(tokenizer),
                interaction_layers=getattr(args_ns, 'interaction_layers', 2),
                interaction_heads=getattr(args_ns, 'interaction_heads', 8),
                aggregation_heads=getattr(args_ns, 'aggregation_heads', 4),
                max_sentences=getattr(args_ns, 'max_sentences', 32),
                final_mlp_hidden_dim=getattr(args_ns, 'final_mlp_hidden_dim', 256),
                dropout_rate=getattr(args_ns, 'dropout_rate', 0.2),
                num_classes=3, freeze_base=False
            ).to(device)
            
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            spacy_nlp_for_lang = load_spacy_model(language)
            if not spacy_nlp_for_lang:
                print(f"Failed to load SpaCy for {language}. Skipping files for this language.")
                loaded_models_cache[language] = None
                continue

            loaded_models_cache[language] = (model, tokenizer, torch.tensor([aspect_target_token_id], device=device), args_ns, spacy_nlp_for_lang)
            print(f"Model, tokenizer, and SpaCy for {language} loaded successfully.")

        cached_data = loaded_models_cache.get(language)
        if cached_data is None:
            print(f"Skipping {input_filename} due to previous failure in loading resources for {language}.")
            continue
        
        model, tokenizer, aspect_id_tensor, args_ns, spacy_nlp = cached_data
        
        try:
            with open(input_filepath, 'r', encoding='utf-8') as f:
                # Load the entire JSON structure from the file
                original_json_structure = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from {input_filepath}: {e}. Skipping this file.")
            continue
        except Exception as e:
            print(f"Error reading file {input_filepath}: {e}. Skipping this file.")
            continue


        # Collect all actual data items (references) from the loaded structure
        all_items_to_process = []
        collect_target_items_recursive(original_json_structure, all_items_to_process)

        if not all_items_to_process:
            print(f"No processable items (with 'article' and 'uuid') found in {input_filepath}.")
        else:
            print(f"Found {len(all_items_to_process)} items to process in {input_filepath}.")
            for item_ref in tqdm(all_items_to_process, desc=f"Predicting for {input_filename}"):
                try:
                    processed_input = preprocess_single_item(
                        item_ref, tokenizer, spacy_nlp, # item_ref is the dictionary
                        max_seq_length=getattr(args_ns, 'max_seq_length', 96),
                        max_sentences=getattr(args_ns, 'max_sentences', 32),
                        use_aspect_marker=getattr(args_ns, 'use_aspect_marker', False)
                    )
                    
                    input_ids = processed_input['input_ids'].to(device)
                    attention_mask = processed_input['attention_mask'].to(device)
                    sentence_mask = processed_input['sentence_mask'].to(device)
                    sentence_position_ids = processed_input['sentence_position_ids'].to(device)

                    with torch.no_grad():
                        logits = model(input_ids, attention_mask, sentence_mask, sentence_position_ids, aspect_id_tensor)
                        prediction_idx = torch.argmax(logits, dim=1).cpu().item() 
                        sentiment_prediction = prediction_idx - 1 
                    
                    item_ref[METHOD_KEY_FOR_JSON] = sentiment_prediction # Modify item_ref in place

                except Exception as e:
                    item_uuid = item_ref.get('uuid', 'UNKNOWN_ITEM_NO_UUID')
                    print(f"Error processing/predicting for item UUID '{item_uuid}': {e}")
                    # Add error information to the item itself
                    item_ref[METHOD_KEY_FOR_JSON] = "ERROR_DURING_PROCESSING"
            print(f"Finished processing {len(all_items_to_process)} items for {input_filename}.")

        # Save the modified original_json_structure (which contains the items with new predictions)
        try:
            with open(output_filepath, 'w', encoding='utf-8') as f:
                json.dump(original_json_structure, f, indent=4, ensure_ascii=False)
            print(f"Saved predictions to: {output_filepath}")
        except Exception as e:
            print(f"Error saving output file {output_filepath}: {e}")
            
        del original_json_structure, all_items_to_process # Explicitly delete large objects
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            
    print("\n--- Prediction process complete. ---")

if __name__ == "__main__":
    try:
        import transformers
        import spacy
    except ImportError as e:
        print(f"Missing essential library: {e}. Please install requirements.")
        exit(1)
        
    main_predict()