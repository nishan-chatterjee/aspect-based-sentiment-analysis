# predict_with_gcm_uncertainty_optimized_v2.py
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, logging as hf_logging
import argparse
from tqdm.auto import tqdm
import warnings
import gc
import statistics
import math
from collections import Counter

# --- Configuration ---
# Suppress unnecessary warnings
hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=FutureWarning, module="thinc.shims.pytorch")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*`torch.load` with `weights_only=False`.*")

# --- Global Variables & Configuration ---
ASPECT_PLACEHOLDER = "[ASPECT_TARGET]"
SPACY_MODELS = {}
N_MC_DROPOUT_SAMPLES = 10
BATCH_SIZE = 64 # Tunable parameter for batch processing

FULL_METHOD_NAME = "global-context-modelling/simplified-dart-xlmr"
METHOD_SLUG = FULL_METHOD_NAME.split('/')[-1]
METHOD_KEY_FOR_JSON = FULL_METHOD_NAME
METHOD_PROB_KEY_FOR_JSON = f"{METHOD_KEY_FOR_JSON}/probabilities"
METHOD_UNCERTAINTY_KEY_FOR_JSON = f"{METHOD_KEY_FOR_JSON}/uncertainty"

BASE_MODEL_DIR_TEMPLATE = "../models/{method_name_full}/{language}"
BASE_INPUT_DATA_DIR = "../data/final/balanced"
BASE_OUTPUT_DATA_DIR_TEMPLATE = "../data/final/balanced-predicted-softmax-uncertainty/{method_slug}"

# --- SpaCy Loading Functions (Unchanged) ---
def load_spacy_model(language):
    lang_key = language.lower()
    if lang_key in SPACY_MODELS and SPACY_MODELS[lang_key] is not None:
        return SPACY_MODELS[lang_key]

    print(f"Loading SpaCy model for language: {language}...")
    model_name = None
    if lang_key == "english": model_name = "en_core_web_sm"
    elif lang_key == "slovenian": model_name = "sl_core_news_sm"
    elif lang_key in ["croatian", "serbian"]: model_name = "hr_core_news_sm"
    else:
        print(f"Warning: Unsupported language '{language}'. Defaulting to 'en_core_web_sm'.")
        model_name = "en_core_web_sm"

    try:
        import spacy
        nlp = spacy.load(model_name)
        if 'sentencizer' not in nlp.pipe_names and 'senter' not in nlp.pipe_names:
            nlp.add_pipe('sentencizer', first=True)
        print(f"SpaCy model '{model_name}' loaded successfully.")
        SPACY_MODELS[lang_key] = nlp
        return nlp
    except Exception as e:
        print(f"Error loading SpaCy model '{model_name}': {e}")
        raise RuntimeError(f"SpaCy model loading failed: {e}")

def split_document(raw_text, nlp_model):
    if nlp_model is None: raise RuntimeError("SpaCy model not available.")
    if not raw_text or not isinstance(raw_text, str): return []
    try:
        doc = nlp_model(raw_text.strip())
        return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    except Exception as e:
        print(f"Error processing text with SpaCy: {e}\nProblematic text snippet: {raw_text[:200]}...")
        return []

# --- Model Definition (Unchanged) ---
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
            for param in self.base_model.parameters(): param.requires_grad = False
        
        self.sentence_pos_embedding = nn.Embedding(max_sentences + 1, self.hidden_dim, padding_idx=0)
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
        contextualized_sentence_summaries *= sentence_mask.unsqueeze(-1).float()

        aspect_placeholder_emb = self.base_model.get_input_embeddings()(aspect_target_token_id.to(input_ids.device))
        if aspect_placeholder_emb.ndim > 2: aspect_placeholder_emb = aspect_placeholder_emb.squeeze(0)
        if aspect_placeholder_emb.ndim == 1: aspect_placeholder_emb = aspect_placeholder_emb.unsqueeze(0)
        
        global_query = aspect_placeholder_emb.unsqueeze(0).repeat(batch_size, 1, 1)

        global_attn_output, _ = self.global_aggregation_attention(
            query=global_query, key=contextualized_sentence_summaries,
            value=contextualized_sentence_summaries, key_padding_mask=sentence_interact_padding_mask
        )
        aggregated_doc_representation = global_attn_output.squeeze(1)
        aggregated_doc_representation = self.dropout(aggregated_doc_representation)
        logits = self.classifier(aggregated_doc_representation)
        return logits

# --- Data Preprocessing & Dataset Definition ---
ASPECT_TAG_START = "<aspect>"
ASPECT_TAG_END = "</aspect>"

def _replace_aspect_with_placeholder_func(text):
    start_idx, end_idx = text.find(ASPECT_TAG_START), text.find(ASPECT_TAG_END)
    while -1 < start_idx < end_idx:
        text = text[:start_idx] + ASPECT_PLACEHOLDER + text[end_idx + len(ASPECT_TAG_END):]
        start_idx, end_idx = text.find(ASPECT_TAG_START), text.find(ASPECT_TAG_END)
    return text.replace(ASPECT_TAG_START, "").replace(ASPECT_TAG_END, "")

def preprocess_single_item(item_data, tokenizer, spacy_nlp, max_seq_length, max_sentences, use_aspect_marker):
    raw_article = item_data.get('article', '')
    article_with_placeholder = _replace_aspect_with_placeholder_func(raw_article)
    
    sentences = split_document(article_with_placeholder, spacy_nlp)
    if len(sentences) > max_sentences: sentences = sentences[:max_sentences]
    
    all_input_ids, all_attention_masks, sentence_pos_ids_list = [], [], []

    for sent_idx, sentence in enumerate(sentences):
        text_to_encode = f"{ASPECT_PLACEHOLDER} {tokenizer.sep_token} {sentence}" if use_aspect_marker else sentence
        encoding = tokenizer.encode_plus(
            text_to_encode, add_special_tokens=True, max_length=max_seq_length,
            padding='max_length', truncation=True, return_attention_mask=True, return_tensors='pt',
        )
        all_input_ids.append(encoding['input_ids'].squeeze(0))
        all_attention_masks.append(encoding['attention_mask'].squeeze(0))
        sentence_pos_ids_list.append(sent_idx + 1)
    
    num_processed = len(all_input_ids)
    pad_needed = max_sentences - num_processed
    if pad_needed > 0:
        pad_ids = torch.full((max_seq_length,), tokenizer.pad_token_id or 0, dtype=torch.long)
        pad_mask = torch.zeros((max_seq_length,), dtype=torch.long)
        all_input_ids.extend([pad_ids] * pad_needed)
        all_attention_masks.extend([pad_mask] * pad_needed)
        sentence_pos_ids_list.extend([0] * pad_needed)

    sentence_mask = torch.zeros(max_sentences, dtype=torch.long)
    if num_processed > 0: sentence_mask[:num_processed] = 1

    return {
        'input_ids': torch.stack(all_input_ids), 'attention_mask': torch.stack(all_attention_masks),
        'sentence_mask': sentence_mask, 'sentence_position_ids': torch.tensor(sentence_pos_ids_list, dtype=torch.long),
    }

class InferenceDataset(Dataset):
    def __init__(self, items, tokenizer, spacy_nlp, args):
        self.items = items
        self.tokenizer = tokenizer
        self.spacy_nlp = spacy_nlp
        self.args = args

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item_ref = self.items[idx]
        processed_data = preprocess_single_item(
            item_ref, self.tokenizer, self.spacy_nlp,
            max_seq_length=getattr(self.args, 'max_seq_length', 96),
            max_sentences=getattr(self.args, 'max_sentences', 32),
            use_aspect_marker=getattr(self.args, 'use_aspect_marker', False)
        )
        processed_data['item_ref'] = item_ref
        return processed_data

# --- NEW: Custom Collate Function ---
def custom_collate_fn(batch):
    """
    Custom collate function to handle batches of dictionaries 
    where one value ('item_ref') is not a tensor.
    """
    # Separate the non-tensor data (item references)
    item_refs = [d.pop('item_ref') for d in batch]
    
    # Use the default collate for the rest of the items, which are all tensors
    collated_tensordata = torch.utils.data.default_collate(batch)
    
    # Add the non-tensor data back into the final batch dictionary
    collated_tensordata['item_ref'] = item_refs
    
    return collated_tensordata

# --- Helper Functions ---
def find_all_model_checkpoints_for_language(method_name_full, language, model_base_dir_template):
    model_dir = model_base_dir_template.format(method_name_full=method_name_full, language=language)
    if not os.path.isdir(model_dir):
        print(f"Error: Model directory not found: {model_dir}"); return []
    found = sorted([os.path.join(model_dir, f) for f in os.listdir(model_dir) if f.startswith("best_model_") and f.endswith(".pt")])
    print(f"Found {len(found)} model checkpoints for {language}." if found else f"Error: No checkpoints found in {model_dir}")
    return found

def calculate_uncertainty_metrics(predictions_list, softmax_probs_list):
    total_samples = len(predictions_list)
    if total_samples == 0: return "ERROR_NO_SAMPLES", {}, {}
    try:
        # The mode can be a numpy.int64, so we cast it to a standard Python int.
        final_prediction = int(statistics.mode(predictions_list))
    except statistics.StatisticsError:
        # If there's a tie for the mode, pick the first prediction. Cast to int.
        final_prediction = int(predictions_list[0]) if predictions_list else 0

    # np.mean will produce a numpy array of numpy.float64
    mean_softmax_probs = np.mean(softmax_probs_list, axis=0)

    # Explicitly cast numpy floats to standard Python floats for JSON serialization.
    mean_prob_dict = {
        "Negative": float(mean_softmax_probs[0]),
        "Neutral": float(mean_softmax_probs[1]),
        "Positive": float(mean_softmax_probs[2])
    }

    counts = Counter(predictions_list)

    # The predictive entropy will be a numpy.float64, cast it to a Python float.
    # Using np.log2 is also safer and more idiomatic with numpy arrays.
    predictive_entropy = float(-np.sum([p * np.log2(p + 1e-9) for p in mean_softmax_probs]))

    uncertainty = {
        # The division might result in a numpy.float64, so cast it.
        "confidence_score": float(counts[final_prediction] / total_samples) if total_samples > 0 else 0.0,
        # counts.get() returns standard Python ints, which are fine.
        "prediction_distribution": {"-1": counts.get(-1, 0), "0": counts.get(0, 0), "1": counts.get(1, 0)},
        "predictive_entropy": predictive_entropy,
        "total_mc_samples": total_samples # This is already a Python int.
    }
    return final_prediction, mean_prob_dict, uncertainty

def collect_target_items_recursive(node, collection_list):
    if isinstance(node, dict):
        if 'article' in node and 'uuid' in node: collection_list.append(node)
        else: [collect_target_items_recursive(node[k], collection_list) for k in node]
    elif isinstance(node, list): [collect_target_items_recursive(e, collection_list) for e in node]

# --- Main Prediction Script ---
def main_predict():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    output_base_dir = BASE_OUTPUT_DATA_DIR_TEMPLATE.format(method_slug=METHOD_SLUG)
    os.makedirs(output_base_dir, exist_ok=True)
    print(f"Output will be saved in: {output_base_dir}")

    input_files_config = [
        {"name": "serbian_test_balanced.json", "language": "serbian"}, {"name": "serbian_train_val_balanced_0.json", "language": "serbian"},
        {"name": "serbian_train_val_balanced_1.json", "language": "serbian"}, {"name": "serbian_train_val_balanced_2.json", "language": "serbian"},
        {"name": "slovenian_test_balanced.json", "language": "slovenian"}, {"name": "slovenian_train_val_balanced_0.json", "language": "slovenian"},
        {"name": "slovenian_train_val_balanced_1.json", "language": "slovenian"}, {"name": "slovenian_train_val_balanced_2.json", "language": "slovenian"},
    ]
    loaded_resources_cache = {}

    for file_config in input_files_config:
        input_filename, language = file_config["name"], file_config["language"]
        input_filepath = os.path.join(BASE_INPUT_DATA_DIR, input_filename)
        output_filepath = os.path.join(output_base_dir, input_filename)

        print(f"\n--- Processing file: {input_filepath} (Language: {language}) ---")
        if not os.path.exists(input_filepath):
            print(f"Warning: Input file not found, skipping: {input_filepath}"); continue

        if language not in loaded_resources_cache:
            print(f"Setting up tokenizer and SpaCy for {language}...")
            checkpoints = find_all_model_checkpoints_for_language(FULL_METHOD_NAME, language, BASE_MODEL_DIR_TEMPLATE)
            if not checkpoints:
                print(f"No models for {language}, skipping subsequent files."); loaded_resources_cache[language] = None; continue
            try:
                first_ckpt = torch.load(checkpoints[0], map_location='cpu')
                args_ns = argparse.Namespace(**first_ckpt.get('args', {}))
                model_id = getattr(args_ns, 'model_name', "xlm-roberta-base")
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                tokenizer.add_special_tokens({'additional_special_tokens': [ASPECT_PLACEHOLDER]})
                aspect_id = tokenizer.convert_tokens_to_ids(ASPECT_PLACEHOLDER)
                if aspect_id == tokenizer.unk_token_id: raise RuntimeError("Aspect placeholder is UNK.")
                spacy_nlp = load_spacy_model(language)
                if not spacy_nlp: raise RuntimeError("SpaCy model failed to load.")
                loaded_resources_cache[language] = {
                    "checkpoints": checkpoints, "tokenizer": tokenizer, "args": args_ns,
                    "aspect_id_tensor": torch.tensor(aspect_id, device=device), "spacy_nlp": spacy_nlp
                }
                print(f"Tokenizer and SpaCy for {language} loaded.")
            except Exception as e:
                print(f"Error initializing resources for {language}: {e}"); loaded_resources_cache[language] = None; continue
        
        cached_res = loaded_resources_cache.get(language)
        if not cached_res: print(f"Skipping {input_filename} due to resource failure."); continue
        
        with open(input_filepath, 'r', encoding='utf-8') as f: original_json = json.load(f)
        all_items = []; collect_target_items_recursive(original_json, all_items)
        if not all_items: print(f"No processable items in {input_filepath}."); continue
        
        print(f"Found {len(all_items)} items. Creating DataLoader...")
        dataset = InferenceDataset(all_items, cached_res["tokenizer"], cached_res["spacy_nlp"], cached_res["args"])
        data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True, collate_fn=custom_collate_fn)

        results = {item['uuid']: {'predictions': [], 'softmaxes': []} for item in all_items}

        for model_path in cached_res["checkpoints"]:
            print(f"\n--- Predicting with model: {os.path.basename(model_path)} ---")
            try:
                ckpt = torch.load(model_path, map_location=device)
                model = SimplifiedDARTModel(
                    model_name=getattr(cached_res["args"], 'model_name', "xlm-roberta-base"), tokenizer_len=len(cached_res["tokenizer"]),
                    interaction_layers=getattr(cached_res["args"], 'interaction_layers', 2), interaction_heads=getattr(cached_res["args"], 'interaction_heads', 8),
                    aggregation_heads=getattr(cached_res["args"], 'aggregation_heads', 4), max_sentences=getattr(cached_res["args"], 'max_sentences', 32),
                    final_mlp_hidden_dim=getattr(cached_res["args"], 'final_mlp_hidden_dim', 256), dropout_rate=getattr(cached_res["args"], 'dropout_rate', 0.2),
                ).to(device)
                model.load_state_dict(ckpt['model_state_dict']); model.train()

                # The autocast context manager enables automatic mixed precision
                with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
                    for batch in tqdm(data_loader, desc=f"Inference (AMP)"):
                        item_refs = batch.pop('item_ref')
                        for k, v in batch.items(): batch[k] = v.to(device)
                        for _ in range(N_MC_DROPOUT_SAMPLES):
                            logits = model(**batch, aspect_target_token_id=cached_res["aspect_id_tensor"])
                            # Note: Softmax is often kept in float32 for stability, autocast handles this.
                            probs = F.softmax(logits, dim=1).cpu() 
                            preds = torch.argmax(probs, dim=1).numpy() - 1
                            for i, item_ref in enumerate(item_refs):
                                results[item_ref['uuid']]['predictions'].append(preds[i])
                                results[item_ref['uuid']]['softmaxes'].append(probs[i].numpy())
            except Exception as e: print(f"Error with model {model_path}: {e}")
            finally: del model, ckpt; gc.collect(); torch.cuda.empty_cache()

        print("\nAggregating results...")
        for item_ref in tqdm(all_items, desc="Finalizing"):
            if (uuid := item_ref.get('uuid')) in results:
                final_pred, mean_probs, uncertainty = calculate_uncertainty_metrics(results[uuid]['predictions'], results[uuid]['softmaxes'])
                item_ref[METHOD_KEY_FOR_JSON], item_ref[METHOD_PROB_KEY_FOR_JSON], item_ref[METHOD_UNCERTAINTY_KEY_FOR_JSON] = final_pred, mean_probs, uncertainty
            else: item_ref[METHOD_KEY_FOR_JSON] = "ERROR_NOT_PROCESSED"

        print(f"Saving to {output_filepath}...")
        with open(output_filepath, 'w', encoding='utf-8') as f: json.dump(original_json, f, indent=4, ensure_ascii=False)
        del original_json, all_items, dataset, data_loader, results; gc.collect()

    print("\n--- Prediction process complete. ---")

if __name__ == "__main__":
    main_predict()