import spacy
import pytextrank # noqa: F401
import re
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from scipy.spatial.distance import cosine
import json
from tqdm import tqdm
import os
import torch

# --- spaCy Model Loading ---
NLP_MODELS = {} # Global cache for loaded models

def get_spacy_model(language: str) -> spacy.language.Language:
    """Load spaCy model based on language, using GPU if available."""
    global NLP_MODELS
    if language in NLP_MODELS:
        return NLP_MODELS[language]

    print(f"Loading spaCy model for {language}...")
    model_name = ""
    if language.lower() == "slovenian":
        model_name = "sl_core_news_sm"
    elif language.lower() in ["croatian", "serbo-croatian", "serbian"]:
        model_name = "hr_core_news_sm"
    else:
        raise ValueError(f"Unsupported language: {language}")

    try:
        nlp = spacy.load(model_name)
        print(f"Loaded {model_name}. Pipeline: {nlp.pipe_names}")
    except OSError:
        raise Exception(f"Please install {model_name}: python -m spacy download {model_name}")

    if "textrank" not in nlp.pipe_names:
        nlp.add_pipe("textrank")
        print("Added textrank pipe.")
    
    NLP_MODELS[language] = nlp
    return nlp

# --- Vector and Similarity Calculations ---
def to_numpy(array_obj: Any) -> np.ndarray:
    """Converts a spaCy vector (which might be CuPy array on GPU) to a NumPy array."""
    if hasattr(array_obj, "get"):
        return array_obj.get()
    return np.asarray(array_obj)

def get_aspect_query_vector(nlp: spacy.language.Language, aspect: str, keywords: List[str]) -> np.ndarray:
    vectors = []
    texts_to_process = [aspect] + [kw.replace("*", "") for kw in keywords if kw.replace("*", "")]
    docs = list(nlp.pipe(texts_to_process))
    for doc in docs:
        if doc.has_vector and doc.vector_norm:
            vectors.append(to_numpy(doc.vector))
    if not vectors:
        vector_size = nlp.vocab.vectors_length if nlp.vocab.vectors_length > 0 else 300
        return np.zeros(vector_size)
    return np.mean(vectors, axis=0)

def compute_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    if np.all(vec1 == 0) or np.all(vec2 == 0):
        return 0.0
    vec1 = vec1.flatten()
    vec2 = vec2.flatten()
    similarity = 1 - cosine(vec1, vec2)
    return float(similarity) if not np.isnan(similarity) else 0.0

# --- Aspect Detection (Modified to return count) ---
def count_aspect_mentions(sentence: spacy.tokens.Span, aspect: str, keywords: List[str]) -> int:
    """Counts mentions of aspect and keywords in the sentence."""
    mention_count = 0
    sentence_lower = sentence.text.lower()
    aspect_lower = aspect.lower()

    # Count <aspect>aspect_lower</aspect> occurrences
    mention_count += len(re.findall(f"<aspect>{re.escape(aspect_lower)}</aspect>", sentence_lower))
        
    # Count plain aspect string (avoid double counting if already tagged)
    # This regex tries to find aspect_lower not already within <aspect> tags
    # It's a simplification; robustly avoiding double counts with regex can be tricky
    plain_aspect_matches = re.finditer(r'(?<!<aspect>)' + re.escape(aspect_lower) + r'(?!</aspect>)', sentence_lower)
    for match in plain_aspect_matches:
        # Check if this match is part of an already counted tagged aspect
        is_part_of_tagged = False
        for tagged_match in re.finditer(f"<aspect>{re.escape(aspect_lower)}</aspect>", sentence_lower):
            if tagged_match.start() <= match.start() and tagged_match.end() >= match.end():
                is_part_of_tagged = True
                break
        if not is_part_of_tagged:
            mention_count += 1
            
    # Count keywords with wildcard handling
    for keyword in keywords:
        clean_keyword_for_regex = keyword.lower().replace("*", ".*")
        search_pattern = clean_keyword_for_regex
        try:
            if not clean_keyword_for_regex.startswith(('.*', r'\b')):
                 search_pattern = r'\b' + search_pattern
            if not clean_keyword_for_regex.endswith(('.*', r'\b')):
                 search_pattern = search_pattern + r'\b'
            mention_count += len(re.findall(search_pattern, sentence_lower))
        except re.error: # Fallback for invalid regex
             mention_count += sentence_lower.count(keyword.lower().replace("*", ""))


    # Count aspect or keywords in named entities (could lead to double counting if not careful)
    # For simplicity, we'll assume the direct string/regex counts are primary.
    # If NER is very reliable and distinct, this could be added more carefully.
    # For now, to avoid overcounting, we'll skip this part or make it very conservative.
    # Consider if the above methods already cover most NER cases.
    # Example: if "Banka Poštanska Štedionica" is an ORG, it's likely caught by aspect_lower or keyword search.

    return mention_count

# --- Summarization Core Logic (Modified) ---
def generate_summaries_batch(
    dataset: List[Dict[str, Any]],
    target_summary_char_length: int,
    min_sentence_score_for_adaptive_summary: float,
    semantic_diversity_threshold: float,
    jaccard_diversity_threshold: float, # Added for clarity
    aspect_mention_boost_factor: float,
    dataset_name: str = "dataset",
    batch_size: int = 64,
    language_to_use: Optional[str] = None
) -> List[Dict[str, Any]]:
    if not dataset:
        return []

    final_language = language_to_use or (dataset[0].get("language") if dataset else None)
    if not final_language:
        raise ValueError("Language not specified and cannot be inferred from dataset.")
    nlp = get_spacy_model(final_language)

    articles = [item["article"] for item in dataset]
    aspects = [item.get("aspect", "") for item in dataset]
    keywords_list = [item.get("keywords", []) for item in dataset]
    
    processed_docs = []
    print(f"Processing {len(articles)} articles for {dataset_name} with nlp.pipe (batch_size={batch_size})...")
    for doc in tqdm(nlp.pipe(articles, batch_size=batch_size), total=len(articles), desc=f"spaCy Processing {dataset_name}"):
        processed_docs.append(doc)

    print(f"Calculating summaries for {dataset_name}...")
    results = []
    for item, doc, aspect, keywords in tqdm(zip(dataset, processed_docs, aspects, keywords_list), total=len(dataset), desc=f"Generating Summaries {dataset_name}"):
        current_item = item.copy()
        sentences = list(doc.sents)
        if not sentences:
            current_item["summary"] = ""
            results.append(current_item)
            continue

        query_vector = get_aspect_query_vector(nlp, aspect, keywords)
        sentence_scores: List[Tuple[spacy.tokens.Span, float, int, np.ndarray]] = [] # Added sentence vector

        for i, sent in enumerate(sentences):
            tr_score = 0.0
            if doc._.textrank:
                for phrase in doc._.textrank.summary(limit_phrases=30, limit_sentences=0):
                    if phrase.text in sent.text or sent.text in phrase.text:
                        tr_score = max(tr_score, phrase.rank)
            
            sent_vector_np = np.zeros(nlp.vocab.vectors_length or 300) # Default vector
            sim_score = 0.0
            if sent.has_vector and sent.vector_norm:
                sent_vector_np = to_numpy(sent.vector)
                sim_score = compute_similarity(sent_vector_np, query_vector)

            position_score = 1.0 / (1 + 0.1 * i)
            
            words = len(sent.text.split())
            length_score = 0.0
            if 5 <= words <= 35: length_score = 1.0
            elif words < 5: length_score = words / 5.0
            else: length_score = max(0.0, 1.0 - (words - 35.0) / 20.0)

            # Weighted Aspect Presence Boost
            mention_count = count_aspect_mentions(sent, aspect, keywords)
            boost = 1.0 + (aspect_mention_boost_factor * mention_count)

            combined_score = (0.3 * tr_score + 0.3 * sim_score +
                             0.2 * position_score + 0.2 * length_score) * boost
            sentence_scores.append((sent, combined_score, i, sent_vector_np))

        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        
        selected_sentences_data: List[Tuple[spacy.tokens.Span, float, int, np.ndarray]] = []
        current_summary_char_count = 0
        sent_words_cache = {}

        for cand_sent, cand_score, cand_pos, cand_vector in sentence_scores:
            # Adaptive length and min score check
            if cand_score < min_sentence_score_for_adaptive_summary and selected_sentences_data: # Allow first sentence even if low score if nothing else
                continue 
            
            # Check if adding this sentence would make the summary too long
            # Allow slight overshoot for the last sentence.
            if selected_sentences_data and (current_summary_char_count + len(cand_sent.text) > target_summary_char_length * 1.25): # Allow 25% overshoot for the last sentence
                if current_summary_char_count >= target_summary_char_length * 0.75: # If we already have a decent summary
                     break # Stop if summary is already reasonably long
                # else, continue to see if this sentence is short enough to fit without much overshoot

            if not selected_sentences_data: # Always add the highest scoring sentence if it meets min score
                if cand_score >= min_sentence_score_for_adaptive_summary or not sentence_scores: # or if it's the only option
                    selected_sentences_data.append((cand_sent, cand_score, cand_pos, cand_vector))
                    current_summary_char_count += len(cand_sent.text)
                    continue
                else: # Top sentence doesn't meet min score, likely no good summary
                    break


            # Stop if target length is reached and we have at least one sentence
            if current_summary_char_count >= target_summary_char_length and selected_sentences_data:
                break

            # Diversity Checks
            cand_sent_text_lower = cand_sent.text.lower()
            if cand_sent_text_lower not in sent_words_cache:
                sent_words_cache[cand_sent_text_lower] = set(re.findall(r'\w+', cand_sent_text_lower))
            current_cand_words = sent_words_cache[cand_sent_text_lower]
            if not current_cand_words: continue

            is_too_similar_jaccard = False
            for sel_sent_data in selected_sentences_data:
                sel_text_lower = sel_sent_data[0].text.lower()
                if sel_text_lower not in sent_words_cache:
                    sent_words_cache[sel_text_lower] = set(re.findall(r'\w+', sel_text_lower))
                sel_words = sent_words_cache[sel_text_lower]
                if not sel_words: continue
                
                intersection = len(current_cand_words.intersection(sel_words))
                union = len(current_cand_words.union(sel_words))
                jaccard = intersection / union if union > 0 else 0
                if jaccard > jaccard_diversity_threshold:
                    is_too_similar_jaccard = True
                    break
            if is_too_similar_jaccard:
                continue

            is_too_similar_semantic = False
            if cand_vector.any(): # Check if candidate vector is not all zeros
                for sel_sent_data in selected_sentences_data:
                    sel_vector = sel_sent_data[3]
                    if sel_vector.any(): # Check if selected vector is not all zeros
                        semantic_sim = compute_similarity(cand_vector, sel_vector)
                        if semantic_sim > semantic_diversity_threshold:
                            is_too_similar_semantic = True
                            break
            if is_too_similar_semantic:
                continue
            
            # If all checks pass, add the sentence
            selected_sentences_data.append((cand_sent, cand_score, cand_pos, cand_vector))
            current_summary_char_count += len(cand_sent.text)

        selected_sentences_data.sort(key=lambda x: x[2]) # Sort by original position
        current_item["summary"] = " ".join(s.text.strip() for s, _, _, _ in selected_sentences_data)
        results.append(current_item)
        
    return results

# --- Dataset Processing Orchestration (Revised) ---
def process_dataset_files_revised(
    base_data_dir: str,
    # New parameters for summarization logic
    target_summary_char_length: int,
    min_sentence_score_for_adaptive_summary: float,
    semantic_diversity_threshold: float,
    jaccard_diversity_threshold: float,
    aspect_mention_boost_factor: float,
    batch_size: int = 64
):
    dataset_configs = [
        {"lang_for_spacy": "Serbo-Croatian", "input_file_name": "serbian.json", "output_file_name": "serbian.json", "tqdm_name": "Serbian Dataset"},
        {"lang_for_spacy": "Slovenian", "input_file_name": "slovenian.json", "output_file_name": "slovenian.json", "tqdm_name": "Slovenian Dataset"},
    ]

    input_dir = os.path.join(base_data_dir, "subsets")
    output_dir = os.path.join(base_data_dir, "final", "summaries", "textrank")
    os.makedirs(output_dir, exist_ok=True)

    print(f"--- Starting TextRank Summarization (Enhanced) ---")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Target summary char length: ~{target_summary_char_length}")
    print(f"Min sentence score for adaptive summary: {min_sentence_score_for_adaptive_summary}")
    print(f"Semantic diversity threshold: {semantic_diversity_threshold}")
    print(f"Jaccard diversity threshold: {jaccard_diversity_threshold}")
    print(f"Aspect mention boost factor: {aspect_mention_boost_factor}")
    print(f"SpaCy batch size: {batch_size}\n")

    for config in dataset_configs:
        lang_spacy = config['lang_for_spacy']
        tqdm_name = config['tqdm_name']
        in_filepath = os.path.join(input_dir, config['input_file_name'])
        out_filepath = os.path.join(output_dir, config['output_file_name'])

        if os.path.exists(out_filepath):
            print(f"Skipping {out_filepath}, output file already exists.")
            continue
        if not os.path.exists(in_filepath):
            print(f"Warning: Input file not found {in_filepath}. Skipping this file.")
            continue
            
        print(f"\nProcessing {tqdm_name} from: {in_filepath}")
        try:
            with open(in_filepath, "r", encoding="utf-8") as f:
                dataset_items = json.load(f)
            if not dataset_items:
                print(f"Warning: No data found in {in_filepath}. Skipping.")
                continue
            
            processed_data = generate_summaries_batch(
                dataset_items,
                target_summary_char_length=target_summary_char_length,
                min_sentence_score_for_adaptive_summary=min_sentence_score_for_adaptive_summary,
                semantic_diversity_threshold=semantic_diversity_threshold,
                jaccard_diversity_threshold=jaccard_diversity_threshold,
                aspect_mention_boost_factor=aspect_mention_boost_factor,
                dataset_name=tqdm_name,
                batch_size=batch_size,
                language_to_use=lang_spacy
            )
            
            print(f"Saving results for {tqdm_name} to {out_filepath}...")
            with open(out_filepath, "w", encoding="utf-8") as f:
                json.dump(processed_data, f, ensure_ascii=False, indent=4)
            print(f"Successfully saved {out_filepath}")

        except Exception as e:
            print(f"Error processing {tqdm_name} ({in_filepath}): {str(e)}")
            import traceback
            traceback.print_exc()
            
    print("\n--- All TextRank Summarization Processing Complete ---")


if __name__ == "__main__":
    # --- Configuration ---
    BASE_DATA_DIRECTORY = "../data"
    SPACY_BATCH_SIZE = 128

    # --- New Summarization Parameters ---
    TARGET_SUMMARY_CHAR_LENGTH = 350  # Approximate target length for the summary
    MIN_SENTENCE_SCORE_FOR_ADAPTIVE_SUMMARY = 0.05 # Sentences below this score (after the first) won't be added
    SEMANTIC_DIVERSITY_THRESHOLD = 0.80  # If cosine_sim(sent1_vec, sent2_vec) > threshold, considered too similar
    JACCARD_DIVERSITY_THRESHOLD = 0.6    # Existing Jaccard threshold (can be tuned)
    ASPECT_MENTION_BOOST_FACTOR = 0.075  # Boost = 1.0 + (factor * num_mentions)

    # --- GPU Setup ---
    if torch.cuda.is_available():
        try:
            torch.backends.cudnn.benchmark = True
            print(f"GPU available: {torch.cuda.get_device_name(0)}")
            spacy.prefer_gpu()
            print("SpaCy configured to prefer GPU (requires cupy and compatible CUDA).")
        except Exception as e:
             print(f"Warning: Error during spaCy GPU setup: {e}. SpaCy will use CPU.")
    else:
        print("No GPU available, spaCy will use CPU.")
    
    # --- Run Processing ---
    process_dataset_files_revised(
        BASE_DATA_DIRECTORY,
        target_summary_char_length=TARGET_SUMMARY_CHAR_LENGTH,
        min_sentence_score_for_adaptive_summary=MIN_SENTENCE_SCORE_FOR_ADAPTIVE_SUMMARY,
        semantic_diversity_threshold=SEMANTIC_DIVERSITY_THRESHOLD,
        jaccard_diversity_threshold=JACCARD_DIVERSITY_THRESHOLD,
        aspect_mention_boost_factor=ASPECT_MENTION_BOOST_FACTOR,
        batch_size=SPACY_BATCH_SIZE
    )