from __future__ import unicode_literals, print_function
import spacy
import warnings
import json
import os
import numpy as np
import plotly.graph_objects as go
from tqdm import tqdm
import plotly.io as pio
import gc # For garbage collection
import torch # For torch.cuda.empty_cache()

# Set default Plotly template
pio.templates.default = "plotly_white"

# Suppress FutureWarnings and general UserWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning) # Suppresses UserWarning from all modules

# --- Configuration ---
BASE_DATA_DIR = "../data/subsets"
PLOT_OUTPUT_DIR = "../data/plots"
SLOVENIAN_DATA_FILE = os.path.join(BASE_DATA_DIR, "slovenian.json")
SERBIAN_DATA_FILE = os.path.join(BASE_DATA_DIR, "serbian.json")

# SpaCy processing parameters
NLP_BATCH_SIZE = 256  # Reduced batch size
MAX_CHARS_PER_CHUNK = 150000 # Max characters per chunk for very long documents

# --- Utility Functions ---
def load_json_data(file_path):
    """Loads data from a JSON file."""
    print(f"Loading data from {file_path}...")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"Successfully loaded {len(data)} items.")
        return data
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return []
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {file_path}")
        return []

def load_spacy_model(language, model_cache={}):
    """Loads and caches SpaCy models."""
    if language in model_cache:
        return model_cache[language]

    model_name = ""
    fallback_module = None

    if language.lower() == "slovenian":
        model_name = "sl_core_news_sm"
        try:
            import sl_core_news_sm
            fallback_module = sl_core_news_sm
        except ImportError:
            print(f"Warning: Fallback module 'sl_core_news_sm' not found for Slovenian.")
    elif language.lower() == "serbian": # Using Croatian model for Serbian
        model_name = "hr_core_news_sm" # Note: This is a large non-transformer model. If OOM persists, consider hr_core_news_md or sm
        try:
            import hr_core_news_sm
            fallback_module = hr_core_news_sm
        except ImportError:
            print(f"Warning: Fallback module 'hr_core_news_sm' not found for Serbian/Croatian.")
    else:
        print(f"Error: Language '{language}' not supported for SpaCy model loading.")
        return None

    print(f"Loading SpaCy model '{model_name}' for {language}...")
    try:
        nlp = spacy.load(model_name)
        print(f"Successfully loaded '{model_name}'.")
    except OSError:
        print(f"Warning: SpaCy model '{model_name}' not found. Attempting fallback...")
        if fallback_module:
            try:
                nlp = fallback_module.load()
                print(f"Successfully loaded '{model_name}' via fallback module.")
            except Exception as e_fallback:
                print(f"Error: Fallback loading for '{model_name}' failed: {e_fallback}")
                return None
        else:
            print(f"Error: SpaCy model '{model_name}' not found and no fallback available. Please install it (e.g., python -m spacy download {model_name}).")
            return None
    except Exception as e:
        print(f"An unexpected error occurred loading SpaCy model '{model_name}': {e}")
        return None
    
    model_cache[language] = nlp
    return nlp

def batch_get_sentence_counts(articles, spacy_model, language_name):
    """
    Processes a list of articles in batches using nlp.pipe(), with chunking for long documents,
    and returns sentence counts.
    """
    if not spacy_model:
        print(f"Error: SpaCy model for {language_name} is not loaded. Cannot count sentences.")
        return []
    if not articles:
        print(f"Warning: No articles provided for {language_name}.")
        return []

    original_articles_text = [str(article) if article is not None else "" for article in articles]
    
    texts_for_pipe = []
    original_article_indices = [] 

    print(f"Preprocessing and chunking articles for {language_name} (max chunk: {MAX_CHARS_PER_CHUNK} chars)...")
    for i, article_text in enumerate(tqdm(original_articles_text, desc=f"Chunking ({language_name})")):
        if not article_text.strip(): 
            texts_for_pipe.append("") 
            original_article_indices.append(i) 
            continue

        if len(article_text) > MAX_CHARS_PER_CHUNK:
            num_chunks = (len(article_text) + MAX_CHARS_PER_CHUNK - 1) // MAX_CHARS_PER_CHUNK
            for j in range(num_chunks):
                chunk = article_text[j*MAX_CHARS_PER_CHUNK : (j+1)*MAX_CHARS_PER_CHUNK]
                if chunk.strip(): 
                    texts_for_pipe.append(chunk)
                    original_article_indices.append(i) 
        else:
            texts_for_pipe.append(article_text)
            original_article_indices.append(i) 

    if not texts_for_pipe:
        print(f"Warning: No processable text found after chunking for {language_name}.")
        return [0] * len(original_articles_text)

    print(f"Processing {len(texts_for_pipe)} text chunks for {language_name} using SpaCy (batch size {NLP_BATCH_SIZE})...")
    
    chunk_sentence_counts = []
    components_to_disable = [] 
    
    doc_stream = spacy_model.pipe(texts_for_pipe, batch_size=NLP_BATCH_SIZE, disable=components_to_disable)
    
    for doc in tqdm(doc_stream, total=len(texts_for_pipe), desc=f"Counting sentences in chunks ({language_name})"):
        try:
            if not doc.text.strip():
                chunk_sentence_counts.append(0)
                continue
            sentences = list(doc.sents)
            chunk_sentence_counts.append(len(sentences))
        except Exception as e:
            print(f"Warning: Error processing a document chunk for {language_name}: {e}. Appending 0 sentences.")
            chunk_sentence_counts.append(0)

    final_sentence_counts = [0] * len(original_articles_text)
    for i in range(len(texts_for_pipe)):
        if texts_for_pipe[i] == "" and original_article_indices[i] < len(final_sentence_counts):
            pass 
        elif original_article_indices[i] < len(final_sentence_counts): 
            final_sentence_counts[original_article_indices[i]] += chunk_sentence_counts[i]
            
    print(f"Cleaning up memory for {language_name}...")
    gc.collect()
    if torch.cuda.is_available(): # Corrected check
        print("Attempting to empty PyTorch CUDA cache...")
        torch.cuda.empty_cache()
            
    return final_sentence_counts

def plot_sentence_distribution(sentence_counts, language_name, output_file_path):
    """
    Generates and saves a histogram of sentence counts per document using Plotly.
    """
    if not sentence_counts:
        print(f"No sentence counts to plot for {language_name}.")
        return

    counts_array = np.array(sentence_counts)
    
    if len(counts_array) == 0:
        print(f"Empty counts_array for {language_name}, skipping plot.")
        return

    q1 = np.percentile(counts_array, 25)
    median = np.median(counts_array)
    q3 = np.percentile(counts_array, 75)
    iqr = q3 - q1
    mean_val = np.mean(counts_array)
    min_val = np.min(counts_array)
    max_val = np.max(counts_array)
    std_dev = np.std(counts_array)

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=counts_array,
        name=f'{language_name.capitalize()} Sentence Counts',
        nbinsx=100, 
        marker_color='#1f77b4'
    ))

    # Main title for the plot (centered, shorter)
    main_plot_title = f'Distribution of Sentences per Document - {language_name.capitalize()}'
    
    # Statistics text block (will be positioned separately or as part of subtitle)
    # For better control, we can use annotations, or adjust the main title's y position
    # and then add the stats block.
    # However, Plotly titles support <br> for line breaks, so we can keep it in one block.

    stats_text_block = (
        f'Total Documents: {len(counts_array):,}<br>'
        f'Mean: {mean_val:.2f}, Median: {median:.2f}, Std Dev: {std_dev:.2f}<br>'
        f'Min: {min_val}, Max: {max_val}<br>'
        f'IQR: {iqr:.2f} (Q1: {q1:.2f}, Q3: {q3:.2f})'
    )
    
    # Combine main title and stats block for the title text
    full_title_text = f'{main_plot_title}<br><br>{stats_text_block}'


    fig.update_layout(
        title_text=full_title_text,
        title_x=0.95,  # Position title towards the right. 0.5 is center, 1 is far right.
                       # Using 0.95 to give a little padding from the edge.
        title_y=0.9,   # Position title towards the top. 0.9 is a common value.
        title_xanchor='right', # Anchor the title text by its right edge at title_x
        title_yanchor='top',   # Anchor the title text by its top edge at title_y
        
        xaxis_title_text='Number of Sentences per Document',
        yaxis_title_text='Number of Documents (Frequency)',
        bargap=0.1,
        # Optional: Adjust margins if title feels cramped
        # margin=dict(t=150) # Increase top margin if title is long and needs space
    )

    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    try:
        fig.write_image(output_file_path, width=1200, height=700)
        print(f"Plot saved to {output_file_path}")
    except Exception as e:
        print(f"Error saving plot to {output_file_path}: {e}")
        print("Please ensure you have `kaleido` installed for static image export (`pip install -U kaleido`).")

def print_summary_stats(language_name, counts):
    if not counts:
        print(f"No counts to summarize for {language_name}.")
        return
    counts_array = np.array(counts)
    print(f"\n--- Summary Statistics for {language_name.capitalize()} ---")
    print(f"Total Documents Processed: {len(counts_array):,}")
    print(f"Average Sentences/Doc:   {np.mean(counts_array):.2f}")
    print(f"Median Sentences/Doc:    {np.median(counts_array):.2f}")
    print(f"Standard Deviation:      {np.std(counts_array):.2f}")
    print(f"Min Sentences/Doc:       {np.min(counts_array)}")
    print(f"Max Sentences/Doc:       {np.max(counts_array)}")
    print(f"25th Percentile (Q1):    {np.percentile(counts_array, 25):.2f}")
    print(f"75th Percentile (Q3):    {np.percentile(counts_array, 75):.2f}")
    print(f"Interquartile Range (IQR): {np.percentile(counts_array, 75) - np.percentile(counts_array, 25):.2f}")
    print("--------------------------------------")

# --- Main Execution ---
def main():
    print("Starting sentence counting and visualization process...")

    # This variable is mostly for informational logging now
    _ = False # Placeholder for gpu_is_active, not strictly needed for logic below
    print("Attempting to enable GPU for SpaCy...")
    try:
        if spacy.prefer_gpu(): 
            print("GPU preferred by SpaCy. Processing will leverage GPU if models and hardware support it.")
            _ = True 
            if not torch.cuda.is_available():
                print("Warning: SpaCy prefers GPU, but torch.cuda.is_available() is False. Check PyTorch CUDA setup.")
        else:
            print("GPU not available or not preferred by SpaCy. Processing will be on CPU.")
    except Exception as e:
        print(f"An error occurred while trying to configure GPU for SpaCy: {e}")
        print("Proceeding with CPU-based processing.")

    # Load data
    slovenian_data = load_json_data(SLOVENIAN_DATA_FILE)
    serbian_data = load_json_data(SERBIAN_DATA_FILE)

    if not slovenian_data and not serbian_data:
        print("No data loaded. Exiting.")
        return

    # Load SpaCy models
    nlp_slovenian = load_spacy_model("slovenian")
    nlp_serbian = load_spacy_model("serbian") 

    # Process Slovenian data
    if slovenian_data and nlp_slovenian:
        slovenian_articles = [item.get('article') for item in slovenian_data if isinstance(item, dict)]
        slovenian_sentence_counts = batch_get_sentence_counts(slovenian_articles, nlp_slovenian, "slovenian")
        if slovenian_sentence_counts:
            slovenian_plot_path = os.path.join(PLOT_OUTPUT_DIR, "slovenian-sentences-per-document-spacy.svg")
            plot_sentence_distribution(slovenian_sentence_counts, "slovenian", slovenian_plot_path)
            print_summary_stats("Slovenian", slovenian_sentence_counts)
    else:
        print("Skipping Slovenian processing due to missing data or model.")

    # Process Serbian data
    if serbian_data and nlp_serbian:
        serbian_articles = [item.get('article') for item in serbian_data if isinstance(item, dict)]
        serbian_sentence_counts = batch_get_sentence_counts(serbian_articles, nlp_serbian, "serbian")
        if serbian_sentence_counts:
            serbian_plot_path = os.path.join(PLOT_OUTPUT_DIR, "serbian-sentences-per-document-spacy.svg")
            plot_sentence_distribution(serbian_sentence_counts, "serbian", serbian_plot_path)
            print_summary_stats("Serbian", serbian_sentence_counts)
    else:
        print("Skipping Serbian processing due to missing data or model.")

    print("\nProcess finished.")

if __name__ == "__main__":
    main()