#!/usr/bin/env python3
"""Build the qualitative selective-deferral analysis notebook."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(os.environ.get("ABSA_RELEASE_ROOT", Path(__file__).resolve().parents[2])).resolve()
OUT_NOTEBOOK = ROOT / "reviews" / "qualitative-analysis.ipynb"


def code_cell(source: str) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def markdown_cell(source: str) -> dict[str, Any]:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def build_notebook() -> dict[str, Any]:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-absa")
    cells = [
        markdown_cell(
            "# Qualitative Analysis: Selective LLM Deferral Hard Cases\n\n"
            "This notebook supports the qualitative/error-analysis story for the revised AspectBench paper. "
            "It focuses on the two selective-deferral systems currently used in the main plots: Slovenian "
            "Longformer Masked and Serbo-Croatian mDeBERTa-v3 Masked. The heavy work is delegated to "
            "`reviews/qualitative_pipeline.py`, which builds cached case tables, embeddings, clusters, "
            "representative examples, and cluster-label prompts."
        ),
        markdown_cell(
            "## How to Generate the Cache\n\n"
            "Run this on a GPU node for BGE-M3 embeddings. The case extraction step is CPU-only; embeddings and "
            "UMAP/HDBSCAN are the expensive parts.\n\n"
            "```bash\n"
            "PYTHON_BIN=/path/to/absa/bin/python\n"
            "CUDA_VISIBLE_DEVICES=1 \"$PYTHON_BIN\" reviews/qualitative_pipeline.py \\\n"
            "  --step all \\\n"
            "  --embedding-model models/embeddings/bge-m3 \\\n"
            "  --view local_windows \\\n"
            "  --scope llm_called \\\n"
            "  --device cuda \\\n"
            "  --batch-size 8 \\\n"
            "  --max-length 2048 \\\n"
            "  --llm-api-base http://127.0.0.1:8000/v1 \\\n"
            "  --llm-model gemma27b\n"
            "```\n\n"
            "If `umap-learn` or `hdbscan` are missing, the helper falls back to PCA + MiniBatchKMeans. "
            "That fallback is useful for debugging, but the paper figure should preferably use UMAP + HDBSCAN. "
            "BERTopic is optional and can be run afterwards with `--step topics` on the same cached embeddings."
        ),
        code_cell(
            "from __future__ import annotations\n\n"
            "import json\n"
            "import math\n"
            "import os\n"
            "import subprocess\n"
            "from collections import Counter\n"
            "from pathlib import Path\n\n"
            "os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib-absa')\n\n"
            "import matplotlib.pyplot as plt\n"
            "import numpy as np\n"
            "import pandas as pd\n"
            "from IPython.display import Markdown, display\n\n"
            "ROOT = Path.cwd().resolve()\n"
            "if ROOT.name == 'reviews':\n"
            "    ROOT = ROOT.parent\n"
            "elif not (ROOT / 'reviews').exists():\n"
            "    raise RuntimeError(f'Run this notebook from the project root or reviews/: {ROOT}')\n\n"
            "OUT_DIR = ROOT / 'reviews/qualitative-analysis'\n"
            "PIPELINE = ROOT / 'reviews/qualitative_pipeline.py'\n"
            "MODEL_SLUG = 'bge-m3'\n"
            "VIEW = 'local_windows'\n"
            "SCOPE = 'llm_called'\n"
            "PREFIX = f'{MODEL_SLUG}_{VIEW}_{SCOPE}'\n"
            "CASES_PATH = OUT_DIR / 'cases.jsonl'\n"
            "RUNS_PATH = OUT_DIR / 'selected_runs.json'\n"
            "CLUSTERS_PATH = OUT_DIR / f'clusters_{PREFIX}.csv'\n"
            "SUMMARY_PATH = OUT_DIR / f'cluster_summary_{PREFIX}.csv'\n"
            "REPS_PATH = OUT_DIR / f'cluster_representatives_{PREFIX}.jsonl'\n"
            "PROMPTS_PATH = OUT_DIR / f'cluster_label_prompts_{PREFIX}.jsonl'\n"
            "LABELS_PATH = OUT_DIR / f'cluster_labels_{PREFIX}.jsonl'\n"
            "TOPICS_PATH = OUT_DIR / f'bertopic_topics_{PREFIX}.csv'\n"
            "TOPIC_DOCS_PATH = OUT_DIR / f'bertopic_docs_{PREFIX}.csv'\n\n"
            "MULTI_ASPECT_PATH = OUT_DIR / 'multi_aspect_diagnostics.csv'\n\n"
            "LABEL_ORDER = ['expert_correct_llm_correct', 'expert_correct_llm_wrong', 'expert_wrong_llm_correct', 'both_wrong', 'expert_correct_llm_abstain', 'expert_wrong_llm_abstain']\n"
            "OUTCOME_COLORS = {\n"
            "    'expert_correct_llm_correct': '#2ca02c',\n"
            "    'expert_correct_llm_wrong': '#d62728',\n"
            "    'expert_wrong_llm_correct': '#1f77b4',\n"
            "    'both_wrong': '#7f7f7f',\n"
            "    'expert_correct_llm_abstain': '#bcbd22',\n"
            "    'expert_wrong_llm_abstain': '#9467bd',\n"
            "}\n\n"
            "NUMERIC_FEATURES = [\n"
            "    'doc_char_len', 'doc_token_len', 'sentence_count', 'declared_mentions', 'exact_mention_count',\n"
            "    'target_sentence_count', 'mention_density_per_1k_tokens', 'first_mention_ratio',\n"
            "    'aspect_token_count', 'aspect_is_acronym', 'quote_count', 'question_count', 'percent_count',\n"
            "    'contrast_count', 'negation_count', 'local_negation_count', 'positive_cue_count',\n"
            "    'negative_cue_count', 'local_positive_cue_count', 'local_negative_cue_count',\n"
            "    'min_sentiment_cue_distance', 'reported_speech_count', 'legal_financial_count',\n"
            "    'primary_confidence', 'primary_entropy', 'hard_score', 'num_aux_disagree', 'num_aux',\n"
            "]\n"
        ),
        code_cell(
            "def read_jsonl(path):\n"
            "    if not path.exists():\n"
            "        return []\n"
            "    with path.open('r', encoding='utf-8') as f:\n"
            "        return [json.loads(line) for line in f if line.strip()]\n\n"
            "def load_cache():\n"
            "    cases = pd.DataFrame(read_jsonl(CASES_PATH))\n"
            "    runs = pd.DataFrame(json.load(open(RUNS_PATH, encoding='utf-8'))) if RUNS_PATH.exists() else pd.DataFrame()\n"
            "    clusters = pd.read_csv(CLUSTERS_PATH) if CLUSTERS_PATH.exists() else pd.DataFrame()\n"
            "    summary = pd.read_csv(SUMMARY_PATH) if SUMMARY_PATH.exists() else pd.DataFrame()\n"
            "    reps = pd.DataFrame(read_jsonl(REPS_PATH))\n"
            "    prompts = pd.DataFrame(read_jsonl(PROMPTS_PATH))\n"
            "    labels = pd.DataFrame(read_jsonl(LABELS_PATH))\n"
            "    topics = pd.read_csv(TOPICS_PATH) if TOPICS_PATH.exists() else pd.DataFrame()\n"
            "    topic_docs = pd.read_csv(TOPIC_DOCS_PATH) if TOPIC_DOCS_PATH.exists() else pd.DataFrame()\n"
            "    return cases, runs, clusters, summary, reps, prompts, labels, topics, topic_docs\n\n"
            "def command_block(step='all'):\n"
            "    return '\\n'.join([\n"
            "        'PYTHON_BIN=/path/to/absa/bin/python',\n"
            "        f'CUDA_VISIBLE_DEVICES=1 \"$PYTHON_BIN\" {PIPELINE.relative_to(ROOT)} \\\\',\n"
            "        f'  --step {step} \\\\',\n"
            "        '  --embedding-model models/embeddings/bge-m3 \\\\',\n"
            "        f'  --view {VIEW} \\\\',\n"
            "        f'  --scope {SCOPE} \\\\',\n"
            "        '  --device cuda \\\\',\n"
            "        '  --batch-size 8 \\\\',\n"
            "        '  --max-length 2048 \\\\',\n"
            "        '  --llm-api-base http://127.0.0.1:8000/v1 \\\\',\n"
            "        '  --llm-model gemma27b',\n"
            "    ])\n\n"
            "cases, runs, clusters, cluster_summary, representatives, prompts, cluster_labels, topics, topic_docs = load_cache()\n"
            "if cases.empty:\n"
            "    display(Markdown('No qualitative cache found yet. Generate it with:'))\n"
            "    print(command_block('all'))\n"
            "else:\n"
            "    display(Markdown(f'Loaded **{len(cases):,}** qualitative cases from `{CASES_PATH.relative_to(ROOT)}`.'))\n"
            "    if not runs.empty:\n"
            "        display(runs)\n"
            "if clusters.empty:\n"
            "    display(Markdown('No cluster cache found yet. After cases/embeddings are generated, run:'))\n"
            "    print(command_block('cluster'))\n"
        ),
        markdown_cell(
            "## Outcome Overview\n\n"
            "The key qualitative split is C vs D: `expert_wrong_llm_correct` shows cases rescued by selective LLM "
            "deferral, while `both_wrong` shows the residual hard set. We also keep raw abstentions separate, then "
            "use the resolved outcome when evaluating abstention as a successful escalation."
        ),
        markdown_cell(
            "## Selected Gates\n\n"
            "The cache is built from one selected gate per language/expert. Unless `--gate-rate` was fixed manually, "
            "the helper chose the best available discrete run by abstain-resolved Macro-F1, then QWK. These are not "
            "continuous bins; they are selected from the existing `0.10`, `0.20`, and `0.30` gate directories."
        ),
        code_cell(
            "if not runs.empty:\n"
            "    gate_cols = ['language_display', 'expert_display', 'prompt_variant', 'gate_rate', 'llm_call_rate', 'f1_macro_abstain_resolved', 'qwk_abstain_resolved', 'abstain_rate', 'override_rate', 'predictions_path']\n"
            "    display(runs[gate_cols].style.format({'gate_rate': '{:.2f}', 'llm_call_rate': '{:.3f}', 'f1_macro_abstain_resolved': '{:.4f}', 'qwk_abstain_resolved': '{:.4f}', 'abstain_rate': '{:.3f}', 'override_rate': '{:.3f}'}, na_rep='-'))\n"
        ),
        markdown_cell(
            "## Fixed-Gate Comparison\n\n"
            "The main cache above can mix gates because it selects the best discrete gate per language. The fixed-gate "
            "directories compare the same gate levels across both languages. This checks whether the qualitative pattern "
            "changes as the routed subset widens from the hardest 10% to 20% and 30%."
        ),
        code_cell(
            "def load_fixed_gate_outputs():\n"
            "    run_frames = []\n"
            "    outcome_frames = []\n"
            "    for gate_dir in sorted((ROOT / 'reviews').glob('qualitative-analysis-gate-*')):\n"
            "        try:\n"
            "            fixed_gate = float(gate_dir.name.rsplit('-', 1)[-1])\n"
            "        except ValueError:\n"
            "            continue\n"
            "        runs_path = gate_dir / 'selected_runs.json'\n"
            "        if runs_path.exists():\n"
            "            run_df = pd.DataFrame(json.load(open(runs_path, encoding='utf-8')))\n"
            "            run_df.insert(0, 'fixed_gate_dir', gate_dir.name)\n"
            "            run_df.insert(1, 'fixed_gate', fixed_gate)\n"
            "            run_frames.append(run_df)\n"
            "        cluster_paths = list(gate_dir.glob('clusters_*_llm_called.csv'))\n"
            "        if cluster_paths:\n"
            "            cdf = pd.read_csv(cluster_paths[0])\n"
            "            counts = cdf.groupby(['language_display', 'error_type_resolved']).size().rename('n').reset_index()\n"
            "            counts['fixed_gate'] = fixed_gate\n"
            "            counts['rate'] = counts['n'] / counts.groupby('language_display')['n'].transform('sum')\n"
            "            outcome_frames.append(counts)\n"
            "    return (\n"
            "        pd.concat(run_frames, ignore_index=True) if run_frames else pd.DataFrame(),\n"
            "        pd.concat(outcome_frames, ignore_index=True) if outcome_frames else pd.DataFrame(),\n"
            "    )\n\n"
            "gate_runs, gate_outcomes = load_fixed_gate_outputs()\n"
            "if gate_runs.empty:\n"
            "    display(Markdown('No fixed-gate qualitative-analysis-gate-* directories found.'))\n"
            "else:\n"
            "    display(gate_runs[['fixed_gate', 'language_display', 'expert_display', 'prompt_variant', 'gate_rate', 'llm_call_rate', 'f1_macro_abstain_resolved', 'qwk_abstain_resolved', 'abstain_rate', 'override_rate']].style.format({'fixed_gate': '{:.2f}', 'gate_rate': '{:.2f}', 'llm_call_rate': '{:.3f}', 'f1_macro_abstain_resolved': '{:.4f}', 'qwk_abstain_resolved': '{:.4f}', 'abstain_rate': '{:.3f}', 'override_rate': '{:.3f}'}, na_rep='-'))\n"
            "    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharex=True)\n"
            "    for language, sub in gate_runs.groupby('language_display'):\n"
            "        sub = sub.sort_values('fixed_gate')\n"
            "        axes[0].plot(sub['fixed_gate'], sub['f1_macro_abstain_resolved'], marker='o', label=language)\n"
            "        axes[1].plot(sub['fixed_gate'], sub['qwk_abstain_resolved'], marker='o', label=language)\n"
            "    axes[0].set_title('Fixed-gate selective deferral: Macro-F1')\n"
            "    axes[1].set_title('Fixed-gate selective deferral: QWK')\n"
            "    for ax in axes:\n"
            "        ax.set_xlabel('Gate rate')\n"
            "        ax.grid(alpha=0.25)\n"
            "        ax.legend()\n"
            "    fig.tight_layout()\n"
            "    plt.show()\n"
            "if not gate_outcomes.empty:\n"
            "    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)\n"
            "    for ax, (language, sub) in zip(axes, gate_outcomes.groupby('language_display')):\n"
            "        for outcome, group in sub.groupby('error_type_resolved'):\n"
            "            group = group.sort_values('fixed_gate')\n"
            "            ax.plot(group['fixed_gate'], group['rate'], marker='o', color=OUTCOME_COLORS.get(outcome, None), label=outcome)\n"
            "        ax.set_title(f'{language}: outcome share by fixed gate')\n"
            "        ax.set_xlabel('Gate rate')\n"
            "        ax.set_ylabel('Share within routed set')\n"
            "        ax.grid(alpha=0.25)\n"
            "        ax.legend(fontsize=8)\n"
            "    fig.tight_layout()\n"
            "    plt.show()\n"
        ),
        code_cell(
            "if not cases.empty:\n"
            "    hard = cases[cases['llm_called'].astype(bool)].copy()\n"
            "    display(Markdown(f'Hard-gated subset: **{len(hard):,}** / {len(cases):,} cases.'))\n"
            "    overview = hard.groupby(['language_display', 'error_type_resolved']).size().rename('n').reset_index()\n"
            "    overview['rate_within_language'] = overview['n'] / overview.groupby('language_display')['n'].transform('sum')\n"
            "    display(overview.sort_values(['language_display', 'n'], ascending=[True, False]).style.format({'rate_within_language': '{:.3f}'}))\n"
            "    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), sharey=True)\n"
            "    for ax, (language, sub) in zip(axes, hard.groupby('language_display')):\n"
            "        counts = sub['error_type_resolved'].value_counts().reindex(LABEL_ORDER).dropna()\n"
            "        ax.bar(counts.index, counts.values, color=[OUTCOME_COLORS.get(x, '#333333') for x in counts.index])\n"
            "        ax.set_title(f'{language}: selective-deferral hard cases')\n"
            "        ax.set_ylabel('Cases')\n"
            "        ax.tick_params(axis='x', rotation=35)\n"
            "        ax.grid(axis='y', alpha=0.25)\n"
            "    fig.tight_layout()\n"
            "    plt.show()\n"
        ),
        markdown_cell(
            "## Feature Contrasts\n\n"
            "These tables are meant to produce candidate explanations, not final claims. Large positive effect sizes "
            "mean the first group has more of that feature than the comparison group. The most paper-relevant contrast "
            "is rescued cases versus both-wrong cases."
        ),
        code_cell(
            "def cohens_d(a, b):\n"
            "    a = pd.to_numeric(a, errors='coerce').dropna().to_numpy(dtype=float)\n"
            "    b = pd.to_numeric(b, errors='coerce').dropna().to_numpy(dtype=float)\n"
            "    if len(a) < 2 or len(b) < 2:\n"
            "        return np.nan\n"
            "    pooled = math.sqrt(((len(a) - 1) * np.var(a, ddof=1) + (len(b) - 1) * np.var(b, ddof=1)) / (len(a) + len(b) - 2))\n"
            "    return (np.mean(a) - np.mean(b)) / pooled if pooled else np.nan\n\n"
            "def contrast_table(df, group_a, group_b, label_a, label_b):\n"
            "    rows = []\n"
            "    a = df[df['error_type_resolved'] == group_a]\n"
            "    b = df[df['error_type_resolved'] == group_b]\n"
            "    for feature in NUMERIC_FEATURES:\n"
            "        if feature not in df.columns:\n"
            "            continue\n"
            "        rows.append({\n"
            "            'feature': feature,\n"
            "            f'{label_a}_mean': pd.to_numeric(a[feature], errors='coerce').mean(),\n"
            "            f'{label_b}_mean': pd.to_numeric(b[feature], errors='coerce').mean(),\n"
            "            'cohens_d': cohens_d(a[feature], b[feature]),\n"
            "            f'{label_a}_n': pd.to_numeric(a[feature], errors='coerce').notna().sum(),\n"
            "            f'{label_b}_n': pd.to_numeric(b[feature], errors='coerce').notna().sum(),\n"
            "        })\n"
            "    return pd.DataFrame(rows).assign(abs_d=lambda x: x['cohens_d'].abs()).sort_values('abs_d', ascending=False)\n\n"
            "if not cases.empty:\n"
            "    hard = cases[cases['llm_called'].astype(bool)].copy()\n"
            "    for language, sub in hard.groupby('language_display'):\n"
            "        display(Markdown(f'### {language}: rescued vs residual hard cases'))\n"
            "        table = contrast_table(sub, 'expert_wrong_llm_correct', 'both_wrong', 'rescued', 'both_wrong')\n"
            "        display(table.head(15).style.format({'rescued_mean': '{:.3f}', 'both_wrong_mean': '{:.3f}', 'cohens_d': '{:+.2f}', 'abs_d': '{:.2f}'}, na_rep='-'))\n"
            "        display(Markdown(f'### {language}: gold/action mix'))\n"
            "        mix = sub.groupby(['error_type_resolved', 'gold_label', 'action']).size().rename('n').reset_index()\n"
            "        display(mix.sort_values('n', ascending=False).head(25))\n"
        ),
        markdown_cell(
            "## Representative Hard Cases\n\n"
            "Quick manual reading samples. These are sorted by high `hard_score`, so they should be genuinely routed "
            "to the LLM rather than incidental label changes."
        ),
        code_cell(
            "def trim(text, n=420):\n"
            "    text = ' '.join(str(text).split())\n"
            "    return text[:n] + ('...' if len(text) > n else '')\n\n"
            "if not cases.empty:\n"
            "    hard = cases[cases['llm_called'].astype(bool)].copy()\n"
            "    hard['snippet'] = hard['local_windows'].map(trim)\n"
            "    cols = ['case_id', 'language_display', 'aspect', 'gold_label', 'expert_label', 'llm_label', 'action', 'primary_confidence', 'hard_score', 'snippet']\n"
            "    for outcome in ['expert_wrong_llm_correct', 'both_wrong', 'expert_correct_llm_wrong']:\n"
            "        sub = hard[hard['error_type_resolved'] == outcome].sort_values('hard_score', ascending=False)\n"
            "        if sub.empty:\n"
            "            continue\n"
            "        display(Markdown(f'### {outcome}'))\n"
            "        display(sub[cols].head(10).style.format({'primary_confidence': '{:.3f}', 'hard_score': '{:.3f}'}, na_rep='-'))\n"
        ),
        markdown_cell(
            "## Cluster Visualization\n\n"
            "The scatter uses the cached 2D projection from the helper. Color by resolved outcome first; cluster IDs are "
            "shown in the summary table below. If the plot looks too fragmented with the fallback KMeans path, install "
            "`umap-learn` and `hdbscan` in the environment and rerun the helper."
        ),
        code_cell(
            "if not clusters.empty:\n"
            "    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True, sharey=True)\n"
            "    for ax, (language, sub) in zip(axes, clusters.groupby('language_display')):\n"
            "        for outcome, group in sub.groupby('error_type_resolved'):\n"
            "            ax.scatter(group['umap_x'], group['umap_y'], s=12, alpha=0.68, color=OUTCOME_COLORS.get(outcome, '#333333'), label=outcome)\n"
            "        ax.set_title(f'{language}: hard-case embedding map')\n"
            "        ax.set_xlabel('UMAP-1')\n"
            "        ax.set_ylabel('UMAP-2')\n"
            "        ax.grid(alpha=0.2)\n"
            "        ax.legend(fontsize=8, loc='best')\n"
            "    fig.tight_layout()\n"
            "    plt.show()\n"
            "    fig, ax = plt.subplots(figsize=(8, 5.5))\n"
            "    scatter = ax.scatter(clusters['umap_x'], clusters['umap_y'], c=clusters['cluster'], s=10, alpha=0.65, cmap='tab20')\n"
            "    ax.set_title('Hard-case clusters, both languages')\n"
            "    ax.set_xlabel('UMAP-1')\n"
            "    ax.set_ylabel('UMAP-2')\n"
            "    ax.grid(alpha=0.2)\n"
            "    plt.colorbar(scatter, ax=ax, label='cluster')\n"
            "    fig.tight_layout()\n"
            "    plt.show()\n"
        ),
        markdown_cell(
            "## Class-Conditioned Embedding Views\n\n"
            "The global scatter is usually dominated by entity/topic templates. These class-conditioned panels ask a "
            "narrower question: within negative, neutral, and positive hard cases, do rescued and residual errors occupy "
            "different regions? Strong overlap means the hard cases are not cleanly separable by semantic embedding alone."
        ),
        code_cell(
            "def plot_class_conditioned_umap(df, language):\n"
            "    sub = df[df['language_display'] == language].copy()\n"
            "    classes = ['negative', 'neutral', 'positive']\n"
            "    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True, sharey=True)\n"
            "    for ax, label in zip(axes, classes):\n"
            "        cls = sub[sub['gold_label'] == label]\n"
            "        if cls.empty:\n"
            "            ax.set_title(f'{language}: {label} (n=0)')\n"
            "            continue\n"
            "        for outcome, group in cls.groupby('error_type_resolved'):\n"
            "            ax.scatter(group['umap_x'], group['umap_y'], s=16, alpha=0.72, color=OUTCOME_COLORS.get(outcome, '#333333'), label=outcome)\n"
            "        ax.set_title(f'{language}: gold {label} (n={len(cls)})')\n"
            "        ax.grid(alpha=0.2)\n"
            "        ax.set_xlabel('UMAP-1')\n"
            "    axes[0].set_ylabel('UMAP-2')\n"
            "    handles, labels = axes[-1].get_legend_handles_labels()\n"
            "    if handles:\n"
            "        fig.legend(handles, labels, loc='upper center', ncol=min(4, len(labels)), frameon=False)\n"
            "    fig.tight_layout(rect=[0, 0, 1, 0.92])\n"
            "    return fig\n\n"
            "def plot_class_outcome_bars(df):\n"
            "    languages = list(df['language_display'].drop_duplicates())\n"
            "    fig, axes = plt.subplots(1, len(languages), figsize=(7 * len(languages), 4.8), sharey=True)\n"
            "    axes = np.atleast_1d(axes)\n"
            "    for ax, language in zip(axes, languages):\n"
            "        sub = df[df['language_display'] == language]\n"
            "        tab = pd.crosstab(sub['gold_label'], sub['error_type_resolved'], normalize='index').reindex(['negative', 'neutral', 'positive']).fillna(0)\n"
            "        tab = tab[[col for col in LABEL_ORDER if col in tab.columns]]\n"
            "        bottom = np.zeros(len(tab))\n"
            "        x = np.arange(len(tab))\n"
            "        for col in tab.columns:\n"
            "            ax.bar(x, tab[col].to_numpy(), bottom=bottom, color=OUTCOME_COLORS.get(col, '#333333'), label=col)\n"
            "            bottom += tab[col].to_numpy()\n"
            "        ax.set_xticks(x)\n"
            "        ax.set_xticklabels(tab.index)\n"
            "        ax.set_title(f'{language}: outcome mix within gold class')\n"
            "        ax.set_ylabel('Share within class')\n"
            "        ax.grid(axis='y', alpha=0.25)\n"
            "    handles, labels = axes[-1].get_legend_handles_labels()\n"
            "    fig.legend(handles, labels, loc='upper center', ncol=min(4, len(labels)), frameon=False)\n"
            "    fig.tight_layout(rect=[0, 0, 1, 0.90])\n"
            "    return fig\n\n"
            "def class_rescue_table(df):\n"
            "    rows = []\n"
            "    for (language, gold_label), group in df.groupby(['language_display', 'gold_label']):\n"
            "        counts = group['error_type_resolved'].value_counts()\n"
            "        expert_wrong = counts.get('expert_wrong_llm_correct', 0) + counts.get('both_wrong', 0)\n"
            "        expert_correct = counts.get('expert_correct_llm_correct', 0) + counts.get('expert_correct_llm_wrong', 0)\n"
            "        rescued = counts.get('expert_wrong_llm_correct', 0)\n"
            "        degraded = counts.get('expert_correct_llm_wrong', 0)\n"
            "        rows.append({\n"
            "            'Language': language,\n"
            "            'Gold class': gold_label,\n"
            "            'Routed n': len(group),\n"
            "            'Expert-wrong n': expert_wrong,\n"
            "            'LLM rescues': rescued,\n"
            "            'Rescue rate | expert wrong': rescued / expert_wrong if expert_wrong else np.nan,\n"
            "            'Expert-correct n': expert_correct,\n"
            "            'LLM-induced errors': degraded,\n"
            "            'Error rate | expert correct': degraded / expert_correct if expert_correct else np.nan,\n"
            "        })\n"
            "    return pd.DataFrame(rows).sort_values(['Language', 'Gold class'])\n\n"
            "if not clusters.empty:\n"
            "    for language in clusters['language_display'].drop_duplicates():\n"
            "        fig = plot_class_conditioned_umap(clusters, language)\n"
            "        plt.show()\n"
            "    fig = plot_class_outcome_bars(clusters)\n"
            "    plt.show()\n"
            "    display(pd.crosstab([clusters['language_display'], clusters['gold_label']], clusters['error_type_resolved']))\n"
            "    display(Markdown('### Class-Conditioned Rescue and Degradation Rates'))\n"
            "    display(Markdown('Blue points correspond to routed examples where the expert was wrong and the LLM-resolved decision was correct. Red points correspond to routed examples where the expert was correct and the LLM override made the final decision wrong. The rates below condition on whether the expert was wrong/correct, and the routed `n` column should be used before making class-specific claims.'))\n"
            "    display(class_rescue_table(clusters).style.format({'Rescue rate | expert wrong': '{:.3f}', 'Error rate | expert correct': '{:.3f}'}, na_rep='-'))\n"
        ),
        markdown_cell(
            "## Multi-Aspect / Entity Competition Diagnostic\n\n"
            "This cached diagnostic uses the aspect keyword inventories in `data/slovenian-aspects.json` and "
            "`data/serbian-aspects.json` to count how many known target aspects appear in each document. It is a "
            "heuristic lexical scan, not entity linking: wildcard keywords can overmatch, and some true aliases may "
            "still be missed. The goal is only to test whether multi-aspect documents are visibly enriched among "
            "routed/residual errors."
        ),
        code_cell(
            "if MULTI_ASPECT_PATH.exists():\n"
            "    multi = pd.read_csv(MULTI_ASPECT_PATH)\n"
            "    for col in ['llm_called', 'multi_aspect_document', 'other_aspect_document', 'target_detected', 'expert_correct', 'llm_resolved_correct']:\n"
            "        if col in multi.columns and multi[col].dtype == object:\n"
            "            multi[col] = multi[col].astype(str).str.lower().map({'true': True, 'false': False})\n"
            "    display(Markdown(f'Loaded multi-aspect diagnostics for **{len(multi):,}** qualitative cases.'))\n"
            "    routed_summary = (\n"
            "        multi.groupby(['language_display', 'llm_called'])\n"
            "        .agg(n=('case_id', 'size'), other_aspect_rate=('other_aspect_document', 'mean'), mean_detected_aspects=('num_detected_aspects', 'mean'), mean_other_aspects=('num_other_detected_aspects', 'mean'))\n"
            "        .reset_index()\n"
            "    )\n"
            "    display(Markdown('### Multi-aspect rate: routed vs non-routed'))\n"
            "    display(routed_summary.style.format({'other_aspect_rate': '{:.3f}', 'mean_detected_aspects': '{:.2f}', 'mean_other_aspects': '{:.2f}'}))\n"
            "    hard_multi = multi[multi['llm_called'] == True].copy()\n"
            "    outcome_summary = (\n"
            "        hard_multi.groupby(['language_display', 'error_type_resolved'])\n"
            "        .agg(n=('case_id', 'size'), other_aspect_rate=('other_aspect_document', 'mean'), mean_detected_aspects=('num_detected_aspects', 'mean'), mean_other_aspects=('num_other_detected_aspects', 'mean'))\n"
            "        .reset_index()\n"
            "        .sort_values(['language_display', 'other_aspect_rate'], ascending=[True, False])\n"
            "    )\n"
            "    display(Markdown('### Multi-aspect rate by routed outcome'))\n"
            "    display(outcome_summary.style.format({'other_aspect_rate': '{:.3f}', 'mean_detected_aspects': '{:.2f}', 'mean_other_aspects': '{:.2f}'}))\n"
            "    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), sharey=True)\n"
            "    for ax, (language, sub) in zip(axes, hard_multi.groupby('language_display')):\n"
            "        order = [x for x in LABEL_ORDER if x in set(sub['error_type_resolved'])]\n"
            "        vals = sub.groupby('error_type_resolved')['other_aspect_document'].mean().reindex(order)\n"
            "        ax.bar(np.arange(len(vals)), vals.to_numpy(), color=[OUTCOME_COLORS.get(x, '#333333') for x in vals.index])\n"
            "        ax.set_xticks(np.arange(len(vals)))\n"
            "        ax.set_xticklabels(vals.index, rotation=35, ha='right')\n"
            "        ax.set_title(f'{language}: other known aspect in document')\n"
            "        ax.set_ylabel('Rate')\n"
            "        ax.grid(axis='y', alpha=0.25)\n"
            "    fig.tight_layout()\n"
            "    plt.show()\n"
            "    examples = hard_multi[(hard_multi['error_type_resolved'] == 'both_wrong') & (hard_multi['num_other_detected_aspects'] >= 3)]\n"
            "    example_cols = ['language_display', 'case_id', 'aspect', 'gold_label', 'error_type_resolved', 'num_detected_aspects', 'num_other_detected_aspects', 'detected_aspects_sample']\n"
            "    if not examples.empty:\n"
            "        display(Markdown('### Example residual errors with several detected known aspects'))\n"
            "        display(examples[example_cols].head(12))\n"
            "    display(Markdown('Interpretation note: if the routed rows are not enriched for `other_aspect_document`, multi-entity competition is a recurring qualitative pattern rather than the main mechanism behind selective deferral.'))\n"
            "else:\n"
            "    display(Markdown('No `multi_aspect_diagnostics.csv` found. Generate it with the helper or rerun the accompanying diagnostic script before using this section.'))\n"
        ),
        markdown_cell(
            "## Cluster Summary and Representative Examples\n\n"
            "Use this as the first pass for qualitative labels: a cluster is interesting when it is enriched for a useful "
            "outcome type, a language, a class, or interpretable numeric features such as mentions, contrast markers, "
            "cue distance, low confidence, or long documents."
        ),
        code_cell(
            "def purity_table(df, key):\n"
            "    rows = []\n"
            "    for cluster_id, group in df.groupby('cluster'):\n"
            "        vc = group[key].value_counts(normalize=True)\n"
            "        rows.append({'cluster': cluster_id, 'n': len(group), 'dominant': vc.index[0], 'purity': vc.iloc[0]})\n"
            "    return pd.DataFrame(rows).sort_values(['purity', 'n'], ascending=[False, False])\n\n"
            "if not clusters.empty:\n"
            "    display(Markdown('### Cluster Purity Diagnostics'))\n"
            "    for key in ['error_type_resolved', 'gold_label', 'language_display']:\n"
            "        display(Markdown(f'#### Dominant {key} per cluster'))\n"
            "        display(purity_table(clusters, key).head(20).style.format({'purity': '{:.3f}'}))\n"
            "    display(Markdown('Interpretation note: high class/topic/language purity with lower outcome purity means the embedding clusters are organizing by subject matter and language more than by whether the LLM rescues the expert.'))\n"
        ),
        code_cell(
            "if not cluster_summary.empty:\n"
            "    display(cluster_summary.sort_values('n', ascending=False).head(30).style.format({'mean_confidence': '{:.3f}', 'mean_doc_tokens': '{:.1f}', 'mean_declared_mentions': '{:.2f}'}, na_rep='-'))\n"
            "if not representatives.empty:\n"
            "    representatives['snippet'] = representatives['local_windows'].map(trim)\n"
            "    for cluster_id in representatives['cluster'].drop_duplicates().head(8):\n"
            "        display(Markdown(f'### Cluster {cluster_id}: representatives'))\n"
            "        display(representatives[representatives['cluster'] == cluster_id][['rank', 'case_id', 'language', 'aspect', 'gold', 'expert', 'llm', 'action', 'error_type_resolved', 'primary_confidence', 'snippet']].head(5).style.format({'primary_confidence': '{:.3f}'}, na_rep='-'))\n"
        ),
        markdown_cell(
            "## LLM Cluster-Label Prompts\n\n"
            "The helper also exports prompts for a separate LLM labeling pass. The prompt asks the model to label clusters "
            "linguistically, cite supporting examples, and avoid sentiment reclassification. This is useful for drafting "
            "a qualitative table, but the labels should still be manually audited."
        ),
        code_cell(
            "if prompts.empty:\n"
            "    display(Markdown('No cluster-label prompts found. Generate them with:'))\n"
            "    print(command_block('prompts'))\n"
            "else:\n"
            "    display(Markdown(f'Loaded **{len(prompts)}** cluster-label prompts from `{PROMPTS_PATH.relative_to(ROOT)}`.'))\n"
            "    first = prompts.iloc[0]\n"
            "    display(Markdown(f'### Prompt for cluster {first[\"cluster\"]}'))\n"
            "    print(first['prompt'][:4000])\n"
        ),
        markdown_cell(
            "## Gemma Cluster Interpretations and BERTopic\n\n"
            "If `--step all` was run with the Gemma server available, the cluster interpretation labels are loaded below. "
            "BERTopic outputs are optional and act as a lexical/topic sanity check against the embedding clusters."
        ),
        code_cell(
            "if not cluster_labels.empty:\n"
            "    display(cluster_labels[['cluster', 'status', 'label_text']].head(30))\n"
            "else:\n"
            "    display(Markdown('No Gemma cluster labels found yet. Generate them with:'))\n"
            "    print(command_block('label'))\n"
            "if not topics.empty:\n"
            "    display(Markdown('### BERTopic topic info'))\n"
            "    display(topics.head(30))\n"
            "else:\n"
            "    display(Markdown('No BERTopic outputs found. Optional command:'))\n"
            "    print(command_block('topics'))\n"
        ),
        markdown_cell(
            "## BERTopic Diagnostic Views\n\n"
            "BERTopic gives more readable, lexical topics than the raw HDBSCAN cluster IDs. These plots use the BERTopic "
            "document-topic assignments to show which topics are dominated by residual errors, rescues, or LLM-induced "
            "errors, and how topics align with gold sentiment classes."
        ),
        code_cell(
            "def topic_enrichment(topic_docs, topics, min_count=20):\n"
            "    rows = []\n"
            "    for topic, group in topic_docs.groupby('topic'):\n"
            "        if len(group) < min_count:\n"
            "            continue\n"
            "        outcome = group['error_type_resolved'].value_counts(normalize=True)\n"
            "        gold = group['gold_label'].value_counts(normalize=True)\n"
            "        lang = group['language_display'].value_counts(normalize=True)\n"
            "        rows.append({\n"
            "            'topic': topic,\n"
            "            'n': len(group),\n"
            "            'rescued_rate': outcome.get('expert_wrong_llm_correct', 0.0),\n"
            "            'both_wrong_rate': outcome.get('both_wrong', 0.0),\n"
            "            'llm_wrong_rate': outcome.get('expert_correct_llm_wrong', 0.0),\n"
            "            'top_gold': gold.index[0],\n"
            "            'gold_purity': gold.iloc[0],\n"
            "            'top_language': lang.index[0],\n"
            "            'language_purity': lang.iloc[0],\n"
            "        })\n"
            "    out = pd.DataFrame(rows)\n"
            "    if not out.empty and not topics.empty:\n"
            "        out = out.merge(topics[['Topic', 'Name', 'Representation']], left_on='topic', right_on='Topic', how='left').drop(columns=['Topic'])\n"
            "    return out\n\n"
            "def plot_topic_rates(enriched, sort_col, title):\n"
            "    sub = enriched[enriched['topic'] != -1].sort_values(sort_col, ascending=False).head(15).copy()\n"
            "    if sub.empty:\n"
            "        return None\n"
            "    labels = sub['Name'].fillna(sub['topic'].astype(str)).astype(str)\n"
            "    fig, ax = plt.subplots(figsize=(11, 6))\n"
            "    y = np.arange(len(sub))\n"
            "    ax.barh(y, sub[sort_col], color='#1f77b4')\n"
            "    ax.set_yticks(y)\n"
            "    ax.set_yticklabels(labels)\n"
            "    ax.invert_yaxis()\n"
            "    ax.set_xlim(0, min(1.0, max(0.05, sub[sort_col].max() + 0.05)))\n"
            "    ax.set_xlabel(sort_col.replace('_', ' '))\n"
            "    ax.set_title(title)\n"
            "    ax.grid(axis='x', alpha=0.25)\n"
            "    fig.tight_layout()\n"
            "    return fig\n\n"
            "def plot_topic_heatmap(topic_docs, value_col, title, top_n=25):\n"
            "    counts = topic_docs['topic'].value_counts()\n"
            "    keep = [topic for topic in counts.index if topic != -1][:top_n]\n"
            "    sub = topic_docs[topic_docs['topic'].isin(keep)]\n"
            "    if sub.empty:\n"
            "        return None\n"
            "    tab = pd.crosstab(sub['topic'], sub[value_col], normalize='index')\n"
            "    fig, ax = plt.subplots(figsize=(10, max(5, 0.28 * len(tab))))\n"
            "    im = ax.imshow(tab.to_numpy(), aspect='auto', cmap='Blues', vmin=0, vmax=1)\n"
            "    ax.set_yticks(np.arange(len(tab.index)))\n"
            "    ax.set_yticklabels(tab.index)\n"
            "    ax.set_xticks(np.arange(len(tab.columns)))\n"
            "    ax.set_xticklabels(tab.columns, rotation=35, ha='right')\n"
            "    ax.set_title(title)\n"
            "    ax.set_xlabel(value_col)\n"
            "    ax.set_ylabel('BERTopic topic')\n"
            "    plt.colorbar(im, ax=ax, label='share within topic')\n"
            "    fig.tight_layout()\n"
            "    return fig\n\n"
            "if not topics.empty and not topic_docs.empty:\n"
            "    enriched = topic_enrichment(topic_docs, topics)\n"
            "    display(Markdown('### Topic enrichment table'))\n"
            "    display(enriched.sort_values('n', ascending=False).head(35).style.format({'rescued_rate': '{:.3f}', 'both_wrong_rate': '{:.3f}', 'llm_wrong_rate': '{:.3f}', 'gold_purity': '{:.3f}', 'language_purity': '{:.3f}'}, na_rep='-'))\n"
            "    for col, title in [('rescued_rate', 'BERTopic topics with highest LLM rescue rate'), ('both_wrong_rate', 'BERTopic topics with highest residual both-wrong rate'), ('llm_wrong_rate', 'BERTopic topics with highest LLM-induced error rate')]:\n"
            "        fig = plot_topic_rates(enriched, col, title)\n"
            "        if fig is not None:\n"
            "            plt.show()\n"
            "    for value_col, title in [('error_type_resolved', 'Topic x outcome mix'), ('gold_label', 'Topic x gold label mix')]:\n"
            "        fig = plot_topic_heatmap(topic_docs, value_col, title)\n"
            "        if fig is not None:\n"
            "            plt.show()\n"
        ),
        markdown_cell(
            "## Notes for the Paper\n\n"
            "- If rescued cases cluster around low confidence, contrast markers, long cue distance, or inflected/aspect sparse mentions, that supports selective LLM deferral as targeted adjudication rather than broad replacement.\n"
            "- If both-wrong cases are enriched for annotation ambiguity, missing target mentions, entity competition, or quoted/reported sentiment, that explains why ordinary multi-expert L2D cannot close the headroom.\n"
            "- If complete LLM failures appear in neutral-heavy or global-sentiment clusters, use them to explain why full deferral degrades: the LLM overgeneralizes article-level affect instead of target-specific sentiment.\n"
            "- Treat cluster labels as hypotheses. The quantitative enrichments and representative examples are the defensible evidence; LLM labels are a drafting aid."
        ),
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=OUT_NOTEBOOK,
        help="Notebook destination (default: release reviews directory).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    notebook = build_notebook()
    write_json(args.output, notebook)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
