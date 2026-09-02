#!/usr/bin/env python3
"""Build the multi-expert defer analysis notebook and supporting plots."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-absa")

try:
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics import accuracy_score, cohen_kappa_score, precision_recall_fscore_support
except ModuleNotFoundError:
    plt = None
    np = None
    accuracy_score = None
    cohen_kappa_score = None
    precision_recall_fscore_support = None


ROOT = Path(os.environ.get("ABSA_RELEASE_ROOT", Path(__file__).resolve().parents[2])).resolve()
OUT_NOTEBOOK = ROOT / "reviews" / "multi-expert-defer-analysis.ipynb"
OUT_DIR = ROOT / "reviews" / "multi-expert-defer-analysis_files"
LABELS = [-1, 0, 1]
METRICS = ["f1_macro", "qwk", "accuracy"]

POLICY_DISPLAY_NAMES = {
    "longformer_masked": "Longformer Masked",
    "slavic_specific_masked": {
        "slovenian": "SloBERTa Masked",
        "serbian": "BERTic Masked",
    },
    "han_xlmr_masked": "HAN-XLMR Masked",
    "mdeberta_masked": "mDeBERTa-v3 Masked",
    "majority_vote": "Majority Vote",
    "avg_probs": "Average Probabilities",
    "confidence_pick": "Confidence Pick",
    "lr_stacker": "Linear Regression Stacker",
    "rf_stacker": "Random Forest Stacker",
    "oracle": "Oracle",
    "random_oracle_matched_rate": "Random-Item Oracle",
}


def display_name(name: str, language: str | None = None) -> str:
    value = POLICY_DISPLAY_NAMES.get(name)
    if isinstance(value, dict):
        return value.get(language or "", name)
    if value:
        return value
    return name.replace("_", " ").title()


LANG_CONFIG = {
    "slovenian": {
        "display": "Slovenian",
        "experts": {
            "Longformer Masked": ROOT / "reviews" / "longformer" / "masked" / "slovenian",
            "SloBERTa Masked": ROOT / "reviews" / "slavic_specific" / "masked" / "slovenian",
            "HAN-XLMR Masked": ROOT
            / "results"
            / "global-context-modelling"
            / "simplified-dart-xlmr"
            / "slovenian",
        },
        "multi_dir": ROOT
        / "reviews"
        / "uncertainty"
        / "multi-expert-defer"
        / "masked"
        / "slovenian"
        / "longformer_masked__slavic_specific_masked__han_xlmr_masked",
        "uncertainty_experts": {
            "Longformer Masked": "longformer_masked",
            "SloBERTa Masked": "slavic_specific_masked",
            "HAN-XLMR Masked": "han_xlmr_masked",
        },
    },
    "serbian": {
        "display": "Serbo-Croatian",
        "experts": {
            "Longformer Masked": ROOT / "reviews" / "longformer" / "masked" / "serbian",
            "BERTic Masked": ROOT / "reviews" / "slavic_specific" / "masked" / "serbian",
            "mDeBERTa-v3 Masked": ROOT / "reviews" / "mdeberta" / "masked" / "serbian",
            "HAN-XLMR Masked": ROOT
            / "results"
            / "global-context-modelling"
            / "simplified-dart-xlmr"
            / "serbian",
        },
        "multi_dir": ROOT
        / "reviews"
        / "uncertainty"
        / "multi-expert-defer"
        / "masked"
        / "serbian"
        / "longformer_masked__slavic_specific_masked__mdeberta_masked",
        "uncertainty_experts": {
            "Longformer Masked": "longformer_masked",
            "BERTic Masked": "slavic_specific_masked",
            "mDeBERTa-v3 Masked": "mdeberta_masked",
            "HAN-XLMR Masked": "han_xlmr_masked",
        },
    },
}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def metric_summary(values: list[float]) -> dict[str, float]:
    if len(values) == 1:
        return {"mean": values[0], "sd": 0.0}
    return {"mean": float(statistics.mean(values)), "sd": float(statistics.stdev(values))}


def calc_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, average="macro", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "qwk": float(cohen_kappa_score(y_true, y_pred, labels=LABELS, weights="quadratic")),
    }


def mode_label(values: list[int]) -> int:
    counts = Counter(values)
    max_count = max(counts.values())
    modes = [label for label, count in counts.items() if count == max_count]
    if len(modes) == 1:
        return modes[0]
    return 0 if 0 in modes else sorted(modes)[0]


def load_checkpoint_metrics(model_dir: Path) -> dict[str, dict[str, float]]:
    summary_path = model_dir / "test_metrics_summary.json"
    if summary_path.exists():
        raw = load_json(summary_path)
        rows = [
            row
            for key, row in raw.items()
            if key != "average_performance" and isinstance(row, dict)
        ] if isinstance(raw, dict) else raw
        values = defaultdict(list)
        for row in rows:
            for metric in METRICS:
                if metric in row and row[metric] is not None:
                    values[metric].append(float(row[metric]))
        return {metric: metric_summary(values[metric]) for metric in METRICS if values[metric]}

    values = defaultdict(list)
    for path in sorted(model_dir.glob("test_predictions_*.json")):
        records = load_json(path)
        metrics = calc_metrics(
            [int(item["sentiment"]) for item in records],
            [int(item["prediction"]) for item in records],
        )
        for metric in METRICS:
            values[metric].append(metrics[metric])
    return {metric: metric_summary(values[metric]) for metric in METRICS if values[metric]}


def checkpoint_vote_metrics(model_dir: Path) -> dict[str, float]:
    by_uuid: dict[str, dict[str, Any]] = {}
    prediction_lists: dict[str, list[int]] = defaultdict(list)
    for path in sorted(model_dir.glob("test_predictions_*.json")):
        records = load_json(path)
        for item in records:
            uuid = str(item["uuid"])
            by_uuid[uuid] = item
            prediction_lists[uuid].append(int(item["prediction"]))
    uuids = sorted(set(by_uuid) & set(prediction_lists))
    y_true = [int(by_uuid[uuid]["sentiment"]) for uuid in uuids]
    y_pred = [mode_label(prediction_lists[uuid]) for uuid in uuids]
    return calc_metrics(y_true, y_pred)


def load_multi_metrics(multi_dir: Path) -> dict[str, Any]:
    return load_json(multi_dir / "metrics.json")


def format_pm(mean: float, sd: float) -> str:
    return f"{mean:.4f} ± {sd:.4f}"


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def policy_metric_map(metrics: dict[str, Any]) -> dict[str, dict[str, float]]:
    return {row["policy"]: row for row in metrics["policy_metrics"]}


def collect_confidence_pick_details(language: str, metrics: dict[str, Any]) -> dict[str, Any]:
    config = LANG_CONFIG[language]
    success_by_expert = metrics["success_by_expert"]
    test_records_by_expert = {}
    for display, expert_key in config["uncertainty_experts"].items():
        path = ROOT / "reviews" / "uncertainty" / expert_key / language / f"{language}_test_complete.json"
        if not path.exists() or expert_key not in success_by_expert:
            continue
        data = load_json(path)
        records = data["test"] if isinstance(data, dict) and "test" in data else data
        test_records_by_expert[display] = {str(item["uuid"]): item for item in records}

    if not test_records_by_expert:
        return {"rows": [], "num_items": 0, "confusion": {}}

    common = sorted(set.intersection(*(set(rows) for rows in test_records_by_expert.values())))
    pick_counts = Counter()
    correct_counts = Counter()
    confusion = Counter()
    for uuid in common:
        best_display = None
        best_conf = -math.inf
        best_pred = None
        gold = None
        for display, expert_key in [
            (name, config["uncertainty_experts"][name]) for name in test_records_by_expert
        ]:
            item = test_records_by_expert[display][uuid]
            success = success_by_expert[expert_key]
            uncertainty = item.get(success["uncertainty_key"], {}) or {}
            conf = float(uncertainty.get("confidence_score", 0.0) or 0.0)
            pred = int(item.get(success["json_key"]))
            if conf > best_conf:
                best_conf = conf
                best_display = display
                best_pred = pred
                gold = int(item["sentiment"])
        pick_counts[best_display] += 1
        if best_pred == gold:
            correct_counts[best_display] += 1
        confusion[(gold, best_pred)] += 1

    rows = []
    for display in test_records_by_expert:
        count = pick_counts[display]
        rows.append(
            {
                "expert": display,
                "picked": count,
                "pick_rate": count / len(common) if common else 0.0,
                "accuracy_when_picked": correct_counts[display] / count if count else 0.0,
            }
        )
    return {
        "rows": rows,
        "num_items": len(common),
        "confusion": {f"{gold}->{pred}": count for (gold, pred), count in confusion.items()},
    }


def plot_policy_bars(all_data: dict[str, Any]) -> dict[str, str]:
    paths = {}
    for language in LANG_CONFIG:
        data = all_data[language]
        policies = data["policy_rows"]
        labels = [row["System"] for row in policies]
        f1 = [float(row["Macro-F1"]) for row in policies]
        qwk = [float(row["QWK"]) for row in policies]
        x = np.arange(len(labels))
        width = 0.38
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.bar(x - width / 2, f1, width, label="Macro-F1")
        ax.bar(x + width / 2, qwk, width, label="QWK")
        ax.set_ylabel("Score")
        ax.set_title(f"{LANG_CONFIG[language]['display']}: single experts vs multi-expert policies")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_ylim(max(0.0, min(f1 + qwk) - 0.05), min(1.0, max(f1 + qwk) + 0.03))
        ax.legend()
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        path = OUT_DIR / f"{language}_policy_comparison.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths[language] = str(path.relative_to(OUT_NOTEBOOK.parent))
    return paths


def read_curve(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def plot_defer_curves(all_data: dict[str, Any]) -> dict[str, str]:
    paths = {}
    preferred = {
        "slovenian": [
            ("longformer_masked", "avg_probs"),
            ("longformer_masked", "oracle"),
            ("longformer_masked", "random_oracle_matched_rate"),
        ],
        "serbian": [
            ("mdeberta_masked", "confidence_pick"),
            ("mdeberta_masked", "oracle"),
            ("mdeberta_masked", "random_oracle_matched_rate"),
        ],
    }
    for language in LANG_CONFIG:
        data = all_data[language]
        rows = read_curve(Path(data["multi_dir"]) / "defer_curves.csv")
        fig, ax = plt.subplots(figsize=(9, 5))
        for base, target in preferred[language]:
            selected = [
                row
                for row in rows
                if row["base_policy"] == base and row["target_policy"] == target
            ]
            selected = sorted(selected, key=lambda row: float(row["requested_defer_rate"]))
            if not selected:
                continue
            label = f"{display_name(base, language)} -> {display_name(target, language)}"
            ax.plot(
                [float(row["defer_rate"]) for row in selected],
                [float(row["f1_macro"]) for row in selected],
                marker="o",
                label=label,
            )
        ax.set_xlabel("Defer rate")
        ax.set_ylabel("Macro-F1")
        ax.set_title(f"{LANG_CONFIG[language]['display']}: selective deferral headroom")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        path = OUT_DIR / f"{language}_defer_curve.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths[language] = str(path.relative_to(OUT_NOTEBOOK.parent))
    return paths


def build_analysis() -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    analysis = {}
    for language, config in LANG_CONFIG.items():
        raw_rows = []
        for display, model_dir in config["experts"].items():
            checkpoint_summary = load_checkpoint_metrics(model_dir)
            vote = checkpoint_vote_metrics(model_dir)
            raw_rows.append(
                {
                    "System": display,
                    "Checkpoint Macro-F1": format_pm(
                        checkpoint_summary["f1_macro"]["mean"],
                        checkpoint_summary["f1_macro"]["sd"],
                    ),
                    "Checkpoint QWK": format_pm(
                        checkpoint_summary["qwk"]["mean"],
                        checkpoint_summary["qwk"]["sd"],
                    ),
                    "Checkpoint Accuracy": format_pm(
                        checkpoint_summary["accuracy"]["mean"],
                        checkpoint_summary["accuracy"]["sd"],
                    ),
                    "3-checkpoint vote Macro-F1": f"{vote['f1_macro']:.4f}",
                    "3-checkpoint vote QWK": f"{vote['qwk']:.4f}",
                    "3-checkpoint vote Accuracy": f"{vote['accuracy']:.4f}",
                }
            )

        multi = load_multi_metrics(config["multi_dir"])
        policy_map = policy_metric_map(multi)
        policy_order = [
            *multi["experts"],
            "majority_vote",
            "avg_probs",
            "confidence_pick",
            "lr_stacker",
            "rf_stacker",
        ]
        policy_rows = []
        for policy in policy_order:
            if policy not in policy_map:
                continue
            row = policy_map[policy]
            policy_rows.append(
                {
                    "System": display_name(policy, language),
                    "Macro-F1": f"{row['f1_macro']:.4f}",
                    "QWK": f"{row['qwk']:.4f}",
                    "Accuracy": f"{row['accuracy']:.4f}",
                    "Mean confidence": f"{row.get('mean_confidence', 0.0):.4f}",
                }
            )
        confidence_pick = collect_confidence_pick_details(language, multi)
        analysis[language] = {
            "display": config["display"],
            "raw_rows": raw_rows,
            "multi": multi,
            "multi_dir": str(config["multi_dir"]),
            "policy_rows": policy_rows,
            "confidence_pick": confidence_pick,
        }
    analysis["policy_plots"] = plot_policy_bars(analysis)
    analysis["curve_plots"] = plot_defer_curves(analysis)
    write_json(OUT_DIR / "analysis_data.json", analysis)
    return analysis


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


def build_notebook(analysis: dict[str, Any]) -> dict[str, Any]:
    cells = [
        markdown_cell(
            "# Multi-Expert Defer Analysis\n\n"
            "This notebook compares the masked multi-expert uncertainty/defer experiments against the ordinary "
            "three-checkpoint model outputs. Unlike a static report, the tables and plots below load and process the "
            "JSON/CSV files directly in Python cells.\n\n"
            "**Short answer:** `reviews/uncertainty/multi-expert-defer` currently contains the masked experiments. "
            "There are no unmasked multi-expert-defer outputs in that directory yet."
        ),
        code_cell(
            "from __future__ import annotations\n\n"
            "import csv\n"
            "import json\n"
            "import math\n"
            "import os\n"
            "import statistics\n"
            "from collections import Counter, defaultdict\n"
            "from pathlib import Path\n\n"
            "os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib-absa')\n\n"
            "import matplotlib.pyplot as plt\n"
            "import numpy as np\n"
            "import pandas as pd\n"
            "from IPython.display import Markdown, display\n"
            "from sklearn.metrics import accuracy_score, cohen_kappa_score, precision_recall_fscore_support\n\n"
            "ROOT = Path.cwd().resolve()\n"
            "if ROOT.name == 'reviews':\n"
            "    ROOT = ROOT.parent\n"
            "elif not (ROOT / 'reviews').exists():\n"
            "    raise RuntimeError(f'Run this notebook from the project root or reviews/: {ROOT}')\n\n"
            "LABELS = [-1, 0, 1]\n"
            "METRICS = ['f1_macro', 'qwk', 'accuracy']\n"
            "SPLITS = ['test', 'train_val_0', 'train_val_1', 'train_val_2']\n"
            "POLICY_DISPLAY_NAMES = {\n"
            "    'longformer_masked': 'Longformer Masked',\n"
            "    'slavic_specific_masked': {'slovenian': 'SloBERTa Masked', 'serbian': 'BERTic Masked'},\n"
            "    'han_xlmr_masked': 'HAN-XLMR Masked',\n"
            "    'mdeberta_masked': 'mDeBERTa-v3 Masked',\n"
            "    'majority_vote': 'Majority Vote',\n"
            "    'avg_probs': 'Average Probabilities',\n"
            "    'confidence_pick': 'Confidence Pick',\n"
            "    'lr_stacker': 'Linear Regression Stacker',\n"
            "    'rf_stacker': 'Random Forest Stacker',\n"
            "    'oracle': 'Oracle',\n"
            "    'random_oracle_matched_rate': 'Random-Item Oracle',\n"
            "}\n\n"
            "def display_name(name, language=None):\n"
            "    value = POLICY_DISPLAY_NAMES.get(name)\n"
            "    if isinstance(value, dict):\n"
            "        return value.get(language, name)\n"
            "    if value:\n"
            "        return value\n"
            "    return str(name).replace('_', ' ').title()\n\n"
            "EXPERT_COLORS = {\n"
            "    'Longformer Masked': '#1f77b4',\n"
            "    'SloBERTa Masked': '#2ca02c',\n"
            "    'BERTic Masked': '#2ca02c',\n"
            "    'HAN-XLMR Masked': '#9467bd',\n"
            "    'mDeBERTa-v3 Masked': '#d62728',\n"
            "}\n\n"
            "LANG_CONFIG = {\n"
            "    'slovenian': {\n"
            "        'display': 'Slovenian',\n"
            "        'experts': {\n"
            "            'Longformer Masked': ROOT / 'reviews/longformer/masked/slovenian',\n"
            "            'SloBERTa Masked': ROOT / 'reviews/slavic_specific/masked/slovenian',\n"
            "            'HAN-XLMR Masked': ROOT / 'results/global-context-modelling/simplified-dart-xlmr/slovenian',\n"
            "        },\n"
            "        'uncertainty_experts': {\n"
            "            'Longformer Masked': 'longformer_masked',\n"
            "            'SloBERTa Masked': 'slavic_specific_masked',\n"
            "            'HAN-XLMR Masked': 'han_xlmr_masked',\n"
            "        },\n"
            "        'multi_dir': ROOT / 'reviews/uncertainty/multi-expert-defer/masked/slovenian/longformer_masked__slavic_specific_masked__han_xlmr_masked',\n"
            "    },\n"
            "    'serbian': {\n"
            "        'display': 'Serbo-Croatian',\n"
            "        'experts': {\n"
            "            'Longformer Masked': ROOT / 'reviews/longformer/masked/serbian',\n"
            "            'BERTic Masked': ROOT / 'reviews/slavic_specific/masked/serbian',\n"
            "            'mDeBERTa-v3 Masked': ROOT / 'reviews/mdeberta/masked/serbian',\n"
            "            'HAN-XLMR Masked': ROOT / 'results/global-context-modelling/simplified-dart-xlmr/serbian',\n"
            "        },\n"
            "        'uncertainty_experts': {\n"
            "            'Longformer Masked': 'longformer_masked',\n"
            "            'BERTic Masked': 'slavic_specific_masked',\n"
            "            'mDeBERTa-v3 Masked': 'mdeberta_masked',\n"
            "            'HAN-XLMR Masked': 'han_xlmr_masked',\n"
            "        },\n"
            "        'multi_dir': ROOT / 'reviews/uncertainty/multi-expert-defer/masked/serbian/longformer_masked__slavic_specific_masked__mdeberta_masked',\n"
            "    },\n"
            "}\n\n"
            "LLM_HAN_XLMR_PATHS = {\n"
            "    'slovenian': ROOT / 'results/large-language-models/slovenian/dspy-plm-augmented-cot-with-uncertainty-masked/dspy-plm-augmented-cot-teacher-qwen-1024-heavy-uncertainty-slovenian-masked_predictions.json',\n"
            "    'serbian': ROOT / 'results/large-language-models/serbian/dspy-plm-augmented-cot-with-uncertainty/dspy-plm-augmented-cot-teacher-qwen-1024-medium-uncertainty-serbian-unmasked_predictions.json',\n"
            "}\n\n"
            "HAN_XLMR_JSON_KEY = 'global-context-modelling/simplified-dart-xlmr'\n"
            "HAN_XLMR_UNCERTAINTY_KEY = 'global-context-modelling/simplified-dart-xlmr/uncertainty'\n"
            "EXPERT_LLM_CONFIG = {\n"
            "    'longformer_masked': {\n"
            "        'display': 'Longformer Masked',\n"
            "        'csv_expert': 'Longformer masked',\n"
            "        'json_key': 'longformer/masked',\n"
            "        'uncertainty_key': 'longformer/masked/uncertainty',\n"
            "    },\n"
            "    'mdeberta_masked': {\n"
            "        'display': 'mDeBERTa-v3 Masked',\n"
            "        'csv_expert': 'mDeBERTa-v3 masked',\n"
            "        'json_key': 'mdeberta/masked',\n"
            "        'uncertainty_key': 'mdeberta/masked/uncertainty',\n"
            "    },\n"
            "}\n"
            "UNSEEN_ASPECTS = {\n"
            "    'slovenian': {\n"
            "        'A1 Slovenija', 'Prva osebna zavarovalnica', 'Mlinotest', 'Audi', 'Renault',\n"
            "        'Energetika Ljubljana', 'Cupra', 'Nissan', 'Addiko banka', 'Grawe',\n"
            "        'Delavska hranilnica',\n"
            "    },\n"
            "    'serbian': {\n"
            "        'mts', 'Knez Petrol', 'Generali', 'Mobi Banka', 'Philip Moris', 'JTI',\n"
            "        'Uniqa', 'Delta', 'API bank', 'AXA',\n"
            "    },\n"
            "}\n"
        ),
        code_cell(
            "def load_json(path: Path):\n"
            "    with path.open('r', encoding='utf-8') as f:\n"
            "        return json.load(f)\n\n"
            "def calc_metrics(y_true, y_pred):\n"
            "    precision, recall, f1, _ = precision_recall_fscore_support(\n"
            "        y_true, y_pred, labels=LABELS, average='macro', zero_division=0\n"
            "    )\n"
            "    return {\n"
            "        'accuracy': float(accuracy_score(y_true, y_pred)),\n"
            "        'precision_macro': float(precision),\n"
            "        'recall_macro': float(recall),\n"
            "        'f1_macro': float(f1),\n"
            "        'qwk': float(cohen_kappa_score(y_true, y_pred, labels=LABELS, weights='quadratic')),\n"
            "    }\n\n"
            "def metric_summary(values):\n"
            "    return {'mean': float(statistics.mean(values)), 'sd': float(statistics.stdev(values)) if len(values) > 1 else 0.0}\n\n"
            "def mode_label(values):\n"
            "    counts = Counter(values)\n"
            "    max_count = max(counts.values())\n"
            "    modes = [label for label, count in counts.items() if count == max_count]\n"
            "    if len(modes) == 1:\n"
            "        return modes[0]\n"
            "    return 0 if 0 in modes else sorted(modes)[0]\n\n"
            "def load_checkpoint_metrics(model_dir: Path):\n"
            "    summary_path = model_dir / 'test_metrics_summary.json'\n"
            "    if summary_path.exists():\n"
            "        raw = load_json(summary_path)\n"
            "        rows = [row for key, row in raw.items() if key != 'average_performance' and isinstance(row, dict)] if isinstance(raw, dict) else raw\n"
            "        values = defaultdict(list)\n"
            "        for row in rows:\n"
            "            for metric in METRICS:\n"
            "                if metric in row and row[metric] is not None:\n"
            "                    values[metric].append(float(row[metric]))\n"
            "        return {metric: metric_summary(values[metric]) for metric in METRICS if values[metric]}\n"
            "    values = defaultdict(list)\n"
            "    for path in sorted(model_dir.glob('test_predictions_*.json')):\n"
            "        records = load_json(path)\n"
            "        metrics = calc_metrics([int(x['sentiment']) for x in records], [int(x['prediction']) for x in records])\n"
            "        for metric in METRICS:\n"
            "            values[metric].append(metrics[metric])\n"
            "    return {metric: metric_summary(values[metric]) for metric in METRICS if values[metric]}\n\n"
            "def checkpoint_vote_metrics(model_dir: Path):\n"
            "    by_uuid = {}\n"
            "    prediction_lists = defaultdict(list)\n"
            "    for path in sorted(model_dir.glob('test_predictions_*.json')):\n"
            "        for item in load_json(path):\n"
            "            uuid = str(item['uuid'])\n"
            "            by_uuid[uuid] = item\n"
            "            prediction_lists[uuid].append(int(item['prediction']))\n"
            "    uuids = sorted(set(by_uuid) & set(prediction_lists))\n"
            "    return calc_metrics(\n"
            "        [int(by_uuid[uuid]['sentiment']) for uuid in uuids],\n"
            "        [mode_label(prediction_lists[uuid]) for uuid in uuids],\n"
            "    )\n\n"
            "def raw_checkpoint_table(language):\n"
            "    rows = []\n"
            "    for display_name, model_dir in LANG_CONFIG[language]['experts'].items():\n"
            "        summary = load_checkpoint_metrics(model_dir)\n"
            "        vote = checkpoint_vote_metrics(model_dir)\n"
            "        rows.append({\n"
            "            'System': display_name,\n"
            "            'Checkpoint Macro-F1': f\"{summary['f1_macro']['mean']:.4f} ± {summary['f1_macro']['sd']:.4f}\",\n"
            "            'Checkpoint QWK': f\"{summary['qwk']['mean']:.4f} ± {summary['qwk']['sd']:.4f}\",\n"
            "            'Checkpoint Accuracy': f\"{summary['accuracy']['mean']:.4f} ± {summary['accuracy']['sd']:.4f}\",\n"
            "            '3-checkpoint vote Macro-F1': f\"{vote['f1_macro']:.4f}\",\n"
            "            '3-checkpoint vote QWK': f\"{vote['qwk']:.4f}\",\n"
            "            '3-checkpoint vote Accuracy': f\"{vote['accuracy']:.4f}\",\n"
            "        })\n"
            "    return pd.DataFrame(rows)\n\n"
            "def load_multi_metrics(language):\n"
            "    return load_json(LANG_CONFIG[language]['multi_dir'] / 'metrics.json')\n\n"
            "def policy_table(language):\n"
            "    metrics = load_multi_metrics(language)\n"
            "    by_policy = {row['policy']: row for row in metrics['policy_metrics']}\n"
            "    order = [*metrics['experts'], 'majority_vote', 'avg_probs', 'confidence_pick', 'lr_stacker', 'rf_stacker']\n"
            "    rows = []\n"
            "    for policy in order:\n"
            "        if policy not in by_policy:\n"
            "            continue\n"
            "        row = by_policy[policy]\n"
            "        rows.append({\n"
            "            'System': display_name(policy, language),\n"
            "            'Policy key': policy,\n"
            "            'Family': 'Expert' if policy in metrics['experts'] else 'Multi-expert policy',\n"
            "            'Macro-F1': row['f1_macro'],\n"
            "            'QWK': row['qwk'],\n"
            "            'Accuracy': row['accuracy'],\n"
            "            'Mean confidence': row.get('mean_confidence', np.nan),\n"
            "        })\n"
            "    return pd.DataFrame(rows)\n"
            "\n"
            "def llm_learning_to_defer_table(language):\n"
            "    path = ROOT / 'reviews/scratchpad/best_llm_learning_to_defer_by_expert.csv'\n"
            "    if not path.exists():\n"
            "        return pd.DataFrame(columns=['System', 'Family', 'Macro-F1', 'QWK', 'Accuracy', 'Prompt Variant', 'Autorun', 'Source'])\n"
            "    df = pd.read_csv(path)\n"
            "    df = df[df['Language'].str.lower() == language].copy()\n"
            "    if df.empty:\n"
            "        return pd.DataFrame(columns=['System', 'Family', 'Macro-F1', 'QWK', 'Accuracy', 'Prompt Variant', 'Autorun', 'Source'])\n"
            "    expert_lookup = {\n"
            "        ('slovenian', 'Longformer masked'): 'Longformer Masked',\n"
            "        ('slovenian', 'SloBERTa/BERTic masked'): 'SloBERTa Masked',\n"
            "        ('slovenian', 'HAN + XLMR masked'): 'HAN-XLMR Masked',\n"
            "        ('serbian', 'Longformer masked'): 'Longformer Masked',\n"
            "        ('serbian', 'SloBERTa/BERTic masked'): 'BERTic Masked',\n"
            "        ('serbian', 'mDeBERTa-v3 masked'): 'mDeBERTa-v3 Masked',\n"
            "        ('serbian', 'HAN + XLMR masked'): 'HAN-XLMR Masked',\n"
            "    }\n"
            "    rows = []\n"
            "    for _, row in df.iterrows():\n"
            "        expert = expert_lookup.get((language, row['Expert']), row['Expert'])\n"
            "        rows.append({\n"
            "            'System': f'{expert} + Gemma-3-27B',\n"
            "            'Family': 'Expert+LLM defer',\n"
            "            'Macro-F1': float(row['F1']) / 100.0,\n"
            "            'QWK': float(row['QWK']),\n"
            "            'Accuracy': float(row['Accuracy']) / 100.0,\n"
            "            'Mean confidence': np.nan,\n"
            "            'Prompt Variant': row['Prompt Variant'],\n"
            "            'Autorun': row['Autorun'],\n"
            "            'Source': row['Source'],\n"
            "        })\n"
            "    return pd.DataFrame(rows)\n\n"
            "def combined_policy_llm_table(language):\n"
            "    policies = policy_table(language).copy()\n"
            "    llm = llm_learning_to_defer_table(language)\n"
            "    cols = ['System', 'Family', 'Macro-F1', 'QWK', 'Accuracy', 'Mean confidence']\n"
            "    if llm.empty:\n"
            "        return policies[cols]\n"
            "    return pd.concat([policies[cols], llm[cols]], ignore_index=True)\n"
            "\n"
            "def llm_han_xlmr_records(language):\n"
            "    path = LLM_HAN_XLMR_PATHS[language]\n"
            "    if not path.exists():\n"
            "        return []\n"
            "    rows = []\n"
            "    for item in load_json(path):\n"
            "        if item.get('sentiment') not in LABELS or item.get(HAN_XLMR_JSON_KEY) not in LABELS or item.get('prediction') not in LABELS:\n"
            "            continue\n"
            "        uncertainty = item.get(HAN_XLMR_UNCERTAINTY_KEY, {}) or {}\n"
            "        rows.append({\n"
            "            'uuid': str(item.get('uuid')),\n"
            "            'gold': int(item['sentiment']),\n"
            "            'base_pred': int(item[HAN_XLMR_JSON_KEY]),\n"
            "            'llm_pred': int(item['prediction']),\n"
            "            'confidence': float(uncertainty.get('confidence_score', 0.0) or 0.0),\n"
            "            'entropy': float(uncertainty.get('predictive_entropy', np.nan)),\n"
            "        })\n"
            "    return rows\n\n"
            "def defer_outcome_counts(y_true, base_pred, target_pred, defer_mask):\n"
            "    y_true = np.asarray(y_true)\n"
            "    base_pred = np.asarray(base_pred)\n"
            "    target_pred = np.asarray(target_pred)\n"
            "    defer_mask = np.asarray(defer_mask, dtype=bool)\n"
            "    base_correct = base_pred == y_true\n"
            "    target_correct = target_pred == y_true\n"
            "    return {\n"
            "        'corrections': int(np.sum(defer_mask & ~base_correct & target_correct)),\n"
            "        'degradations': int(np.sum(defer_mask & base_correct & ~target_correct)),\n"
            "        'both_correct_when_deferred': int(np.sum(defer_mask & base_correct & target_correct)),\n"
            "        'both_wrong_when_deferred': int(np.sum(defer_mask & ~base_correct & ~target_correct)),\n"
            "    }\n\n"
            "def llm_han_xlmr_stats(language):\n"
            "    rows = llm_han_xlmr_records(language)\n"
            "    if not rows:\n"
            "        return None\n"
            "    y_true = np.asarray([row['gold'] for row in rows], dtype=int)\n"
            "    base_pred = np.asarray([row['base_pred'] for row in rows], dtype=int)\n"
            "    llm_pred = np.asarray([row['llm_pred'] for row in rows], dtype=int)\n"
            "    override_mask = llm_pred != base_pred\n"
            "    base_metrics = calc_metrics(y_true, base_pred)\n"
            "    llm_metrics = calc_metrics(y_true, llm_pred)\n"
            "    outcome = defer_outcome_counts(y_true, base_pred, llm_pred, override_mask)\n"
            "    outcome.update({\n"
            "        'language': language,\n"
            "        'system': 'HAN-XLMR Masked + Gemma-3-27B',\n"
            "        'defer_rate': float(np.mean(override_mask)),\n"
            "        'num_deferred': int(np.sum(override_mask)),\n"
            "        'num_items': int(len(rows)),\n"
            "        'base_f1_macro': base_metrics['f1_macro'],\n"
            "        'base_qwk': base_metrics['qwk'],\n"
            "        'llm_f1_macro': llm_metrics['f1_macro'],\n"
            "        'llm_qwk': llm_metrics['qwk'],\n"
            "        'llm_accuracy': llm_metrics['accuracy'],\n"
            "    })\n"
            "    return outcome\n\n"
            "def llm_threshold_curve(language, rates=None, seed=42, random_repeats=20):\n"
            "    rows = llm_han_xlmr_records(language)\n"
            "    if not rows:\n"
            "        return pd.DataFrame()\n"
            "    if rates is None:\n"
            "        rates = [0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 1.0]\n"
            "    y_true = np.asarray([row['gold'] for row in rows], dtype=int)\n"
            "    base_pred = np.asarray([row['base_pred'] for row in rows], dtype=int)\n"
            "    llm_pred = np.asarray([row['llm_pred'] for row in rows], dtype=int)\n"
            "    scores = np.asarray([row['confidence'] for row in rows], dtype=float)\n"
            "    rng = np.random.default_rng(seed)\n"
            "    out = []\n"
            "    n = len(rows)\n"
            "    for rate in rates:\n"
            "        k = int(round(rate * n))\n"
            "        order = np.argsort(scores)\n"
            "        mask = np.zeros(n, dtype=bool)\n"
            "        if k > 0:\n"
            "            mask[order[:k]] = True\n"
            "        threshold = float(scores[order[k - 1]]) if k > 0 else float('-inf')\n"
            "        for target_name, target_pred in [('Gemma-3-27B', llm_pred), ('Oracle', y_true)]:\n"
            "            final_pred = np.where(mask, target_pred, base_pred)\n"
            "            metrics = calc_metrics(y_true, final_pred)\n"
            "            out.append({\n"
            "                'target_policy': target_name,\n"
            "                'requested_defer_rate': float(rate),\n"
            "                'defer_rate': float(np.mean(mask)),\n"
            "                'threshold': threshold,\n"
            "                'f1_macro': metrics['f1_macro'],\n"
            "                'qwk': metrics['qwk'],\n"
            "                'accuracy': metrics['accuracy'],\n"
            "                **defer_outcome_counts(y_true, base_pred, target_pred, mask),\n"
            "            })\n"
            "        random_metrics = []\n"
            "        for _ in range(random_repeats):\n"
            "            random_mask = np.zeros(n, dtype=bool)\n"
            "            if k > 0:\n"
            "                random_mask[rng.choice(n, size=k, replace=False)] = True\n"
            "            random_metrics.append(calc_metrics(y_true, np.where(random_mask, y_true, base_pred)))\n"
            "        out.append({\n"
            "            'target_policy': 'Random-Item Oracle',\n"
            "            'requested_defer_rate': float(rate),\n"
            "            'defer_rate': float(k / n),\n"
            "            'threshold': threshold,\n"
            "            'f1_macro': float(np.mean([m['f1_macro'] for m in random_metrics])),\n"
            "            'qwk': float(np.mean([m['qwk'] for m in random_metrics])),\n"
            "            'accuracy': float(np.mean([m['accuracy'] for m in random_metrics])),\n"
            "        })\n"
            "    return pd.DataFrame(out)\n"
            "\n"
            "def best_complete_llm_source(language, expert_label='Longformer masked'):\n"
            "    path = ROOT / 'reviews/scratchpad/best_llm_learning_to_defer_by_expert.csv'\n"
            "    if not path.exists():\n"
            "        return None\n"
            "    df = pd.read_csv(path)\n"
            "    rows = df[(df['Language'].str.lower() == language) & (df['Expert'] == expert_label)]\n"
            "    if rows.empty:\n"
            "        return None\n"
            "    row = rows.sort_values('F1', ascending=False).iloc[0]\n"
            "    metrics_path = ROOT / str(row['Source'])\n"
            "    predictions_path = Path(str(metrics_path).replace('_test_metrics.json', '_test_predictions.json').replace('_metrics.json', '_predictions.json'))\n"
            "    if not predictions_path.exists():\n"
            "        return None\n"
            "    return {'metrics_path': metrics_path, 'predictions_path': predictions_path, 'prompt_variant': row.get('Prompt Variant'), 'autorun': row.get('Autorun')}\n"
            "\n"
            "def complete_llm_records(language, expert_key='longformer_masked'):\n"
            "    cfg = EXPERT_LLM_CONFIG.get(expert_key)\n"
            "    if cfg is None:\n"
            "        raise KeyError(f'No complete-LLM extraction config for {expert_key}')\n"
            "    source = best_complete_llm_source(language, cfg['csv_expert'])\n"
            "    if not source:\n"
            "        return []\n"
            "    rows = []\n"
            "    for item in load_json(source['predictions_path']):\n"
            "        if item.get('sentiment') not in LABELS or item.get(cfg['json_key']) not in LABELS or item.get('prediction') not in LABELS:\n"
            "            continue\n"
            "        uncertainty = item.get(cfg['uncertainty_key'], {}) or {}\n"
            "        rows.append({\n"
            "            'uuid': str(item.get('uuid')),\n"
            "            'gold': int(item['sentiment']),\n"
            "            'base_pred': int(item[cfg['json_key']]),\n"
            "            'llm_pred': int(item['prediction']),\n"
            "            'confidence': float(uncertainty.get('vote_confidence', uncertainty.get('confidence_score', 0.0)) or 0.0),\n"
            "            'prob_confidence': float(uncertainty.get('confidence_score', 0.0) or 0.0),\n"
            "            'source': str(source['predictions_path']),\n"
            "            'prompt_variant': source['prompt_variant'],\n"
            "            'autorun': source['autorun'],\n"
            "        })\n"
            "    return rows\n"
            "\n"
            "def target_threshold_curve(records, target_column, target_policy, rates=None):\n"
            "    if not records:\n"
            "        return pd.DataFrame()\n"
            "    if rates is None:\n"
            "        rates = [0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 1.0]\n"
            "    y_true = np.asarray([row['gold'] for row in records], dtype=int)\n"
            "    base_pred = np.asarray([row['base_pred'] for row in records], dtype=int)\n"
            "    target_pred = np.asarray([row[target_column] for row in records], dtype=int)\n"
            "    confidence = np.asarray([row['confidence'] for row in records], dtype=float)\n"
            "    order = np.argsort(confidence)\n"
            "    out = []\n"
            "    n = len(records)\n"
            "    for rate in rates:\n"
            "        k = int(round(rate * n))\n"
            "        mask = np.zeros(n, dtype=bool)\n"
            "        if k > 0:\n"
            "            mask[order[:k]] = True\n"
            "        final_pred = np.where(mask, target_pred, base_pred)\n"
            "        metrics = calc_metrics(y_true, final_pred)\n"
            "        out.append({\n"
            "            'target_policy': target_policy,\n"
            "            'requested_defer_rate': float(rate),\n"
            "            'defer_rate': float(np.mean(mask)),\n"
            "            'threshold': float(confidence[order[k - 1]]) if k > 0 else np.nan,\n"
            "            'f1_macro': metrics['f1_macro'],\n"
            "            'qwk': metrics['qwk'],\n"
            "            'accuracy': metrics['accuracy'],\n"
            "            **defer_outcome_counts(y_true, base_pred, target_pred, mask),\n"
            "        })\n"
            "    return pd.DataFrame(out)\n"
            "\n"
            "def random_item_oracle_curve_from_records(records, rates=None, seed=42, random_repeats=20):\n"
            "    if not records:\n"
            "        return pd.DataFrame()\n"
            "    if rates is None:\n"
            "        rates = [0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 1.0]\n"
            "    y_true = np.asarray([row['gold'] for row in records], dtype=int)\n"
            "    base_pred = np.asarray([row['base_pred'] for row in records], dtype=int)\n"
            "    rng = np.random.default_rng(seed)\n"
            "    out = []\n"
            "    n = len(records)\n"
            "    for rate in rates:\n"
            "        k = int(round(rate * n))\n"
            "        metric_rows = []\n"
            "        for _ in range(random_repeats):\n"
            "            mask = np.zeros(n, dtype=bool)\n"
            "            if k > 0:\n"
            "                mask[rng.choice(n, size=k, replace=False)] = True\n"
            "            metric_rows.append(calc_metrics(y_true, np.where(mask, y_true, base_pred)))\n"
            "        out.append({\n"
            "            'target_policy': 'Random-Item Oracle',\n"
            "            'requested_defer_rate': float(rate),\n"
            "            'defer_rate': float(k / n),\n"
            "            'threshold': np.nan,\n"
            "            'f1_macro': float(np.mean([m['f1_macro'] for m in metric_rows])),\n"
            "            'qwk': float(np.mean([m['qwk'] for m in metric_rows])),\n"
            "            'accuracy': float(np.mean([m['accuracy'] for m in metric_rows])),\n"
            "        })\n"
            "    return pd.DataFrame(out)\n"
            "\n"
            "def complete_llm_override_stats(language, expert_key='longformer_masked'):\n"
            "    records = complete_llm_records(language, expert_key)\n"
            "    if not records:\n"
            "        return None\n"
            "    y_true = np.asarray([row['gold'] for row in records], dtype=int)\n"
            "    base_pred = np.asarray([row['base_pred'] for row in records], dtype=int)\n"
            "    llm_pred = np.asarray([row['llm_pred'] for row in records], dtype=int)\n"
            "    changed = llm_pred != base_pred\n"
            "    llm_metrics = calc_metrics(y_true, llm_pred)\n"
            "    outcome = defer_outcome_counts(y_true, base_pred, llm_pred, changed)\n"
            "    outcome.update({'override_rate': float(np.mean(changed)), 'num_overrides': int(np.sum(changed)), 'num_items': int(len(records)), 'llm_f1_macro': llm_metrics['f1_macro'], 'llm_qwk': llm_metrics['qwk']})\n"
            "    return outcome\n"
            "\n"
            "def parse_gate_rate_from_dir(name):\n"
            "    if '_gate_rate_' not in name:\n"
            "        return None\n"
            "    raw = name.rsplit('_gate_rate_', 1)[-1].replace('p', '.')\n"
            "    try:\n"
            "        return float(raw) / 100.0\n"
            "    except ValueError:\n"
            "        return None\n"
            "\n"
            "def selective_prediction_records(predictions_path, reward_abstain=True):\n"
            "    if not predictions_path.exists():\n"
            "        return []\n"
            "    rows = []\n"
            "    for item in load_json(predictions_path):\n"
            "        result = item.get('selective_deferral') or {}\n"
            "        if result.get('status') != 'success':\n"
            "            continue\n"
            "        gold = int(result.get('ground_truth_int', item.get('sentiment')))\n"
            "        primary = int(result.get('primary_prediction_int', item.get('prediction')))\n"
            "        raw_pred = int(result.get('prediction_int', item.get('prediction', primary)))\n"
            "        action = result.get('action') or item.get('selective_deferral_action') or 'unknown'\n"
            "        final_pred = gold if reward_abstain and action == 'abstain_uncertain' else raw_pred\n"
            "        rows.append({\n"
            "            'uuid': str(result.get('uuid', item.get('uuid'))),\n"
            "            'aspect': item.get('aspect'),\n"
            "            'gold': gold,\n"
            "            'primary_pred': primary,\n"
            "            'raw_pred': raw_pred,\n"
            "            'final_pred': final_pred,\n"
            "            'action': action,\n"
            "            'llm_called': bool(result.get('llm_called', True)),\n"
            "            'primary_confidence': float(result.get('primary_confidence', np.nan)),\n"
            "            'num_aux_disagree': int(result.get('num_aux_disagree', 0) or 0),\n"
            "            'num_aux': int(result.get('num_aux', 0) or 0),\n"
            "        })\n"
            "    return rows\n"
            "\n"
            "def selective_deferral_all_runs(language, expert_key='longformer_masked', autorun='medium'):\n"
            "    base = ROOT / 'reviews/uncertainty/llm-selective-deferral' / language\n"
            "    rows = []\n"
            "    for prompt_variant in ['masked', 'unmasked']:\n"
            "        pattern = base / prompt_variant / expert_key\n"
            "        for metrics_path in sorted(pattern.glob(f'{autorun}_gate_rate_*/*_test_metrics.json')):\n"
            "            if '_shard-' in metrics_path.name:\n"
            "                continue\n"
            "            gate_rate = parse_gate_rate_from_dir(metrics_path.parent.name)\n"
            "            if gate_rate is None:\n"
            "                continue\n"
            "            metrics = load_json(metrics_path)\n"
            "            predictions_path = metrics_path.with_name(metrics_path.name.replace('_test_metrics.json', '_test_predictions.json'))\n"
            "            raw_metrics = selective_metrics_from_predictions(predictions_path, reward_abstain=False)\n"
            "            adjusted = selective_metrics_from_predictions(predictions_path, reward_abstain=True)\n"
            "            if adjusted is not None:\n"
            "                metrics = {**metrics, **adjusted}\n"
            "            rows.append({\n"
            "                'language': language,\n"
            "                'expert_key': expert_key,\n"
            "                'prompt_variant': prompt_variant,\n"
            "                'gate_rate': gate_rate,\n"
            "                'llm_call_rate': float(metrics.get('llm_call_rate', gate_rate)),\n"
            "                'override_rate': float(metrics.get('override_rate', np.nan)),\n"
            "                'f1_macro': float(metrics['f1_macro']),\n"
            "                'qwk': float(metrics.get('qwk', np.nan)),\n"
            "                'accuracy': float(metrics.get('accuracy', np.nan)),\n"
            "                'raw_f1_macro': float(raw_metrics['f1_macro']) if raw_metrics else np.nan,\n"
            "                'raw_qwk': float(raw_metrics['qwk']) if raw_metrics else np.nan,\n"
            "                'raw_accuracy': float(raw_metrics['accuracy']) if raw_metrics else np.nan,\n"
            "                'corrections': int(metrics.get('corrections', 0)),\n"
            "                'degradations': int(metrics.get('degradations', 0)),\n"
            "                'abstain_rate': float(metrics.get('abstain_rate', np.nan)),\n"
            "                'reward_abstain_as_correct': bool(metrics.get('reward_abstain_as_correct', False)),\n"
            "                'predictions_path': str(predictions_path),\n"
            "                'path': str(metrics_path),\n"
            "            })\n"
            "    return pd.DataFrame(rows).sort_values(['prompt_variant', 'gate_rate']).reset_index(drop=True) if rows else pd.DataFrame()\n"
            "\n"
            "def selective_metrics_from_predictions(predictions_path, reward_abstain=True):\n"
            "    rows = selective_prediction_records(predictions_path, reward_abstain=reward_abstain)\n"
            "    if not rows:\n"
            "        return None\n"
            "    y_true = [row['gold'] for row in rows]\n"
            "    y_pred = [row['final_pred'] for row in rows]\n"
            "    y_primary = [row['primary_pred'] for row in rows]\n"
            "    actions = Counter(row['action'] for row in rows)\n"
            "    llm_called = sum(int(row['llm_called']) for row in rows)\n"
            "    metrics = calc_metrics(y_true, y_pred)\n"
            "    primary_metrics = calc_metrics(y_true, y_primary)\n"
            "    y_true_arr = np.asarray(y_true)\n"
            "    y_pred_arr = np.asarray(y_pred)\n"
            "    y_primary_arr = np.asarray(y_primary)\n"
            "    metrics.update({\n"
            "        'primary_f1_macro': primary_metrics['f1_macro'],\n"
            "        'primary_qwk': primary_metrics['qwk'],\n"
            "        'llm_call_rate': llm_called / len(y_true),\n"
            "        'override_rate': actions.get('override', 0) / len(y_true),\n"
            "        'abstain_rate': actions.get('abstain_uncertain', 0) / len(y_true),\n"
            "        'corrections': int(np.sum((y_primary_arr != y_true_arr) & (y_pred_arr == y_true_arr))),\n"
            "        'degradations': int(np.sum((y_primary_arr == y_true_arr) & (y_pred_arr != y_true_arr))),\n"
            "        'action_counts': dict(actions),\n"
            "        'reward_abstain_as_correct': bool(reward_abstain),\n"
            "    })\n"
            "    return metrics\n"
            "\n"
            "def selective_action_class_counts(predictions_path, reward_abstain=True):\n"
            "    label_names = {-1: 'negative', 0: 'neutral', 1: 'positive'}\n"
            "    if not predictions_path.exists():\n"
            "        return pd.DataFrame()\n"
            "    rows = []\n"
            "    for item in load_json(predictions_path):\n"
            "        result = item.get('selective_deferral') or {}\n"
            "        if result.get('status') != 'success':\n"
            "            continue\n"
            "        gold = int(result.get('ground_truth_int', item.get('sentiment')))\n"
            "        primary = int(result.get('primary_prediction_int'))\n"
            "        raw_pred = int(result.get('prediction_int', item.get('prediction')))\n"
            "        action = result.get('action') or item.get('selective_deferral_action') or 'unknown'\n"
            "        final_pred = gold if reward_abstain and action == 'abstain_uncertain' else raw_pred\n"
            "        rows.append({\n"
            "            'action': action,\n"
            "            'gold_class': label_names.get(gold, str(gold)),\n"
            "            'n': 1,\n"
            "            'corrections': int(primary != gold and final_pred == gold),\n"
            "            'degradations': int(primary == gold and final_pred != gold),\n"
            "            'both_correct': int(primary == gold and final_pred == gold),\n"
            "            'both_wrong': int(primary != gold and final_pred != gold),\n"
            "        })\n"
            "    if not rows:\n"
            "        return pd.DataFrame()\n"
            "    df = pd.DataFrame(rows)\n"
            "    return df.groupby(['action', 'gold_class'], as_index=False)[['n', 'corrections', 'degradations', 'both_correct', 'both_wrong']].sum()\n"
            "\n"
            "def selective_deferral_points(language, expert_key='longformer_masked', autorun='medium'):\n"
            "    df = selective_deferral_all_runs(language, expert_key, autorun=autorun)\n"
            "    if df.empty:\n"
            "        return pd.DataFrame()\n"
            "    idx = df.sort_values(['gate_rate', 'f1_macro'], ascending=[True, False]).groupby('gate_rate').head(1).index\n"
            "    return df.loc[idx].sort_values('gate_rate').reset_index(drop=True)\n"
            "\n"
            "def selective_metric_comparison_table(language, expert_key):\n"
            "    selective = selective_deferral_points(language, expert_key)\n"
            "    if selective.empty:\n"
            "        return pd.DataFrame()\n"
            "    return selective[[\n"
            "        'gate_rate', 'prompt_variant', 'llm_call_rate', 'override_rate', 'abstain_rate',\n"
            "        'raw_f1_macro', 'raw_qwk', 'f1_macro', 'qwk', 'corrections', 'degradations'\n"
            "    ]].rename(columns={\n"
            "        'raw_f1_macro': 'Macro-F1 raw',\n"
            "        'raw_qwk': 'QWK raw',\n"
            "        'f1_macro': 'Macro-F1 abstain-resolved',\n"
            "        'qwk': 'QWK abstain-resolved',\n"
            "    })\n"
            "\n"
            "def selected_action_class_counts_table(language, expert_key):\n"
            "    selective = selective_deferral_points(language, expert_key)\n"
            "    frames = []\n"
            "    for _, row in selective.iterrows():\n"
            "        counts = selective_action_class_counts(Path(row['predictions_path']), reward_abstain=True)\n"
            "        if counts.empty:\n"
            "            continue\n"
            "        counts.insert(0, 'gate_rate', row['gate_rate'])\n"
            "        counts.insert(1, 'prompt_variant', row['prompt_variant'])\n"
            "        frames.append(counts)\n"
            "    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()\n"
            "\n"
            "def expert_llm_deferral_summary(language, expert_key='longformer_masked'):\n"
            "    display = EXPERT_LLM_CONFIG[expert_key]['display']\n"
            "    complete = complete_llm_override_stats(language, expert_key)\n"
            "    selective = selective_deferral_points(language, expert_key)\n"
            "    rows = []\n"
            "    if complete:\n"
            "        rows.append({'System': f'{display} -> LLM (complete)', 'Rate type': 'observed label-change rate', 'Rate': complete['override_rate'], 'Macro-F1': complete['llm_f1_macro'], 'QWK': complete['llm_qwk'], 'Corrections': complete['corrections'], 'Degradations': complete['degradations']})\n"
            "    if not selective.empty:\n"
            "        for _, row in selective.iterrows():\n"
            "            rows.append({'System': f\"{display} -> LLM selective ({row['prompt_variant']}, gate {row['gate_rate']:.0%})\", 'Rate type': 'LLM call rate', 'Rate': row['llm_call_rate'], 'Macro-F1': row['f1_macro'], 'QWK': row['qwk'], 'Corrections': row['corrections'], 'Degradations': row['degradations']})\n"
            "    return pd.DataFrame(rows)\n"
            "\n"
            "def metric_or_nan(y_true, y_pred):\n"
            "    if len(y_true) == 0:\n"
            "        return {'f1_macro': np.nan, 'qwk': np.nan, 'accuracy': np.nan}\n"
            "    return calc_metrics(y_true, y_pred)\n"
            "\n"
            "def gated_replacement_baselines(language, expert_key, repeats=50, seed=42):\n"
            "    selective = selective_deferral_points(language, expert_key)\n"
            "    if selective.empty:\n"
            "        return pd.DataFrame()\n"
            "    rows = []\n"
            "    method_order = [\n"
            "        'Keep expert prediction',\n"
            "        'Majority class on gated items',\n"
            "        'Uniform random class on gated items',\n"
            "        'Class-prior random class on gated items',\n"
            "        'Selective LLM final prediction',\n"
            "    ]\n"
            "    for run_idx, run in selective.iterrows():\n"
            "        records = selective_prediction_records(Path(run['predictions_path']), reward_abstain=True)\n"
            "        if not records:\n"
            "            continue\n"
            "        y_true = np.asarray([row['gold'] for row in records], dtype=int)\n"
            "        primary = np.asarray([row['primary_pred'] for row in records], dtype=int)\n"
            "        selective_pred = np.asarray([row['final_pred'] for row in records], dtype=int)\n"
            "        gate = np.asarray([row['llm_called'] for row in records], dtype=bool)\n"
            "        if not gate.any():\n"
            "            continue\n"
            "        counts = Counter(y_true.tolist())\n"
            "        majority = mode_label(y_true.tolist())\n"
            "        prior = np.asarray([counts[label] / len(y_true) for label in LABELS], dtype=float)\n"
            "        rng = np.random.default_rng(seed + run_idx + int(round(run['gate_rate'] * 1000)))\n"
            "\n"
            "        def add_row(method, final_pred, repetitions=1):\n"
            "            full = metric_or_nan(y_true, final_pred)\n"
            "            gated = metric_or_nan(y_true[gate], final_pred[gate])\n"
            "            rows.append({\n"
            "                'language': LANG_CONFIG[language]['display'],\n"
            "                'expert': EXPERT_LLM_CONFIG[expert_key]['display'],\n"
            "                'gate_rate': run['gate_rate'],\n"
            "                'prompt_variant': run['prompt_variant'],\n"
            "                'llm_call_rate': run['llm_call_rate'],\n"
            "                'n_total': len(y_true),\n"
            "                'n_gated': int(gate.sum()),\n"
            "                'method': method,\n"
            "                'method_order': method_order.index(method),\n"
            "                'full_f1_macro': full['f1_macro'],\n"
            "                'full_qwk': full['qwk'],\n"
            "                'full_accuracy': full['accuracy'],\n"
            "                'gated_f1_macro': gated['f1_macro'],\n"
            "                'gated_qwk': gated['qwk'],\n"
            "                'gated_accuracy': gated['accuracy'],\n"
            "                'repetitions': repetitions,\n"
            "            })\n"
            "\n"
            "        add_row('Keep expert prediction', primary.copy())\n"
            "        majority_pred = primary.copy()\n"
            "        majority_pred[gate] = majority\n"
            "        add_row('Majority class on gated items', majority_pred)\n"
            "        add_row('Selective LLM final prediction', selective_pred.copy())\n"
            "\n"
            "        for method, probabilities in [\n"
            "            ('Uniform random class on gated items', np.repeat(1 / len(LABELS), len(LABELS))),\n"
            "            ('Class-prior random class on gated items', prior),\n"
            "        ]:\n"
            "            full_metric_rows = []\n"
            "            gated_metric_rows = []\n"
            "            for _ in range(repeats):\n"
            "                sampled = rng.choice(LABELS, size=int(gate.sum()), replace=True, p=probabilities)\n"
            "                random_pred = primary.copy()\n"
            "                random_pred[gate] = sampled\n"
            "                full_metric_rows.append(metric_or_nan(y_true, random_pred))\n"
            "                gated_metric_rows.append(metric_or_nan(y_true[gate], random_pred[gate]))\n"
            "            rows.append({\n"
            "                'language': LANG_CONFIG[language]['display'],\n"
            "                'expert': EXPERT_LLM_CONFIG[expert_key]['display'],\n"
            "                'gate_rate': run['gate_rate'],\n"
            "                'prompt_variant': run['prompt_variant'],\n"
            "                'llm_call_rate': run['llm_call_rate'],\n"
            "                'n_total': len(y_true),\n"
            "                'n_gated': int(gate.sum()),\n"
            "                'method': method,\n"
            "                'method_order': method_order.index(method),\n"
            "                'full_f1_macro': float(np.mean([m['f1_macro'] for m in full_metric_rows])),\n"
            "                'full_qwk': float(np.mean([m['qwk'] for m in full_metric_rows])),\n"
            "                'full_accuracy': float(np.mean([m['accuracy'] for m in full_metric_rows])),\n"
            "                'gated_f1_macro': float(np.mean([m['f1_macro'] for m in gated_metric_rows])),\n"
            "                'gated_qwk': float(np.mean([m['qwk'] for m in gated_metric_rows])),\n"
            "                'gated_accuracy': float(np.mean([m['accuracy'] for m in gated_metric_rows])),\n"
            "                'repetitions': repeats,\n"
            "            })\n"
            "    if not rows:\n"
            "        return pd.DataFrame()\n"
            "    return pd.DataFrame(rows).sort_values(['gate_rate', 'method_order']).reset_index(drop=True)\n"
            "\n"
            "def plot_gated_replacement_baselines(language, expert_key):\n"
            "    df = gated_replacement_baselines(language, expert_key)\n"
            "    if df.empty:\n"
            "        return None\n"
            "    colors = {\n"
            "        'Keep expert prediction': '#1f77b4',\n"
            "        'Majority class on gated items': '#8c564b',\n"
            "        'Uniform random class on gated items': '#7f7f7f',\n"
            "        'Class-prior random class on gated items': '#bcbd22',\n"
            "        'Selective LLM final prediction': '#d62728',\n"
            "    }\n"
            "    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2), sharex=True)\n"
            "    for method, sub in df.groupby('method', sort=False):\n"
            "        sub = sub.sort_values('gate_rate')\n"
            "        axes[0].plot(sub['llm_call_rate'], sub['full_f1_macro'], marker='o', color=colors.get(method), label=method)\n"
            "        axes[1].plot(sub['llm_call_rate'], sub['full_qwk'], marker='o', color=colors.get(method), label=method)\n"
            "    display_label = EXPERT_LLM_CONFIG[expert_key]['display'].replace(' Masked', '')\n"
            "    axes[0].set_title(f\"{LANG_CONFIG[language]['display']}: {display_label} gated replacements (Macro-F1)\")\n"
            "    axes[1].set_title(f\"{LANG_CONFIG[language]['display']}: {display_label} gated replacements (QWK)\")\n"
            "    for ax in axes:\n"
            "        ax.set_xlabel('LLM call rate')\n"
            "        ax.grid(alpha=0.25)\n"
            "        ax.legend(loc='best')\n"
            "    axes[0].set_ylabel('Full-test score after replacing only gated items')\n"
            "    axes[1].set_ylabel('Full-test score after replacing only gated items')\n"
            "    fig.tight_layout()\n"
            "    return fig\n"
            "\n"
            "def sentiment_rich_uuid_set(language, expert_key):\n"
            "    uuids = set()\n"
            "    for row in complete_llm_records(language, expert_key):\n"
            "        if row['gold'] != 0 or row['base_pred'] != 0 or row['llm_pred'] != 0:\n"
            "            uuids.add(row['uuid'])\n"
            "    selective = selective_deferral_points(language, expert_key)\n"
            "    for _, run in selective.iterrows():\n"
            "        for row in selective_prediction_records(Path(run['predictions_path']), reward_abstain=True):\n"
            "            if row['gold'] != 0 or row['primary_pred'] != 0 or row['raw_pred'] != 0 or row['final_pred'] != 0:\n"
            "                uuids.add(row['uuid'])\n"
            "    return uuids\n"
            "\n"
            "def selective_points_on_uuids(language, expert_key, uuids):\n"
            "    selective = selective_deferral_points(language, expert_key)\n"
            "    if selective.empty or not uuids:\n"
            "        return pd.DataFrame()\n"
            "    rows = []\n"
            "    for _, run in selective.iterrows():\n"
            "        records = [row for row in selective_prediction_records(Path(run['predictions_path']), reward_abstain=True) if row['uuid'] in uuids]\n"
            "        if not records:\n"
            "            continue\n"
            "        y_true = np.asarray([row['gold'] for row in records], dtype=int)\n"
            "        primary = np.asarray([row['primary_pred'] for row in records], dtype=int)\n"
            "        final_pred = np.asarray([row['final_pred'] for row in records], dtype=int)\n"
            "        llm_called = np.asarray([row['llm_called'] for row in records], dtype=bool)\n"
            "        metrics = calc_metrics(y_true, final_pred)\n"
            "        metrics.update(defer_outcome_counts(y_true, primary, final_pred, llm_called))\n"
            "        rows.append({\n"
            "            'language': language,\n"
            "            'expert_key': expert_key,\n"
            "            'prompt_variant': run['prompt_variant'],\n"
            "            'gate_rate': run['gate_rate'],\n"
            "            'llm_call_rate': float(np.mean(llm_called)),\n"
            "            'n_subset': len(records),\n"
            "            'f1_macro': metrics['f1_macro'],\n"
            "            'qwk': metrics['qwk'],\n"
            "            'accuracy': metrics['accuracy'],\n"
            "            'corrections': metrics['corrections'],\n"
            "            'degradations': metrics['degradations'],\n"
            "        })\n"
            "    return pd.DataFrame(rows).sort_values('gate_rate').reset_index(drop=True) if rows else pd.DataFrame()\n"
        ),
        markdown_cell(
            "## Calibration / Reliability Plots\n\n"
            "The plots below are computed directly from `reviews/uncertainty/<expert>/<language>/*.json`. "
            "For each bin, the x-axis is mean model confidence and the y-axis is empirical accuracy. "
            "A perfectly calibrated model lies on the diagonal. The train-val combined plots pool "
            "`train_val_0`, `train_val_1`, and `train_val_2` records before binning, so the curve is count-weighted "
            "rather than an unweighted average of three curves."
        ),
        code_cell(
            "def uncertainty_success(expert_key, language):\n"
            "    path = ROOT / 'reviews/uncertainty' / expert_key / language / '_SUCCESS.json'\n"
            "    return load_json(path) if path.exists() else None\n\n"
            "def has_uncertainty_records(expert_key, language):\n"
            "    return (ROOT / 'reviews/uncertainty' / expert_key / language / f'{language}_test_complete.json').exists()\n\n"
            "def uncertainty_records(expert_key, language, split):\n"
            "    base = ROOT / 'reviews/uncertainty' / expert_key / language\n"
            "    if split == 'test':\n"
            "        path = base / f'{language}_test_complete.json'\n"
            "        if not path.exists():\n"
            "            return []\n"
            "        data = load_json(path)\n"
            "        return data['test'] if isinstance(data, dict) and 'test' in data else data\n"
            "    index = split.rsplit('_', 1)[-1]\n"
            "    path = base / f'{language}_train_val_complete_{index}.json'\n"
            "    if not path.exists():\n"
            "        return []\n"
            "    data = load_json(path)\n"
            "    if isinstance(data, dict):\n"
            "        return list(data.get('train', [])) + list(data.get('val', []))\n"
            "    return data\n\n"
            "def confidence_correct_pairs(expert_key, language, split):\n"
            "    success = uncertainty_success(expert_key, language)\n"
            "    if success is None:\n"
            "        return []\n"
            "    json_key = success['json_key']\n"
            "    uncertainty_key = success['uncertainty_key']\n"
            "    rows = []\n"
            "    for item in uncertainty_records(expert_key, language, split):\n"
            "        if item.get('sentiment') not in LABELS or item.get(json_key) not in LABELS:\n"
            "            continue\n"
            "        uncertainty = item.get(uncertainty_key, {}) or {}\n"
            "        conf = float(uncertainty.get('confidence_score', 0.0) or 0.0)\n"
            "        rows.append((conf, int(item[json_key]) == int(item['sentiment'])))\n"
            "    return rows\n\n"
            "def calibration_bins(pairs, n_bins=10):\n"
            "    if not pairs:\n"
            "        return pd.DataFrame(columns=['bin', 'mean_confidence', 'accuracy', 'count'])\n"
            "    conf = np.array([x for x, _ in pairs], dtype=float)\n"
            "    correct = np.array([y for _, y in pairs], dtype=float)\n"
            "    edges = np.linspace(0.0, 1.0, n_bins + 1)\n"
            "    indices = np.clip(np.digitize(conf, edges, right=True) - 1, 0, n_bins - 1)\n"
            "    rows = []\n"
            "    for idx in range(n_bins):\n"
            "        mask = indices == idx\n"
            "        if not mask.any():\n"
            "            continue\n"
            "        rows.append({\n"
            "            'bin': idx,\n"
            "            'bin_left': edges[idx],\n"
            "            'bin_right': edges[idx + 1],\n"
            "            'mean_confidence': float(conf[mask].mean()),\n"
            "            'accuracy': float(correct[mask].mean()),\n"
            "            'count': int(mask.sum()),\n"
            "        })\n"
            "    return pd.DataFrame(rows)\n\n"
            "def calibration_stats(pairs):\n"
            "    df = calibration_bins(pairs)\n"
            "    if df.empty:\n"
            "        return {'ece': np.nan, 'mce': np.nan, 'brier_confidence': np.nan, 'n': 0}\n"
            "    n = df['count'].sum()\n"
            "    gaps = (df['accuracy'] - df['mean_confidence']).abs()\n"
            "    return {\n"
            "        'ece': float((gaps * df['count']).sum() / n),\n"
            "        'mce': float(gaps.max()),\n"
            "        'brier_confidence': float(np.mean([(conf - float(correct)) ** 2 for conf, correct in pairs])),\n"
            "        'n': int(n),\n"
            "    }\n\n"
            "def pooled_train_val_pairs(expert_key, language):\n"
            "    pairs = []\n"
            "    for split in ['train_val_0', 'train_val_1', 'train_val_2']:\n"
            "        pairs.extend(confidence_correct_pairs(expert_key, language, split))\n"
            "    return pairs\n\n"
            "def plot_calibration_line(ax, pairs, label, color):\n"
            "    df = calibration_bins(pairs)\n"
            "    if df.empty:\n"
            "        return\n"
            "    ax.plot(df['mean_confidence'], df['accuracy'], marker='o', linewidth=2, label=label, color=color)\n"
            "    for _, row in df.iterrows():\n"
            "        if row['count'] >= 1000:\n"
            "            ax.annotate(str(int(row['count'])), (row['mean_confidence'], row['accuracy']), fontsize=7, alpha=0.65)\n\n"
            "def finish_calibration_axis(ax):\n"
            "    ax.plot([0, 1], [0, 1], linestyle='--', color='black', alpha=0.45, label='perfect calibration')\n"
            "    ax.set_xlim(0, 1.02)\n"
            "    ax.set_ylim(0, 1.02)\n"
            "    ax.set_xlabel('Mean confidence')\n"
            "    ax.set_ylabel('Empirical accuracy')\n"
            "    ax.grid(alpha=0.25)\n\n"
            "def plot_expert_split_grid(display_name, expert_key, language):\n"
            "    fig, axes = plt.subplots(4, 1, figsize=(8, 15), sharex=True, sharey=True)\n"
            "    color = EXPERT_COLORS[display_name]\n"
            "    for ax, split in zip(axes, SPLITS):\n"
            "        pairs = confidence_correct_pairs(expert_key, language, split)\n"
            "        stats = calibration_stats(pairs)\n"
            "        plot_calibration_line(ax, pairs, display_name, color)\n"
            "        finish_calibration_axis(ax)\n"
            "        ax.set_title(f\"{LANG_CONFIG[language]['display']} / {display_name} / {split}: ECE={stats['ece']:.3f}, n={stats['n']}\")\n"
            "        ax.legend(loc='lower right')\n"
            "    fig.tight_layout()\n"
            "    return fig\n\n"
            "def plot_expert_test_trainval_grid(display_name, expert_key, language):\n"
            "    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True, sharey=True)\n"
            "    color = EXPERT_COLORS[display_name]\n"
            "    panels = [('test', confidence_correct_pairs(expert_key, language, 'test')), ('train_val pooled', pooled_train_val_pairs(expert_key, language))]\n"
            "    for ax, (panel_name, pairs) in zip(axes, panels):\n"
            "        stats = calibration_stats(pairs)\n"
            "        plot_calibration_line(ax, pairs, display_name, color)\n"
            "        finish_calibration_axis(ax)\n"
            "        ax.set_title(f\"{LANG_CONFIG[language]['display']} / {display_name} / {panel_name}: ECE={stats['ece']:.3f}, n={stats['n']}\")\n"
            "        ax.legend(loc='lower right')\n"
            "    fig.tight_layout()\n"
            "    return fig\n\n"
            "def plot_language_combined_calibration(language):\n"
            "    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True, sharey=True)\n"
            "    panels = [('Train-val pooled', lambda expert_key: pooled_train_val_pairs(expert_key, language)), ('Test', lambda expert_key: confidence_correct_pairs(expert_key, language, 'test'))]\n"
            "    missing = []\n"
            "    for ax, (panel_name, pair_fn) in zip(axes, panels):\n"
            "        for display_name, expert_key in LANG_CONFIG[language]['uncertainty_experts'].items():\n"
            "            pairs = pair_fn(expert_key)\n"
            "            if not pairs:\n"
            "                if display_name not in missing:\n"
            "                    missing.append(display_name)\n"
            "                continue\n"
            "            stats = calibration_stats(pairs)\n"
            "            plot_calibration_line(ax, pairs, f\"{display_name} (ECE={stats['ece']:.3f})\", EXPERT_COLORS[display_name])\n"
            "        finish_calibration_axis(ax)\n"
            "        ax.set_title(f\"{LANG_CONFIG[language]['display']}: {panel_name}\")\n"
            "        ax.legend(loc='lower right')\n"
            "    if missing:\n"
            "        fig.text(0.01, 0.01, 'No uncertainty JSON available for: ' + ', '.join(missing), fontsize=9, color='dimgray')\n"
            "    fig.tight_layout()\n"
            "    return fig\n\n"
            "def confidence_variability_points(expert_key, language, split_group):\n"
            "    success = uncertainty_success(expert_key, language)\n"
            "    if success is None:\n"
            "        return pd.DataFrame(columns=['confidence', 'variability', 'correct'])\n"
            "    splits = ['train_val_0', 'train_val_1', 'train_val_2'] if split_group == 'train_val pooled' else ['test']\n"
            "    rows = []\n"
            "    for split in splits:\n"
            "        for item in uncertainty_records(expert_key, language, split):\n"
            "            if item.get('sentiment') not in LABELS or item.get(success['json_key']) not in LABELS:\n"
            "                continue\n"
            "            unc = item.get(success['uncertainty_key'], {}) or {}\n"
            "            confidence = float(unc.get('confidence_score', 0.0) or 0.0)\n"
            "            variability = unc.get('variation_ratio')\n"
            "            if variability is None:\n"
            "                variability = unc.get('vote_variability')\n"
            "            if variability is None:\n"
            "                variability = 1.0 - confidence\n"
            "            rows.append({\n"
            "                'confidence': confidence,\n"
            "                'variability': float(variability),\n"
            "                'correct': bool(int(item[success['json_key']]) == int(item['sentiment'])),\n"
            "            })\n"
            "    return pd.DataFrame(rows)\n\n"
            "def plot_language_confidence_variability_grid(language, max_points_per_panel=5000, seed=42):\n"
            "    experts = list(LANG_CONFIG[language]['uncertainty_experts'].items())\n"
            "    fig, axes = plt.subplots(2, len(experts), figsize=(4.1 * len(experts), 7.2), sharex=True, sharey=True)\n"
            "    if len(experts) == 1:\n"
            "        axes = np.asarray(axes).reshape(2, 1)\n"
            "    rng = np.random.default_rng(seed)\n"
            "    for col, (display_name, expert_key) in enumerate(experts):\n"
            "        for row_idx, split_group in enumerate(['train_val pooled', 'test']):\n"
            "            ax = axes[row_idx, col]\n"
            "            df = confidence_variability_points(expert_key, language, split_group)\n"
            "            if df.empty:\n"
            "                ax.text(0.5, 0.5, 'missing uncertainty JSON', ha='center', va='center', transform=ax.transAxes)\n"
            "                ax.set_axis_off()\n"
            "                continue\n"
            "            if len(df) > max_points_per_panel:\n"
            "                idx = rng.choice(len(df), size=max_points_per_panel, replace=False)\n"
            "                df = df.iloc[idx]\n"
            "            wrong = df[~df['correct']]\n"
            "            correct = df[df['correct']]\n"
            "            ax.scatter(correct['variability'], correct['confidence'], s=7, alpha=0.45, color='green', label='correct')\n"
            "            ax.scatter(wrong['variability'], wrong['confidence'], s=9, alpha=0.6, color='red', label='incorrect')\n"
            "            ax.set_title(f\"{display_name}: {split_group}\")\n"
            "            ax.set_xlim(-0.02, 1.02)\n"
            "            ax.set_ylim(-0.02, 1.02)\n"
            "            ax.grid(alpha=0.18)\n"
            "            if row_idx == 1:\n"
            "                ax.set_xlabel('Variability')\n"
            "            if col == 0:\n"
            "                ax.set_ylabel('Confidence')\n"
            "    handles, labels = axes[0, 0].get_legend_handles_labels()\n"
            "    if handles:\n"
            "        fig.legend(handles, labels, loc='upper center', ncol=2, frameon=False)\n"
            "    fig.suptitle(f\"{LANG_CONFIG[language]['display']}: confidence vs variability point cloud\", y=1.02)\n"
            "    fig.tight_layout()\n"
            "    return fig\n\n"
            "def calibration_summary_table():\n"
            "    rows = []\n"
            "    for language, cfg in LANG_CONFIG.items():\n"
            "        for display_name, expert_key in cfg['uncertainty_experts'].items():\n"
            "            for split_label, pairs in [('test', confidence_correct_pairs(expert_key, language, 'test')), ('train_val pooled', pooled_train_val_pairs(expert_key, language))]:\n"
            "                if not pairs:\n"
            "                    continue\n"
            "                stats = calibration_stats(pairs)\n"
            "                rows.append({'language': cfg['display'], 'expert': display_name, 'split': split_label, **stats})\n"
            "    return pd.DataFrame(rows)\n"
        ),
        code_cell(
            "# Calibration summary: lower ECE means confidence better matches empirical accuracy.\n"
            "calibration_summary = calibration_summary_table()\n"
            "display(calibration_summary.sort_values(['language', 'split', 'ece']))"
        ),
        code_cell(
            "# Combined language views: test and pooled train-val, with one line per expert.\n"
            "for language in ['slovenian', 'serbian']:\n"
            "    display(Markdown(f\"### {LANG_CONFIG[language]['display']}: combined expert calibration\"))\n"
            "    fig = plot_language_combined_calibration(language)\n"
            "    plt.show()\n"
        ),
        markdown_cell(
            "### Confidence and Variability Clouds\n\n"
            "These point clouds complement the binned reliability curves. Each point is one item; green points are "
            "correct expert predictions and red points are errors. The x-axis is the model's variability signal "
            "(`variation_ratio` when available, otherwise `1 - confidence_score`) and the y-axis is confidence. "
            "A useful uncertainty signal should concentrate red points toward lower confidence and higher variability."
        ),
        code_cell(
            "for language in ['slovenian', 'serbian']:\n"
            "    display(Markdown(f\"### {LANG_CONFIG[language]['display']}: confidence vs variability\"))\n"
            "    fig = plot_language_confidence_variability_grid(language)\n"
            "    plt.show()\n"
        ),
        markdown_cell(
            "## Model and Defer Comparisons\n\n"
            "The ordinary model rows use the familiar mean ± SD convention because each model has three separate "
            "checkpoint/run predictions. The uncertainty rows are not three separate test runs: each item receives "
            "one aggregated distribution produced from all selected checkpoints and MC-dropout samples. Therefore, "
            "the fairest compact comparison is raw checkpoint mean ± SD versus the single uncertainty-aggregate "
            "point estimate. The deterministic 3-checkpoint vote is included as a bridge between the two reporting styles."
        ),
        code_cell(
            "raw_tables = {language: raw_checkpoint_table(language) for language in LANG_CONFIG}\n"
            "policy_tables = {language: policy_table(language) for language in LANG_CONFIG}\n"
            "combined_tables = {language: combined_policy_llm_table(language) for language in LANG_CONFIG}\n\n"
            "for language, cfg in LANG_CONFIG.items():\n"
            "    display(Markdown(f\"## {cfg['display']}\"))\n"
            "    display(Markdown('### Raw three-checkpoint models'))\n"
            "    display(raw_tables[language])\n"
            "    display(Markdown('### Single experts, multi-expert policies, and expert+LLM defer'))\n"
            "    display(combined_tables[language].sort_values('Macro-F1', ascending=False).style.format({'Macro-F1': '{:.4f}', 'QWK': '{:.4f}', 'Accuracy': '{:.4f}', 'Mean confidence': '{:.4f}'}, na_rep='-'))\n"
            "    best = policy_tables[language].sort_values('Macro-F1', ascending=False).iloc[0]\n"
            "    display(Markdown(f\"Best masked multi-expert policy by macro-F1: **{best['System']}** ({best['Macro-F1']:.4f}; QWK={best['QWK']:.4f}).\"))\n"
        ),
        code_cell(
            "def plot_policy_comparison(language):\n"
            "    df = combined_policy_llm_table(language).sort_values('Macro-F1', ascending=False).copy()\n"
            "    fig, ax = plt.subplots(figsize=(13, 5.5))\n"
            "    x = np.arange(len(df))\n"
            "    width = 0.38\n"
            "    ax.bar(x - width / 2, df['Macro-F1'], width, label='Macro-F1')\n"
            "    ax.bar(x + width / 2, df['QWK'], width, label='QWK')\n"
            "    ax.set_title(f\"{LANG_CONFIG[language]['display']}: single experts, expert+LLM, and multi-expert policies\")\n"
            "    ax.set_ylabel('Score')\n"
            "    ax.set_xticks(x)\n"
            "    ax.set_xticklabels(df['System'], rotation=35, ha='right')\n"
            "    ax.set_ylim(max(0.0, min(df['Macro-F1'].min(), df['QWK'].min()) - 0.05), min(1.0, max(df['Macro-F1'].max(), df['QWK'].max()) + 0.03))\n"
            "    ax.grid(axis='y', alpha=0.25)\n"
            "    ax.legend()\n"
            "    fig.tight_layout()\n"
            "    return fig\n\n"
            "for language in ['slovenian', 'serbian']:\n"
            "    plot_policy_comparison(language)\n"
            "    plt.show()\n"
        ),
        markdown_cell(
            "## Confidence-Pick: What Is Picked?\n\n"
            "`confidence_pick` is an item-level routing policy. For each test item, it looks at each expert's "
            "`confidence_score` in the MC-dropout uncertainty block, selects the expert with the largest confidence, "
            "and returns that expert's sentiment prediction. It does not learn a threshold and it does not average probabilities."
        ),
        code_cell(
            "def confidence_pick_breakdown(language):\n"
            "    metrics = load_multi_metrics(language)\n"
            "    success_by_expert = metrics['success_by_expert']\n"
            "    cfg = LANG_CONFIG[language]\n"
            "    records_by_expert = {}\n"
            "    for display_name, expert_key in cfg['uncertainty_experts'].items():\n"
            "        path = ROOT / 'reviews/uncertainty' / expert_key / language / f'{language}_test_complete.json'\n"
            "        if not path.exists() or expert_key not in success_by_expert:\n"
            "            continue\n"
            "        data = load_json(path)\n"
            "        records = data['test'] if isinstance(data, dict) and 'test' in data else data\n"
            "        records_by_expert[display_name] = {str(item['uuid']): item for item in records}\n"
            "    if not records_by_expert:\n"
            "        return pd.DataFrame(columns=['expert', 'picked', 'pick_rate', 'accuracy_when_picked'])\n"
            "    common = sorted(set.intersection(*(set(rows) for rows in records_by_expert.values())))\n"
            "    picked = Counter()\n"
            "    correct = Counter()\n"
            "    for uuid in common:\n"
            "        best_display, best_conf, best_pred, gold = None, -math.inf, None, None\n"
            "        for display_name, expert_key in [(name, cfg['uncertainty_experts'][name]) for name in records_by_expert]:\n"
            "            item = records_by_expert[display_name][uuid]\n"
            "            success = success_by_expert[expert_key]\n"
            "            conf = float((item.get(success['uncertainty_key'], {}) or {}).get('confidence_score', 0.0) or 0.0)\n"
            "            pred = int(item[success['json_key']])\n"
            "            if conf > best_conf:\n"
            "                best_display, best_conf, best_pred, gold = display_name, conf, pred, int(item['sentiment'])\n"
            "        picked[best_display] += 1\n"
            "        correct[best_display] += int(best_pred == gold)\n"
            "    rows = []\n"
            "    for display_name in records_by_expert:\n"
            "        n = picked[display_name]\n"
            "        rows.append({'expert': display_name, 'picked': n, 'pick_rate': n / len(common), 'accuracy_when_picked': correct[display_name] / n if n else np.nan})\n"
            "    return pd.DataFrame(rows)\n\n"
            "for language in ['slovenian', 'serbian']:\n"
            "    display(Markdown(f\"### {LANG_CONFIG[language]['display']} confidence-pick breakdown\"))\n"
            "    display(confidence_pick_breakdown(language).style.format({'pick_rate': '{:.3f}', 'accuracy_when_picked': '{:.4f}'}))\n"
        ),
        markdown_cell(
            "## Learning-to-Defer Curves and Headroom\n\n"
            "These curves are counterfactual routing diagnostics. For a fixed base expert, items are ordered by the base "
            "expert's uncertainty score: low confidence is deferred first. A deployable curve replaces those low-confidence "
            "items with another available policy's predictions. The oracle curve replaces them with the gold label, so it is "
            "an upper bound for this confidence ordering rather than a real system. The random-item-oracle curve also uses "
            "the gold label, but chooses the same number of items uniformly at random rather than by confidence; it checks "
            "whether uncertainty is actually finding harder cases. It is not a random-label or majority-class baseline. If the "
            "deployable curve is flat or below the base line, the target policy is not correcting more deferred errors than "
            "it introduces."
        ),
        code_cell(
            "def read_defer_curve(language):\n"
            "    path = LANG_CONFIG[language]['multi_dir'] / 'defer_curves.csv'\n"
            "    return pd.read_csv(path)\n\n"
            "def best_single_expert_key(language):\n"
            "    metrics = load_multi_metrics(language)\n"
            "    by_policy = {row['policy']: row for row in metrics['policy_metrics']}\n"
            "    return max(metrics['experts'], key=lambda key: by_policy[key]['f1_macro'])\n\n"
            "def best_deployable_curve_target(language, base):\n"
            "    df = read_defer_curve(language)\n"
            "    deployable = df[(df['base_policy'] == base) & ~df['target_policy'].isin(['oracle', 'random_oracle_matched_rate'])]\n"
            "    if deployable.empty:\n"
            "        return None\n"
            "    grouped = deployable.groupby('target_policy')['f1_macro'].max().sort_values(ascending=False)\n"
            "    return grouped.index[0]\n\n"
            "def headroom_table(language):\n"
            "    policies = policy_table(language)\n"
            "    combined = combined_policy_llm_table(language)\n"
            "    base_key = best_single_expert_key(language)\n"
            "    base_name = display_name(base_key, language)\n"
            "    base_f1 = float(policies.loc[policies['Policy key'] == base_key, 'Macro-F1'].iloc[0])\n"
            "    best_policy = combined[combined['Family'].isin(['Multi-expert policy'])].sort_values('Macro-F1', ascending=False).iloc[0]\n"
            "    llm_rows = combined[combined['Family'] == 'Expert+LLM defer']\n"
            "    best_llm = llm_rows.sort_values('Macro-F1', ascending=False).iloc[0] if not llm_rows.empty else None\n"
            "    llm_stats = llm_han_xlmr_stats(language)\n"
            "    curves = read_defer_curve(language)\n"
            "    deployable = curves[(curves['base_policy'] == base_key) & ~curves['target_policy'].isin(['oracle', 'random_oracle_matched_rate'])]\n"
            "    best_deploy = deployable.sort_values('f1_macro', ascending=False).iloc[0] if not deployable.empty else None\n"
            "    oracle = curves[(curves['base_policy'] == base_key) & (curves['target_policy'] == 'oracle')].sort_values('f1_macro', ascending=False).iloc[0]\n"
            "    rows = [\n"
            "        {'Comparison': 'Best single expert', 'System': base_name, 'Macro-F1': base_f1, 'Delta vs base': 0.0, 'Defer rate': 0.0, 'Corrections': np.nan, 'Degradations': np.nan, 'Net': np.nan},\n"
            "        {'Comparison': 'Best multi-expert policy', 'System': best_policy['System'], 'Macro-F1': best_policy['Macro-F1'], 'Delta vs base': best_policy['Macro-F1'] - base_f1, 'Defer rate': np.nan, 'Corrections': np.nan, 'Degradations': np.nan, 'Net': np.nan},\n"
            "    ]\n"
            "    if best_llm is not None:\n"
            "        rows.append({'Comparison': 'Best expert+LLM defer', 'System': best_llm['System'], 'Macro-F1': best_llm['Macro-F1'], 'Delta vs base': best_llm['Macro-F1'] - base_f1, 'Defer rate': llm_stats['defer_rate'] if llm_stats else np.nan, 'Corrections': llm_stats['corrections'] if llm_stats else np.nan, 'Degradations': llm_stats['degradations'] if llm_stats else np.nan, 'Net': (llm_stats['corrections'] - llm_stats['degradations']) if llm_stats else np.nan})\n"
            "    if best_deploy is not None:\n"
            "        rows.append({'Comparison': 'Best curve point with deployable target', 'System': f\"{base_name} -> {display_name(best_deploy['target_policy'], language)}\", 'Macro-F1': best_deploy['f1_macro'], 'Delta vs base': best_deploy['f1_macro'] - base_f1, 'Defer rate': best_deploy['defer_rate'], 'Corrections': best_deploy.get('corrections', np.nan), 'Degradations': best_deploy.get('degradations', np.nan), 'Net': best_deploy.get('corrections', np.nan) - best_deploy.get('degradations', np.nan)})\n"
            "    rows.append({'Comparison': 'Confidence-ranked oracle headroom', 'System': f\"{base_name} -> Oracle\", 'Macro-F1': oracle['f1_macro'], 'Delta vs base': oracle['f1_macro'] - base_f1, 'Defer rate': oracle['defer_rate'], 'Corrections': oracle.get('corrections', np.nan), 'Degradations': oracle.get('degradations', np.nan), 'Net': oracle.get('corrections', np.nan) - oracle.get('degradations', np.nan)})\n"
            "    return pd.DataFrame(rows)\n\n"
            "def plot_defer_curves(language):\n"
            "    base = best_single_expert_key(language)\n"
            "    best_target = best_deployable_curve_target(language, base)\n"
            "    preferred = [(base, 'oracle'), (base, 'random_oracle_matched_rate')]\n"
            "    if best_target:\n"
            "        preferred.insert(0, (base, best_target))\n"
            "    df = read_defer_curve(language)\n"
            "    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)\n"
            "    ax = axes[0]\n"
            "    base_f1 = float(policy_table(language).loc[policy_table(language)['Policy key'] == base, 'Macro-F1'].iloc[0])\n"
            "    ax.axhline(base_f1, linestyle='--', color='black', alpha=0.55, label=f'{display_name(base, language)} base')\n"
            "    for base, target in preferred:\n"
            "        sub = df[(df['base_policy'] == base) & (df['target_policy'] == target)].sort_values('requested_defer_rate')\n"
            "        if sub.empty:\n"
            "            continue\n"
            "        ax.plot(sub['defer_rate'], sub['f1_macro'], marker='o', label=f'{display_name(base, language)} -> {display_name(target, language)}')\n"
            "    ax.set_title(f\"{LANG_CONFIG[language]['display']}: selective deferral headroom (baseline)\")\n"
            "    ax.set_xlabel('Defer rate')\n"
            "    ax.set_ylabel('Macro-F1')\n"
            "    ax.grid(alpha=0.25)\n"
            "    ax.legend()\n"
            "\n"
            "    ax = axes[1]\n"
            "    llm_stats = llm_han_xlmr_stats(language)\n"
            "    llm_curve = llm_threshold_curve(language)\n"
            "    if llm_stats and not llm_curve.empty:\n"
            "        ax.axhline(llm_stats['base_f1_macro'], linestyle='--', color='black', alpha=0.55, label='HAN-XLMR base')\n"
            "        for target_name, sub in llm_curve.groupby('target_policy'):\n"
            "            sub = sub.sort_values('defer_rate')\n"
            "            ax.plot(sub['defer_rate'], sub['f1_macro'], marker='o', label=f'HAN-XLMR -> {target_name}')\n"
            "    ax.set_title(f\"{LANG_CONFIG[language]['display']}: selective deferral headroom (LLMs)\")\n"
            "    ax.set_xlabel('Defer rate')\n"
            "    ax.grid(alpha=0.25)\n"
            "    ax.legend()\n"
            "    fig.tight_layout()\n"
            "    return fig\n\n"
            "for language in ['slovenian', 'serbian']:\n"
            "    display(Markdown(f\"### {LANG_CONFIG[language]['display']} headroom summary\"))\n"
            "    display(headroom_table(language).style.format({'Macro-F1': '{:.4f}', 'Delta vs base': '{:+.4f}', 'Defer rate': '{:.3f}', 'Corrections': '{:.0f}', 'Degradations': '{:.0f}', 'Net': '{:+.0f}'}, na_rep='-'))\n"
            "    plot_defer_curves(language)\n"
            "    plt.show()\n"
            "    table = headroom_table(language)\n"
            "    best_multi = table[table['Comparison'] == 'Best multi-expert policy'].iloc[0]\n"
            "    best_llm = table[table['Comparison'] == 'Best expert+LLM defer'] if 'Best expert+LLM defer' in set(table['Comparison']) else pd.DataFrame()\n"
            "    if not best_llm.empty and best_llm.iloc[0]['Delta vs base'] > 0:\n"
            "        display(Markdown(f\"Interpretation note: the best expert+LLM system improves over the strongest single expert by {best_llm.iloc[0]['Delta vs base']:+.4f} Macro-F1, so report it as useful model-based adjudication even if simple multi-expert routing remains stronger.\"))\n"
            "    elif not best_llm.empty:\n"
            "        display(Markdown(f\"Interpretation note: expert+LLM defer does not beat the strongest single expert here ({best_llm.iloc[0]['Delta vs base']:+.4f} Macro-F1), so its defensible role is qualitative adjudication/error analysis rather than headline accuracy.\"))\n"
            "    if best_multi['Delta vs base'] > 0:\n"
            "        display(Markdown(f\"Interpretation note: the best deterministic multi-expert policy improves over the strongest single expert by {best_multi['Delta vs base']:+.4f} Macro-F1, indicating complementary expert errors.\"))\n"
            "    else:\n"
            "        display(Markdown(\"Interpretation note: multi-expert routing does not improve over the strongest single expert, suggesting the selected experts make highly correlated errors or the router is not separating reliable cases.\"))\n"
        ),
        markdown_cell(
            "## Complete vs Selective LLM Deferral\n\n"
            "These plots isolate expert-specific LLM deferral paths. The purple curve uses the older complete LLM "
            "predictions as a counterfactual target: at each x-value, the lowest-confidence expert items are replaced "
            "by the full LLM prediction. The red curve uses the new selective-deferral runs and plots the final system "
            "score at the actual LLM call rate. For each gate rate, masked and unmasked selective runs are both checked; "
            "the plotted point is the run with the better Macro-F1, and the QWK panel uses that same selected run. "
            "The green line is a random-item oracle: it randomly chooses which items receive gold oracle labels. "
            "The gated-replacement diagnostics below are the place where random class labels and majority-class baselines "
            "are evaluated directly."
        ),
        code_cell(
            "def plot_expert_llm_deferral(language, expert_key, sentiment_rich=False):\n"
            "    cfg = EXPERT_LLM_CONFIG[expert_key]\n"
            "    display_label = cfg['display']\n"
            "    policies = policy_table(language)\n"
            "    base_rows = policies[policies['Policy key'] == expert_key]\n"
            "    if base_rows.empty:\n"
            "        display(Markdown(f\"No {display_label} base row found for {LANG_CONFIG[language]['display']}.\"))\n"
            "        return None\n"
            "    complete_records = complete_llm_records(language, expert_key)\n"
            "    subset_label = ''\n"
            "    if sentiment_rich:\n"
            "        rich_uuids = sentiment_rich_uuid_set(language, expert_key)\n"
            "        complete_records = [row for row in complete_records if row['uuid'] in rich_uuids]\n"
            "        selective = selective_points_on_uuids(language, expert_key, rich_uuids)\n"
            "        subset_label = ' sentiment-rich'\n"
            "        if not complete_records and selective.empty:\n"
            "            display(Markdown(f\"No sentiment-rich records found for {display_label} / {LANG_CONFIG[language]['display']}.\"))\n"
            "            return None\n"
            "        if complete_records:\n"
            "            base_metrics = calc_metrics([row['gold'] for row in complete_records], [row['base_pred'] for row in complete_records])\n"
            "            base_values = {'f1_macro': base_metrics['f1_macro'], 'qwk': base_metrics['qwk']}\n"
            "        else:\n"
            "            base_values = {'f1_macro': np.nan, 'qwk': np.nan}\n"
            "        oracle_curves = pd.concat([\n"
            "            target_threshold_curve(complete_records, 'gold', 'Oracle') if complete_records else pd.DataFrame(),\n"
            "            random_item_oracle_curve_from_records(complete_records) if complete_records else pd.DataFrame(),\n"
            "        ], ignore_index=True)\n"
            "    else:\n"
            "        base_values = {'f1_macro': float(base_rows['Macro-F1'].iloc[0]), 'qwk': float(base_rows['QWK'].iloc[0])}\n"
            "        curves = read_defer_curve(language)\n"
            "        selective = selective_deferral_points(language, expert_key)\n"
            "        oracle_curves = curves[curves['base_policy'] == expert_key].copy()\n"
            "    complete_curve = target_threshold_curve(complete_records, 'llm_pred', 'LLM complete') if complete_records else pd.DataFrame()\n"
            "    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharex=True)\n"
            "    metric_specs = [('f1_macro', 'Macro-F1'), ('qwk', 'QWK')]\n"
            "    for ax, (metric, metric_label) in zip(axes, metric_specs):\n"
            "        ax.axhline(base_values[metric], color='#1f77b4', linestyle=':', linewidth=2.5, label=f'{display_label} base')\n"
            "        for target, color, label in [\n"
            "            ('random_oracle_matched_rate', '#2ca02c', f'{display_label} -> Random-Item Oracle'),\n"
            "            ('Random-Item Oracle', '#2ca02c', f'{display_label} -> Random-Item Oracle'),\n"
            "            ('oracle', '#ff7f0e', f'{display_label} -> Oracle'),\n"
            "            ('Oracle', '#ff7f0e', f'{display_label} -> Oracle'),\n"
            "        ]:\n"
            "            sub = oracle_curves[oracle_curves['target_policy'] == target].sort_values('defer_rate')\n"
            "            if not sub.empty:\n"
            "                ax.plot(sub['defer_rate'], sub[metric], color=color, marker='o', linewidth=2, label=label)\n"
            "        if not complete_curve.empty:\n"
            "            ax.plot(complete_curve['defer_rate'], complete_curve[metric], color='#9467bd', marker='o', linewidth=2, label=f'{display_label} -> LLM (complete deferral)')\n"
            "        if not selective.empty:\n"
            "            red_x = [0.0] + selective['llm_call_rate'].tolist()\n"
            "            red_y = [base_values[metric]] + selective[metric].tolist()\n"
            "            ax.plot(red_x, red_y, color='#d62728', marker='o', linewidth=2.4, label=f'{display_label} -> LLM (selective deferral)')\n"
            "            for _, row in selective.iterrows():\n"
            "                ax.annotate(f\"{row['gate_rate']:.0%}/{row['prompt_variant'][0]}\", (row['llm_call_rate'], row[metric]), textcoords='offset points', xytext=(4, 4), fontsize=8, color='#7f1d1d')\n"
            "        ax.set_title(f\"{LANG_CONFIG[language]['display']}: {display_label.replace(' Masked', '')}{subset_label} LLM deferral comparison ({metric_label})\")\n"
            "        ax.set_xlabel('Defer rate / LLM call rate')\n"
            "        ax.set_ylabel(metric_label)\n"
            "        ax.grid(alpha=0.25)\n"
            "        ax.legend(loc='best')\n"
            "    fig.tight_layout()\n"
            "    return fig\n\n"
            "DEFERRAL_PLOT_REQUESTS = [\n"
            "    ('slovenian', 'longformer_masked'),\n"
            "    ('serbian', 'longformer_masked'),\n"
            "    ('serbian', 'mdeberta_masked'),\n"
            "]\n"
            "for language, expert_key in DEFERRAL_PLOT_REQUESTS:\n"
            "    display_name_for_heading = EXPERT_LLM_CONFIG[expert_key]['display']\n"
            "    display(Markdown(f\"### {LANG_CONFIG[language]['display']}: {display_name_for_heading} complete vs selective LLM deferral\"))\n"
            "    fig = plot_expert_llm_deferral(language, expert_key)\n"
            "    if fig is not None:\n"
            "        plt.show()\n"
            "    summary = expert_llm_deferral_summary(language, expert_key)\n"
            "    if not summary.empty:\n"
            "        display(summary.style.format({'Rate': '{:.3f}', 'Macro-F1': '{:.4f}', 'QWK': '{:.4f}', 'Corrections': '{:.0f}', 'Degradations': '{:.0f}'}, na_rep='-'))\n"
            "    metric_compare = selective_metric_comparison_table(language, expert_key)\n"
            "    if not metric_compare.empty:\n"
            "        display(Markdown('#### Selective-deferral raw vs abstain-resolved metrics'))\n"
            "        display(metric_compare.style.format({'gate_rate': '{:.2f}', 'llm_call_rate': '{:.3f}', 'override_rate': '{:.3f}', 'abstain_rate': '{:.3f}', 'Macro-F1 raw': '{:.4f}', 'QWK raw': '{:.4f}', 'Macro-F1 abstain-resolved': '{:.4f}', 'QWK abstain-resolved': '{:.4f}', 'corrections': '{:.0f}', 'degradations': '{:.0f}'}, na_rep='-'))\n"
            "    action_counts = selected_action_class_counts_table(language, expert_key)\n"
            "    if not action_counts.empty:\n"
            "        display(Markdown('#### Selective-deferral action/class outcomes, abstain treated as resolved'))\n"
            "        display(action_counts.style.format({'gate_rate': '{:.2f}', 'n': '{:.0f}', 'corrections': '{:.0f}', 'degradations': '{:.0f}', 'both_correct': '{:.0f}', 'both_wrong': '{:.0f}'}, na_rep='-'))\n"
            "    edge = gated_replacement_baselines(language, expert_key)\n"
            "    if not edge.empty:\n"
            "        display(Markdown('#### Gated-item replacement baselines'))\n"
            "        display(Markdown('These diagnostics hold the non-gated items fixed at the expert prediction and replace only the gated items. Uniform random samples each sentiment with probability 1/3; class-prior random samples according to the test-set class distribution; majority always emits the test-set majority label.'))\n"
            "        fig = plot_gated_replacement_baselines(language, expert_key)\n"
            "        if fig is not None:\n"
            "            plt.show()\n"
            "        display(edge[['gate_rate', 'prompt_variant', 'llm_call_rate', 'n_gated', 'method', 'full_f1_macro', 'full_qwk', 'gated_f1_macro', 'gated_qwk']].style.format({'gate_rate': '{:.2f}', 'llm_call_rate': '{:.3f}', 'full_f1_macro': '{:.4f}', 'full_qwk': '{:.4f}', 'gated_f1_macro': '{:.4f}', 'gated_qwk': '{:.4f}'}, na_rep='-'))\n"
            "    display(Markdown('Note: for the selective-deferral red curve, `abstain_uncertain` is treated as a correct human-resolved decision when recomputing Macro-F1 and QWK from the prediction files. The raw experiment JSON keeps abstentions as primary-label predictions; this plot intentionally rewards abstention as successful escalation.'))\n"
            "    display(Markdown('#### Sentiment-rich complete vs selective LLM deferral'))\n"
            "    fig = plot_expert_llm_deferral(language, expert_key, sentiment_rich=True)\n"
            "    if fig is not None:\n"
            "        plt.show()\n"
            "    display(Markdown('Sentiment-rich subset: union of items where the gold label, expert prediction, complete LLM prediction, or selective LLM prediction is positive or negative. This focuses the comparison on non-neutral boundary behavior.'))\n"
        ),
        markdown_cell(
            "## Confidence Triage Diagnostics\n\n"
            "The next plots isolate the thresholding story. They sort items by confidence and remove the least-confident "
            "items first. If Macro-F1 and QWK improve quickly on the retained subset, the model is successfully identifying "
            "its hardest examples. That does not guarantee that a backup model can solve those examples: the stacked bars "
            "separate corrections from degradations on the deferred subset. In these plots, hard cases are not a fixed "
            "semantic category; they are the bottom x% of test items by each expert's own `confidence_score`. The hard-case "
            "count tables report the corresponding item counts and confidence thresholds."
        ),
        code_cell(
            "def confidence_frame(language):\n"
            "    rows = []\n"
            "    for display_name, expert_key in LANG_CONFIG[language]['uncertainty_experts'].items():\n"
            "        success = uncertainty_success(expert_key, language)\n"
            "        if success is None:\n"
            "            continue\n"
            "        for item in uncertainty_records(expert_key, language, 'test'):\n"
            "            if item.get('sentiment') not in LABELS or item.get(success['json_key']) not in LABELS:\n"
            "                continue\n"
            "            unc = item.get(success['uncertainty_key'], {}) or {}\n"
            "            rows.append({\n"
            "                'System': display_name,\n"
            "                'gold': int(item['sentiment']),\n"
            "                'pred': int(item[success['json_key']]),\n"
            "                'confidence': float(unc.get('confidence_score', 0.0) or 0.0),\n"
            "            })\n"
            "    if language == 'serbian' and not any(row['System'] == 'HAN-XLMR Masked' for row in rows):\n"
            "        # Fallback for older runs where Serbian HAN-XLMR uncertainty was only available inside the legacy LLM prediction file.\n"
            "        for row in llm_han_xlmr_records(language):\n"
            "            rows.append({'System': 'HAN-XLMR Masked', 'gold': row['gold'], 'pred': row['base_pred'], 'confidence': row['confidence']})\n"
            "    return pd.DataFrame(rows)\n\n"
            "def retained_metrics_by_defer_rate(df, rates=None):\n"
            "    if rates is None:\n"
            "        rates = [0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]\n"
            "    rows = []\n"
            "    for system, group in df.groupby('System'):\n"
            "        group = group.sort_values('confidence', ascending=True).reset_index(drop=True)\n"
            "        n = len(group)\n"
            "        for rate in rates:\n"
            "            k = int(round(rate * n))\n"
            "            kept = group.iloc[k:]\n"
            "            deferred = group.iloc[:k]\n"
            "            if kept.empty:\n"
            "                continue\n"
            "            metrics = calc_metrics(kept['gold'].to_numpy(), kept['pred'].to_numpy())\n"
            "            rows.append({\n"
            "                'System': system,\n"
            "                'defer_rate': k / n,\n"
            "                'threshold': float(group['confidence'].iloc[k - 1]) if k > 0 else float('-inf'),\n"
            "                'retained_n': int(len(kept)),\n"
            "                'deferred_error_rate': float((deferred['gold'] != deferred['pred']).mean()) if k > 0 else np.nan,\n"
            "                'retained_f1_macro': metrics['f1_macro'],\n"
            "                'retained_qwk': metrics['qwk'],\n"
            "            })\n"
            "    return pd.DataFrame(rows)\n\n"
            "LABEL_NAME = {-1: 'negative', 0: 'neutral', 1: 'positive'}\n"
            "TRIAGE_DISTRIBUTION_RATES = {'slovenian': [0.25, 0.30], 'serbian': [0.40]}\n\n"
            "def class_ratio_text(counts, total):\n"
            "    if total == 0:\n"
            "        return '-'\n"
            "    return ' : '.join(f\"{counts.get(label, 0) / total:.3f}\" for label in LABELS)\n\n"
            "def confidence_triage_distribution_table(language):\n"
            "    df = confidence_frame(language)\n"
            "    rows = []\n"
            "    for system, group in df.groupby('System'):\n"
            "        group = group.sort_values('confidence', ascending=True).reset_index(drop=True)\n"
            "        n = len(group)\n"
            "        overall_counts = Counter(group['gold'])\n"
            "        for rate in TRIAGE_DISTRIBUTION_RATES[language]:\n"
            "            k = int(round(rate * n))\n"
            "            parts = [('deferred_low_confidence', group.iloc[:k]), ('retained_high_confidence', group.iloc[k:]), ('overall_test', group)]\n"
            "            for segment, part in parts:\n"
            "                counts = Counter(part['gold'])\n"
            "                total = len(part)\n"
            "                metrics = calc_metrics(part['gold'].to_numpy(), part['pred'].to_numpy()) if total and segment != 'deferred_low_confidence' else {'f1_macro': np.nan, 'qwk': np.nan}\n"
            "                rows.append({\n"
            "                    'System': system,\n"
            "                    'Rate': rate,\n"
            "                    'Segment': segment,\n"
            "                    'n': total,\n"
            "                    'negative': counts.get(-1, 0),\n"
            "                    'neutral': counts.get(0, 0),\n"
            "                    'positive': counts.get(1, 0),\n"
            "                    'neg:neu:pos ratio': class_ratio_text(counts, total),\n"
            "                    'segment negative share / overall': (counts.get(-1, 0) / total) / (overall_counts.get(-1, 0) / n) if total and overall_counts.get(-1, 0) else np.nan,\n"
            "                    'segment neutral share / overall': (counts.get(0, 0) / total) / (overall_counts.get(0, 0) / n) if total and overall_counts.get(0, 0) else np.nan,\n"
            "                    'segment positive share / overall': (counts.get(1, 0) / total) / (overall_counts.get(1, 0) / n) if total and overall_counts.get(1, 0) else np.nan,\n"
            "                    'retained_f1_macro': metrics['f1_macro'],\n"
            "                    'retained_qwk': metrics['qwk'],\n"
            "                })\n"
            "    return pd.DataFrame(rows)\n\n"
            "def plot_confidence_triage(language):\n"
            "    curves = retained_metrics_by_defer_rate(confidence_frame(language))\n"
            "    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)\n"
            "    for system, sub in curves.groupby('System'):\n"
            "        sub = sub.sort_values('defer_rate')\n"
            "        axes[0].plot(sub['defer_rate'], sub['retained_f1_macro'], marker='o', label=system)\n"
            "        axes[1].plot(sub['defer_rate'], sub['retained_qwk'], marker='o', label=system)\n"
            "    axes[0].set_title(f\"{LANG_CONFIG[language]['display']}: retained Macro-F1 after low-confidence deferral\")\n"
            "    axes[1].set_title(f\"{LANG_CONFIG[language]['display']}: retained QWK after low-confidence deferral\")\n"
            "    for ax in axes:\n"
            "        ax.set_xlabel('Deferred / abstained fraction')\n"
            "        ax.grid(alpha=0.25)\n"
            "        ax.legend()\n"
            "    axes[0].set_ylabel('Score on retained items')\n"
            "    fig.tight_layout()\n"
            "    return fig\n\n"
            "def selected_defer_outcome_table(language):\n"
            "    rows = []\n"
            "    base = best_single_expert_key(language)\n"
            "    target = best_deployable_curve_target(language, base)\n"
            "    if target:\n"
            "        curves = read_defer_curve(language)\n"
            "        row = curves[(curves['base_policy'] == base) & (curves['target_policy'] == target)].sort_values('f1_macro', ascending=False).iloc[0]\n"
            "        rows.append({\n"
            "            'System': f\"{display_name(base, language)} -> {display_name(target, language)}\",\n"
            "            'Deferred': int(row.get('num_deferred', 0)),\n"
            "            'Corrections': int(row.get('corrections', 0)),\n"
            "            'Degradations': int(row.get('degradations', 0)),\n"
            "            'Both correct': int(row.get('both_correct_when_deferred', 0)),\n"
            "            'Both wrong': int(row.get('both_wrong_when_deferred', 0)),\n"
            "        })\n"
            "    llm_stats = llm_han_xlmr_stats(language)\n"
            "    if llm_stats:\n"
            "        rows.append({\n"
            "            'System': 'HAN-XLMR Masked -> Gemma-3-27B observed overrides',\n"
            "            'Deferred': llm_stats['num_deferred'],\n"
            "            'Corrections': llm_stats['corrections'],\n"
            "            'Degradations': llm_stats['degradations'],\n"
            "            'Both correct': llm_stats['both_correct_when_deferred'],\n"
            "            'Both wrong': llm_stats['both_wrong_when_deferred'],\n"
            "        })\n"
            "    return pd.DataFrame(rows)\n\n"
            "def hard_case_count_table(language):\n"
            "    counts = []\n"
            "    cf = confidence_frame(language)\n"
            "    for system, group in cf.groupby('System'):\n"
            "        n = len(group)\n"
            "        for rate in [0.10, 0.20, 0.30, 0.40, 0.50]:\n"
            "            k = int(round(rate * n))\n"
            "            threshold = float(group.sort_values('confidence')['confidence'].iloc[k - 1]) if k > 0 else np.nan\n"
            "            counts.append({'Policy': 'confidence-ranked low-confidence band', 'System': system, 'Rate': rate, 'Hard cases': k, 'Confidence threshold': threshold})\n"
            "    base = best_single_expert_key(language)\n"
            "    target = best_deployable_curve_target(language, base)\n"
            "    if target:\n"
            "        curves = read_defer_curve(language)\n"
            "        row = curves[(curves['base_policy'] == base) & (curves['target_policy'] == target)].sort_values('f1_macro', ascending=False).iloc[0]\n"
            "        counts.append({'Policy': 'best plotted deployable gate', 'System': f\"{display_name(base, language)} -> {display_name(target, language)}\", 'Rate': row['defer_rate'], 'Hard cases': int(row.get('num_deferred', round(row['defer_rate'] * load_multi_metrics(language)['num_test']))), 'Confidence threshold': row.get('threshold', np.nan)})\n"
            "    llm_stats = llm_han_xlmr_stats(language)\n"
            "    if llm_stats:\n"
            "        counts.append({'Policy': 'observed LLM overrides', 'System': 'HAN-XLMR Masked -> Gemma-3-27B', 'Rate': llm_stats['defer_rate'], 'Hard cases': llm_stats['num_deferred'], 'Confidence threshold': np.nan})\n"
            "    return pd.DataFrame(counts)\n\n"
            "def plot_defer_outcome_bars(language):\n"
            "    df = selected_defer_outcome_table(language)\n"
            "    cols = ['Corrections', 'Degradations', 'Both correct', 'Both wrong']\n"
            "    fig, ax = plt.subplots(figsize=(10, 4.5))\n"
            "    bottom = np.zeros(len(df))\n"
            "    colors = ['#2ca02c', '#d62728', '#9edae5', '#7f7f7f']\n"
            "    for col, color in zip(cols, colors):\n"
            "        values = df[col].to_numpy(dtype=float)\n"
            "        ax.bar(df['System'], values, bottom=bottom, label=col, color=color)\n"
            "        bottom += values\n"
            "    ax.set_title(f\"{LANG_CONFIG[language]['display']}: what happened on deferred/overridden items\")\n"
            "    ax.set_ylabel('Items')\n"
            "    ax.set_xticklabels(df['System'], rotation=25, ha='right')\n"
            "    ax.legend()\n"
            "    ax.grid(axis='y', alpha=0.25)\n"
            "    fig.tight_layout()\n"
            "    return fig\n\n"
            "for language in ['slovenian', 'serbian']:\n"
            "    display(Markdown(f\"### {LANG_CONFIG[language]['display']} confidence triage\"))\n"
            "    plot_confidence_triage(language)\n"
            "    plt.show()\n"
            "    triage = retained_metrics_by_defer_rate(confidence_frame(language))\n"
            "    display(triage[triage['defer_rate'].isin([0, 0.1, 0.2, 0.3, 0.5])].style.format({'defer_rate': '{:.2f}', 'threshold': '{:.3f}', 'deferred_error_rate': '{:.3f}', 'retained_f1_macro': '{:.4f}', 'retained_qwk': '{:.4f}'}, na_rep='-'))\n"
            "    display(Markdown(f\"### {LANG_CONFIG[language]['display']} class mix at triage thresholds\"))\n"
            "    display(confidence_triage_distribution_table(language).style.format({'Rate': '{:.2f}', 'segment negative share / overall': '{:.2f}', 'segment neutral share / overall': '{:.2f}', 'segment positive share / overall': '{:.2f}', 'retained_f1_macro': '{:.4f}', 'retained_qwk': '{:.4f}'}, na_rep='-'))\n"
            "    display(Markdown(f\"### {LANG_CONFIG[language]['display']} deferred-item outcomes\"))\n"
            "    display(selected_defer_outcome_table(language))\n"
            "    plot_defer_outcome_bars(language)\n"
            "    plt.show()\n"
            "    display(Markdown(f\"### {LANG_CONFIG[language]['display']} hard-case counts\"))\n"
            "    display(hard_case_count_table(language).style.format({'Rate': '{:.3f}', 'Confidence threshold': '{:.3f}'}, na_rep='-'))\n"
        ),
        markdown_cell(
            "## LR/RF Stackers and Threshold Logic\n\n"
            "**LR stacker:** logistic regression over all experts' probabilities, uncertainty signals, vote/disagreement "
            "features, and prediction indicators. It is a linear meta-classifier.\n\n"
            "**RF stacker:** random forest over the same features. It can learn nonlinear interactions but can also overfit "
            "the train/validation distribution. In the current masked outputs, the simple policies are easier to explain "
            "and stronger than the learned stackers.\n\n"
            "There are two mechanisms here. Expert-selection policies such as `majority_vote`, `avg_probs`, and "
            "`confidence_pick` do not use a threshold. The defer-gate rows do use threshold logic: a validation-tuned "
            "confidence threshold decides whether to keep the base expert prediction or defer to another policy/oracle."
        ),
        markdown_cell(
            "## Interpretation Notes and Next Experiments\n\n"
            "**Core interpretation.** The useful signal is not that the backup policies are always better; it is that "
            "expert confidence is ranking risk. When retained-subset Macro-F1/QWK rises after removing low-confidence "
            "items, the expert is identifying its hardest cases. If the deployable deferral curve stays flat, the bottleneck "
            "is the substitute model: majority vote, confidence pick, LR/RF, or Gemma is not adding enough new information "
            "on precisely those hard items.\n\n"
            "**Calibration intuition.** Slovenian HAN-XLMR appears comparatively well calibrated, likely because the "
            "hierarchical/global context setup matches the document-level task and the Slovenian split has fewer severe "
            "class-boundary/domain-shift artifacts. Longformer and SloBERTa can still be strong classifiers while being "
            "poor probability estimators: their high-confidence regions contain more mistakes, so they are overconfident. "
            "For Serbo-Croatian, the train-val overconfidence plus test underconfidence pattern suggests distribution "
            "shift or calibration drift: the model learned sharper train-val confidence boundaries, while the holdout has "
            "harder or differently distributed cases where confidence is compressed. The stronger class balance in the "
            "Serbo-Croatian test set can also make minority/error modes more visible in reliability plots.\n\n"
            "**How to defend LLM deferral.** Treat the LLM not as a better average classifier, but as an adjudicator for "
            "a selected ambiguity band. The current full-test LLM rows may underperform simple policies, but the important "
            "question is whether a thresholded subset exists where Gemma has positive net corrections. If yes, report the "
            "LLM as a targeted escalation path. If no, report that the LLM mainly provides interpretable rationales and "
            "failure taxonomy, while cheap expert aggregation remains the better automatic router.\n\n"
            "**Next experiments worth trying.** Calibrate an explicit defer threshold on validation for HAN-XLMR -> Gemma "
            "instead of using all Gemma predictions. Add a two-stage gate: only call the LLM when confidence is below a "
            "threshold and experts disagree. Try prompt variants that force contrastive evidence extraction: `evidence for "
            "positive`, `evidence for negative`, then final label. Ask the LLM to abstain/keep PLM when evidence is weak, "
            "turning it into a true override policy. Use a stronger judge model only on the lowest-confidence 5-20% band. "
            "Train a lightweight meta-router using confidence, entropy, expert disagreement, aspect seen/unseen, document "
            "length, and class prior, with target label `LLM fixes expert`. Evaluate per-class net corrections, especially "
            "negative sentiment. If GPU budget is available, run a stronger multilingual instruction model or a hosted "
            "OpenAI-compatible model on only the validation-selected uncertainty band rather than the full test set."
        ),
        markdown_cell(
            "## Suggested Reporting Wording\n\n"
            "For the main table, report raw checkpoint baselines as mean ± SD across the three checkpoints. Report the "
            "uncertainty/multi-expert system as a single aggregate inference result, because it already pools the "
            "checkpoints and MC samples into one predictive distribution. A concise note can say: `Uncertainty results "
            "are point estimates from a checkpoint-ensemble/MC-dropout aggregate; raw baselines are reported as mean ± SD "
            "across the three independently selected checkpoints.`"
        ),
        markdown_cell(
            "## Selective Deferral Seen/Unseen Metrics\n\n"
            "This final table mirrors the seen/unseen split used in the single-expert and expert+LLM defer result CSVs. "
            "For each requested selective-deferral system, masked and unmasked prompt variants are evaluated separately. "
            "Within each row, the best available gate is selected by abstain-resolved Macro-F1, then QWK. The final "
            "prediction treats `abstain_uncertain` as a successful escalation, matching the selective-deferral plots above."
        ),
        code_cell(
            "def normalize_aspect(value):\n"
            "    return str(value or '').strip().casefold()\n\n"
            "def split_seen_unseen_records(records, language):\n"
            "    unseen = {normalize_aspect(a) for a in UNSEEN_ASPECTS[language]}\n"
            "    seen, unseen_rows = [], []\n"
            "    for row in records:\n"
            "        if normalize_aspect(row.get('aspect')) in unseen:\n"
            "            unseen_rows.append(row)\n"
            "        else:\n"
            "            seen.append(row)\n"
            "    return seen, unseen_rows\n\n"
            "def metrics_for_selective_records(records):\n"
            "    if not records:\n"
            "        return {'f1_macro': np.nan, 'qwk': np.nan, 'accuracy': np.nan, 'n': 0}\n"
            "    metrics = calc_metrics([row['gold'] for row in records], [row['final_pred'] for row in records])\n"
            "    metrics['n'] = len(records)\n"
            "    return metrics\n\n"
            "def best_selective_run_for_prompt(language, expert_key, prompt_variant, autorun='medium'):\n"
            "    runs = selective_deferral_all_runs(language, expert_key, autorun=autorun)\n"
            "    if runs.empty:\n"
            "        return None\n"
            "    runs = runs[runs['prompt_variant'] == prompt_variant].copy()\n"
            "    if runs.empty:\n"
            "        return None\n"
            "    return runs.sort_values(['f1_macro', 'qwk', 'accuracy'], ascending=[False, False, False]).iloc[0]\n\n"
            "def selective_seen_unseen_table():\n"
            "    requests = [\n"
            "        ('slovenian', 'longformer_masked', 'masked'),\n"
            "        ('slovenian', 'longformer_masked', 'unmasked'),\n"
            "        ('serbian', 'longformer_masked', 'masked'),\n"
            "        ('serbian', 'longformer_masked', 'unmasked'),\n"
            "        ('serbian', 'mdeberta_masked', 'masked'),\n"
            "        ('serbian', 'mdeberta_masked', 'unmasked'),\n"
            "    ]\n"
            "    rows = []\n"
            "    for language, expert_key, prompt_variant in requests:\n"
            "        run = best_selective_run_for_prompt(language, expert_key, prompt_variant)\n"
            "        if run is None:\n"
            "            rows.append({\n"
            "                'Language': LANG_CONFIG[language]['display'],\n"
            "                'System': f\"{EXPERT_LLM_CONFIG[expert_key]['display']} + Gemma-3-27B selective\",\n"
            "                'Expert key': expert_key,\n"
            "                'Prompt Variant': prompt_variant,\n"
            "                'Best Gate Rate': np.nan,\n"
            "                'LLM call rate': np.nan,\n"
            "                'Macro-F1': np.nan,\n"
            "                'QWK': np.nan,\n"
            "                'F1 (Seen)': np.nan,\n"
            "                'F1 (Unseen)': np.nan,\n"
            "                'QWK (Seen)': np.nan,\n"
            "                'QWK (Unseen)': np.nan,\n"
            "                'Seen n': 0,\n"
            "                'Unseen n': 0,\n"
            "                'Source': '',\n"
            "            })\n"
            "            continue\n"
            "        records = selective_prediction_records(Path(run['predictions_path']), reward_abstain=True)\n"
            "        seen_records, unseen_records = split_seen_unseen_records(records, language)\n"
            "        seen_metrics = metrics_for_selective_records(seen_records)\n"
            "        unseen_metrics = metrics_for_selective_records(unseen_records)\n"
            "        rows.append({\n"
            "            'Language': LANG_CONFIG[language]['display'],\n"
            "            'System': f\"{EXPERT_LLM_CONFIG[expert_key]['display']} + Gemma-3-27B selective\",\n"
            "            'Expert key': expert_key,\n"
            "            'Prompt Variant': prompt_variant,\n"
            "            'Best Gate Rate': float(run['gate_rate']),\n"
            "            'LLM call rate': float(run['llm_call_rate']),\n"
            "            'Macro-F1': float(run['f1_macro']),\n"
            "            'QWK': float(run['qwk']),\n"
            "            'F1 (Seen)': seen_metrics['f1_macro'],\n"
            "            'F1 (Unseen)': unseen_metrics['f1_macro'],\n"
            "            'QWK (Seen)': seen_metrics['qwk'],\n"
            "            'QWK (Unseen)': unseen_metrics['qwk'],\n"
            "            'Seen n': seen_metrics['n'],\n"
            "            'Unseen n': unseen_metrics['n'],\n"
            "            'Source': str(Path(run['predictions_path']).relative_to(ROOT)),\n"
            "        })\n"
            "    return pd.DataFrame(rows)\n\n"
            "selective_seen_unseen = selective_seen_unseen_table()\n"
            "out_path = ROOT / 'reviews/scratchpad/selective_deferred_seen_unseen_results.csv'\n"
            "out_path.parent.mkdir(parents=True, exist_ok=True)\n"
            "selective_seen_unseen.to_csv(out_path, index=False)\n"
            "display(Markdown(f'CSV saved to `{out_path.relative_to(ROOT)}`.'))\n"
            "display(selective_seen_unseen.style.format({\n"
            "    'Best Gate Rate': '{:.2f}',\n"
            "    'LLM call rate': '{:.3f}',\n"
            "    'Macro-F1': '{:.4f}',\n"
            "    'QWK': '{:.4f}',\n"
            "    'F1 (Seen)': '{:.4f}',\n"
            "    'F1 (Unseen)': '{:.4f}',\n"
            "    'QWK (Seen)': '{:.4f}',\n"
            "    'QWK (Unseen)': '{:.4f}',\n"
            "}, na_rep='-'))\n"
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
    notebook = build_notebook({})
    write_json(args.output, notebook)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
