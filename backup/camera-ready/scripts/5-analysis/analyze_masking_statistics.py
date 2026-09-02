#!/usr/bin/env python3
"""Paired statistical tests for masking effects.

This uses per-aspect macro-F1 as the paired observation, mirroring the earlier
statistical-analysis note. For three-run neural models, per-aspect scores are
averaged across the three test prediction files before testing masked vs.
unmasked, so each aspect contributes one paired observation per language/model
family.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

try:
    from baycomp import SignedRankTest
except Exception:  # pragma: no cover - depends on local env
    SignedRankTest = None


ROOT = Path(os.environ.get("ABSA_RELEASE_ROOT", Path(__file__).resolve().parents[2])).resolve()
OUT_DIR = ROOT / "reviews" / "scratchpad"
ROPE = 0.01
RNG = np.random.default_rng(42)


def load_builder_module():
    path = ROOT / "scripts" / "5-analysis" / "build_final_results_notebook.py"
    spec = importlib.util.spec_from_file_location("final_results_builder", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = load_builder_module()
SPEC_BY_KEY = {spec["key"]: spec for spec in builder.TABLE_SPECS}


MASKING_PAIRS = [
    {
        "pair": "Document Embeddings + MLP",
        "unmasked": "bge_m3_mlp_whole",
        "masked": "bge_m3_mlp_masked",
        "notes": "Whole-document BGE-M3 MLP vs aspect-masked BGE-M3 MLP.",
    },
    {
        "pair": "XLMR LLM Summary",
        "unmasked": "xlmr_gemma_summary",
        "masked": "xlmr_gemma_summary_masked",
        "notes": "Gemma-summary input vs masked Gemma-summary input.",
    },
    {
        "pair": "HAN + XLMR",
        "unmasked": "han_with_aspect_markers",
        "masked": "han_simplified_dart",
        "notes": "Matches current paper Baseline vs Masked rows; not a pure single-factor masking ablation because the HAN variants also differ architecturally.",
    },
    {
        "pair": "Longformer",
        "unmasked": "longformer_unmasked",
        "masked": "longformer_masked",
        "notes": "additional-comparison long-context encoder.",
    },
    {
        "pair": "mDeBERTa-v3",
        "unmasked": "mdeberta_unmasked",
        "masked": "mdeberta_masked",
        "notes": "additional-comparison mDeBERTa baseline.",
    },
    {
        "pair": "mT5",
        "unmasked": "mt5_unmasked",
        "masked": "mt5_masked",
        "notes": "additional-comparison text-to-text baseline.",
    },
    {
        "pair": "Language-specific Encoder",
        "unmasked": "slavic_specific_unmasked",
        "masked": "slavic_specific_masked",
        "notes": "SloBERTa for Slovenian and BERTic for Serbo-Croatian.",
    },
    {
        "pair": "LLM Softmax Fusion",
        "unmasked": "llm_softmax_fusion",
        "masked": "llm_softmax_fusion_masked",
        "notes": "Prompting/fusion row; single prediction file per language.",
    },
    {
        "pair": "LLM Uncertainty Fusion",
        "unmasked": "llm_uncertainty_fusion",
        "masked": "llm_uncertainty_fusion_masked",
        "notes": "Prompting/fusion row; single prediction file per language.",
    },
]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def paths_for_key(key: str, language: str) -> list[Path]:
    spec = SPEC_BY_KEY[key]
    return builder.prediction_paths_for_spec(spec, language)


def nanmean_or_nan(values: list[float]) -> float:
    clean = [v for v in values if not (isinstance(v, float) and math.isnan(v))]
    if not clean:
        return math.nan
    return float(np.mean(clean))


def table_metric_delta(pair: dict[str, str], language: str) -> dict[str, float]:
    unmasked = builder.build_overall_row(SPEC_BY_KEY[pair["unmasked"]], language)
    masked = builder.build_overall_row(SPEC_BY_KEY[pair["masked"]], language)
    return {
        "table_delta_f1": (
            masked["metrics"]["f1_macro"]["mean"] - unmasked["metrics"]["f1_macro"]["mean"]
        ),
        "table_delta_qwk": (
            masked["metrics"]["qwk"]["mean"] - unmasked["metrics"]["qwk"]["mean"]
        ),
    }


def per_aspect_scores(paths: list[Path]) -> dict[str, dict[str, float]]:
    by_aspect: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for path in paths:
        records = load_json(path)
        records_by_aspect: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in records:
            records_by_aspect[str(item.get("aspect", ""))].append(item)
        for aspect, aspect_records in records_by_aspect.items():
            metrics = builder.metrics_for_record_subset(aspect_records)
            by_aspect[aspect]["f1_macro"].append(metrics["f1_macro"])
            by_aspect[aspect]["qwk"].append(metrics["qwk"])

    output = {}
    for aspect, metrics in by_aspect.items():
        output[aspect] = {
            "f1_macro": nanmean_or_nan(metrics["f1_macro"]),
            "qwk": nanmean_or_nan(metrics["qwk"]),
        }
    return output


def wilcoxon(values: np.ndarray, alternative: str) -> float | None:
    nonzero = values[np.abs(values) > 1e-12]
    if len(nonzero) == 0:
        return 1.0
    try:
        return float(stats.wilcoxon(nonzero, alternative=alternative).pvalue)
    except ValueError:
        return None


def bootstrap_mean_delta(values: np.ndarray, n_bootstrap: int = 10000) -> dict[str, float]:
    if len(values) == 0:
        return {"ci_low": math.nan, "ci_high": math.nan, "p_gt_0": math.nan, "p_gt_rope": math.nan}
    samples = RNG.choice(values, size=(n_bootstrap, len(values)), replace=True)
    means = samples.mean(axis=1)
    return {
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
        "p_gt_0": float(np.mean(means > 0)),
        "p_gt_rope": float(np.mean(means > ROPE)),
    }


def bayesian_signed_rank(masked: np.ndarray, unmasked: np.ndarray) -> dict[str, float | None]:
    if SignedRankTest is None:
        return {"p_masked": None, "p_rope": None, "p_unmasked": None}
    try:
        probs = SignedRankTest(masked, unmasked, rope=ROPE).probs()
        return {
            "p_masked": float(probs[0]),
            "p_rope": float(probs[1]),
            "p_unmasked": float(probs[2]),
        }
    except Exception:
        return {"p_masked": None, "p_rope": None, "p_unmasked": None}


def verdict(row: dict[str, Any]) -> str:
    mean_delta = row["mean_delta_f1"]
    p_masked = row.get("bayes_p_masked")
    p_unmasked = row.get("bayes_p_unmasked")
    p_rope = row.get("bayes_p_rope")
    ci_low = row.get("bootstrap_ci_low")
    ci_high = row.get("bootstrap_ci_high")
    if p_masked is not None and p_masked >= 0.95 and ci_low is not None and ci_low > 0:
        return "masked better"
    if p_unmasked is not None and p_unmasked >= 0.95 and ci_high is not None and ci_high < 0:
        return "unmasked better"
    if p_rope is not None and p_rope >= 0.95:
        return "practically equivalent"
    if mean_delta > ROPE:
        return "leans masked"
    if mean_delta < -ROPE:
        return "leans unmasked"
    return "small/mixed"


def analyze_pair(pair: dict[str, str], language: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    unmasked_paths = paths_for_key(pair["unmasked"], language)
    masked_paths = paths_for_key(pair["masked"], language)
    unmasked_scores = per_aspect_scores(unmasked_paths)
    masked_scores = per_aspect_scores(masked_paths)
    aspects = sorted(set(unmasked_scores) & set(masked_scores))

    detail_rows = []
    for aspect in aspects:
        base = unmasked_scores[aspect]
        masked = masked_scores[aspect]
        detail_rows.append(
            {
                "pair": pair["pair"],
                "language": language,
                "aspect": aspect,
                "unmasked_f1": base["f1_macro"],
                "masked_f1": masked["f1_macro"],
                "delta_f1": masked["f1_macro"] - base["f1_macro"],
                "unmasked_qwk": base["qwk"],
                "masked_qwk": masked["qwk"],
                "delta_qwk": masked["qwk"] - base["qwk"],
            }
        )

    deltas = np.array([row["delta_f1"] for row in detail_rows], dtype=float)
    qwk_deltas = np.array([row["delta_qwk"] for row in detail_rows], dtype=float)
    masked = np.array([row["masked_f1"] for row in detail_rows], dtype=float)
    unmasked = np.array([row["unmasked_f1"] for row in detail_rows], dtype=float)
    boot = bootstrap_mean_delta(deltas)
    bayes = bayesian_signed_rank(masked, unmasked)
    table_deltas = table_metric_delta(pair, language)
    wins = int(np.sum(deltas > 1e-12))
    losses = int(np.sum(deltas < -1e-12))
    ties = int(np.sum(np.abs(deltas) <= 1e-12))
    sign_p = (
        float(stats.binomtest(wins, wins + losses, 0.5, alternative="greater").pvalue)
        if wins + losses
        else 1.0
    )
    row = {
        "pair": pair["pair"],
        "language": language,
        "n_aspects": len(deltas),
        "table_delta_f1": table_deltas["table_delta_f1"],
        "table_delta_qwk": table_deltas["table_delta_qwk"],
        "mean_delta_f1": float(np.mean(deltas)),
        "median_delta_f1": float(np.median(deltas)),
        "mean_delta_qwk": float(np.nanmean(qwk_deltas)),
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "sign_test_p_greater": sign_p,
        "wilcoxon_p_greater": wilcoxon(deltas, "greater"),
        "wilcoxon_p_two_sided": wilcoxon(deltas, "two-sided"),
        "bootstrap_ci_low": boot["ci_low"],
        "bootstrap_ci_high": boot["ci_high"],
        "bootstrap_p_gt_0": boot["p_gt_0"],
        "bootstrap_p_gt_rope": boot["p_gt_rope"],
        "bayes_p_masked": bayes["p_masked"],
        "bayes_p_rope": bayes["p_rope"],
        "bayes_p_unmasked": bayes["p_unmasked"],
        "notes": pair.get("notes", ""),
    }
    row["verdict"] = verdict(row)
    return row, detail_rows


def analyze_all() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    summary_rows = []
    detail_rows = []
    for pair in MASKING_PAIRS:
        for language in builder.LANGUAGES:
            row, details = analyze_pair(pair, language)
            summary_rows.append(row)
            detail_rows.extend(details)

    aggregate_rows = []
    for pair_name in sorted({row["pair"] for row in detail_rows}):
        pair_details = [row for row in detail_rows if row["pair"] == pair_name]
        aggregate_rows.append(analyze_deltas(pair_name, "pooled_languages", pair_details))

    aggregate_rows.append(analyze_deltas("ALL_MASKING_PAIRS", "pooled_aspects", detail_rows))

    family_language_deltas = [
        {
            "pair": "ALL_MASKING_PAIRS",
            "language": "family_language_means",
            "delta_f1": row["mean_delta_f1"],
            "masked_f1": row["mean_delta_f1"],
            "unmasked_f1": 0.0,
            "delta_qwk": row["mean_delta_qwk"],
        }
        for row in summary_rows
    ]
    aggregate_rows.append(
        analyze_deltas("ALL_MASKING_PAIRS", "family_language_means", family_language_deltas)
    )
    return summary_rows, detail_rows, aggregate_rows


def analyze_deltas(pair_name: str, language: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    deltas = np.array([row["delta_f1"] for row in rows], dtype=float)
    qwk_deltas = np.array([row["delta_qwk"] for row in rows], dtype=float)
    masked = np.array([row["masked_f1"] for row in rows], dtype=float)
    unmasked = np.array([row["unmasked_f1"] for row in rows], dtype=float)
    boot = bootstrap_mean_delta(deltas)
    bayes = bayesian_signed_rank(masked, unmasked)
    wins = int(np.sum(deltas > 1e-12))
    losses = int(np.sum(deltas < -1e-12))
    ties = int(np.sum(np.abs(deltas) <= 1e-12))
    row = {
        "pair": pair_name,
        "language": language,
        "n_aspects": len(deltas),
        "table_delta_f1": math.nan,
        "table_delta_qwk": math.nan,
        "mean_delta_f1": float(np.mean(deltas)),
        "median_delta_f1": float(np.median(deltas)),
        "mean_delta_qwk": float(np.nanmean(qwk_deltas)),
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "sign_test_p_greater": (
            float(stats.binomtest(wins, wins + losses, 0.5, alternative="greater").pvalue)
            if wins + losses
            else 1.0
        ),
        "wilcoxon_p_greater": wilcoxon(deltas, "greater"),
        "wilcoxon_p_two_sided": wilcoxon(deltas, "two-sided"),
        "bootstrap_ci_low": boot["ci_low"],
        "bootstrap_ci_high": boot["ci_high"],
        "bootstrap_p_gt_0": boot["p_gt_0"],
        "bootstrap_p_gt_rope": boot["p_gt_rope"],
        "bayes_p_masked": bayes["p_masked"],
        "bayes_p_rope": bayes["p_rope"],
        "bayes_p_unmasked": bayes["p_unmasked"],
        "notes": "Aggregate exploratory analysis; interpret with dependence caveats.",
    }
    row["verdict"] = verdict(row)
    return row


def fmt_pct(value: float | None, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "NA"
    return f"{value * 100:.{digits}f}"


def fmt_float(value: float | None, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "NA"
    return f"{value:.{digits}f}"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, Any]]) -> str:
    headers = [
        "Pair",
        "Lang",
        "n",
        "ΔTable F1 pp",
        "ΔAspect F1 pp",
        "95% boot CI pp",
        "ΔTable QWK",
        "ΔQWK",
        "W/L/T",
        "Wilcoxon p>",
        "Bayes P(masked)",
        "Bayes P(eq)",
        "Bayes P(unmasked)",
        "Verdict",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        ci = f"[{fmt_pct(row['bootstrap_ci_low'])}, {fmt_pct(row['bootstrap_ci_high'])}]"
        lines.append(
            "| {pair} | {language} | {n} | {tdelta} | {delta} | {ci} | {tdqwk} | {dqwk} | {w}/{l}/{t} | {wp} | {bp_m} | {bp_e} | {bp_u} | {verdict} |".format(
                pair=row["pair"],
                language=row["language"],
                n=row["n_aspects"],
                tdelta=fmt_pct(row.get("table_delta_f1")),
                delta=fmt_pct(row["mean_delta_f1"]),
                ci=ci,
                tdqwk=fmt_float(row.get("table_delta_qwk")),
                dqwk=fmt_float(row["mean_delta_qwk"]),
                w=row["wins"],
                l=row["losses"],
                t=row["ties"],
                wp=fmt_float(row["wilcoxon_p_greater"], 4),
                bp_m=fmt_float(row["bayes_p_masked"], 3),
                bp_e=fmt_float(row["bayes_p_rope"], 3),
                bp_u=fmt_float(row["bayes_p_unmasked"], 3),
                verdict=row["verdict"],
            )
        )
    return "\n".join(lines)


def write_markdown(summary_rows: list[dict[str, Any]], aggregate_rows: list[dict[str, Any]]) -> None:
    by_verdict = defaultdict(int)
    for row in summary_rows:
        by_verdict[row["verdict"]] += 1
    positive = sum(1 for row in summary_rows if row["mean_delta_f1"] > 0)
    negative = sum(1 for row in summary_rows if row["mean_delta_f1"] < 0)
    table_positive = sum(1 for row in summary_rows if row["table_delta_f1"] > 0)
    table_negative = sum(1 for row in summary_rows if row["table_delta_f1"] < 0)

    lines = [
        "# Masking Statistical Tests",
        "",
        "Generated by `reviews/analyze_masking_statistics.py`.",
        "",
        "Method: paired per-aspect macro-F1. For three-run models, the per-aspect score is averaged across runs before masked and unmasked systems are compared. ROPE is ±0.01 macro-F1.",
        "",
        f"Bayesian signed-rank via `baycomp`: {'available' if SignedRankTest is not None else 'not available'}",
        "",
        "## Executive Read",
        "",
        f"Across the {len(summary_rows)} family-language masking comparisons, the main-table aggregate macro-F1 improves in {table_positive} and decreases in {table_negative}. When aspects are weighted equally for paired statistical testing, {positive} have positive mean ΔF1 and {negative} have negative mean ΔF1.",
        "",
        "This supports a cautious claim: masking often improves the aggregate benchmark score, especially for the additional-comparison encoder baselines, but the effect is not uniformly positive across aspects, model families, and languages. The safest paper wording is that masking is a useful regularizer/ablation that frequently improves aggregate F1 and often improves QWK, not that it is a statistically consistent universal improvement.",
        "",
        "Important distinction: `ΔTable F1` is the difference in the paper-table aggregate macro-F1. `ΔAspect F1` is the mean of paired per-aspect macro-F1 differences used for statistical testing. They can disagree when a method improves high-volume aspects but hurts smaller aspects, or vice versa.",
        "",
        "## Pair-by-Language Tests",
        "",
        markdown_table(summary_rows),
        "",
        "## Aggregate Exploratory Tests",
        "",
        markdown_table(aggregate_rows),
        "",
        "## Interpretation Notes",
        "",
        "- Treat the HAN + XLMR comparison carefully: the current paper's Baseline/Masked rows also differ in architecture, so this is not a pure masking-only ablation.",
        "- The strongest clean masking story is among additional-comparison encoders: Longformer, mDeBERTa-v3, and mT5 improve with masking in both languages; language-specific encoders are already strong and the unmasked version slightly edges masked in macro-F1, though Slovenian QWK improves with masking.",
        "- The LLM uncertainty-fusion Serbian masked result is a clear counterexample to a universal masking-improves claim.",
        "- If the manuscript needs one sentence: `Across model families, masking usually improves or preserves macro-F1 and often improves QWK, but paired tests show this effect is model-dependent rather than universal.`",
    ]
    (OUT_DIR / "masking_statistical_tests.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUT_DIR,
        help="Directory for tables and the Markdown report.",
    )
    return parser.parse_args()


def main() -> None:
    global OUT_DIR
    args = parse_args()
    OUT_DIR = args.output_dir
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows, detail_rows, aggregate_rows = analyze_all()
    write_csv(OUT_DIR / "masking_statistical_tests.csv", summary_rows)
    write_csv(OUT_DIR / "masking_statistical_tests_by_aspect.csv", detail_rows)
    write_csv(OUT_DIR / "masking_statistical_tests_aggregate.csv", aggregate_rows)
    (OUT_DIR / "masking_statistical_tests.json").write_text(
        json.dumps(
            {
                "rope": ROPE,
                "baycomp_available": SignedRankTest is not None,
                "summary": summary_rows,
                "aggregate": aggregate_rows,
                "detail": detail_rows,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    write_markdown(summary_rows, aggregate_rows)
    print(f"Wrote {OUT_DIR / 'masking_statistical_tests.md'}")
    print(f"Wrote {OUT_DIR / 'masking_statistical_tests.csv'}")


if __name__ == "__main__":
    main()
