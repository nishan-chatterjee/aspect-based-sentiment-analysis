#!/usr/bin/env python3
"""Multi-expert learning-to-defer experiments from uncertainty JSON files.

This is a cheap second-stage experiment: it consumes the already generated
checkpoint-ensemble + MC-dropout predictions for several experts, trains small
stacking models on train/val, and evaluates validation-tuned defer gates on
the test set.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    precision_recall_fscore_support,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT_DIR = Path(os.environ.get("ABSA_RELEASE_ROOT", Path(__file__).resolve().parents[2])).resolve()
UNCERTAINTY_ROOT = ROOT_DIR / "reviews" / "uncertainty"
OUTPUT_ROOT = UNCERTAINTY_ROOT / "multi-expert-defer"
LABELS = [-1, 0, 1]
PROB_LABELS = ("Negative", "Neutral", "Positive")

DEFAULT_EXPERTS_BY_LANGUAGE = {
    "slovenian": ["longformer_masked", "slavic_specific_masked", "han_xlmr_masked"],
    "serbian": ["longformer_masked", "slavic_specific_masked", "mdeberta_masked"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", required=True, choices=["slovenian", "serbian"])
    parser.add_argument("--setting", default="masked", choices=["masked", "unmasked"])
    parser.add_argument("--experts", nargs="+", default=None)
    parser.add_argument("--uncertainty-root", default=str(UNCERTAINTY_ROOT))
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--split-index", type=int, default=0)
    parser.add_argument("--metric", default="f1_macro", choices=["f1_macro", "qwk", "accuracy"])
    parser.add_argument("--max-train-items", type=int, default=None)
    parser.add_argument("--max-val-items", type=int, default=None)
    parser.add_argument("--limit-test-items", type=int, default=None)
    parser.add_argument("--defer-rates", default="0,0.05,0.10,0.15,0.20,0.25,0.30,0.40,0.50")
    parser.add_argument("--random-baseline-repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(data), f, indent=2, ensure_ascii=False)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def split_path(root: Path, expert: str, language: str, split_name: str) -> Path:
    base = root / expert / language
    if split_name == "test":
        return base / f"{language}_test_complete.json"
    if split_name.startswith("train_val_"):
        index = split_name.rsplit("_", 1)[-1]
        return base / f"{language}_train_val_complete_{index}.json"
    raise ValueError(f"Unsupported split name: {split_name}")


def load_records(root: Path, expert: str, language: str, split_name: str, subset: str | None) -> list[dict[str, Any]]:
    path = split_path(root, expert, language, split_name)
    data = load_json(path)
    if split_name == "test":
        if isinstance(data, dict) and "test" in data:
            return data["test"]
        if isinstance(data, list):
            return data
    if subset and isinstance(data, dict) and subset in data:
        return data[subset]
    raise ValueError(f"Could not find records in {path} for split={split_name} subset={subset}")


def load_success(root: Path, expert: str, language: str) -> dict[str, str]:
    path = root / expert / language / "_SUCCESS.json"
    data = load_json(path)
    return {
        "expert": expert,
        "json_key": data["json_key"],
        "probabilities_key": data["probabilities_key"],
        "uncertainty_key": data["uncertainty_key"],
    }


def index_by_uuid(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(item.get("uuid")): item for item in records if item.get("uuid") is not None}


def label_from_probs(probs: np.ndarray) -> int:
    return LABELS[int(np.argmax(probs))]


def probs_from_item(item: dict[str, Any], probabilities_key: str) -> np.ndarray:
    raw = item.get(probabilities_key, {}) or {}
    probs = np.array([float(raw.get(label, 0.0) or 0.0) for label in PROB_LABELS], dtype=np.float64)
    total = float(probs.sum())
    if total <= 0:
        return np.array([0.0, 1.0, 0.0], dtype=np.float64)
    return probs / total


def entropy(probs: np.ndarray) -> float:
    return float(-np.sum(probs * np.log2(probs + 1e-12)))


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def vectorize_item(
    uuid: str,
    by_expert: dict[str, dict[str, dict[str, Any]]],
    success_by_expert: dict[str, dict[str, str]],
    experts: list[str],
) -> dict[str, Any] | None:
    items = [by_expert[expert].get(uuid) for expert in experts]
    if any(item is None for item in items):
        return None
    first = items[0]
    if first is None or first.get("sentiment") not in LABELS:
        return None

    y = int(first["sentiment"])
    features: list[float] = []
    expert_preds: dict[str, int] = {}
    expert_probs: dict[str, list[float]] = {}
    expert_confidences: dict[str, float] = {}
    flattened_probs = []
    uncertainty_rows = []

    for expert, item in zip(experts, items):
        assert item is not None
        success = success_by_expert[expert]
        probs = probs_from_item(item, success["probabilities_key"])
        uncertainty = item.get(success["uncertainty_key"], {}) or {}
        pred = item.get(success["json_key"])
        pred = int(pred) if pred in LABELS else label_from_probs(probs)
        conf = safe_float(uncertainty.get("confidence_score"), float(np.max(probs)))
        vote_conf = safe_float(uncertainty.get("vote_confidence"), conf)
        variation = safe_float(uncertainty.get("variation_ratio"), 1.0 - vote_conf)
        pred_entropy = safe_float(uncertainty.get("predictive_entropy"), entropy(probs))
        expected_entropy = safe_float(uncertainty.get("expected_entropy"), pred_entropy)
        mutual_info = safe_float(uncertainty.get("mutual_information"), max(0.0, pred_entropy - expected_entropy))
        margin = float(np.sort(probs)[-1] - np.sort(probs)[-2])

        expert_preds[expert] = pred
        expert_probs[expert] = probs.tolist()
        expert_confidences[expert] = conf
        flattened_probs.append(probs)
        uncertainty_rows.append([conf, vote_conf, variation, pred_entropy, expected_entropy, mutual_info, margin])
        features.extend(probs.tolist())
        features.extend(uncertainty_rows[-1])
        features.extend([1.0 if pred == label else 0.0 for label in LABELS])

    prob_matrix = np.vstack(flattened_probs)
    mean_probs = prob_matrix.mean(axis=0)
    std_probs = prob_matrix.std(axis=0)
    pred_counts = Counter(expert_preds.values())
    vote_features = np.array([pred_counts.get(label, 0) / len(experts) for label in LABELS], dtype=np.float64)
    confs = np.array([row[0] for row in uncertainty_rows], dtype=np.float64)
    entropies = np.array([row[3] for row in uncertainty_rows], dtype=np.float64)
    mutual_infos = np.array([row[5] for row in uncertainty_rows], dtype=np.float64)

    features.extend(mean_probs.tolist())
    features.extend(std_probs.tolist())
    features.extend(vote_features.tolist())
    features.extend(
        [
            entropy(mean_probs),
            float(np.max(mean_probs)),
            float(np.sort(mean_probs)[-1] - np.sort(mean_probs)[-2]),
            float(len(pred_counts)),
            float(max(pred_counts.values()) / len(experts)),
            float(np.max(confs)),
            float(np.min(confs)),
            float(np.mean(confs)),
            float(np.max(confs) - np.min(confs)),
            float(np.min(entropies)),
            float(np.mean(entropies)),
            float(np.max(entropies)),
            float(np.max(entropies) - np.min(entropies)),
            float(np.mean(mutual_infos)),
            float(np.max(mutual_infos)),
        ]
    )

    return {
        "uuid": uuid,
        "y": y,
        "features": features,
        "expert_preds": expert_preds,
        "expert_probs": expert_probs,
        "expert_confidences": expert_confidences,
        "mean_probs": mean_probs.tolist(),
        "mean_confidence": float(np.max(mean_probs)),
        "num_unique_expert_predictions": len(pred_counts),
        "expert_vote_counts": {str(label): int(pred_counts.get(label, 0)) for label in LABELS},
    }


def build_dataset(
    root: Path,
    experts: list[str],
    language: str,
    split_name: str,
    subset: str | None,
    limit: int | None,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, str]]]:
    success_by_expert = {expert: load_success(root, expert, language) for expert in experts}
    by_expert = {
        expert: index_by_uuid(load_records(root, expert, language, split_name, subset))
        for expert in experts
    }
    common_uuids = sorted(set.intersection(*(set(rows) for rows in by_expert.values())))
    rows = [
        row
        for uuid in common_uuids
        if (row := vectorize_item(uuid, by_expert, success_by_expert, experts)) is not None
    ]
    if limit is not None and limit > 0 and limit < len(rows):
        rng = random.Random(seed)
        rng.shuffle(rows)
        rows = rows[:limit]
    return rows, success_by_expert


def rows_to_xy(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray([row["features"] for row in rows], dtype=np.float64),
        np.asarray([row["y"] for row in rows], dtype=np.int64),
    )


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, average="macro", zero_division=0
    )
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, average="weighted", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(p_macro),
        "recall_macro": float(r_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(p_weighted),
        "recall_weighted": float(r_weighted),
        "f1_weighted": float(f1_weighted),
        "qwk": float(cohen_kappa_score(y_true, y_pred, labels=LABELS, weights="quadratic")),
        "num_samples_evaluated": int(len(y_true)),
        "per_class_report": classification_report(
            y_true,
            y_pred,
            labels=LABELS,
            target_names=["negative", "neutral", "positive"],
            output_dict=True,
            zero_division=0,
        ),
    }


def expert_policy(rows: list[dict[str, Any]], expert: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    preds = np.asarray([row["expert_preds"][expert] for row in rows], dtype=np.int64)
    probs = np.asarray([row["expert_probs"][expert] for row in rows], dtype=np.float64)
    scores = np.asarray([row["expert_confidences"][expert] for row in rows], dtype=np.float64)
    return preds, probs, scores


def avg_probs_policy(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    probs = np.asarray([row["mean_probs"] for row in rows], dtype=np.float64)
    preds = np.asarray([label_from_probs(row) for row in probs], dtype=np.int64)
    scores = np.max(probs, axis=1)
    return preds, probs, scores


def majority_vote_policy(rows: list[dict[str, Any]], experts: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    preds = []
    probs = []
    scores = []
    for row in rows:
        counts = Counter(row["expert_preds"][expert] for expert in experts)
        max_count = max(counts.values())
        tied = [label for label, count in counts.items() if count == max_count]
        if len(tied) == 1:
            pred = tied[0]
        else:
            pred = max(
                tied,
                key=lambda label: sum(
                    row["expert_confidences"][expert]
                    for expert in experts
                    if row["expert_preds"][expert] == label
                ),
            )
        vote_probs = np.array([counts.get(label, 0) / len(experts) for label in LABELS], dtype=np.float64)
        preds.append(pred)
        probs.append(vote_probs)
        scores.append(float(max_count / len(experts)))
    return np.asarray(preds), np.asarray(probs), np.asarray(scores)


def confidence_pick_policy(rows: list[dict[str, Any]], experts: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    preds = []
    probs = []
    scores = []
    for row in rows:
        expert = max(experts, key=lambda name: row["expert_confidences"][name])
        preds.append(row["expert_preds"][expert])
        probs.append(row["expert_probs"][expert])
        scores.append(row["expert_confidences"][expert])
    return np.asarray(preds), np.asarray(probs), np.asarray(scores)


def model_policy(model: Any, rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, _ = rows_to_xy(rows)
    preds = model.predict(x)
    probs = model.predict_proba(x)
    class_to_col = {int(label): idx for idx, label in enumerate(model.classes_)}
    aligned = np.zeros((len(rows), len(LABELS)), dtype=np.float64)
    for out_idx, label in enumerate(LABELS):
        if label in class_to_col:
            aligned[:, out_idx] = probs[:, class_to_col[label]]
    scores = np.max(aligned, axis=1)
    return preds.astype(np.int64), aligned, scores


def policy_bundle(
    name: str,
    y_true: np.ndarray,
    preds: np.ndarray,
    probs: np.ndarray,
    scores: np.ndarray,
) -> dict[str, Any]:
    metrics = calculate_metrics(y_true, preds)
    metrics.update(
        {
            "policy": name,
            "mean_confidence": float(np.mean(scores)),
            "median_confidence": float(np.median(scores)),
            "min_confidence": float(np.min(scores)),
        }
    )
    return {"name": name, "preds": preds, "probs": probs, "scores": scores, "metrics": metrics}


def metric_value(metrics: dict[str, Any], metric: str) -> float:
    return float(metrics.get(metric, float("-inf")))


def tune_threshold(
    y_true: np.ndarray,
    base_preds: np.ndarray,
    target_preds: np.ndarray,
    scores: np.ndarray,
    metric: str,
) -> dict[str, Any]:
    candidates = sorted(set(np.quantile(scores, np.linspace(0.0, 1.0, 101)).tolist()))
    best: dict[str, Any] | None = None
    for threshold in candidates:
        defer_mask = scores <= threshold
        final_preds = np.where(defer_mask, target_preds, base_preds)
        metrics = calculate_metrics(y_true, final_preds)
        result = {
            "threshold": float(threshold),
            "defer_rate": float(np.mean(defer_mask)),
            "metrics": metrics,
        }
        if best is None or metric_value(metrics, metric) > metric_value(best["metrics"], metric):
            best = result
    assert best is not None
    return best


def apply_threshold_defer(
    y_true: np.ndarray,
    base_preds: np.ndarray,
    target_preds: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    defer_mask = scores <= threshold
    final_preds = np.where(defer_mask, target_preds, base_preds)
    base_correct = base_preds == y_true
    target_correct = target_preds == y_true
    return {
        "defer_rate": float(np.mean(defer_mask)),
        "num_deferred": int(np.sum(defer_mask)),
        "metrics": calculate_metrics(y_true, final_preds),
        "corrections": int(np.sum(defer_mask & ~base_correct & target_correct)),
        "degradations": int(np.sum(defer_mask & base_correct & ~target_correct)),
        "both_correct_when_deferred": int(np.sum(defer_mask & base_correct & target_correct)),
        "both_wrong_when_deferred": int(np.sum(defer_mask & ~base_correct & ~target_correct)),
    }


def curve_rows(
    y_true: np.ndarray,
    base_preds: np.ndarray,
    target_preds: np.ndarray,
    scores: np.ndarray,
    rates: list[float],
    base_name: str,
    target_name: str,
) -> list[dict[str, Any]]:
    output = []
    for rate in rates:
        if rate <= 0:
            threshold = float("-inf")
        elif rate >= 1:
            threshold = float("inf")
        else:
            threshold = float(np.quantile(scores, rate))
        result = apply_threshold_defer(y_true, base_preds, target_preds, scores, threshold)
        flat_metrics = {
            key: value
            for key, value in result["metrics"].items()
            if key != "per_class_report"
        }
        row = {
            "base_policy": base_name,
            "target_policy": target_name,
            "requested_defer_rate": rate,
            "threshold": threshold,
            **{k: v for k, v in result.items() if k != "metrics"},
            **flat_metrics,
        }
        output.append(row)
    return output


def random_oracle_curve(
    y_true: np.ndarray,
    base_preds: np.ndarray,
    rates: list[float],
    repeats: int,
    seed: int,
    base_name: str,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    rows = []
    n = len(y_true)
    for rate in rates:
        k = int(round(rate * n))
        metrics_at_rate = []
        for _ in range(repeats):
            mask = np.zeros(n, dtype=bool)
            if k > 0:
                mask[rng.choice(n, size=k, replace=False)] = True
            final_preds = np.where(mask, y_true, base_preds)
            metrics_at_rate.append(calculate_metrics(y_true, final_preds))
        rows.append(
            {
                "base_policy": base_name,
                "target_policy": "random_oracle_matched_rate",
                "requested_defer_rate": rate,
                "defer_rate": float(k / n),
                "num_deferred": k,
                "accuracy": float(np.mean([m["accuracy"] for m in metrics_at_rate])),
                "f1_macro": float(np.mean([m["f1_macro"] for m in metrics_at_rate])),
                "qwk": float(np.mean([m["qwk"] for m in metrics_at_rate])),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: to_jsonable(row.get(key)) for key in keys})


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# Multi-Expert Defer: {payload['language']} ({payload['setting']})",
        "",
        "## Experts",
        "",
        *[f"- `{expert}`" for expert in payload["experts"]],
        "",
        "## Test Policies",
        "",
        "| Policy | Macro-F1 | QWK | Accuracy | Mean confidence |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in payload["policy_metrics"]:
        lines.append(
            "| {policy} | {f1:.4f} | {qwk:.4f} | {acc:.4f} | {conf:.4f} |".format(
                policy=row["policy"],
                f1=row["f1_macro"],
                qwk=row["qwk"],
                acc=row["accuracy"],
                conf=row.get("mean_confidence", 0.0),
            )
        )
    lines.extend(["", "## Best Validation-Tuned Defer Gates", ""])
    lines.extend(["| Base | Target | Threshold | Test defer rate | Macro-F1 | QWK | Corrections | Degradations |"])
    lines.extend(["|---|---|---:|---:|---:|---:|---:|---:|"])
    for row in payload["best_defer_gates"]:
        test = row["test"]
        lines.append(
            "| {base} | {target} | {thr:.4f} | {rate:.3f} | {f1:.4f} | {qwk:.4f} | {corr} | {deg} |".format(
                base=row["base_policy"],
                target=row["target_policy"],
                thr=row["threshold"],
                rate=test["defer_rate"],
                f1=test["metrics"]["f1_macro"],
                qwk=test["metrics"]["qwk"],
                corr=test["corrections"],
                deg=test["degradations"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_rates(raw: str) -> list[float]:
    rates = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if chunk:
            rates.append(min(1.0, max(0.0, float(chunk))))
    return sorted(set(rates))


def main() -> None:
    args = parse_args()
    started_at = time.time()
    root = Path(args.uncertainty_root)
    experts = args.experts or DEFAULT_EXPERTS_BY_LANGUAGE[args.language]
    out_dir = Path(args.output_root) / args.setting / args.language / ("__".join(experts))
    metrics_path = out_dir / "metrics.json"
    if metrics_path.exists() and not args.force:
        print(f"Skipping existing run: {metrics_path}", flush=True)
        return

    print("--- Multi-Expert Learning-to-Defer ---", flush=True)
    print(f"Language: {args.language}", flush=True)
    print(f"Setting: {args.setting}", flush=True)
    print(f"Experts: {', '.join(experts)}", flush=True)
    print(f"Uncertainty root: {root}", flush=True)
    print(f"Output dir: {out_dir}", flush=True)

    train_rows, success_by_expert = build_dataset(
        root,
        experts,
        args.language,
        f"train_val_{args.split_index}",
        "train",
        args.max_train_items,
        args.seed,
    )
    val_rows, _ = build_dataset(
        root,
        experts,
        args.language,
        f"train_val_{args.split_index}",
        "val",
        args.max_val_items,
        args.seed + 1,
    )
    test_rows, _ = build_dataset(
        root,
        experts,
        args.language,
        "test",
        None,
        args.limit_test_items,
        args.seed + 2,
    )
    if not train_rows or not val_rows or not test_rows:
        raise RuntimeError("Prepared an empty train/val/test dataset.")

    x_train, y_train = rows_to_xy(train_rows)
    x_val, y_val = rows_to_xy(val_rows)
    _, y_test = rows_to_xy(test_rows)
    print(f"Rows: train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}", flush=True)

    lr = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=args.seed,
        ),
    )
    rf = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=4,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=args.seed,
    )
    lr.fit(x_train, y_train)
    rf.fit(x_train, y_train)

    def bundles_for(rows: list[dict[str, Any]], y: np.ndarray) -> dict[str, dict[str, Any]]:
        bundles = {}
        for expert in experts:
            pred, prob, score = expert_policy(rows, expert)
            bundles[expert] = policy_bundle(expert, y, pred, prob, score)
        for name, fn in [
            ("avg_probs", lambda: avg_probs_policy(rows)),
            ("majority_vote", lambda: majority_vote_policy(rows, experts)),
            ("confidence_pick", lambda: confidence_pick_policy(rows, experts)),
            ("lr_stacker", lambda: model_policy(lr, rows)),
            ("rf_stacker", lambda: model_policy(rf, rows)),
        ]:
            pred, prob, score = fn()
            bundles[name] = policy_bundle(name, y, pred, prob, score)
        return bundles

    val_bundles = bundles_for(val_rows, y_val)
    test_bundles = bundles_for(test_rows, y_test)
    policy_metrics = [bundle["metrics"] for bundle in test_bundles.values()]
    best_single = max(experts, key=lambda name: metric_value(val_bundles[name]["metrics"], args.metric))
    target_names = ["avg_probs", "majority_vote", "confidence_pick", "lr_stacker", "rf_stacker"]
    base_names = list(experts) + [best_single]
    base_names = list(dict.fromkeys(base_names))
    best_defer_gates = []
    curve_output = []
    rates = parse_rates(args.defer_rates)

    for base_name in base_names:
        base_val = val_bundles[base_name]
        base_test = test_bundles[base_name]
        for target_name in target_names + ["oracle"]:
            if target_name == base_name:
                continue
            val_target_preds = y_val if target_name == "oracle" else val_bundles[target_name]["preds"]
            test_target_preds = y_test if target_name == "oracle" else test_bundles[target_name]["preds"]
            tuned = tune_threshold(
                y_val,
                base_val["preds"],
                val_target_preds,
                base_val["scores"],
                args.metric,
            )
            test_result = apply_threshold_defer(
                y_test,
                base_test["preds"],
                test_target_preds,
                base_test["scores"],
                tuned["threshold"],
            )
            best_defer_gates.append(
                {
                    "base_policy": base_name,
                    "target_policy": target_name,
                    "selection_metric": args.metric,
                    "threshold": tuned["threshold"],
                    "val": tuned,
                    "test": test_result,
                }
            )
            curve_output.extend(
                curve_rows(
                    y_test,
                    base_test["preds"],
                    test_target_preds,
                    base_test["scores"],
                    rates,
                    base_name,
                    target_name,
                )
            )
        curve_output.extend(
            random_oracle_curve(
                y_test,
                base_test["preds"],
                rates,
                args.random_baseline_repeats,
                args.seed,
                base_name,
            )
        )

    best_defer_gates = sorted(
        best_defer_gates,
        key=lambda row: metric_value(row["test"]["metrics"], args.metric),
        reverse=True,
    )
    payload = {
        "language": args.language,
        "setting": args.setting,
        "experts": experts,
        "split_index": args.split_index,
        "selection_metric": args.metric,
        "num_train": len(train_rows),
        "num_val": len(val_rows),
        "num_test": len(test_rows),
        "success_by_expert": success_by_expert,
        "best_single_expert_on_val": best_single,
        "policy_metrics": sorted(policy_metrics, key=lambda row: metric_value(row, args.metric), reverse=True),
        "best_defer_gates": best_defer_gates,
        "elapsed_seconds": time.time() - started_at,
    }
    write_json(metrics_path, payload)
    write_csv(out_dir / "defer_curves.csv", curve_output)
    write_summary(out_dir / "summary.md", payload)
    print(f"Wrote {metrics_path}", flush=True)
    print(f"Wrote {out_dir / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()
