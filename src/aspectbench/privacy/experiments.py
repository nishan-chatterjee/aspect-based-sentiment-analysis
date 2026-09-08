"""Aggregate-only privacy and memorization experiments for AspectBench.

The helpers in this module intentionally never serialize source text, entity
names, UUIDs, or per-example scores.  They quantify signals that can motivate
additional review; they do not provide a formal privacy guarantee.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import unicodedata
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import NearestNeighbors

from .audit import membership_attack_report


ASPECT_TAG_RE = re.compile(r"<aspect>.*?</aspect>", flags=re.IGNORECASE | re.DOTALL)
TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)
LABELS = (-1, 0, 1)
CLASS_KEYS = {
    -1: "-1 (negative)",
    0: "0 (neutral)",
    1: "1 (positive)",
}


def normalize_private_text(value: Any) -> str:
    """Normalize private text for comparison without returning it in reports."""

    value = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return " ".join(value.split())


def private_digest(*values: Any) -> str:
    material = "\0".join(normalize_private_text(value) for value in values)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _safe_rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def _quantiles(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {}
    return {
        "mean": float(array.mean()),
        "p50": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "maximum": float(array.max()),
    }


def _hash_set(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> set[str]:
    return {private_digest(*(row.get(field, "") for field in fields)) for row in rows}


def _duplicate_summary(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> dict[str, Any]:
    values = [private_digest(*(row.get(field, "") for field in fields)) for row in rows]
    counts = Counter(values)
    duplicate_rows = sum(count for count in counts.values() if count > 1)
    return {
        "rows": len(values),
        "unique": len(counts),
        "duplicate_rows": duplicate_rows,
        "duplicate_row_fraction": _safe_rate(duplicate_rows, len(values)),
        "largest_duplicate_group": max(counts.values(), default=0),
    }


def exact_overlap_report(
    train: Sequence[Mapping[str, Any]],
    validation: Sequence[Mapping[str, Any]],
    test: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Report hashed exact overlaps and within-cohort duplicates."""

    cohorts = {"train": train, "validation": validation, "test": test}
    definitions = {
        "uuid": ("uuid",),
        "normalized_article": ("article",),
        "normalized_article_and_aspect": ("article", "aspect"),
    }
    report: dict[str, Any] = {"within_cohort": {}, "cross_cohort": {}}
    for name, rows in cohorts.items():
        report["within_cohort"][name] = {
            key: _duplicate_summary(rows, fields)
            for key, fields in definitions.items()
        }
    for left_name, right_name in (("train", "validation"), ("train", "test"), ("validation", "test")):
        left, right = cohorts[left_name], cohorts[right_name]
        pair_key = f"{left_name}_vs_{right_name}"
        report["cross_cohort"][pair_key] = {}
        for definition, fields in definitions.items():
            left_hashes = _hash_set(left, fields)
            right_values = [
                private_digest(*(row.get(field, "") for field in fields)) for row in right
            ]
            overlap_rows = sum(value in left_hashes for value in right_values)
            report["cross_cohort"][pair_key][definition] = {
                "right_rows_overlapping_left": int(overlap_rows),
                "right_fraction_overlapping_left": _safe_rate(overlap_rows, len(right_values)),
                "unique_overlap_count": len(left_hashes & set(right_values)),
            }
    return report


def _sample_rows(rows: Sequence[Mapping[str, Any]], limit: int, rng: np.random.Generator):
    if limit <= 0 or len(rows) <= limit:
        return list(rows)
    indices = rng.choice(len(rows), limit, replace=False)
    return [rows[int(index)] for index in indices]


def near_duplicate_report(
    train: Sequence[Mapping[str, Any]],
    evaluation: Sequence[Mapping[str, Any]],
    *,
    seed: int = 42,
    train_limit: int = 5000,
    evaluation_limit: int = 2000,
    max_features: int = 60000,
) -> dict[str, Any]:
    """Estimate nearest train-document similarity using character TF-IDF.

    The report contains similarity distributions only.  No nearest-neighbour
    identities or text are retained.
    """

    rng = np.random.default_rng(seed)
    train_sample = _sample_rows(train, train_limit, rng)
    evaluation_sample = _sample_rows(evaluation, evaluation_limit, rng)
    train_text = [normalize_private_text(row.get("article")) for row in train_sample]
    eval_text = [normalize_private_text(row.get("article")) for row in evaluation_sample]
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=max_features,
        sublinear_tf=True,
        dtype=np.float32,
    )
    train_matrix = vectorizer.fit_transform(train_text)
    eval_matrix = vectorizer.transform(eval_text)
    neighbors = NearestNeighbors(n_neighbors=1, metric="cosine", algorithm="brute")
    neighbors.fit(train_matrix)
    distances, _ = neighbors.kneighbors(eval_matrix, return_distance=True)
    similarities = np.clip(1.0 - distances[:, 0], 0.0, 1.0)
    return {
        "train_sample_n": len(train_sample),
        "evaluation_sample_n": len(evaluation_sample),
        "feature_count": int(train_matrix.shape[1]),
        "nearest_train_similarity": _quantiles(similarities),
        "fraction_at_or_above": {
            str(threshold): float(np.mean(similarities >= threshold))
            for threshold in (0.80, 0.90, 0.95, 0.99)
        },
        "method": "character TF-IDF (3-5 grams), sampled nearest train document",
    }


def _js_divergence(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left = left / max(left.sum(), 1.0)
    right = right / max(right.sum(), 1.0)
    middle = (left + right) / 2.0
    terms = []
    for distribution in (left, right):
        mask = distribution > 0
        terms.append(float(np.sum(distribution[mask] * np.log(distribution[mask] / middle[mask]))))
    return (terms[0] + terms[1]) / 2.0


def _aspect_label_baseline(
    train: Sequence[Mapping[str, Any]], evaluation: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    global_counts = Counter(int(row["sentiment"]) for row in train)
    aspect_counts: dict[str, Counter[int]] = defaultdict(Counter)
    for row in train:
        aspect_counts[normalize_private_text(row.get("aspect"))][int(row["sentiment"])] += 1
    fallback = max(LABELS, key=lambda label: global_counts[label])
    gold, predicted, seen = [], [], []
    for row in evaluation:
        aspect = normalize_private_text(row.get("aspect"))
        counts = aspect_counts.get(aspect)
        prediction = max(LABELS, key=lambda label: counts[label]) if counts else fallback
        gold.append(int(row["sentiment"]))
        predicted.append(prediction)
        seen.append(bool(counts))
    return {
        "n": len(gold),
        "seen_aspect_fraction": float(np.mean(seen)) if seen else float("nan"),
        "accuracy": float(accuracy_score(gold, predicted)),
        "macro_f1": float(f1_score(gold, predicted, labels=list(LABELS), average="macro", zero_division=0)),
        "interpretation": "Dataset property: train aspect-majority label baseline, not a model attack.",
    }


def aspect_distribution_report(
    train: Sequence[Mapping[str, Any]],
    validation: Sequence[Mapping[str, Any]],
    test: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize aspect support, drift, and aspect-only label predictiveness."""

    cohorts = {"train": train, "validation": validation, "test": test}
    counters = {
        name: Counter(normalize_private_text(row.get("aspect")) for row in rows)
        for name, rows in cohorts.items()
    }
    vocabulary = sorted(set().union(*(counter for counter in counters.values())))
    vectors = {
        name: np.asarray([counter[value] for value in vocabulary], dtype=np.float64)
        for name, counter in counters.items()
    }
    train_aspects = set(counters["train"])
    support = {}
    for name, rows in cohorts.items():
        seen_rows = sum(normalize_private_text(row.get("aspect")) in train_aspects for row in rows)
        support[name] = {
            "rows": len(rows),
            "unique_aspects": len(counters[name]),
            "rows_with_train_seen_aspect": int(seen_rows),
            "train_seen_fraction": _safe_rate(seen_rows, len(rows)),
        }
    return {
        "support": support,
        "jensen_shannon_divergence_nats": {
            "train_vs_validation": _js_divergence(vectors["train"], vectors["validation"]),
            "train_vs_test": _js_divergence(vectors["train"], vectors["test"]),
        },
        "aspect_only_label_baseline": {
            "validation": _aspect_label_baseline(train, validation),
            "test": _aspect_label_baseline(train, test),
        },
    }


def distribution_classifier_report(
    members: Sequence[Mapping[str, Any]],
    nonmembers: Sequence[Mapping[str, Any]],
    *,
    seed: int = 42,
    repeats: int = 5,
    max_features: int = 50000,
) -> dict[str, Any]:
    """Measure observable train/nonmember text distribution shift.

    A high AUC warns that membership attacks may be exploiting split drift
    rather than parameter memorization.
    """

    rows = list(members) + list(nonmembers)
    text = [normalize_private_text(row.get("article")) for row in rows]
    labels = np.concatenate([np.ones(len(members)), np.zeros(len(nonmembers))])
    splitter = StratifiedShuffleSplit(n_splits=repeats, test_size=0.30, random_state=seed)
    aucs, feature_counts = [], []
    placeholder = np.zeros((len(labels), 1), dtype=np.int8)
    for train_index, test_index in splitter.split(placeholder, labels):
        vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), min_df=2,
            max_features=max_features, sublinear_tf=True, dtype=np.float32,
        )
        train_features = vectorizer.fit_transform([text[int(index)] for index in train_index])
        test_features = vectorizer.transform([text[int(index)] for index in test_index])
        classifier = LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=500, solver="liblinear",
            random_state=seed,
        )
        classifier.fit(train_features, labels[train_index])
        probability = classifier.predict_proba(test_features)[:, 1]
        auc = float(roc_auc_score(labels[test_index], probability))
        aucs.append(auc)
        feature_counts.append(int(train_features.shape[1]))
    return {
        "member_n": len(members),
        "nonmember_n": len(nonmembers),
        "feature_count": _quantiles(feature_counts),
        "held_out_auc": _quantiles(aucs),
        # Retained for report compatibility. The classifier has a predefined
        # member label, so fold-wise AUCs are no longer reflected around 0.5.
        "absolute_direction_auc": _quantiles(aucs),
        "interpretation": (
            "High AUC indicates cohort distribution shift. It is a confounder for "
            "membership inference and is not evidence of model memorization by itself."
        ),
    }


def _digest_text(value: str) -> bytes:
    return hashlib.blake2b(value.encode("utf-8"), digest_size=16).digest()


def _tokens(value: Any) -> list[str]:
    return TOKEN_RE.findall(normalize_private_text(value))


def _shingle_digests(tokens: Sequence[str], width: int) -> set[bytes]:
    if len(tokens) < width:
        return set()
    return {
        _digest_text(" ".join(tokens[index : index + width]))
        for index in range(len(tokens) - width + 1)
    }


def tracked_release_paths(repository_root: str | Path) -> list[Path]:
    """Return Git-tracked text-like release files, excluding private/runtime trees."""

    root = Path(repository_root).resolve()
    result = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"],
        check=True,
        stdout=subprocess.PIPE,
    )
    excluded_roots = {"data", "models", "outputs"}
    excluded_prefixes = {("huggingface", "models"), ("huggingface", "validation-runs")}
    suffixes = {
        ".cff", ".csv", ".ipynb", ".jinja", ".json", ".jsonl", ".md",
        ".py", ".sh", ".tex", ".toml", ".tsv", ".txt", ".yaml", ".yml",
    }
    paths = []
    for raw in result.stdout.decode("utf-8").split("\0"):
        if not raw:
            continue
        relative = Path(raw)
        if relative.parts[0] in excluded_roots:
            continue
        if any(relative.parts[: len(prefix)] == prefix for prefix in excluded_prefixes):
            continue
        path = root / relative
        if path.is_file() and path.suffix.lower() in suffixes:
            paths.append(path)
    return paths


def release_surface_overlap_report(
    rows: Sequence[Mapping[str, Any]],
    *,
    repository_root: str | Path,
    candidate_paths: Sequence[str | Path] | None = None,
    shingle_tokens: int = 24,
    maximum_file_bytes: int = 64 * 1024 * 1024,
) -> dict[str, Any]:
    """Check publishable text files for corpus-derived material.

    The report contains counts and relative file paths only. It never emits a
    source record, aspect name, matching shingle, or digest.
    """

    if shingle_tokens < 8:
        raise ValueError("shingle_tokens must be at least 8 to limit incidental matches.")
    root = Path(repository_root).resolve()
    candidates = (
        tracked_release_paths(root) if candidate_paths is None
        else [Path(path).resolve() for path in candidate_paths]
    )
    article_hashes: set[bytes] = set()
    article_shingles: set[bytes] = set()
    aspect_hashes_by_length: dict[int, set[bytes]] = defaultdict(set)
    for row in rows:
        article = normalize_private_text(row.get("article"))
        article_hash = _digest_text(article)
        if article and article_hash not in article_hashes:
            article_hashes.add(article_hash)
            article_shingles.update(_shingle_digests(_tokens(article), shingle_tokens))
        aspect = normalize_private_text(row.get("aspect"))
        aspect_tokens = _tokens(aspect)
        if aspect and aspect_tokens:
            aspect_hashes_by_length[len(aspect_tokens)].add(_digest_text(" ".join(aspect_tokens)))

    matches, skipped = [], []
    for path in candidates:
        try:
            relative = str(path.relative_to(root))
        except ValueError:
            relative = path.name
        try:
            size = path.stat().st_size
            if size > maximum_file_bytes:
                skipped.append({"path": relative, "reason": "over_size_limit", "size_bytes": size})
                continue
            content = path.read_text(encoding="utf-8", errors="ignore")
        except OSError as error:
            skipped.append({"path": relative, "reason": type(error).__name__})
            continue
        normalized = normalize_private_text(content)
        candidate_tokens = _tokens(normalized)
        leaf_values = [normalize_private_text(line) for line in content.splitlines() if line.strip()]
        if path.suffix.lower() in {".json", ".ipynb"}:
            try:
                payload = json.loads(content)
                stack = [payload]
                leaf_values = []
                while stack:
                    value = stack.pop()
                    if isinstance(value, Mapping):
                        stack.extend(value.values())
                    elif isinstance(value, list):
                        stack.extend(value)
                    elif isinstance(value, str):
                        leaf_values.append(normalize_private_text(value))
            except ValueError:
                pass
        exact_articles = {
            digest for value in leaf_values
            if value and (digest := _digest_text(value)) in article_hashes
        }
        shingle_matches = _shingle_digests(candidate_tokens, shingle_tokens) & article_shingles
        aspect_matches: set[bytes] = set()
        for length, private_hashes in aspect_hashes_by_length.items():
            if length > len(candidate_tokens):
                continue
            # One-token aspects are only counted as exact structured/line
            # values; otherwise common nouns create misleading matches.
            if length == 1:
                aspect_matches.update(
                    digest for value in leaf_values
                    if value and len(_tokens(value)) == 1
                    and (digest := _digest_text(value)) in private_hashes
                )
                continue
            for index in range(len(candidate_tokens) - length + 1):
                digest = _digest_text(" ".join(candidate_tokens[index : index + length]))
                if digest in private_hashes:
                    aspect_matches.add(digest)
        if exact_articles or shingle_matches or aspect_matches:
            matches.append(
                {
                    "path": relative,
                    "exact_article_value_count": len(exact_articles),
                    "distinct_long_shingle_match_count": len(shingle_matches),
                    "distinct_aspect_match_count": len(aspect_matches),
                }
            )
    return {
        "candidate_file_count": len(candidates),
        "scanned_file_count": len(candidates) - len(skipped),
        "skipped_files": skipped,
        "private_unique_article_count": len(article_hashes),
        "private_unique_aspect_count": sum(len(values) for values in aspect_hashes_by_length.values()),
        "shingle_tokens": shingle_tokens,
        "files_with_matches": matches,
        "contains_source_text_or_names": False,
        "interpretation": (
            "Any match in a tracked file requires review. Long shingles indicate possible verbatim "
            "corpus excerpts; aspect matches can also be legitimate public entity mentions."
        ),
    }


def safe_prediction_scores(
    outputs: Sequence[Mapping[str, Any]],
    source_rows: Sequence[Mapping[str, Any]],
    train_aspect_counts: Mapping[str, int],
) -> list[dict[str, Any]]:
    """Convert predictions to ephemeral numeric attack features."""

    safe = []
    for output, source in zip(outputs, source_rows, strict=True):
        probabilities = np.asarray(
            [output["class_probabilities"][CLASS_KEYS[label]] for label in LABELS],
            dtype=np.float64,
        )
        probabilities = np.clip(probabilities, 1e-12, 1.0)
        probabilities /= probabilities.sum()
        gold_index = int(source["sentiment"]) + 1
        ordered = np.sort(probabilities)
        aspect_count = int(train_aspect_counts.get(normalize_private_text(source.get("aspect")), 0))
        mc = output.get("uncertainty_across_classes", {}).get("mc_dropout", {})
        safe.append(
            {
                "gold_log_probability": float(np.log(probabilities[gold_index])),
                "loss_score": float(np.log(probabilities[gold_index])),
                "confidence": float(probabilities.max()),
                "negative_entropy": float(np.sum(probabilities * np.log(probabilities))),
                "margin": float(ordered[-1] - ordered[-2]),
                "correct": int(probabilities.argmax() == gold_index),
                "label": int(source["sentiment"]),
                "length_log": float(np.log1p(len(normalize_private_text(source.get("article"))))),
                "aspect_frequency_log": float(np.log1p(aspect_count)),
                "aspect_seen": int(aspect_count > 0),
                "negative_mc_mutual_information": -float(mc.get("mutual_information_bits", 0.0)),
                "mc_prediction_agreement": float(mc.get("prediction_agreement", 1.0)),
                "probabilities": probabilities.tolist(),
            }
        )
    return safe


def _combined_attack(
    member_rows: Sequence[Mapping[str, Any]],
    nonmember_rows: Sequence[Mapping[str, Any]],
    *, seed: int, include_covariates: bool = False,
) -> dict[str, Any]:
    feature_names = ["gold_log_probability", "confidence", "negative_entropy", "margin"]
    for optional in ("negative_mc_mutual_information", "mc_prediction_agreement"):
        if all(optional in row for row in list(member_rows) + list(nonmember_rows)):
            feature_names.append(optional)
    if include_covariates:
        feature_names.extend(["length_log", "aspect_frequency_log", "aspect_seen"])
    rows = list(member_rows) + list(nonmember_rows)
    features = np.asarray([[float(row[name]) for name in feature_names] for row in rows])
    labels = np.concatenate([np.ones(len(member_rows)), np.zeros(len(nonmember_rows))])
    splitter = StratifiedShuffleSplit(n_splits=10, test_size=0.30, random_state=seed)
    aucs = []
    for train_index, test_index in splitter.split(features, labels):
        train_features = features[train_index]
        mean = train_features.mean(axis=0)
        scale = train_features.std(axis=0)
        scale[scale == 0] = 1.0
        classifier = LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=500, solver="liblinear",
            random_state=seed,
        )
        classifier.fit((train_features - mean) / scale, labels[train_index])
        probability = classifier.predict_proba((features[test_index] - mean) / scale)[:, 1]
        aucs.append(float(roc_auc_score(labels[test_index], probability)))
    return {
        "features": list(feature_names),
        "held_out_auc": _quantiles(aucs),
        "repeats": len(aucs),
    }


def _subgroup_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    output = {}
    for seen_value, name in ((1, "seen_aspect"), (0, "unseen_aspect")):
        group = [row for row in rows if int(row["aspect_seen"]) == seen_value]
        if not group:
            output[name] = {"n": 0}
            continue
        output[name] = {
            "n": len(group),
            "accuracy": float(np.mean([row["correct"] for row in group])),
            "confidence_mean": float(np.mean([row["confidence"] for row in group])),
            "gold_log_probability_mean": float(np.mean([row["gold_log_probability"] for row in group])),
        }
    return output


def _covariate_match(
    member_rows: Sequence[Mapping[str, Any]],
    nonmember_rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    """Frequency-match cohorts on label, length, and train aspect support."""

    combined = list(member_rows) + list(nonmember_rows)
    lengths = np.asarray([float(row["length_log"]) for row in combined])
    length_edges = np.unique(np.quantile(lengths, [0.25, 0.50, 0.75]))

    def frequency_bucket(value: float) -> int:
        count = max(0, int(round(np.expm1(value))))
        if count == 0:
            return 0
        if count == 1:
            return 1
        if count <= 4:
            return 2
        return 3

    def key(row: Mapping[str, Any]) -> tuple[int, int, int]:
        return (
            int(row["label"]),
            int(np.digitize(float(row["length_log"]), length_edges)),
            frequency_bucket(float(row["aspect_frequency_log"])),
        )

    member_groups: dict[tuple[int, int, int], list[Mapping[str, Any]]] = defaultdict(list)
    nonmember_groups: dict[tuple[int, int, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in member_rows:
        member_groups[key(row)].append(row)
    for row in nonmember_rows:
        nonmember_groups[key(row)].append(row)
    rng = np.random.default_rng(seed)
    matched_members, matched_nonmembers = [], []
    for group_key in sorted(set(member_groups) & set(nonmember_groups)):
        left, right = member_groups[group_key], nonmember_groups[group_key]
        count = min(len(left), len(right))
        matched_members.extend(left[int(index)] for index in rng.choice(len(left), count, replace=False))
        matched_nonmembers.extend(right[int(index)] for index in rng.choice(len(right), count, replace=False))
    return matched_members, matched_nonmembers


def model_membership_report(
    member_rows: Sequence[Mapping[str, Any]],
    nonmember_rows: Sequence[Mapping[str, Any]],
    *,
    seed: int = 42,
    bootstrap_samples: int = 1000,
    permutation_samples: int = 1000,
) -> dict[str, Any]:
    attacks = {}
    attack_features = ["gold_log_probability", "confidence", "negative_entropy", "margin"]
    for optional in ("negative_mc_mutual_information", "mc_prediction_agreement"):
        if all(optional in row for row in list(member_rows) + list(nonmember_rows)):
            attack_features.append(optional)
    for feature in attack_features:
        member_values = [float(row[feature]) for row in member_rows]
        nonmember_values = [float(row[feature]) for row in nonmember_rows]
        forward = membership_attack_report(
            member_values,
            nonmember_values,
            seed=seed,
            bootstrap_samples=bootstrap_samples,
            permutation_samples=permutation_samples,
        )
        if forward["auc"] < 0.5:
            selected = membership_attack_report(
                [-value for value in member_values],
                [-value for value in nonmember_values],
                seed=seed,
                bootstrap_samples=bootstrap_samples,
                permutation_samples=permutation_samples,
            )
            selected["score_direction"] = "lower_original_score_means_member"
            selected["raw_forward_auc"] = forward["auc"]
        else:
            selected = forward
            selected["score_direction"] = "higher_original_score_means_member"
            selected["raw_forward_auc"] = forward["auc"]
        attacks[feature] = selected
    output = {
        "univariate_attacks": attacks,
        "combined_attack": _combined_attack(member_rows, nonmember_rows, seed=seed),
        "combined_attack_with_observable_covariates": _combined_attack(
            member_rows, nonmember_rows, seed=seed, include_covariates=True
        ),
        "member_accuracy": float(np.mean([row["correct"] for row in member_rows])),
        "nonmember_accuracy": float(np.mean([row["correct"] for row in nonmember_rows])),
        "nonmember_seen_unseen": _subgroup_summary(nonmember_rows),
    }
    matched_members, matched_nonmembers = _covariate_match(
        member_rows, nonmember_rows, seed=seed
    )
    if min(len(matched_members), len(matched_nonmembers)) >= 30:
        output["covariate_matched_attack"] = {
            "member_n": len(matched_members),
            "nonmember_n": len(matched_nonmembers),
            "matched_on": ["sentiment_label", "article_length_quartile", "train_aspect_frequency_bucket"],
            "combined_attack": _combined_attack(matched_members, matched_nonmembers, seed=seed + 17),
            "univariate_attacks": {
                feature: membership_attack_report(
                    [float(row[feature]) for row in matched_members],
                    [float(row[feature]) for row in matched_nonmembers],
                    seed=seed + 17,
                    bootstrap_samples=bootstrap_samples,
                    permutation_samples=permutation_samples,
                )
                for feature in attack_features
            },
        }
    else:
        output["covariate_matched_attack"] = {
            "available": False,
            "member_n": len(matched_members),
            "nonmember_n": len(matched_nonmembers),
            "reason": "Fewer than 30 matched examples per cohort.",
        }
    return output


def probabilities_from_outputs(outputs: Sequence[Mapping[str, Any]]) -> np.ndarray:
    return np.asarray(
        [
            [output["class_probabilities"][CLASS_KEYS[label]] for label in LABELS]
            for output in outputs
        ],
        dtype=np.float64,
    )


def counterfactual_aggregate(
    original_outputs: Sequence[Mapping[str, Any]],
    changed_outputs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    original = probabilities_from_outputs(original_outputs)
    changed = probabilities_from_outputs(changed_outputs)
    middle = (original + changed) / 2.0
    original_safe = np.clip(original, 1e-12, 1.0)
    changed_safe = np.clip(changed, 1e-12, 1.0)
    middle_safe = np.clip(middle, 1e-12, 1.0)
    divergences = 0.5 * (
        np.sum(original_safe * np.log(original_safe / middle_safe), axis=1)
        + np.sum(changed_safe * np.log(changed_safe / middle_safe), axis=1)
    )
    flips = np.argmax(original, axis=1) != np.argmax(changed, axis=1)
    return {
        "n": len(original),
        "prediction_flip_rate": float(np.mean(flips)),
        "js_divergence_nats": _quantiles(divergences),
        "mean_absolute_probability_change": float(np.mean(np.abs(original - changed))),
    }


def pseudonymized_rows(
    rows: Sequence[Mapping[str, Any]], *, language: str
) -> list[dict[str, Any]]:
    pseudonym = "Sintetični Cilj" if language == "sl" else "Sintetička Meta"
    output = []
    for row in rows:
        clone = dict(row)
        clone["article"] = ASPECT_TAG_RE.sub(f"<aspect>{pseudonym}</aspect>", str(row["article"]))
        clone["aspect"] = pseudonym
        output.append(clone)
    return output


def permuted_aspect_rows(
    rows: Sequence[Mapping[str, Any]], *, seed: int = 42
) -> list[dict[str, Any]]:
    if len(rows) < 2:
        return [dict(row) for row in rows]
    rng = np.random.default_rng(seed)
    aspects = [str(row.get("aspect", "")) for row in rows]
    order = np.roll(rng.permutation(len(rows)), 1)
    output = []
    for index, row in enumerate(rows):
        replacement = aspects[int(order[index])]
        if normalize_private_text(replacement) == normalize_private_text(row.get("aspect")):
            replacement = aspects[(int(order[index]) + 1) % len(aspects)]
        clone = dict(row)
        clone["article"] = ASPECT_TAG_RE.sub(f"<aspect>{replacement}</aspect>", str(row["article"]))
        clone["aspect"] = replacement
        output.append(clone)
    return output


def neutral_context_rows(
    aspects: Iterable[str], *, language: str
) -> list[dict[str, Any]]:
    template = (
        "V tem nevtralnem preizkusu je omenjen <aspect>{}</aspect>."
        if language == "sl"
        else "U ovom neutralnom testu pominje se <aspect>{}</aspect>."
    )
    return [
        {"article": template.format(aspect), "aspect": aspect, "sentiment": 0}
        for aspect in aspects
    ]


def neutral_probe_aggregate(outputs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    probabilities = probabilities_from_outputs(outputs)
    predicted = np.argmax(probabilities, axis=1)
    return {
        "n": len(outputs),
        "mean_class_probabilities": {
            str(label): float(probabilities[:, index].mean())
            for index, label in enumerate(LABELS)
        },
        "predicted_class_fractions": {
            str(label): float(np.mean(predicted == index))
            for index, label in enumerate(LABELS)
        },
        "mean_confidence": float(np.max(probabilities, axis=1).mean()),
        "interpretation": "OOD property probe; skew can reflect corpus priors and is not verbatim extraction.",
    }


def finite_json(value: Any) -> Any:
    """Recursively replace non-finite floats before strict JSON serialization."""

    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(item) for item in value]
    if isinstance(value, np.generic):
        return finite_json(value.item())
    return value
