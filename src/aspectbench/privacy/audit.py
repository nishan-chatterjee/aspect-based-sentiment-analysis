"""Aggregate-only checks for accidental artifacts and membership signal."""

from __future__ import annotations

from collections.abc import Sequence
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def _wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> list[float]:
    if trials <= 0:
        return [float("nan"), float("nan")]
    proportion = successes / trials
    denominator = 1.0 + (z * z / trials)
    center = (proportion + z * z / (2.0 * trials)) / denominator
    radius = (
        z
        * np.sqrt(proportion * (1.0 - proportion) / trials + z * z / (4.0 * trials * trials))
        / denominator
    )
    return [float(max(0.0, center - radius)), float(min(1.0, center + radius))]


def _low_fpr_operating_points(
    members: np.ndarray,
    nonmembers: np.ndarray,
    targets: Sequence[float] = (0.001, 0.01, 0.05),
) -> dict[str, Any]:
    """Report empirical TPR at low FPR without pretending sub-resolution precision."""

    ordered = np.sort(nonmembers)[::-1]
    output: dict[str, Any] = {}
    for target in targets:
        allowed_false_positives = int(np.floor(target * len(nonmembers)))
        key = f"{100 * target:g}%"
        if allowed_false_positives < 1:
            output[key] = {
                "available": False,
                "target_fpr": target,
                "nonmember_n": len(nonmembers),
                "empirical_fpr_resolution": 1.0 / len(nonmembers),
                "reason": "The nonmember cohort is too small to resolve this false-positive rate.",
            }
            continue
        boundary_index = min(allowed_false_positives, len(ordered) - 1)
        high = ordered[allowed_false_positives - 1]
        low = ordered[boundary_index]
        threshold = float((high + low) / 2.0) if high != low else float(high)
        member_positive = members >= threshold
        nonmember_positive = nonmembers >= threshold
        true_positives = int(member_positive.sum())
        false_positives = int(nonmember_positive.sum())
        output[key] = {
            "available": True,
            "target_fpr": target,
            "threshold": threshold,
            "member_n": len(members),
            "nonmember_n": len(nonmembers),
            "true_positives": true_positives,
            "false_positives": false_positives,
            "tpr": float(true_positives / len(members)),
            "tpr_wilson_95_ci": _wilson_interval(true_positives, len(members)),
            "empirical_fpr": float(false_positives / len(nonmembers)),
            "empirical_fpr_resolution": 1.0 / len(nonmembers),
            "calibration_warning": (
                "Threshold and rate use the same finite nonmember cohort; confirm promising "
                "results with larger held-out controls or shadow/reference models."
            ),
        }
    return output


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_tensor_inventory(path: str | Path) -> dict[str, Any]:
    """Verify that a release checkpoint contains tensors rather than run payloads."""

    import torch

    checkpoint = Path(path)
    try:
        payload = torch.load(
            checkpoint, map_location="cpu", weights_only=True, mmap=True
        )
    except TypeError:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"Expected a non-empty state dictionary: {checkpoint}")
    non_tensors = {
        str(key): type(value).__name__
        for key, value in payload.items()
        if not torch.is_tensor(value)
    }
    tensors = [value for value in payload.values() if torch.is_tensor(value)]
    return {
        "path": str(checkpoint),
        "size_bytes": checkpoint.stat().st_size,
        "sha256": _sha256(checkpoint),
        "tensor_count": len(tensors),
        "parameter_count": int(sum(value.numel() for value in tensors)),
        "dtypes": sorted({str(value.dtype) for value in tensors}),
        "non_tensor_entries": non_tensors,
        "tensor_only": not non_tensors and len(tensors) == len(payload),
    }


def membership_attack_report(
    member_scores: Sequence[float],
    nonmember_scores: Sequence[float],
    *,
    seed: int = 42,
    bootstrap_samples: int = 1000,
    permutation_samples: int = 1000,
) -> dict[str, Any]:
    """Evaluate a fixed high-score-means-member black-box attack.

    The bootstrap interval measures sampling uncertainty. The permutation test
    asks whether the observed AUC exceeds chance under exchangeability; neither
    statistic proves that a model is safe against stronger attacks.
    """

    members = np.asarray(member_scores, dtype=np.float64)
    nonmembers = np.asarray(nonmember_scores, dtype=np.float64)
    if not len(members) or not len(nonmembers):
        raise ValueError("Both member and nonmember score arrays must be non-empty.")
    if not np.isfinite(members).all() or not np.isfinite(nonmembers).all():
        raise ValueError("Membership scores must all be finite.")
    labels = np.concatenate(
        [np.ones(len(members), dtype=np.int8), np.zeros(len(nonmembers), dtype=np.int8)]
    )
    scores = np.concatenate([members, nonmembers])
    auc = float(roc_auc_score(labels, scores))
    false_positive, true_positive, thresholds = roc_curve(labels, scores)
    advantages = true_positive - false_positive
    best = int(np.argmax(advantages))
    rng = np.random.default_rng(seed)
    bootstraps = []
    for _ in range(bootstrap_samples):
        member_sample = rng.choice(members, len(members), replace=True)
        nonmember_sample = rng.choice(nonmembers, len(nonmembers), replace=True)
        bootstrap_labels = np.concatenate(
            [np.ones(len(members)), np.zeros(len(nonmembers))]
        )
        bootstraps.append(
            roc_auc_score(
                bootstrap_labels,
                np.concatenate([member_sample, nonmember_sample]),
            )
        )
    permuted_at_least_observed = 0
    observed_distance = abs(auc - 0.5)
    permuted_at_least_as_extreme = 0
    for _ in range(permutation_samples):
        permuted = rng.permutation(labels)
        permuted_auc = roc_auc_score(permuted, scores)
        permuted_at_least_observed += int(permuted_auc >= auc)
        permuted_at_least_as_extreme += int(abs(permuted_auc - 0.5) >= observed_distance)
    return {
        "member_n": len(members),
        "nonmember_n": len(nonmembers),
        "auc": auc,
        "auc_bootstrap_95_ci": [
            float(np.quantile(bootstraps, 0.025)),
            float(np.quantile(bootstraps, 0.975)),
        ],
        "permutation_p_one_sided": (permuted_at_least_observed + 1)
        / (permutation_samples + 1),
        "permutation_p_two_sided": (permuted_at_least_as_extreme + 1)
        / (permutation_samples + 1),
        "maximum_tpr_minus_fpr": float(advantages[best]),
        "threshold_at_maximum_advantage": float(thresholds[best]),
        "low_fpr_operating_points": _low_fpr_operating_points(members, nonmembers),
        "member_mean": float(members.mean()),
        "nonmember_mean": float(nonmembers.mean()),
        "interpretation": (
            "Exploratory signal only. AUC near 0.5 is reassuring for this attack, "
            "but does not establish absence of memorization or resistance to other attacks."
        ),
    }
