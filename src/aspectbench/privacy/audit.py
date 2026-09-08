"""Aggregate-only checks for accidental artifacts and membership signal."""

from __future__ import annotations

from collections.abc import Sequence
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


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
    for _ in range(permutation_samples):
        permuted = rng.permutation(labels)
        permuted_at_least_observed += int(roc_auc_score(permuted, scores) >= auc)
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
        "maximum_tpr_minus_fpr": float(advantages[best]),
        "threshold_at_maximum_advantage": float(thresholds[best]),
        "member_mean": float(members.mean()),
        "nonmember_mean": float(nonmembers.mean()),
        "interpretation": (
            "Exploratory signal only. AUC near 0.5 is reassuring for this attack, "
            "but does not establish absence of memorization or resistance to other attacks."
        ),
    }
