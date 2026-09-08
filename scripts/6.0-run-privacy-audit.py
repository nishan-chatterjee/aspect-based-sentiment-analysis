#!/usr/bin/env python3
"""Run one resumable dataset/variant privacy audit on one GPU."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aspectbench.inference.hf_bridge import (  # noqa: E402
    checkpoint_status,
    create_engine,
    load_release_modules,
    release_coordinates,
)
from aspectbench.privacy import checkpoint_tensor_inventory  # noqa: E402
from aspectbench.privacy.experiments import (  # noqa: E402
    aspect_distribution_report,
    counterfactual_aggregate,
    distribution_classifier_report,
    exact_overlap_report,
    finite_json,
    model_membership_report,
    near_duplicate_report,
    neutral_context_rows,
    neutral_probe_aggregate,
    normalize_private_text,
    permuted_aspect_rows,
    pseudonymized_rows,
    release_surface_overlap_report,
    safe_prediction_scores,
)
from aspectbench.registry import select_models  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate-only model privacy and memorization audit."
    )
    parser.add_argument("--dataset", required=True, choices=("hbs", "sl"))
    parser.add_argument("--variant", required=True, choices=("masked", "unmasked"))
    parser.add_argument("--models", nargs="+", default=["all"])
    parser.add_argument("--repository-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--model-root", type=Path, default=Path("huggingface/models"))
    parser.add_argument("--base-model-root", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/privacy-audit"))
    parser.add_argument("--run-id", default="release-candidate")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--han-batch-size", type=int, default=1)
    parser.add_argument("--mc-passes", type=int, default=8)
    parser.add_argument("--max-per-cohort", type=int, default=384)
    parser.add_argument("--counterfactual-limit", type=int, default=256)
    parser.add_argument("--neutral-probe-limit", type=int, default=128)
    parser.add_argument("--similarity-train-limit", type=int, default=5000)
    parser.add_argument("--similarity-eval-limit", type=int, default=2000)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--permutation-samples", type=int, default=1000)
    parser.add_argument("--release-shingle-tokens", type=int, default=24)
    parser.add_argument("--release-scan-max-mib", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-unavailable", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-restricted-slovene", action="store_true")
    parser.add_argument("--data-only", action="store_true")
    return parser.parse_args()


def load_payload(path: Path) -> dict[str, list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected split mapping in {path}")
    return payload


def selected_split(
    repository_root: Path,
    model_root: Path,
    model: str,
    language: str,
    variant: str,
) -> int:
    release_model, release_language = release_coordinates(model, language)
    availability_path = model_root / release_model / "availability.json"
    if availability_path.is_file():
        availability = json.loads(availability_path.read_text())
        match = next(
            (
                row
                for row in availability["entries"]
                if row["language"] == release_language and row["mode"] == variant
            ),
            None,
        )
        if match and match.get("selected_split") is not None:
            return int(match["selected_split"])
    # Older availability manifests did not copy the selected run even though
    # the authoritative release registry retained it. Never guess split 0: a
    # wrong membership set invalidates the attack.
    _, registry = load_release_modules(repository_root)
    run = registry.CHECKPOINTS[(release_model, release_language, variant)].get("run")
    if run is None:
        raise ValueError(
            f"No selected split provenance for {model}/{language}/{variant}; "
            "membership inference cannot safely label train members."
        )
    return int(run)


def private_record_key(row: dict[str, Any]) -> tuple[str, str]:
    return normalize_private_text(row.get("article")), normalize_private_text(row.get("aspect"))


def matched_label_samples(
    members: list[dict[str, Any]],
    nonmembers: list[dict[str, Any]],
    limit: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    """Sample equal-size cohorts with exactly matched sentiment counts.

    Equal thirds waste most records when the neutral class is rare. This
    allocation preserves exact label matching while distributing the requested
    capacity proportionally across every label's paired availability.
    """

    rng = np.random.default_rng(seed)
    member_groups = {
        label: [row for row in members if int(row["sentiment"]) == label]
        for label in (-1, 0, 1)
    }
    nonmember_groups = {
        label: [row for row in nonmembers if int(row["sentiment"]) == label]
        for label in (-1, 0, 1)
    }
    capacities = {
        label: min(len(member_groups[label]), len(nonmember_groups[label]))
        for label in (-1, 0, 1)
    }
    if any(capacity == 0 for capacity in capacities.values()):
        raise ValueError("Every membership cohort must contain all three sentiment labels.")
    target = min(limit, sum(capacities.values()))
    allocation = {label: 1 for label in capacities}
    remaining = target - len(allocation)
    while remaining > 0:
        available = {
            label: capacities[label] - allocation[label]
            for label in capacities
            if capacities[label] > allocation[label]
        }
        if not available:
            break
        total_available = sum(available.values())
        additions = {
            label: min(
                capacity,
                max(0, int(np.floor(remaining * capacity / total_available))),
            )
            for label, capacity in available.items()
        }
        if not any(additions.values()):
            label = max(available, key=available.get)
            additions[label] = 1
        for label, addition in additions.items():
            addition = min(addition, remaining)
            allocation[label] += addition
            remaining -= addition
            if remaining == 0:
                break
    sampled_members, sampled_nonmembers = [], []
    for label in (-1, 0, 1):
        count = allocation[label]
        left = rng.choice(len(member_groups[label]), count, replace=False)
        right = rng.choice(len(nonmember_groups[label]), count, replace=False)
        sampled_members.extend(member_groups[label][int(index)] for index in left)
        sampled_nonmembers.extend(nonmember_groups[label][int(index)] for index in right)
    rng.shuffle(sampled_members)
    rng.shuffle(sampled_nonmembers)
    return sampled_members, sampled_nonmembers, {
        str(label): allocation[label] for label in (-1, 0, 1)
    }


def comparable_cohorts(
    train: list[dict[str, Any]],
    validation: list[dict[str, Any]],
    test: list[dict[str, Any]],
    *,
    limit: int,
    seed: int,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    # Any evaluation row duplicated anywhere in the full training partition is
    # a member for this threat model, even if that train row was not sampled.
    # Excluding only overlaps with sampled members would mislabel known members
    # as nonmembers and inflate or distort an attack.
    full_train_keys = {private_record_key(row) for row in train}
    validation_filtered = [row for row in validation if private_record_key(row) not in full_train_keys]
    test_filtered = [row for row in test if private_record_key(row) not in full_train_keys]
    validation_members, validation_sample, validation_labels = matched_label_samples(
        train, validation_filtered, limit, seed + 1000
    )
    test_members, test_sample, test_labels = matched_label_samples(
        train, test_filtered, limit, seed + 2000
    )
    return validation_members, validation_sample, test_members, test_sample, {
        "validation_exact_member_rows_excluded": len(validation) - len(validation_filtered),
        "test_exact_member_rows_excluded": len(test) - len(test_filtered),
        "validation_label_counts_per_membership_cohort": validation_labels,
        "test_label_counts_per_membership_cohort": test_labels,
    }


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(finite_json(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def infer(
    engine: Any,
    rows: list[dict[str, Any]],
    batch_size: int,
    seed: int,
    mc_passes: int = 0,
):
    return engine.predict_batch(rows, batch_size=batch_size, mc_passes=mc_passes, seed=seed)


def property_inference(scores: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    seen = [row for row in scores if row["aspect_seen"]]
    unseen = [row for row in scores if not row["aspect_seen"]]
    if min(len(seen), len(unseen)) < 20:
        return {
            "available": False,
            "seen_n": len(seen),
            "unseen_n": len(unseen),
            "reason": "Fewer than 20 examples in one aspect-support group.",
        }
    from aspectbench.privacy import membership_attack_report

    reports = {}
    for feature in ("confidence", "negative_entropy", "margin"):
        seen_values = [row[feature] for row in seen]
        unseen_values = [row[feature] for row in unseen]
        forward = membership_attack_report(
            seen_values,
            unseen_values,
            seed=args.seed,
            bootstrap_samples=args.bootstrap_samples,
            permutation_samples=args.permutation_samples,
        )
        if forward["auc"] < 0.5:
            forward = membership_attack_report(
                [-value for value in seen_values],
                [-value for value in unseen_values],
                seed=args.seed,
                bootstrap_samples=args.bootstrap_samples,
                permutation_samples=args.permutation_samples,
            )
            forward["score_direction"] = "lower_original_score_means_seen"
        else:
            forward["score_direction"] = "higher_original_score_means_seen"
        reports[feature] = forward
    return {
        "available": True,
        "seen_n": len(seen),
        "unseen_n": len(unseen),
        "attacks": reports,
        "interpretation": "Ability to infer whether an aspect was seen in training; not exact record extraction.",
    }


def review_triggers(report: dict[str, Any]) -> list[dict[str, Any]]:
    triggers: list[dict[str, Any]] = []
    for cohort in ("validation_nonmember", "test_nonmember"):
        membership = report.get("model_audit", {}).get("membership", {}).get(cohort, {})
        for feature, result in membership.get("univariate_attacks", {}).items():
            interval = result["auc_bootstrap_95_ci"]
            if (
                result["auc"] >= 0.60
                and interval[0] > 0.50
                and result.get("permutation_p_two_sided", 1.0) <= 0.05
            ):
                triggers.append({"type": "membership", "cohort": cohort, "feature": feature, "auc": result["auc"], "ci": interval})
        combined = membership.get("combined_attack", {}).get("held_out_auc", {})
        if combined.get("mean", 0.0) >= 0.60:
            triggers.append({"type": "combined_membership", "cohort": cohort, "auc_mean": combined["mean"]})
        matched = membership.get("covariate_matched_attack", {}).get("combined_attack", {}).get("held_out_auc", {})
        if matched.get("mean", 0.0) >= 0.60:
            triggers.append({"type": "covariate_matched_membership", "cohort": cohort, "auc_mean": matched["mean"]})
    overlap = report["data_audit"]["exact_overlap"]["cross_cohort"]["train_vs_test"]["normalized_article_and_aspect"]
    if overlap["right_rows_overlapping_left"]:
        triggers.append({"type": "exact_train_test_overlap", **overlap})
    near = report["data_audit"]["near_duplicates_after_exact_exclusion"]["test"]["fraction_at_or_above"]["0.95"]
    if near >= 0.01:
        triggers.append({"type": "near_duplicate_rate", "test_fraction_at_0.95": near})
    surface = report["data_audit"].get("tracked_release_surface", {})
    for match in surface.get("files_with_matches", []):
        if match["exact_article_value_count"] or match["distinct_long_shingle_match_count"]:
            triggers.append({"type": "tracked_corpus_text_match", **match})
        elif match["distinct_aspect_match_count"]:
            triggers.append({"type": "tracked_aspect_name_match", **match})
    return triggers


def run_model(
    args: argparse.Namespace,
    model: str,
    model_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    prefix = "slovene" if args.dataset == "sl" else "hbs"
    split = selected_split(
        args.repository_root, model_root, model, args.dataset, args.variant
    )
    split_payload = load_payload(args.repository_root / "data" / args.dataset / f"{prefix}_train_val_{split}.json")
    test = load_payload(args.repository_root / "data" / args.dataset / f"{prefix}_test.json")["test"]
    train, validation = split_payload["train"], split_payload["val"]
    validation_members, validation_sample, test_members, test_sample, exclusions = comparable_cohorts(
        train, validation, test, limit=args.max_per_cohort, seed=args.seed + split
    )
    data_cache = output_dir.parent / "_data" / f"split-{split}.json"
    data_audit = json.loads(data_cache.read_text(encoding="utf-8")) if data_cache.is_file() else None
    if not data_audit or any(
        key not in data_audit
        for key in ("tracked_release_surface", "near_duplicates_after_exact_exclusion")
    ):
        data_audit = {
            "cohort_sizes": {"train": len(train), "validation": len(validation), "test": len(test)},
            "sample_sizes": {
                "validation_members": len(validation_members),
                "validation_nonmembers": len(validation_sample),
                "test_members": len(test_members),
                "test_nonmembers": len(test_sample),
            },
            "membership_overlap_exclusions": exclusions,
            "exact_overlap": exact_overlap_report(train, validation, test),
            "near_duplicates": {
                "validation": near_duplicate_report(
                    train, validation, seed=args.seed + split,
                    train_limit=args.similarity_train_limit,
                    evaluation_limit=args.similarity_eval_limit,
                ),
                "test": near_duplicate_report(
                    train, test, seed=args.seed + 100 + split,
                    train_limit=args.similarity_train_limit,
                    evaluation_limit=args.similarity_eval_limit,
                ),
            },
            "near_duplicates_after_exact_exclusion": {
                "validation": near_duplicate_report(
                    train, validation_sample, seed=args.seed + 10 + split,
                    train_limit=args.similarity_train_limit,
                    evaluation_limit=args.similarity_eval_limit,
                ),
                "test": near_duplicate_report(
                    train, test_sample, seed=args.seed + 110 + split,
                    train_limit=args.similarity_train_limit,
                    evaluation_limit=args.similarity_eval_limit,
                ),
            },
            "aspect_distribution": aspect_distribution_report(train, validation, test),
            "text_distribution_classifier": {
                "validation": distribution_classifier_report(validation_members, validation_sample, seed=args.seed),
                "test": distribution_classifier_report(test_members, test_sample, seed=args.seed + 1),
            },
            "tracked_release_surface": release_surface_overlap_report(
                [*train, *validation, *test],
                repository_root=args.repository_root,
                shingle_tokens=args.release_shingle_tokens,
                maximum_file_bytes=args.release_scan_max_mib * 1024 * 1024,
            ),
        }
        atomic_json(data_cache, data_audit)
    report: dict[str, Any] = {
        "schema_version": 3,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "variant": args.variant,
        "model": model,
        "selected_split": split,
        "seed": args.seed,
        "mc_passes": args.mc_passes,
        "data_audit": data_audit,
        "limitations": [
            "No formal differential-privacy guarantee.",
            "A passed empirical audit cannot rule out untested attacks.",
            "Distribution shift can inflate membership inference performance.",
            "Counterfactual and neutral-context probes are out of distribution.",
            "Retrospective canary exposure is invalid because no canaries were inserted before training.",
            "The classifier interface cannot generate or reconstruct arbitrary source articles.",
        ],
    }
    if args.data_only:
        report["model_audit"] = {"skipped": True, "reason": "--data-only"}
        report["review_triggers"] = review_triggers(report)
        return report

    status = checkpoint_status(args.repository_root, model_root, model, args.dataset, args.variant)
    if not status["available"]:
        raise FileNotFoundError(status["reason"])
    checkpoint = Path(status["weight_file"])
    inventory = checkpoint_tensor_inventory(checkpoint)
    inventory["path"] = str(checkpoint.relative_to(args.repository_root))
    engine = create_engine(
        repository_root=args.repository_root,
        model_root=model_root,
        base_model_root=args.base_model_root,
        model=model,
        language=args.dataset,
        variant=args.variant,
        device=args.device,
    )
    batch_size = args.han_batch_size if getattr(engine, "backend", "") == "han" else args.batch_size
    train_aspect_counts = Counter(normalize_private_text(row.get("aspect")) for row in train)
    validation_member_outputs = infer(
        engine, validation_members, batch_size, args.seed, args.mc_passes
    )
    test_member_outputs = infer(
        engine, test_members, batch_size, args.seed + 1, args.mc_passes
    )
    validation_outputs = infer(engine, validation_sample, batch_size, args.seed, args.mc_passes)
    test_outputs = infer(engine, test_sample, batch_size, args.seed, args.mc_passes)
    validation_member_scores = safe_prediction_scores(
        validation_member_outputs, validation_members, train_aspect_counts
    )
    test_member_scores = safe_prediction_scores(
        test_member_outputs, test_members, train_aspect_counts
    )
    validation_scores = safe_prediction_scores(validation_outputs, validation_sample, train_aspect_counts)
    test_scores = safe_prediction_scores(test_outputs, test_sample, train_aspect_counts)
    membership = {
        "validation_nonmember": model_membership_report(
            validation_member_scores, validation_scores, seed=args.seed,
            bootstrap_samples=args.bootstrap_samples,
            permutation_samples=args.permutation_samples,
        ),
        "test_nonmember": model_membership_report(
            test_member_scores, test_scores, seed=args.seed + 1,
            bootstrap_samples=args.bootstrap_samples,
            permutation_samples=args.permutation_samples,
        ),
    }
    cf_n = min(args.counterfactual_limit, len(test_members), len(test_sample))
    member_cf, test_cf = test_members[:cf_n], test_sample[:cf_n]
    member_original, test_original = test_member_outputs[:cf_n], test_outputs[:cf_n]
    counterfactuals = {
        "member_pseudonym": counterfactual_aggregate(
            member_original,
            infer(engine, pseudonymized_rows(member_cf, language=args.dataset), batch_size, args.seed),
        ),
        "test_pseudonym": counterfactual_aggregate(
            test_original,
            infer(engine, pseudonymized_rows(test_cf, language=args.dataset), batch_size, args.seed),
        ),
        "member_aspect_permutation": counterfactual_aggregate(
            member_original,
            infer(engine, permuted_aspect_rows(member_cf, seed=args.seed), batch_size, args.seed),
        ),
        "test_aspect_permutation": counterfactual_aggregate(
            test_original,
            infer(engine, permuted_aspect_rows(test_cf, seed=args.seed), batch_size, args.seed),
        ),
    }
    train_unique = list(dict.fromkeys(str(row.get("aspect", "")) for row in train))
    test_unseen = list(
        dict.fromkeys(
            str(row.get("aspect", ""))
            for row in test
            if normalize_private_text(row.get("aspect")) not in train_aspect_counts
        )
    )
    rng = np.random.default_rng(args.seed)
    rng.shuffle(train_unique)
    rng.shuffle(test_unseen)
    neutral = {
        "train_seen_aspects": neutral_probe_aggregate(
            infer(
                engine,
                neutral_context_rows(train_unique[: args.neutral_probe_limit], language=args.dataset),
                batch_size,
                args.seed,
            )
        )
    }
    if test_unseen:
        neutral["test_unseen_aspects"] = neutral_probe_aggregate(
            infer(
                engine,
                neutral_context_rows(test_unseen[: args.neutral_probe_limit], language=args.dataset),
                batch_size,
                args.seed,
            )
        )
    report["model_audit"] = {
        "checkpoint_inventory": inventory,
        "membership": membership,
        "aspect_support_property_inference": {
            "validation": property_inference(validation_scores, args),
            "test": property_inference(test_scores, args),
        },
        "aspect_name_counterfactuals": counterfactuals,
        "neutral_context_aspect_probe": neutral,
    }
    report["review_triggers"] = review_triggers(report)
    del engine, validation_member_outputs, test_member_outputs, validation_outputs, test_outputs
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    return report


def main() -> None:
    args = parse_args()
    args.repository_root = args.repository_root.resolve()
    if args.mc_passes == 1 or args.mc_passes < 0:
        raise ValueError("--mc-passes must be 0 or at least 2.")
    if args.max_per_cohort < 3:
        raise ValueError("--max-per-cohort must be at least 3 (one row per sentiment label).")
    for name in ("max_per_cohort", "counterfactual_limit", "neutral_probe_limit"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if args.dataset == "sl" and not args.allow_restricted_slovene:
        raise RuntimeError(
            "Slovenian data is restricted. Pass --allow-restricted-slovene only in an authorized internal environment."
        )
    model_root = args.model_root
    if not model_root.is_absolute():
        model_root = args.repository_root / model_root
    output_root = args.output_root
    if not output_root.is_absolute():
        output_root = args.repository_root / output_root
    output_dir = output_root / args.run_id / args.dataset / args.variant
    specs = select_models(args.models, language=args.dataset, variant=args.variant)
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(
        output_dir / "manifest.json",
        {
            "schema_version": 3,
            "run_id": args.run_id,
            "dataset": args.dataset,
            "variant": args.variant,
            "models": [spec.name for spec in specs],
        "seed": args.seed,
        "mc_passes": args.mc_passes,
            "aggregate_only": True,
        },
    )
    failures = []
    for spec in specs:
        model_dir = output_dir / spec.name
        success = model_dir / "_SUCCESS.json"
        failed = model_dir / "_FAILED.json"
        if args.resume and success.is_file():
            failed.unlink(missing_ok=True)
            print(f"[{spec.name}] already complete; skipping", flush=True)
            continue
        status = checkpoint_status(args.repository_root, model_root, spec.name, args.dataset, args.variant)
        if not args.data_only and not status["available"]:
            message = status["reason"]
            if args.skip_unavailable:
                print(f"[{spec.name}] unavailable; skipping: {message}", flush=True)
                atomic_json(model_dir / "_SKIPPED.json", {"model": spec.name, "reason": message})
                continue
            raise FileNotFoundError(message)
        try:
            failed.unlink(missing_ok=True)
            print(f"[{spec.name}] starting", flush=True)
            report = run_model(args, spec.name, model_root, model_dir)
            atomic_json(model_dir / "aggregate-report.json", report)
            atomic_json(
                success,
                {
                    "model": spec.name,
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                    "report": "aggregate-report.json",
                    "review_trigger_count": len(report["review_triggers"]),
                },
            )
            failed.unlink(missing_ok=True)
            print(f"[{spec.name}] complete; triggers={len(report['review_triggers'])}", flush=True)
        except Exception as error:  # keep other models resumable
            failures.append({"model": spec.name, "error_type": type(error).__name__, "message": str(error)})
            atomic_json(failed, failures[-1])
            print(f"[{spec.name}] FAILED: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
    atomic_json(
        output_dir / "progress.json",
        {
            "completed": sorted(path.parent.name for path in output_dir.glob("*/_SUCCESS.json")),
            "skipped": sorted(path.parent.name for path in output_dir.glob("*/_SKIPPED.json")),
            "failures": failures,
        },
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
