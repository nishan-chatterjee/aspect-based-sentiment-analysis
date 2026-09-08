#!/usr/bin/env python3
"""Aggregate completed per-model privacy reports without source records."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--require-four-shards", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    reports = []
    for path in sorted(run_dir.glob("*/*/*/aggregate-report.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        reports.append((path, report))
    completed_shards = sorted({(report["dataset"], report["variant"]) for _, report in reports})
    expected = {(dataset, variant) for dataset in ("hbs", "sl") for variant in ("masked", "unmasked")}
    missing_shards = sorted(expected - set(completed_shards))
    failures = [str(path.relative_to(run_dir)) for path in run_dir.glob("*/*/*/_FAILED.json")]
    if args.require_four_shards and missing_shards:
        raise RuntimeError(f"No completed model report for shards: {missing_shards}")
    rows, triggers = [], []
    for path, report in reports:
        membership = report.get("model_audit", {}).get("membership", {})
        for cohort in ("validation_nonmember", "test_nonmember"):
            attack = membership.get(cohort, {})
            univariate = attack.get("univariate_attacks", {})
            best_feature, best_auc = None, None
            for feature, result in univariate.items():
                if best_auc is None or result["auc"] > best_auc:
                    best_feature, best_auc = feature, result["auc"]
            best_result = univariate.get(best_feature, {}) if best_feature else {}
            one_percent = best_result.get("low_fpr_operating_points", {}).get("1%", {})
            point_one_percent = best_result.get("low_fpr_operating_points", {}).get("0.1%", {})
            rows.append(
                {
                    "dataset": report["dataset"],
                    "variant": report["variant"],
                    "model": report["model"],
                    "selected_split": report["selected_split"],
                    "nonmember_cohort": cohort,
                    "strongest_univariate_feature": best_feature,
                    "strongest_univariate_auc": best_auc,
                    "strongest_attack_permutation_p_two_sided": best_result.get("permutation_p_two_sided"),
                    "strongest_attack_tpr_at_1pct_fpr": one_percent.get("tpr"),
                    "strongest_attack_empirical_fpr_at_1pct": one_percent.get("empirical_fpr"),
                    "strongest_attack_tpr_at_0_1pct_fpr": point_one_percent.get("tpr"),
                    "low_fpr_resolution": one_percent.get("empirical_fpr_resolution") or point_one_percent.get("empirical_fpr_resolution"),
                    "combined_attack_auc_mean": attack.get("combined_attack", {}).get("held_out_auc", {}).get("mean"),
                    "covariate_matched_attack_auc_mean": attack.get("covariate_matched_attack", {}).get("combined_attack", {}).get("held_out_auc", {}).get("mean"),
                    "member_accuracy": attack.get("member_accuracy"),
                    "nonmember_accuracy": attack.get("nonmember_accuracy"),
                    "review_trigger_count": len(report.get("review_triggers", [])),
                }
            )
        for trigger in report.get("review_triggers", []):
            triggers.append(
                {
                    "dataset": report["dataset"],
                    "variant": report["variant"],
                    "model": report["model"],
                    **trigger,
                }
            )
    summary = {
        "schema_version": 3,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "aggregate_only": True,
        "report_count": len(reports),
        "completed_shards": [list(value) for value in completed_shards],
        "missing_shards": [list(value) for value in missing_shards],
        "failed_markers": failures,
        "membership_summary": rows,
        "review_triggers": triggers,
        "release_interpretation": (
            "No trigger is a privacy verdict. Review data overlap and distribution-shift results "
            "before attributing membership AUC to parameter memorization. Passing results do not "
            "constitute differential privacy or rule out untested attacks."
        ),
    }
    output = run_dir / "aggregate-summary.json"
    atomic_json(output, summary)
    print(json.dumps({"output": str(output), "reports": len(reports), "triggers": len(triggers), "missing_shards": missing_shards}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
