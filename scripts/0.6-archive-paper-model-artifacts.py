#!/usr/bin/env python3
"""Resumably preserve all three paper checkpoints and available prediction artifacts."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess


@dataclass(frozen=True)
class Item:
    source: Path
    relative: Path
    kind: str


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def inventory(source_root: Path, release_root: Path, include_training_state: bool) -> list[Item]:
    items: list[Item] = []
    language_dirs = {"hbs": "serbian", "sl": "slovenian"}
    historical = {
        "xlmr": ("additional-tasks/xlmr_truncated/masked", ("masked",)),
        "han-xlmr": ("results/global-context-modelling/simplified-dart-xlmr", ("masked",)),
        "longformer": ("reviews/longformer", ("masked", "unmasked")),
        "mdeberta-v3": ("reviews/mdeberta", ("masked", "unmasked")),
        "mt5": ("reviews/mt5", ("masked", "unmasked")),
        "slavic-specific": ("reviews/slavic_specific", ("masked", "unmasked")),
    }
    for family, (base, variants) in historical.items():
        for dataset, source_language in language_dirs.items():
            model = "bertic" if family == "slavic-specific" and dataset == "hbs" else "sloberta" if family == "slavic-specific" else family
            for variant in variants:
                folder = source_root / base / variant / source_language if family not in {"xlmr", "han-xlmr"} else source_root / base / source_language
                for split in range(3):
                    target = Path("models/_paper-splits") / model / dataset / variant / f"split-{split}"
                    for name, destination, kind in (
                        (f"best_model_{split}.pt", "best-model.pt", "checkpoint"),
                        (f"test_predictions_{split}.json", "test-predictions.json", "test_predictions"),
                        (f"training_metrics_{split}.json", "training-metrics.json", "training_metrics"),
                    ):
                        path = folder / name
                        if path.is_file():
                            items.append(Item(path, target / destination, kind))
    recovery_run = "xlmr-han-paper-recovery"
    for model in ("xlmr", "han-xlmr"):
        for dataset, release_language in (("hbs", "hbs"), ("sl", "slovenian")):
            folder = release_root / "huggingface/models" / model / "training/runs" / recovery_run / release_language / "unmasked"
            for split in range(3):
                source = folder / f"split-{split}"
                target = Path("models/_paper-splits") / model / dataset / "unmasked" / f"split-{split}"
                names = [("best-model.pt", "checkpoint"), ("test-predictions.json", "test_predictions"), ("training-report.json", "training_metrics")]
                if include_training_state:
                    names.append(("last-training-state.pt", "training_state"))
                for name, kind in names:
                    if (source / name).is_file():
                        items.append(Item(source / name, target / name, kind))
    bge_run = "bge-m3-paper-recovery"
    for dataset, release_language in (("hbs", "hbs"), ("sl", "slovenian")):
        for variant in ("masked", "unmasked"):
            folder = release_root / "huggingface/models/bge-m3-mlp/training/runs" / bge_run / release_language / variant
            for split in range(3):
                source = folder / f"split-{split}"
                target = Path("models/_paper-splits/bge-m3-mlp") / dataset / variant / f"split-{split}"
                names = [("best-model.pt", "checkpoint"), ("test-predictions.json", "test_predictions"), ("training-report.json", "training_metrics")]
                if include_training_state:
                    names.append(("last-training-state.pt", "training_state"))
                for name, kind in names:
                    if (source / name).is_file():
                        items.append(Item(source / name, target / name, kind))
    uncertainty = {
        "xlmr": source_root / "additional-tasks/uncertainty/xlmr_truncated_masked",
        "han-xlmr": source_root / "reviews/uncertainty/han_xlmr_masked",
        "longformer": source_root / "reviews/uncertainty/longformer_masked",
        "mdeberta-v3": source_root / "reviews/uncertainty/mdeberta_masked",
        "slavic-specific": source_root / "reviews/uncertainty/slavic_specific_masked",
    }
    for family, base in uncertainty.items():
        for dataset, source_language in language_dirs.items():
            model = "bertic" if family == "slavic-specific" and dataset == "hbs" else "sloberta" if family == "slavic-specific" else family
            folder = base / source_language
            if folder.is_dir():
                for path in folder.glob("*.json"):
                    relative = Path("models/_paper-splits") / model / dataset / "masked/uncertainty/paper-mc" / path.name
                    items.append(Item(path, relative, "mc_uncertainty"))
    return items


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True, help="Read-only historical absa root.")
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--include-training-state", action="store_true")
    parser.add_argument("--sha256", action="store_true")
    args = parser.parse_args()
    roots = [args.release_root.resolve(), args.archive_root.resolve()]
    items = inventory(args.source_root.resolve(), roots[0], args.include_training_state)
    rows = []
    for item in items:
        row = {"source": str(item.source), "relative_destination": str(item.relative), "kind": item.kind, "size_bytes": item.source.stat().st_size}
        if args.sha256:
            row["sha256"] = digest(item.source)
        row["destinations"] = []
        for root in roots:
            destination = root / item.relative
            state = "present" if destination.is_file() and destination.stat().st_size == item.source.stat().st_size else "pending"
            if args.execute and state == "pending":
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_suffix(destination.suffix + ".partial")
                subprocess.run(["cp", "--reflink=auto", "--preserve=mode,timestamps", str(item.source), str(temporary)], check=True)
                os.replace(temporary, destination)
                state = "copied"
            row["destinations"].append({"root": str(root), "path": str(destination), "status": state})
        rows.append(row)
    checkpoints = sum(row["kind"] == "checkpoint" for row in rows)
    payload = {"schema_version": 1, "generated_at": datetime.now(timezone.utc).isoformat(), "checkpoint_count": checkpoints, "item_count": len(rows), "include_training_state": args.include_training_state, "items": rows}
    for root in roots:
        output = root / "models/_paper-splits/artifact-inventory.json"
        if args.execute:
            output.parent.mkdir(parents=True, exist_ok=True)
            temporary = output.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(payload, indent=2) + "\n")
            os.replace(temporary, output)
    print(json.dumps({"checkpoint_count": checkpoints, "item_count": len(rows), "execute": args.execute}, indent=2))
    if checkpoints != 84:
        raise SystemExit(f"Expected 84 split checkpoints, found {checkpoints}")


if __name__ == "__main__":
    main()
