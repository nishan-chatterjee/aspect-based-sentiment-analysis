#!/usr/bin/env python3
"""Resolve canonical splits or generate reproducible train/validation distributions."""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Callable

from sklearn.model_selection import train_test_split


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=("hbs", "sl"))
    parser.add_argument("--input", type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-count", type=int, choices=(1, 3), default=3)
    parser.add_argument("--seeds", nargs="+", type=int, default=(1729, 6174, 8191))
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument(
        "--loader",
        help="Optional module:function callable accepting Path and returning rows or a train/val mapping.",
    )
    return parser.parse_args()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def resolve_loader(value: str | None) -> Callable[[Path], Any]:
    if not value:
        return lambda path: json.loads(path.read_text(encoding="utf-8"))
    if ":" not in value:
        raise ValueError("--loader must use module:function syntax.")
    module_name, function_name = value.split(":", 1)
    loader = getattr(importlib.import_module(module_name), function_name)
    if not callable(loader):
        raise TypeError(f"Custom loader is not callable: {value}")
    return loader


def pool_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict) and isinstance(payload.get("records"), list):
        rows = payload["records"]
    elif isinstance(payload, dict) and isinstance(payload.get("train"), list):
        rows = list(payload["train"]) + list(payload.get("val", payload.get("validation", [])))
    else:
        raise ValueError("Loader must return a row list, {'records': [...]}, or {'train': [...], 'val': [...]}.")
    if not rows or any(row.get("sentiment") not in (-1, 0, 1) for row in rows):
        raise ValueError("Every source row must have sentiment -1, 0, or 1.")
    return [dict(row) for row in rows]


def main() -> None:
    args = arguments()
    if len(args.seeds) < args.split_count:
        raise ValueError("Provide at least --split-count seeds.")
    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("--validation-fraction must be between zero and one.")
    entries = []
    if args.input is None:
        prefix = "hbs" if args.dataset == "hbs" else "slovene"
        for index in range(args.split_count):
            path = (args.data_root / args.dataset / f"{prefix}_train_val_{index}.json").resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            entries.append({"split": index, "seed": args.seeds[index], "path": str(path), "generated": False})
    else:
        rows = pool_records(resolve_loader(args.loader)(args.input.resolve()))
        labels = [int(row["sentiment"]) for row in rows]
        for index, seed in enumerate(args.seeds[: args.split_count]):
            train, validation = train_test_split(
                rows,
                test_size=args.validation_fraction,
                random_state=seed,
                shuffle=True,
                stratify=labels,
            )
            path = (args.output_dir / f"train-val-{index}.json").resolve()
            atomic_json(path, {"train": train, "val": validation})
            entries.append({"split": index, "seed": seed, "path": str(path), "generated": True})
    manifest = {
        "schema_version": 1,
        "dataset": args.dataset,
        "split_count": args.split_count,
        "seeds": list(args.seeds[: args.split_count]),
        "source_input": str(args.input.resolve()) if args.input else None,
        "loader": args.loader or "built-in JSON loader",
        "loader_contract": "callable(path: pathlib.Path) -> list[dict] or train/val mapping",
        "entries": entries,
    }
    output = args.output_dir / "prepared-splits.json"
    atomic_json(output, manifest)
    print(output.resolve())


if __name__ == "__main__":
    try:
        main()
    except (ImportError, OSError, TypeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
