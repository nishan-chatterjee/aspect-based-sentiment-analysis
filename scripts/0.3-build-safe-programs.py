#!/usr/bin/env python3
"""Build the audited 18-program public grid from an authorized internal archive."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from aspectbench.deferral.programs import SIGNATURE_INSTRUCTIONS
from aspectbench.runtime.runs import atomic_json


EXPERTS = {
    "slovenian": {
        "han_xlmr_masked": "han-xlmr",
        "longformer_masked": "longformer",
        "slavic_specific_masked": "sloberta",
    },
    "serbian": {
        "han_xlmr_masked": "han-xlmr",
        "longformer_masked": "longformer",
        "mdeberta_masked": "mdeberta-v3",
        "slavic_specific_masked": "bertic",
    },
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_payload(payload: Any) -> str:
    rendered = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(rendered).hexdigest()


def expected_sources(internal_root: Path):
    review_root = internal_root / "reviews/uncertainty/llm-selective-deferral"
    for language, experts in EXPERTS.items():
        for variant in ("masked", "unmasked"):
            for source_expert, public_expert in experts.items():
                directory = review_root / language / variant / source_expert / "medium"
                matches = sorted(directory.glob("optimized_program_*.json"))
                if len(matches) != 1:
                    raise FileNotFoundError(f"Expected one program under {directory}; found {len(matches)}")
                yield matches[0], public_expert, ("sl" if language == "slovenian" else "hbs"), variant
    extra_root = internal_root / "additional-tasks/uncertainty/llm-selective-deferral"
    for language in ("slovenian", "serbian"):
        for variant in ("masked", "unmasked"):
            directory = extra_root / language / variant / "xlmr_truncated_masked" / "medium"
            matches = sorted(directory.glob("optimized_program_*.json"))
            if len(matches) != 1:
                raise FileNotFoundError(f"Expected one program under {directory}; found {len(matches)}")
            yield matches[0], "xlmr", ("sl" if language == "slovenian" else "hbs"), variant


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--internal-root", required=True)
    parser.add_argument("--output-root", default=str(ROOT / "selective-deferral-programs"))
    args = parser.parse_args()
    internal_root = Path(args.internal_root).resolve()
    output_root = Path(args.output_root).resolve()
    inventory = []
    for source, expert, language, variant in expected_sources(internal_root):
        payload = json.loads(source.read_text(encoding="utf-8"))
        signature = payload["predict"]["signature"]
        original_instructions = str(signature.get("instructions", ""))
        sanitized = "**Example:**" in original_instructions
        if sanitized:
            signature["instructions"] = SIGNATURE_INSTRUCTIONS
        for key in ("traces", "train", "demos"):
            payload["predict"][key] = []
        rendered = json.dumps(payload, ensure_ascii=False)
        if "**Example:**" in rendered:
            raise ValueError(f"Embedded example remains after sanitation: {source}")
        destination = output_root / "precalibrated" / expert / language / variant
        destination.mkdir(parents=True, exist_ok=True)
        atomic_json(destination / "program.json", payload)
        metadata = {
            "schema_version": 1,
            "model": expert,
            "language": language,
            "prompt_variant": variant,
            "optimizer": "MIPROv2-medium",
            "release_status": "sanitized-post-release" if sanitized else "paper-original-audited",
            "behavioral_parity_with_paper": not sanitized,
            "source_sha256": sha256(source),
            "program_sha256": sha256_payload(payload),
            "source_locator": str(source.relative_to(internal_root)),
            "privacy_audit": {
                "demos_cleared": True,
                "traces_cleared": True,
                "train_cleared": True,
                "embedded_example_removed": sanitized,
            },
        }
        atomic_json(destination / "metadata.json", metadata)
        inventory.append(metadata)
    inventory.sort(key=lambda row: (row["language"], row["model"], row["prompt_variant"]))
    if len(inventory) != 18:
        raise RuntimeError(f"Expected 18 programs, built {len(inventory)}")
    atomic_json(output_root / "inventory.json", {"schema_version": 1, "programs": inventory})
    print(f"Built {len(inventory)} audited programs under {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
