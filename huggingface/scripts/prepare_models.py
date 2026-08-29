#!/usr/bin/env python3
"""Export selected checkpoints as model-only .pt state dictionaries."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from model_registry import LANGUAGES, MODES, MODEL_SPECS, all_slots, model_spec, weight_path


CARD_ARTICLE = (
    "Tokom šestonedeljnog testiranja, redakcija je više puta kontaktirala "
    "<aspect>Primer Grupu</aspect> zbog nove usluge. Prvi odgovor "
    "<aspect>Primer Grupe</aspect> stigao je istog dana, a tehnički tim je "
    "zatim otklonio prijavljenu grešku bez dodatnih troškova. U završnom "
    "upitniku većina korisnika ocenila je podršku kao jasnu i pouzdanu."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--model", choices=sorted(MODEL_SPECS), action="append")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def export_state(source: Path, destination: Path, force: bool) -> dict[str, Any]:
    if destination.exists() and not force:
        loaded = torch.load(destination, map_location="cpu", weights_only=True, mmap=True)
        if not isinstance(loaded, dict) or not loaded:
            raise ValueError(f"Existing output is not a non-empty state dict: {destination}")
        return {
            "status": "existing",
            "size": destination.stat().st_size,
            "sha256": sha256(destination),
            "tensor_count": len(loaded),
        }

    checkpoint = torch_load(source)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint is not a dictionary: {source}")
    state = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    if not isinstance(state, dict) or not state:
        raise ValueError(f"No model state found in {source}")
    if not all(isinstance(value, torch.Tensor) for value in state.values()):
        raise TypeError(f"Model state contains non-tensor values: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, destination)
    del checkpoint, state
    gc.collect()

    verified = torch.load(destination, map_location="cpu", weights_only=True, mmap=True)
    forbidden = [
        key
        for key in verified
        if any(word in key.lower() for word in ("optimizer", "scheduler", "scaler"))
    ]
    if forbidden:
        raise ValueError(f"Training-state keys leaked into {destination}: {forbidden[:8]}")
    return {
        "status": "exported",
        "size": destination.stat().st_size,
        "sha256": sha256(destination),
        "tensor_count": len(verified),
    }


def family_readme(model_name: str, entries: list[dict[str, Any]]) -> str:
    spec = MODEL_SPECS[model_name]
    available = sum(item["available"] for item in entries)
    rows = []
    for item in entries:
        score = (
            f"{item['validation_macro_f1']:.4f}"
            if item["validation_macro_f1"] is not None
            else "—"
        )
        status = "Available" if item["available"] else "Checkpoint file unavailable"
        rows.append(
            f"| {item['language']} | {item['mode']} | {status} | {score} |"
        )
    table = "\n".join(rows)
    example = next((item for item in entries if item["available"]), entries[0])
    example_language = example["language"]
    example_mode = example["mode"]
    repo_id = spec["hf_repo"]
    if available == 0:
        return f"""---
library_name: pytorch
tags:
- aspect-based-sentiment-analysis
- south-slavic
- text-classification
license: other
---

# AspectBench {spec['display_name']}

This repository reserves the release location for the HBS and Slovenian
document-level aspect-based sentiment analysis checkpoints for this family.

## Checkpoint status

| Language | Mode | Status | Best validation Macro-F1 |
|---|---|---|---:|
{table}

The validation results survived, but none of the four selected trained MLP-head
files are present in the source tree or model archive. Consequently this
repository currently contains metadata only and cannot be used for inference.
The heads must be recovered or retrained before `masked.pt` and `unmasked.pt`
can be published. No checkpoint from another architecture or mode is used as a
substitute.

Input articles will use the same literal target markup as the other
AspectBench families:

```text
{CARD_ARTICLE}
```

See the shared toolkit at
[`nishan-chatterjee/aspect-based-sentiment-analysis`](https://huggingface.co/nishan-chatterjee/aspect-based-sentiment-analysis)
for the available model families and validation tooling.
"""
    return f"""---
library_name: pytorch
tags:
- aspect-based-sentiment-analysis
- south-slavic
- text-classification
license: other
---

# AspectBench {spec['display_name']}

Model-only checkpoints for HBS and Slovenian document-level aspect-based
sentiment analysis. This repository contains {available}/4 language-mode
checkpoint slots. It is used with the shared inference toolkit in
[`nishan-chatterjee/aspect-based-sentiment-analysis`](https://huggingface.co/nishan-chatterjee/aspect-based-sentiment-analysis).

## Input format

Every article must mark the target span with literal tags, even when using an
unmasked checkpoint:

```text
{CARD_ARTICLE}
```

- `masked`: the tagged text is replaced with `[ASPECT]`; the model does not see
  the target name.
- `unmasked`: the tags are removed and the model sees the target name.
- Gold `sentiment` is optional: `-1` = negative, `0` = neutral, `1` = positive.
  It is reported in the result but never used to produce the prediction.

## Available checkpoints

| Language | Mode | Status | Best validation Macro-F1 |
|---|---|---|---:|
{table}

`availability.json` contains the machine-readable selection record. A missing
checkpoint is never replaced with a checkpoint from another mode or language.

## Getting started

Create the portable environment from the toolkit repository:

```bash
conda env create -f environment.yml
conda activate aspectbench
```

`environment.yml` is maintained once in the shared toolkit rather than copied
into every model repository, preventing dependency versions from drifting
between family releases.

Or install the runtime packages in an existing environment:

```bash
python -m pip install -U torch transformers accelerate huggingface-hub sentencepiece numpy spacy sentence-transformers
```

Download the toolkit and this model repository into the expected directory
layout:

```python
from pathlib import Path
from huggingface_hub import snapshot_download

ROOT = Path("huggingface")
snapshot_download(
    repo_id="nishan-chatterjee/aspect-based-sentiment-analysis",
    local_dir=ROOT,
)
snapshot_download(
    repo_id="{repo_id}",
    local_dir=ROOT / "models" / "{model_name}",
)
```

The model repository includes the tokenizer and configuration assets required
to reconstruct the architecture. No separate base-model cache is needed.

## Python / Jupyter prediction

```python
from pathlib import Path
import sys

ROOT = Path("huggingface").resolve()
sys.path.insert(0, str(ROOT / "scripts"))

from inference import InferenceEngine

engine = InferenceEngine(
    model_name="{model_name}",
    language="{example_language}",
    mode="{example_mode}",
    model_root=ROOT / "models",
    device="cuda",  # use "cpu" when no GPU is available
)

prediction = engine.predict(
    {{
        "article": "{CARD_ARTICLE}",
        "sentiment": 1,
    }},
    mc_passes=10,
)
prediction
```

For a real batch, reuse the loaded engine:

```python
records = [
    {{"article": "{CARD_ARTICLE}", "sentiment": 1}},
    {{"article": "Pritužbe na <aspect>Drugi Sistem</aspect> nisu riješene.", "sentiment": -1}},
]
predictions = engine.predict_batch(records, batch_size=2, mc_passes=10)
```

## Command-line prediction

Run from the toolkit directory:

```bash
python scripts/predict.py \\
  --model-name {model_name} \\
  --language {example_language} \\
  --mode {example_mode} \\
  --model-root models \\
  --device cuda \\
  --mc-passes 10 \\
  --article '{CARD_ARTICLE}' \\
  --sentiment 1
```

## Output fields

| Field | Meaning |
|---|---|
| `input_article` | Original article, including `<aspect>` tags. |
| `tagged_aspects` | Target strings extracted from the tags. |
| `aspect_used` | Target representation actually supplied to the model. |
| `gold_sentiment` | Optional user-supplied reference label. |
| `predicted_sentiment` | Predicted integer label: `-1`, `0`, or `1`. |
| `predicted_sentiment_name` | Human-readable class name. |
| `class_probabilities` | Probability assigned to every sentiment class. |
| `uncertainty_across_classes` | Entropy, confidence, probability margin, and—when MC dropout is enabled—mutual information and vote statistics. |
| `inference` | Device, MC-dropout flag, and checkpoint path. |

The `.pt` files contain model tensors only. Optimizer, scheduler, and
gradient-scaler state is excluded.
"""


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_root).resolve()
    output_root = Path(args.output_root).resolve()
    selected_models = set(args.model or MODEL_SPECS)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "selection_rule": "highest validation Macro-F1 among three train-validation runs; test metrics unused",
        "expected_slots": len(MODEL_SPECS) * len(LANGUAGES) * len(MODES),
        "entries": [],
    }

    for model_name in MODEL_SPECS:
        family_entries = []
        for current_model, language, mode, selection in all_slots():
            if current_model != model_name:
                continue
            entry = {
                "model": model_name,
                "language": language,
                "mode": mode,
                "validation_macro_f1": selection["validation_macro_f1"],
                "available": bool(selection["available"]),
                "unavailable_reason": selection["unavailable_reason"],
                "base_model": model_spec(model_name, language)["base_model"],
                "weight_path": f"{model_name}/{language}/{mode}.pt",
            }
            if selection["available"] and model_name in selected_models:
                source = source_root / selection["source"]
                if not source.is_file():
                    raise FileNotFoundError(source)
                result = export_state(
                    source,
                    weight_path(output_root, model_name, language, mode),
                    force=args.force,
                )
                entry.update(result)
            elif selection["available"]:
                entry["status"] = "not_selected_for_this_run"
            else:
                entry["status"] = "unavailable"
            manifest["entries"].append(entry)
            family_entries.append(entry)
            print(f"{model_name}/{language}/{mode}: {entry['status']}", flush=True)

        family_dir = output_root / model_name
        family_dir.mkdir(parents=True, exist_ok=True)
        (family_dir / "availability.json").write_text(
            json.dumps({"model": model_name, "entries": family_entries}, indent=2) + "\n",
            encoding="utf-8",
        )
        (family_dir / "README.md").write_text(
            family_readme(model_name, family_entries), encoding="utf-8"
        )

    manifest["available_slots"] = sum(
        entry["available"] for entry in manifest["entries"]
    )
    manifest["unavailable_slots"] = manifest["expected_slots"] - manifest["available_slots"]
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
