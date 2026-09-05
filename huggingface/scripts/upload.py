#!/usr/bin/env python3
"""Upload the toolkit and genuine model weights to private personal HF repos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi

from model_registry import COLLECTION_SLUG, MODEL_SPECS, NAMESPACE, TOOLKIT_REPO


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ROOT = SCRIPT_DIR.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--model", action="append", choices=sorted(MODEL_SPECS))
    parser.add_argument(
        "--toolkit-only",
        action="store_true",
        help="Upload only the shared toolkit repository, not model repositories.",
    )
    parser.add_argument("--token", help="Normally omitted; the saved HF token is used.")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Perform uploads. Without this flag, only print the intended operations.",
    )
    return parser.parse_args()


def available_count(family_dir: Path) -> int:
    availability = json.loads((family_dir / "availability.json").read_text())
    return sum(bool(entry["available"]) for entry in availability["entries"])


def assert_private(api: HfApi, repo_id: str, token: str | None) -> None:
    info = api.repo_info(repo_id=repo_id, repo_type="model", token=token)
    if not info.private:
        raise RuntimeError(f"Privacy verification failed: {repo_id} is public.")


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    api = HfApi(token=args.token)
    identity = api.whoami(token=args.token)
    username = identity.get("name") or identity.get("fullname")
    if username != NAMESPACE:
        raise RuntimeError(
            f"Authenticated as {username!r}; refusing to upload outside {NAMESPACE!r}."
        )
    if args.toolkit_only and args.model:
        raise RuntimeError("--toolkit-only cannot be combined with --model.")
    selected = [] if args.toolkit_only else (args.model or list(MODEL_SPECS))
    operations: list[tuple[str, Path, bool]] = [(TOOLKIT_REPO, root, True)]
    for model_name in selected:
        family = root / "models" / model_name
        if available_count(family) == 0:
            print(
                f"METADATA ONLY {model_name}: no saved checkpoint weights exist.",
                flush=True,
            )
        operations.append((MODEL_SPECS[model_name]["hf_repo"], family, False))

    for repo_id, folder, is_toolkit in operations:
        if not repo_id.startswith(f"{NAMESPACE}/"):
            raise RuntimeError(f"Namespace guard rejected {repo_id}.")
        print(f"PRIVATE UPLOAD {folder} -> {repo_id}", flush=True)
        if not args.execute:
            continue
        api.create_repo(
            repo_id=repo_id,
            repo_type="model",
            private=True,
            exist_ok=True,
            token=args.token,
        )
        api.update_repo_settings(
            repo_id=repo_id, repo_type="model", private=True, token=args.token
        )
        if is_toolkit:
            api.upload_folder(
                repo_id=repo_id,
                repo_type="model",
                folder_path=folder,
                ignore_patterns=[
                    "models/**",
                    "validation-report.json",
                    "validation-runs/**",
                    "logs/**",
                    "**/__pycache__/**",
                    "**/*.pyc",
                    ".gitignore",
                ],
                commit_message="Restructure reusable AspectBench inference toolkit",
                token=args.token,
            )
        else:
            api.upload_folder(
                repo_id=repo_id,
                repo_type="model",
                folder_path=folder,
                ignore_patterns=[
                    "**/__pycache__/**",
                    "**/*.pyc",
                    "training/**",
                ],
                commit_message="Add canonical HBS and Slovenian checkpoints",
                token=args.token,
            )
        assert_private(api, repo_id, args.token)
        api.add_collection_item(
            collection_slug=COLLECTION_SLUG,
            item_id=repo_id,
            item_type="model",
            exists_ok=True,
            token=args.token,
        )
        print(f"VERIFIED PRIVATE {repo_id}", flush=True)

    if not args.execute:
        print("Dry run only. Re-run with --execute to upload.", flush=True)


if __name__ == "__main__":
    main()
