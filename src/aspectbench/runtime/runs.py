"""Filesystem protocol shared by inference, training, and DSPy workflows."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import re
from typing import Any, Iterable


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def record_id(record: dict[str, Any]) -> str:
    """Return a stable ID without exposing input text in filenames."""

    explicit = record.get("id") or record.get("uuid") or record.get("record_id")
    if explicit is not None and str(explicit).strip():
        value = re.sub(r"[^A-Za-z0-9._-]+", "-", str(explicit).strip())
        return value[:96]
    content = "\0".join(
        (str(record.get("article", "")), str(record.get("aspect", "")))
    )
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:20]


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


class RunLayout:
    """A restart-safe run rooted under ``models/_runs`` by default."""

    def __init__(
        self,
        operation: str,
        run_id: str,
        *,
        run_root: str | Path = "models/_runs",
        resume: bool = True,
    ) -> None:
        self.operation = operation
        self.run_id = run_id
        self.root = Path(run_root) / operation / run_id
        if self.root.exists() and not resume:
            raise FileExistsError(
                f"Run already exists: {self.root}. Use --resume or another --run-id."
            )
        self.logs = self.root / "_logs"
        self.shards = self.root / "shards"
        self.logs.mkdir(parents=True, exist_ok=True)
        self.shards.mkdir(parents=True, exist_ok=True)
        self.progress_path = self.root / "progress.json"
        self.manifest_path = self.root / "manifest.json"

    def logger(self, name: str = "run") -> logging.Logger:
        logger_name = f"aspectbench.{self.operation}.{self.run_id}.{name}"
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
            file_handler = logging.FileHandler(self.logs / f"{name}.log", encoding="utf-8")
            file_handler.setFormatter(formatter)
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            logger.addHandler(stream_handler)
            logger.propagate = False
        return logger

    def write_manifest(self, payload: dict[str, Any]) -> None:
        existing: dict[str, Any] = {}
        if self.manifest_path.is_file():
            existing = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        manifest = {**existing, **payload}
        manifest.setdefault("schema_version", 1)
        manifest.setdefault("operation", self.operation)
        manifest.setdefault("run_id", self.run_id)
        manifest.setdefault("created_at", utc_now())
        manifest["updated_at"] = utc_now()
        atomic_json(self.manifest_path, manifest)

    def update_progress(self, **values: Any) -> None:
        progress: dict[str, Any] = {}
        if self.progress_path.is_file():
            progress = json.loads(self.progress_path.read_text(encoding="utf-8"))
        progress.update(values)
        progress["updated_at"] = utc_now()
        atomic_json(self.progress_path, progress)

    def shard_path(self, namespace: str, index: int) -> Path:
        return self.shards / namespace / f"shard-{index:06d}.json"

    def completed_ids(self, namespace: str, *, include_failed: bool = True) -> set[str]:
        completed: set[str] = set()
        directory = self.shards / namespace
        if not directory.is_dir():
            return completed
        for path in sorted(directory.glob("shard-*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows = payload if isinstance(payload, list) else payload.get("records", [])
            completed.update(
                str(row["record_id"])
                for row in rows
                if "record_id" in row
                and (include_failed or row.get("dspy_status") != "failed-fallback")
            )
        return completed

    def write_shard(self, namespace: str, index: int, rows: list[dict[str, Any]]) -> Path:
        path = self.shard_path(namespace, index)
        atomic_json(path, rows)
        return path

    def collect(self, namespace: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        directory = self.shards / namespace
        for path in sorted(directory.glob("shard-*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows.extend(payload if isinstance(payload, list) else payload["records"])
        return rows

    def collect_latest(self, namespace: str) -> list[dict[str, Any]]:
        """Return the last stored version of each record (used after retries)."""

        latest: dict[str, dict[str, Any]] = {}
        order: list[str] = []
        for row in self.collect(namespace):
            identifier = str(row.get("record_id", ""))
            if identifier not in latest:
                order.append(identifier)
            latest[identifier] = row
        return [latest[identifier] for identifier in order]


def chunks(rows: Iterable[Any], size: int) -> Iterable[list[Any]]:
    if size < 1:
        raise ValueError("shard size must be at least 1")
    bucket: list[Any] = []
    for row in rows:
        bucket.append(row)
        if len(bucket) == size:
            yield bucket
            bucket = []
    if bucket:
        yield bucket
