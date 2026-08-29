#!/usr/bin/env python3
"""Numbered cluster-friendly wrapper for `aspectbench score`."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aspectbench.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main(["score", *sys.argv[1:]]))
