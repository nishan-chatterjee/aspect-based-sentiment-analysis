#!/usr/bin/env python3
"""Numbered cluster entry point for full training plus MC uncertainty."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from aspectbench.cli import main

raise SystemExit(main(["train", *sys.argv[1:]]))
