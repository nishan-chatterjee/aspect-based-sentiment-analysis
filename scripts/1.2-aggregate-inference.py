#!/usr/bin/env python3
"""Numbered entry point for combining independently scheduled experts."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aspectbench.cli import main

raise SystemExit(main(["aggregate-inference", *sys.argv[1:]]))
