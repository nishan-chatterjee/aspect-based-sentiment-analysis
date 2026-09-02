#!/usr/bin/env python3
"""One-update save/reload smoke without a full training run."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from aspectbench.cli import main

raise SystemExit(main(["train-smoke", *sys.argv[1:]]))
