#!/usr/bin/env python3
"""Export a local qualitative error-review queue from prediction JSON."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from aspectbench.cli import main

raise SystemExit(main(["qualitative", *sys.argv[1:]]))
