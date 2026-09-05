#!/usr/bin/env python3
"""Finalize BGE release metadata after all four grid jobs finish."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aspectbench.training.bge_m3 import finalize_main

raise SystemExit(finalize_main())
