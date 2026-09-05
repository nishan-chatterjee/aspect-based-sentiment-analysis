#!/usr/bin/env python3
"""Train one dataset/variant BGE-M3 MLP grid across three splits."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aspectbench.training.bge_m3 import train_main

raise SystemExit(train_main())
