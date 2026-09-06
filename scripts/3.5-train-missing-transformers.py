#!/usr/bin/env python3
"""Train one XLM-R/HAN-XLM-R dataset recovery job over all three splits."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aspectbench.training.transformer_recovery import train_main

raise SystemExit(train_main())
