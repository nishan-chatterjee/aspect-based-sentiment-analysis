#!/usr/bin/env python3
"""Finalize the four recovered XLM-R/HAN-XLM-R unmasked heads."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aspectbench.training.transformer_recovery import finalize_main

raise SystemExit(finalize_main())
