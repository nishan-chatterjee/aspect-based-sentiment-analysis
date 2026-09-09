# Refactor and release tracker

Updated: 2026-09-09

| Work item | Status | Evidence / next action |
|---|---|---|
| Public HF model load/inference/MC matrix | complete | 28/28 release slots passed under `huggingface/validation-runs/hf-release-smoke/` |
| BGE-M3-MLP recovery | complete | 12 split heads retained in the local training run; four selected release heads promoted |
| XLM-R/HAN unmasked recovery | complete | 12 split checkpoints retained; four selected release heads promoted |
| Privacy audit implementation | complete | aggregate-only broad and low-FPR queues, tracked-release scan, data controls, tests |
| Privacy audit GPU execution | ready | run `6.1` on the four-A40 allocation; preserve notebook/HTML with `6.3` |
| One/three-distribution Pathway B | ready | `3.7` preparation, `3.8` four-GPU grid, `3.9` best-split activation |
| Public DSPy program matrix smoke | ready | `2.5` runs 18 programs and records 10 explicit unavailable skips |
| DSPy optimization/reload smoke | ready | `4.3` writes only to ignored `optimized/` output |
| Historical 3× checkpoint/MC archive | complete | 349 mapped artifacts are present at the expected sizes in both private trees, including all 84 split checkpoints; no partial copies remain; source `absa` tree was not altered |
| Aggregate local data manifests | ready | `0.5-build-data-statistics.py` writes counts, sizes, and SHA-256 only; raw data remains ignored |
| Full GPU results interpretation | waiting on run | inspect aggregate privacy and DSPy summaries after the interactive jobs finish |
