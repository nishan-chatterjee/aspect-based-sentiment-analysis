---
library_name: pytorch
tags:
- aspect-based-sentiment-analysis
- south-slavic
- text-classification
license: other
---

# AspectBench BGE-M3 dense + MLP

Selected model-only heads for normalized 1024-dimensional `BAAI/bge-m3`
document embeddings. Each released head is the best validation Macro-F1 result
among three fixed train/validation splits; test results were not used for
selection. The shared inference toolkit reconstructs the 1024→512→256→3 MLP.

| Language | Mode | Status | Selected validation Macro-F1 | Mean test Macro-F1 (3 splits) | Mean test QWK (3 splits) |
|---|---|---|---:|---:|---:|
| hbs | masked | Available (retrained) | 0.8851 | 0.7866 | 0.7572 |
| hbs | unmasked | Available (retrained) | 0.8856 | 0.7861 | 0.7551 |
| slovenian | masked | Available (retrained) | 0.7529 | 0.6528 | 0.6538 |
| slovenian | unmasked | Available (retrained) | 0.7578 | 0.6737 | 0.6458 |

Masked training reproduces the historical implementation used for the paper:
tagged mentions become `[ASPECT_MENTION]` and `[ASPECT_NAME]` is appended.
Unmasked training removes the literal XML-like aspect tags. Complete metrics,
per-class results, seen/unseen reports, seeds, and paper deltas are retained in
the private ignored training run before upload.

## Use

Download this repository beneath the toolkit at
`huggingface/models/bge-m3-mlp/`, then run `aspectbench infer --models
bge-m3-mlp ...`. The checkpoint files contain tensors only—no optimizer state,
dataset rows, or cached embeddings.

See the shared toolkit at
[`nishan-chatterjee/aspect-based-sentiment-analysis`](https://huggingface.co/nishan-chatterjee/aspect-based-sentiment-analysis)
for input examples, uncertainty output, and validation commands.
