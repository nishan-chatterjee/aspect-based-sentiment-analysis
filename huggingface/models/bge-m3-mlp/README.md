---
library_name: pytorch
tags:
- aspect-based-sentiment-analysis
- south-slavic
- text-classification
license: cc-by-nc-4.0
---

# AspectBench BGE-M3 dense + MLP

> [!NOTE]
> AspectBench fine-tuned checkpoint contributions / trained heads are CC BY-NC 4.0: attribution is required, and commercial use requires separate permission. Noncommercial describes the use's purpose, not its user's affiliation. Third-party base assets and datasets retain their own terms; see the License section.

Selected model-only heads for normalized 1024-dimensional `BAAI/bge-m3`
document embeddings. Each released head is the best validation Macro-F1 result
among three fixed train/validation splits; test results were not used for
selection. The shared inference toolkit reconstructs the 1024→512→256→3 MLP.

## License

Copyright (c) 2026 the AspectBench model contributors.

The AspectBench fine-tuned checkpoint contributions / trained heads are licensed under
[Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/).
The [official legal code](https://creativecommons.org/licenses/by-nc/4.0/legalcode.en) is incorporated by reference.
Noncommercial research, evaluation, adaptation and redistribution are allowed
subject to attribution and the license terms. Commercial use is not granted
under this license; contact the project for separate permission.

Noncommercial describes the purpose of a use, not whether its user is a
university or a company. Academic affiliation does not automatically make a
commercial project noncommercial. The legal code controls, including its
exceptions and limitations. The material is provided as-is without warranties.

This notice does not relicense third-party base weights, tokenizers, code or
configuration assets, and does not revoke any rights previously granted.
AspectBench adapted the listed base models for aspect-based sentiment analysis;
retain their original attribution and license notices when redistributing.

- HBS and Slovenian: [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) — upstream MIT.

The frozen BGE-M3 encoder is downloaded separately and remains upstream MIT; the noncommercial grant covers the trained AspectBench MLP heads.

## What this repository contains

This repository contains four small tensor-only MLP heads (about 2.63 MB each),
one per language/mode combination. It does **not** duplicate the frozen
`BAAI/bge-m3` encoder. At inference time, the shared toolkit downloads or uses
a cached copy of `BAAI/bge-m3`, creates a normalized 1024-dimensional document
embedding, and applies the selected 1024→512→256→3 head. Thus the deployable
model is the public frozen BGE-M3 encoder plus one released AspectBench head.

The heads are ready to use through `InferenceEngine` or the `aspectbench` CLI;
they are not standalone Sentence Transformers models and should not be passed
directly to `AutoModel.from_pretrained()`. No corpus rows, embedding caches,
optimizer state, logs, or row-level predictions are included.

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

```bash
python huggingface/scripts/download.py --model bge-m3-mlp
CUDA_VISIBLE_DEVICES=0 aspectbench infer --models bge-m3-mlp --dataset hbs \
  --variant unmasked --input-doc 'Poziv za <aspect>Primer Grupu</aspect> je uspeo.' \
  --mc-passes 8
```

See the shared toolkit at
[`nishan-chatterjee/aspect-based-sentiment-analysis`](https://huggingface.co/nishan-chatterjee/aspect-based-sentiment-analysis)
for input examples, uncertainty output, and validation commands.

## Citation

Please cite the accompanying article when using AspectBench:

```bibtex
@article{chatterjee2026aspectbench,
  title   = {Evaluating Fine-Tuned, Embedding-Based, and Zero-Shot Models for
             Aspect-Based Sentiment Analysis in South Slavic News},
  author  = {Chatterjee, Nishan and Koloski, Boshko and Doucet, Antoine and
             Pollak, Senja and Purver, Matthew},
  journal = {Frontiers in Artificial Intelligence},
  volume  = {9},
  year    = {2026},
  doi     = {10.3389/frai.2026.1844418},
  url     = {https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2026.1844418/abstract}
}
```
