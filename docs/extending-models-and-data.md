# Extending models and datasets

## Add a model

Use a kebab-case public name such as `new-long-encoder`.

1. Add `src/aspectbench/models/new_long_encoder.py` with one `ModelSpec`.
2. Import that spec in `src/aspectbench/registry.py` and append it to `_SPECS`.
3. Add its loading/preprocessing implementation to the validated
   `huggingface/scripts/` release engine, including all language/variant
   checkpoint slots in `model_registry.py`.
4. Add released weights under
   `huggingface/models/new-long-encoder/{hbs,sl}/{masked,unmasked}.pt`, or add
   their authenticated Hugging Face download mapping.
5. Add registry, missing-checkpoint, one-record inference, and one-update
   smoke tests.

Architecture code is not combined into a generic encoder adapter: XLM-R,
Longformer, mDeBERTa-v3, BERTić, and SloBERTa have distinct modules. Numbered
files under `scripts/` remain thin aggregators and should not contain model
implementations.

To include the model in every `all` convenience launcher, append its canonical
name to the language-specific lists in `scripts/run-aspectbench.sh`,
`scripts/1.1-all-inference.sh`, `scripts/3.1-all-finetuning.sh`,
`scripts/3.1-all-finetuning-smoke.sh`, and the DSPy matrix launchers where that
model has a compatible program. Add its `best` variant mapping to
`configs/models/best-variants.json` and the shell launchers.

## Add a dataset

Add `configs/data/my-dataset.json` and place local files under
`data/my-dataset/`. Records use:

```json
{
  "uuid": "stable-id",
  "article": "Text with <aspect>target</aspect> tags.",
  "aspect": "target",
  "sentiment": 1
}
```

Labels are `-1`, `0`, and `1`. Input may be a JSON object, list, JSONL file, or
an object with `records`, `train`, `val`, or `test` arrays. A new language needs
an explicit registry alias and a supported checkpoint grid; do not silently
map it to HBS or Slovenian.

HBS data can be imported with `scripts/0.1-download-hbs.sh`. Slovenian data is
local/private and can be copied from authorized storage with
`scripts/0.2-import-sl-data.sh`; both destination trees are Git-ignored.
