# User-trained models

This directory is the default destination for newly trained checkpoints. Its
contents are ignored by Git.

```text
models/MODEL/LANGUAGE/VARIANT/smoke/RUN-ID/
  best-model.pt
  smoke-report.json

models/MODEL/LANGUAGE/VARIANT/RUN-ID/
  best-model.pt
  last-training-state.pt
  training-report.json
  uncertainty/SPLIT/
    predictions-with-uncertainty.json
    shard-*.json
    _SUCCESS.json

models/_runs/OPERATION/RUN-ID/
  manifest.json
  progress.json
  predictions.json
  _logs/
  shards/

models/_active/HUGGINGFACE-FAMILY/{hbs,slovenian}/{masked,unmasked}.pt
  # stable local pointers created after full training
```

Use canonical kebab-case model names and `hbs`/`sl` language identifiers.
Released fine-tuned PLM checkpoints are separate and remain under
`huggingface/`. Local GGUF serving assets may also be stored under
`models/gemma3-27b-qat/` and `models/qwen2.5-72b/`; all contents of this
directory are ignored by Git. The `_runs` and `smoke` subtrees are
independently removable and every numbered workflow resumes by default.
Use `--model-root models/_active` for later inference or DSPy inference with
the most recently activated user-trained checkpoints. The per-run best model
remains the durable source; `_active` is a replaceable deployment view.
