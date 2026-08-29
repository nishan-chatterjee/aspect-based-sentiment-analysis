# User-trained models

This directory is the default destination for newly trained checkpoints. Its
contents are ignored by Git.

```text
models/MODEL/LANGUAGE/VARIANT/smoke/RUN-ID/
  best-model.pt
  smoke-report.json

models/_runs/OPERATION/RUN-ID/
  manifest.json
  progress.json
  predictions.json
  _logs/
  shards/
```

Use canonical kebab-case model names and `hbs`/`sl` language identifiers.
Released pretrained checkpoints are separate and remain under `huggingface/`.
The `_runs` and `smoke` subtrees are independently removable and every
numbered workflow resumes from them by default.
