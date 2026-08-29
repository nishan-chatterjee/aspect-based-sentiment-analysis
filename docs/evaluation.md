# Evaluation and aspect-level reporting

Use `aspectbench.evaluation.build_evaluation_report` for every model, ensemble,
or selective-deferral output. Centralizing this calculation prevents notebooks
and training scripts from drifting into slightly different metric definitions.

The label space is always the original `(-1, 0, 1)` ordering. Macro-F1 uses all
three fixed labels even when a class is absent from a subset; this makes missing
minority-class support visible instead of silently improving a score. QWK uses
quadratic weights and returns JSON `null` when it is mathematically undefined,
as can happen for a single-class per-aspect subset.

The report contains:

- overall accuracy, Macro-F1, weighted F1, and QWK;
- per-class precision, recall, F1, and support;
- a fixed-order confusion matrix;
- class proportions, absent classes, majority fraction, and maximum/minimum
  non-zero support ratio;
- the same metrics for every normalized target aspect;
- Macro-F1, QWK, and per-class F1 macro-averaged across aspects;
- seen and unseen reports when training/validation records are supplied.

Seen/unseen status is derived from the supplied training and validation records
for the relevant split. It is not inferred from the held-out test set and is not
hard-coded to an author-specific target list.

```python
from aspectbench.data import load_records
from aspectbench.evaluation import build_evaluation_report

predictions = load_records("predictions.json")
training = load_records("hbs_train_val_0.json", keys=("train", "val"))
report = build_evaluation_report(predictions, training_records=training)
```

For a file-based command:

```bash
python scripts/5.0-score-predictions.py \
  --predictions predictions.json \
  --train-data hbs_train_val_0.json \
  --output report.json
```

For authorized local article review, export errors with the most confident
mistakes first:

```bash
python scripts/5.1-qualitative-review.py \
  --predictions models/_runs/inference/RUN-ID/predictions.json \
  --output models/_runs/analysis/RUN-ID/qualitative-review.json \
  --limit 100
```

That record-level output may reproduce source text and is therefore kept under
the ignored `models/` tree; do not commit it for Slovenian data.
