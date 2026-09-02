# Private release archive

The companion local directory
`/Utilisateurs/nchatt01/GitHub/aspect-based-sentiment-analysis-release-archive`
mirrors the public working tree at its root. It therefore has the same
top-level `backup`, `configs`, `data`, `docs`, `examples`, `huggingface`,
`models`, `notebooks`, `provenance`, `scripts`, `selective-deferral-programs`,
`src`, and `tests` layout rather than a nested `release-code/` wrapper.

The distinction is in ignored contents:

- `data/hbs/` contains the authorized CLARIN release JSONs;
- `data/sl/` contains the restricted Slovenian JSONs;
- `huggingface/models/` contains the local released fine-tuned weights;
- `models/` contains GGUF assets, user checkpoints, and run state;
- `selective-deferral-programs/private-originals/` may contain authorized
  internal originals, including programs excluded from public Git.

This archive is private even though its code layout mirrors the public repo.
Do not initialize, force-add, or push it: Slovenian records and some original
DSPy instructions are not publication-safe. Public Git receives only the
audited example-free grid under
`selective-deferral-programs/precalibrated/`.
