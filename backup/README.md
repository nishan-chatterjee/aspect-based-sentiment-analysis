# Curated pre-refactor backup

This directory preserves useful experiment entry points from the repository's
former `paper-release-archive-2026`, `backup`, and `legacy` histories. It is
provenance, not the supported AspectBench API; use the numbered root-level
`scripts/` and `src/aspectbench/` for new runs.

```text
backup/
  camera-ready/
    scripts/               final experiment-era training/analysis entry points
    hpc-tasks/             original interactive, SLURM, and Apptainer helpers
    configs/chat-templates/ Gemma and Qwen serving templates
  historical/scripts/     first-submission and exploratory entry points
  legacy/scripts/         earliest Git-history scripts
```

The restoration is intentionally curated. It excludes all corpus records,
record-level predictions, model weights, logs, caches, private DSPy originals,
paper/rebuttal sources, correspondence, and output-bearing notebooks. Files
with the old review-cycle organization terminology were renamed to neutral
`additional comparison` wording, including the encoder and mT5 baseline
entry points. No claim is made that historical scripts are portable without
path/configuration changes; they are retained to trace how the clean modules
were derived.

The three original Slovenian DSPy programs that contained article examples
are not present here. Audited example-free release programs live under
`selective-deferral-programs/precalibrated/`; local authorized originals belong
only in a private archive and remain ignored by Git.
