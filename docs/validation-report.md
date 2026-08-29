# Refactor validation report

Validated on 2026-08-30 before branch replacement:

- 13 CPU unit tests passed (registry, fixed-label metrics, per-aspect and
  seen/unseen reports, resumable runtime, gating, and public-program audit);
- every Python entry point compiled and every shell launcher passed `bash -n`;
- both vLLM and llama.cpp launchers resolved correct dry-run commands, including
  two-GPU tensor-parallel/split flags;
- all 18 public DSPy JSONs passed the example audit and representative original
  and sanitized programs loaded successfully with DSPy 3.2.1;
- released BERTić HBS masked weights loaded on CPU and classified the synthetic
  tagged smoke article as positive with 0.9919 confidence using two MC-dropout
  passes; the repeated command resumed without model inference;
- the one-update smoke produced finite loss, gradients on 201 parameters, a
  nonzero parameter update, a 442,562,375-byte checkpoint, strict reload, and a
  successful post-reload prediction; a repeated run skipped the update.

No GPU was exposed in the refactor environment (`nvidia-smi` was unavailable),
so the documented interactive GPU commands remain the final cluster checks.
The preserved Hugging Face release had previously passed 20 GPU model slots
with zero failures and eight expected missing-checkpoint skips.
