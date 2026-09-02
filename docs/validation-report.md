# Refactor validation report

Final local release audit: 2026-09-02.

- 17 CPU unit tests passed, covering the model registry and one/few/all input,
  fixed-label and imbalance-aware metrics, per-aspect and seen/unseen reports,
  resumable runtime state, deferral gating, public-program privacy inventory,
  pre-calibrated/optimized program resolution, metadata mismatch rejection,
  and the user-checkpoint `_active` layout.
- Every Python entry point compiled in the writable validation worktree and
  every shell launcher passed `bash -n`. The llama.cpp launcher resolved the
  documented 196,608/16 dry-run command.
- All 18 tracked DSPy programs have adjacent metadata and passed the
  example-free audit. User-created programs resolve separately under the
  Git-ignored `selective-deferral-programs/optimized/` tree.
- `aspectbench models --models all` succeeded after editable installation in
  both the `absa` and `vllm` conda environments and listed all eight canonical
  model families.
- All ten copied HBS/Slovenian JSON SHA-256 digests match their authorized
  source files. Git reports the records as ignored.
- The copied Gemma 27B and Qwen 2.5 72B GGUFs match source SHA-256 values:
  `7b131f721b95bd06ef4066e18ed2febed15ad6620f61c6eaf37832837ca109fe`
  and `8199a697259eb9f2b8fd484682a8a5ecfbb17eb0600eefa198ff0430e11410a0`.
  Git reports both model trees as ignored.
- The private release archive mirrors every tracked public file except its
  intentionally private root README. Its runnable data/model paths contain
  independent copies with matching checksums and distinct inodes, while
  complete release packages and all internal prompt originals remain in
  explicitly private subtrees.
- GitHub `origin/main` matched the audited local head after a normal
  fast-forward push; the remote exposes no `backup` or `legacy` branch.

Earlier model-level validation on 2026-08-30 also established that released
BERTić HBS masked weights loaded on CPU, classified the synthetic tagged smoke
article as positive with 0.9919 confidence using two MC-dropout passes, and
resumed without duplicate inference. The real one-update smoke produced finite
loss, gradients on 201 parameters, a nonzero update, a 442,562,375-byte
checkpoint, strict reload, and a successful post-reload prediction.

No GPU was exposed in the refactor environment (`nvidia-smi` was unavailable),
so the documented interactive-node commands remain the required GPU checks.
The preserved Hugging Face release had previously passed 20 GPU model slots
with zero failures and eight expected missing-checkpoint skips.
