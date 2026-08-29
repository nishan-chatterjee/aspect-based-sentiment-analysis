# Selective-deferral programs

Only programs that pass the private-record audit may be tracked here.

Public layout:

```text
selective-deferral-programs/MODEL/LANGUAGE/PROMPT-VARIANT/
  program.json
  metadata.json
```

Exact optimized originals that contain restricted examples belong in
`private-originals/`, which is ignored by Git. A sanitized or re-optimized
program receives a new checksum and metadata that labels it `sanitized` or
`post-release`; it must not claim byte or behavioral identity with the paper
program.

The inventory contains the 18 paper-facing model/language/prompt-variant
programs. Four originals contained embedded article examples and are published
with example-free instructions as `sanitized-post-release`; they deliberately
do not claim byte or behavioral identity with the paper programs. The other 14
are `paper-original-audited`. All demos, traces, and training arrays are empty.

Rebuild the grid from an authorized internal archive with
`scripts/0.3-build-safe-programs.py`. The public inventory records checksums and
relative source locators, never private records or author-specific paths.
