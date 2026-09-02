# Selective-deferral programs

Only programs that pass the private-record audit may be tracked here.

Audited, release-time programs are immutable and tracked:

```text
selective-deferral-programs/precalibrated/MODEL/LANGUAGE/PROMPT-VARIANT/
  program.json
  metadata.json
```

Programs created by a user from the same or a new authorized dataset are kept
separately and ignored by Git:

```text
selective-deferral-programs/optimized/MODEL/LANGUAGE/PROMPT-VARIANT/RUN-ID/
  program.json
  metadata.json
```

Select the first tree with `defer-query --program-source precalibrated`, or the
second with `--program-source optimized --program-run-id RUN-ID`. An explicit
`--program PATH` remains available for advanced use. Metadata compatibility is
checked before inference; bypass it only deliberately with
`--allow-program-mismatch`.

Exact historical originals that contain restricted examples belong only in an
authorized private archive. A sanitized or re-optimized program receives a new
checksum and metadata that labels it `sanitized` or `post-release`; it must not
claim byte or behavioral identity with the paper program.

The inventory contains the 18 paper-facing model/language/prompt-variant
programs. Four originals contained embedded article examples and are published
with example-free instructions as `sanitized-post-release`; they deliberately
do not claim byte or behavioral identity with the paper programs. The other 14
are `paper-original-audited`. All demos, traces, and training arrays are empty.

Rebuild the grid from an authorized internal archive with
`scripts/0.3-build-safe-programs.py`. The public inventory records checksums and
relative source locators, never private records or author-specific paths.
