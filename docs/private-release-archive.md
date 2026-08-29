# Private release archive

The companion local directory
`/Utilisateurs/nchatt01/GitHub/aspect-based-sentiment-analysis-release-archive`
is intentionally not a public Git repository. It contains:

- `release-code/`: a snapshot of this clean implementation;
- `data/clarin-release/`: the complete HBS release package;
- `data/slovene-release/`: the restricted Slovenian release package;
- `prompt-programs/all-originals/`: every internal
  `optimized_program*.json`, preserving its path relative to the internal
  source tree;
- `prompt-programs/MANIFEST.sha256`: integrity hashes for those originals.

The archive README labels the whole directory private. Do not initialize or
push it as a public repository: Slovenian records and some original DSPy
instructions are not publication-safe. Public Git receives only the sanitized
program grid under `selective-deferral-programs/`.
