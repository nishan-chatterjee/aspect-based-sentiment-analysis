# Local inference outputs

Inference exports are written to `inference/DATASET/RUN-ID/`. The detailed
file contains one row per model/variant/record; the adjacent `-ensemble.json`
file contains one record with expert predictions, majority voting, and
confidence voting. All generated files in this directory are ignored by Git
because they may reproduce source articles.
