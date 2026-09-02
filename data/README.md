# Data

No real dataset records are distributed through Git.

Expected local layouts are:

```text
data/hbs/
  hbs_train_val_0.json
  hbs_train_val_1.json
  hbs_train_val_2.json
  hbs_test.json
  hbs_aspects.json

data/sl/
  slovene_train_val_0.json
  slovene_train_val_1.json
  slovene_train_val_2.json
  slovene_test.json
  slovene_aspects.json
```

HBS is obtained from the authenticated CLARIN.SI release. Slovenian records
must be imported from storage for which the user is authorized. Do not place
Nextcloud tokens, private download URLs, or credentials in configuration files.

```bash
# Already-downloaded CLARIN archive, or use --url / CLARIN_DOWNLOAD_URL.
bash scripts/0.1-download-hbs.sh --archive /authorized/path/hbs-release.zip

# Authorized local Slovenian release directory.
bash scripts/0.2-import-sl-data.sh --source /authorized/path/slovene-release
```

`CLARIN_BEARER_TOKEN`, when required, is read only from the environment. Both
destination directories are ignored by Git.

The local working repository and the private release archive may contain
actual authorized copies in these exact paths for smoke tests. Their presence
does not make them publishable: `git status --ignored` should report every
real JSON as ignored, and no data record may be force-added.

Dataset descriptors live in `configs/data/`. A custom dataset may use the same
record contract: `uuid`, `article`, `aspect`, and `sentiment`, with original
sentiment labels `-1`, `0`, and `1`.
