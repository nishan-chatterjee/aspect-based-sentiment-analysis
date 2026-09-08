#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/Utilisateurs/nchatt01/.conda/envs/absa/bin/python}"
RUN_ID="${RUN_ID:-privacy-release-audit}"
REPORT_ROOT="${REPORT_ROOT:-outputs/privacy-audit}"
SOURCE_NOTEBOOK="${SOURCE_NOTEBOOK:-notebooks/model-privacy-extraction-and-distribution-audit.ipynb}"
EXECUTED_DIR="$REPORT_ROOT/$RUN_ID/notebooks"
NOTEBOOK_NAME="$(basename "$SOURCE_NOTEBOOK" .ipynb)"
EXECUTED_NOTEBOOK="$EXECUTED_DIR/$NOTEBOOK_NAME.executed.ipynb"

if [[ "$REPORT_ROOT" = /* ]]; then
  AUDIT_RUN_DIR="$REPORT_ROOT/$RUN_ID"
else
  AUDIT_RUN_DIR="$ROOT/$REPORT_ROOT/$RUN_ID"
fi

mkdir -p "$EXECUTED_DIR"
export ASPECTBENCH_ROOT="$ROOT"
export PRIVACY_AUDIT_RUN_DIR="$AUDIT_RUN_DIR"
export PRIVACY_RUN_ID="$RUN_ID"

"$PYTHON_BIN" -m jupyter nbconvert \
  --to notebook \
  --execute "$SOURCE_NOTEBOOK" \
  --ExecutePreprocessor.timeout=1800 \
  --output "$EXECUTED_NOTEBOOK"

"$PYTHON_BIN" -m jupyter nbconvert \
  --to html "$EXECUTED_NOTEBOOK" \
  --output "$EXECUTED_DIR/$NOTEBOOK_NAME.html"

echo "Preserved executed notebook: $EXECUTED_NOTEBOOK"
echo "Preserved HTML report: $EXECUTED_DIR/$NOTEBOOK_NAME.html"
