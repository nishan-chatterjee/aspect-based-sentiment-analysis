#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/Utilisateurs/nchatt01/.conda/envs/absa/bin/python}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
RUN_ID="${RUN_ID:-hf-release-smoke}"
MODELS="${MODELS:-mt5,longformer,mdeberta-v3,han-xlmr,xlmr,slavic-specific,bge-m3-mlp}"
MC_PASSES="${MC_PASSES:-8}"
BATCH_SIZE="${BATCH_SIZE:-10}"
DOWNLOAD_FROM_HF="${DOWNLOAD_FROM_HF:-1}"
BASE_MODEL_ROOT="${BASE_MODEL_ROOT:-}"
DRY_RUN="${DRY_RUN:-0}"

if [[ "$MODELS" == "all" ]]; then
  MODELS="mt5,longformer,mdeberta-v3,han-xlmr,xlmr,slavic-specific,bge-m3-mlp"
fi

IFS=',' read -r -a gpu_array <<< "$GPU_IDS"
if [[ ${#gpu_array[@]} -ne 4 ]]; then
  echo "GPU_IDS must contain exactly four comma-separated GPU IDs." >&2
  exit 2
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "PYTHON_BIN is not executable: $PYTHON_BIN" >&2
  exit 2
fi
if [[ "$MC_PASSES" == "1" || "$MC_PASSES" =~ ^- ]]; then
  echo "MC_PASSES must be 0 or at least 2." >&2
  exit 2
fi
for path in \
  huggingface/examples/hbs-tagged-examples.json \
  huggingface/examples/sl-tagged-synthetic-examples.json; do
  [[ -s "$path" ]] || { echo "Missing example file: $path" >&2; exit 2; }
done

export ABSA_PYTHON="$PYTHON_BIN"
export ABSA_MODELS="$MODELS"
export ABSA_MC_PASSES="$MC_PASSES"
export ABSA_BATCH_SIZE="$BATCH_SIZE"
export ABSA_DEVICE=cuda
export ABSA_VALIDATION_RUN_ID="$RUN_ID"
export ABSA_DOWNLOAD_FROM_HF="$DOWNLOAD_FROM_HF"
export ABSA_REQUIRE_COMPLETE_MATRIX=1
export GPU_IDS
if [[ -n "$BASE_MODEL_ROOT" ]]; then
  export ABSA_BASE_MODEL_ROOT="$BASE_MODEL_ROOT"
fi

echo "Starting four-GPU Hugging Face release smoke test: run=$RUN_ID"
echo "Models: $MODELS"
echo "Remote download: $DOWNLOAD_FROM_HF; MC passes: $MC_PASSES"
echo "This tests both example files, both variants, and every language slot."

if [[ "$DRY_RUN" == "1" ]]; then
  echo "DRY RUN: ABSA_VALIDATION_RUN_ID=$RUN_ID ABSA_MODELS=$MODELS GPU_IDS=$GPU_IDS bash huggingface/scripts/run_validation_interactive.sh 4"
  exit 0
fi

bash huggingface/scripts/run_validation_interactive.sh 4

echo "Smoke test complete."
echo "Preserved report: huggingface/validation-runs/$RUN_ID/validation-report.json"
echo "Preserved logs: huggingface/validation-runs/$RUN_ID/logs"
