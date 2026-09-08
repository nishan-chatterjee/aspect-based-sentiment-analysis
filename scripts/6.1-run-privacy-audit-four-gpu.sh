#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/Utilisateurs/nchatt01/.conda/envs/absa/bin/python}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
RUN_ID="${RUN_ID:-privacy-release-audit}"
MODELS="${MODELS:-all}"
MODEL_ROOT="${MODEL_ROOT:-huggingface/models}"
BASE_MODEL_ROOT="${BASE_MODEL_ROOT:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/privacy-audit}"
MAX_PER_COHORT="${MAX_PER_COHORT:-384}"
COUNTERFACTUAL_LIMIT="${COUNTERFACTUAL_LIMIT:-256}"
NEUTRAL_PROBE_LIMIT="${NEUTRAL_PROBE_LIMIT:-128}"
SIMILARITY_TRAIN_LIMIT="${SIMILARITY_TRAIN_LIMIT:-5000}"
SIMILARITY_EVAL_LIMIT="${SIMILARITY_EVAL_LIMIT:-2000}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-1000}"
PERMUTATION_SAMPLES="${PERMUTATION_SAMPLES:-1000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
HAN_BATCH_SIZE="${HAN_BATCH_SIZE:-1}"
MC_PASSES="${MC_PASSES:-8}"
SEED="${SEED:-42}"
ALLOW_RESTRICTED_SLOVENE="${ALLOW_RESTRICTED_SLOVENE:-0}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
DRY_RUN="${DRY_RUN:-0}"
export OMP_NUM_THREADS TOKENIZERS_PARALLELISM=false

IFS=',' read -r -a gpu_array <<< "$GPU_IDS"
if [[ ${#gpu_array[@]} -ne 4 ]]; then
  echo "GPU_IDS must contain exactly four comma-separated GPU IDs." >&2
  exit 2
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "PYTHON_BIN is not executable: $PYTHON_BIN" >&2
  exit 2
fi
if [[ "$ALLOW_RESTRICTED_SLOVENE" != "1" ]]; then
  echo "This four-job audit includes restricted Slovenian data." >&2
  echo "Set ALLOW_RESTRICTED_SLOVENE=1 only on an authorized internal node." >&2
  exit 2
fi
for path in \
  data/hbs/hbs_train_val_{0,1,2}.json data/hbs/hbs_test.json \
  data/sl/slovene_train_val_{0,1,2}.json data/sl/slovene_test.json; do
  if [[ ! -s "$path" ]]; then
    echo "Required data file is missing or empty: $path" >&2
    exit 2
  fi
done
"$PYTHON_BIN" -c 'import numpy, pandas, sklearn, torch, transformers' || {
  echo "The absa environment is missing an audit dependency." >&2
  exit 2
}
if [[ "$DRY_RUN" != "1" ]]; then
  cuda_count="$(CUDA_VISIBLE_DEVICES="$GPU_IDS" "$PYTHON_BIN" -c 'import torch; print(torch.cuda.device_count())')"
  if [[ ! "$cuda_count" =~ ^[0-9]+$ || "$cuda_count" -lt 4 ]]; then
    echo "The selected environment sees $cuda_count GPUs after applying GPU_IDS; four are required." >&2
    exit 2
  fi
fi

log_root="$OUTPUT_ROOT/$RUN_ID/_logs"
if [[ "$DRY_RUN" != "1" ]]; then
  mkdir -p "$log_root"
fi

datasets=(hbs hbs sl sl)
variants=(masked unmasked masked unmasked)
pids=()
labels=()

terminate_children() {
  for pid in "${pids[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap terminate_children INT TERM

echo "Starting privacy audit: run=$RUN_ID GPUs=$GPU_IDS models=$MODELS"
echo "GPU map: hbs/masked, hbs/unmasked, sl/masked, sl/unmasked"
echo "Each model writes an aggregate report and _SUCCESS marker; rerunning the same RUN_ID resumes."

for index in 0 1 2 3; do
  dataset="${datasets[$index]}"
  variant="${variants[$index]}"
  gpu="${gpu_array[$index]}"
  label="$dataset-$variant"
  log="$log_root/$label.log"
  command=(
    "$PYTHON_BIN" -u scripts/6.0-run-privacy-audit.py
    --repository-root "$ROOT"
    --model-root "$MODEL_ROOT"
    --output-root "$OUTPUT_ROOT"
    --run-id "$RUN_ID"
    --dataset "$dataset"
    --variant "$variant"
    --models "$MODELS"
    --device cuda
    --batch-size "$BATCH_SIZE"
    --han-batch-size "$HAN_BATCH_SIZE"
    --mc-passes "$MC_PASSES"
    --max-per-cohort "$MAX_PER_COHORT"
    --counterfactual-limit "$COUNTERFACTUAL_LIMIT"
    --neutral-probe-limit "$NEUTRAL_PROBE_LIMIT"
    --similarity-train-limit "$SIMILARITY_TRAIN_LIMIT"
    --similarity-eval-limit "$SIMILARITY_EVAL_LIMIT"
    --bootstrap-samples "$BOOTSTRAP_SAMPLES"
    --permutation-samples "$PERMUTATION_SAMPLES"
    --seed "$SEED"
    --resume
    --skip-unavailable
  )
  if [[ "$dataset" == "sl" ]]; then
    command+=(--allow-restricted-slovene)
  fi
  if [[ -n "$BASE_MODEL_ROOT" ]]; then
    command+=(--base-model-root "$BASE_MODEL_ROOT")
  fi
  echo "[$label] GPU $gpu -> $log"
  if [[ "$DRY_RUN" == "1" ]]; then
    printf 'CUDA_VISIBLE_DEVICES=%q ' "$gpu"
    printf '%q ' "${command[@]}"
    printf '\n'
    continue
  fi
  CUDA_VISIBLE_DEVICES="$gpu" "${command[@]}" >"$log" 2>&1 &
  pids+=("$!")
  labels+=("$label")
done

if [[ "$DRY_RUN" == "1" ]]; then
  echo "Dry run complete; no audit processes or output directories were created."
  exit 0
fi

status=0
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    echo "[${labels[$index]}] complete"
  else
    code=$?
    echo "[${labels[$index]}] failed with exit code $code; inspect $log_root/${labels[$index]}.log" >&2
    status=1
  fi
done

"$PYTHON_BIN" scripts/6.2-aggregate-privacy-audit.py \
  --run-dir "$OUTPUT_ROOT/$RUN_ID" || status=1

if [[ $status -ne 0 ]]; then
  echo "At least one audit shard failed. Re-run the identical command and RUN_ID to resume." >&2
  exit 1
fi

echo "Privacy audit complete."
echo "Aggregate report: $OUTPUT_ROOT/$RUN_ID/aggregate-summary.json"
echo "Logs: $log_root"
