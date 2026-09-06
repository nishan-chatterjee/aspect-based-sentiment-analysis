#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/Utilisateurs/nchatt01/.conda/envs/absa/bin/python}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
RUN_ID="${RUN_ID:-xlmr-han-paper-recovery}"
DATA_ROOT="${DATA_ROOT:-data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-huggingface/models}"
BASE_MODEL="${BASE_MODEL:-xlm-roberta-base}"
BASE_MODEL_ROOT="${BASE_MODEL_ROOT:-}"
REVISION="${REVISION:-e73636d4f797dec63c3081bb6ed5c7b0bb3f2089}"
EPOCHS="${EPOCHS:-10}"
XLMR_BATCH_SIZE="${XLMR_BATCH_SIZE:-32}"
HAN_BATCH_SIZE="${HAN_BATCH_SIZE:-2}"
HAN_EFFECTIVE_BATCH_SIZE="${HAN_EFFECTIVE_BATCH_SIZE:-32}"
PRECISION="${PRECISION:-float16}"
CHECKPOINT_EVERY_STEPS="${CHECKPOINT_EVERY_STEPS:-100}"
SEED="${SEED:-42}"
DRY_RUN="${DRY_RUN:-0}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OMP_NUM_THREADS

IFS=',' read -r -a gpu_array <<< "$GPU_IDS"
if [[ ${#gpu_array[@]} -ne 4 ]]; then
  echo "GPU_IDS must contain exactly four comma-separated GPU IDs." >&2
  exit 2
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "PYTHON_BIN is not executable: $PYTHON_BIN" >&2
  exit 2
fi
"$PYTHON_BIN" -c 'import sklearn, spacy, torch, transformers; import hr_core_news_sm, sl_core_news_sm' || {
  echo "The absa environment is missing a required transformer/HAN training package." >&2
  exit 2
}
if [[ "$DRY_RUN" != "1" ]]; then
  cuda_count="$("$PYTHON_BIN" -c 'import torch; print(torch.cuda.device_count())')"
  if [[ ! "$cuda_count" =~ ^[0-9]+$ || "$cuda_count" -lt 4 ]]; then
    echo "The selected Python environment sees $cuda_count CUDA devices; four are required." >&2
    exit 2
  fi
fi
for path in \
  "$DATA_ROOT"/hbs/hbs_train_val_{0,1,2}.json "$DATA_ROOT"/hbs/hbs_test.json \
  "$DATA_ROOT"/sl/slovene_train_val_{0,1,2}.json "$DATA_ROOT"/sl/slovene_test.json; do
  if [[ ! -s "$path" ]]; then
    echo "Required data file is missing or empty: $path" >&2
    exit 2
  fi
done
if (( HAN_EFFECTIVE_BATCH_SIZE < HAN_BATCH_SIZE )); then
  echo "HAN_EFFECTIVE_BATCH_SIZE must be at least HAN_BATCH_SIZE." >&2
  exit 2
fi
han_accumulation=$(( (HAN_EFFECTIVE_BATCH_SIZE + HAN_BATCH_SIZE - 1) / HAN_BATCH_SIZE ))

log_root="$OUTPUT_ROOT/_recovery/runs/$RUN_ID/_logs"
if [[ "$DRY_RUN" != "1" ]] && ! mkdir -p "$log_root"; then
  echo "Cannot create training log directory: $log_root" >&2
  exit 2
fi

families=(xlmr xlmr han-xlmr han-xlmr)
datasets=(hbs sl hbs sl)
pids=()
labels=()

terminate_children() {
  for pid in "${pids[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap terminate_children INT TERM

echo "Starting missing-head recovery: run=$RUN_ID GPUs=$GPU_IDS"
echo "GPU map: xlmr/hbs, xlmr/sl, han-xlmr/hbs, han-xlmr/sl"
echo "Every job trains splits 0, 1, and 2; completed splits and step checkpoints resume."
for index in 0 1 2 3; do
  family="${families[$index]}"
  dataset="${datasets[$index]}"
  gpu="${gpu_array[$index]}"
  label="$family-$dataset-unmasked"
  log="$log_root/$label.log"
  if [[ "$family" == "xlmr" ]]; then
    batch_size="$XLMR_BATCH_SIZE"
    accumulation=1
    learning_rate=2e-5
    max_length=512
  else
    batch_size="$HAN_BATCH_SIZE"
    accumulation="$han_accumulation"
    learning_rate=1e-5
    max_length=96
  fi
  command=(
    "$PYTHON_BIN" -u scripts/3.5-train-missing-transformers.py
    --repository-root "$ROOT"
    --family "$family"
    --dataset "$dataset"
    --split-indices 0 1 2
    --run-id "$RUN_ID"
    --data-root "$DATA_ROOT"
    --output-root "$OUTPUT_ROOT"
    --base-model "$BASE_MODEL"
    --revision "$REVISION"
    --device cuda
    --epochs "$EPOCHS"
    --batch-size "$batch_size"
    --accumulation-steps "$accumulation"
    --learning-rate "$learning_rate"
    --weight-decay 0.01
    --max-length "$max_length"
    --max-sentences 128
    --interaction-layers 2
    --interaction-heads 8
    --aggregation-heads 4
    --final-mlp-hidden-dim 256
    --dropout 0.2
    --precision "$PRECISION"
    --checkpoint-every-steps "$CHECKPOINT_EVERY_STEPS"
    --seed "$SEED"
    --resume
    --require-spacy
  )
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
  echo "Dry run complete; no directories or training processes were created."
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
if [[ $status -ne 0 ]]; then
  echo "At least one grid job failed. Re-run the identical command and RUN_ID to resume." >&2
  exit 1
fi

if ! "$PYTHON_BIN" scripts/3.6-finalize-missing-transformers.py \
  --repository-root "$ROOT" --run-id "$RUN_ID" --require-complete; then
  echo "Training completed, but finalization failed; no upload was attempted." >&2
  exit 1
fi

echo "All four unmasked heads trained and promoted."
echo "Comparison: $OUTPUT_ROOT/_recovery/runs/$RUN_ID/comparison-to-paper.json"
echo "Canonical heads: $OUTPUT_ROOT/{xlmr,han-xlmr}/{hbs,slovenian}/unmasked.pt"
