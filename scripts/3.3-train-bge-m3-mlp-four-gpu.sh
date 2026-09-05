#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/Utilisateurs/nchatt01/.conda/envs/absa/bin/python}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
RUN_ID="${RUN_ID:-bge-m3-paper-recovery}"
EPOCHS="${EPOCHS:-15}"
EMBEDDING_BATCH_SIZE="${EMBEDDING_BATCH_SIZE:-8}"
EMBEDDING_SHARD_SIZE="${EMBEDDING_SHARD_SIZE:-512}"
EMBEDDING_PRECISION="${EMBEDDING_PRECISION:-float32}"
MLP_BATCH_SIZE="${MLP_BATCH_SIZE:-64}"
SEED="${SEED:-42}"
MAX_LENGTH="${MAX_LENGTH:-8192}"
BASE_MODEL="${BASE_MODEL:-BAAI/bge-m3}"
REVISION="${REVISION:-5617a9f61b028005a4858fdac845db406aefb181}"
DATA_ROOT="${DATA_ROOT:-data}"
OUTPUT_ROOT="huggingface/models/bge-m3-mlp"
CACHE_ROOT="${CACHE_ROOT:-huggingface/models/bge-m3-mlp/training/cache}"
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
$PYTHON_BIN -c 'import sentence_transformers, sklearn, torch' || {
  echo "The absa environment is missing a required BGE training package." >&2
  exit 2
}
if [[ "$DRY_RUN" != "1" ]]; then
  cuda_count="$($PYTHON_BIN -c 'import torch; print(torch.cuda.device_count())')"
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

run_root="$OUTPUT_ROOT/training/runs/$RUN_ID"
log_root="$run_root/_logs"
if [[ "$DRY_RUN" != "1" ]] && ! mkdir -p "$log_root"; then
  echo "Cannot create training log directory: $log_root" >&2
  exit 2
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

echo "Starting BGE-M3 release recovery: run=$RUN_ID GPUs=$GPU_IDS"
echo "Each GPU embeds one dataset/variant once, then trains splits 0, 1, and 2."
for index in 0 1 2 3; do
  dataset="${datasets[$index]}"
  variant="${variants[$index]}"
  gpu="${gpu_array[$index]}"
  label="$dataset-$variant"
  log="$log_root/$label.log"
  command=(
    "$PYTHON_BIN" -u scripts/3.3-train-bge-m3-mlp.py
    --dataset "$dataset"
    --variant "$variant"
    --split-indices 0 1 2
    --run-id "$RUN_ID"
    --data-root "$DATA_ROOT"
    --output-root "$OUTPUT_ROOT"
    --cache-root "$CACHE_ROOT"
    --base-model "$BASE_MODEL"
    --device cuda
    --max-length "$MAX_LENGTH"
    --embedding-precision "$EMBEDDING_PRECISION"
    --embedding-batch-size "$EMBEDDING_BATCH_SIZE"
    --embedding-shard-size "$EMBEDDING_SHARD_SIZE"
    --epochs "$EPOCHS"
    --batch-size "$MLP_BATCH_SIZE"
    --learning-rate 1e-4
    --weight-decay 0.01
    --hidden-dim1 512
    --hidden-dim2 256
    --dropout 0.3
    --seed "$SEED"
    --resume
  )
  if [[ -n "$REVISION" ]]; then
    command+=(--revision "$REVISION")
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
  echo "Dry run complete; no training processes were started."
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
  echo "At least one grid job failed. Re-run this same command and RUN_ID to resume." >&2
  exit 1
fi

if ! "$PYTHON_BIN" scripts/3.4-finalize-bge-m3-mlp.py \
  --repository-root "$ROOT" --run-id "$RUN_ID" --require-complete; then
  echo "All jobs finished, but release finalization failed; no upload was attempted." >&2
  exit 1
fi

echo "All four heads trained and promoted."
echo "Comparison: $run_root/comparison-to-paper.json"
echo "Canonical heads: huggingface/models/bge-m3-mlp/{hbs,slovenian}/{masked,unmasked}.pt"
