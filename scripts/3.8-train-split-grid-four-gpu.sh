#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-/Utilisateurs/nchatt01/.conda/envs/absa/bin/python}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
DATASET="${DATASET:?Set DATASET=hbs or DATASET=sl}"
MODELS="${MODELS:-all}"
VARIANT="${VARIANT:-both}"
RUN_ID="${RUN_ID:?Set a stable RUN_ID for resume}"
SPLIT_COUNT="${SPLIT_COUNT:-3}"
SPLIT_INPUT="${SPLIT_INPUT:-}"
SPLIT_LOADER="${SPLIT_LOADER:-}"
SPLIT_SEEDS="${SPLIT_SEEDS:-1729 6174 8191}"
VALIDATION_FRACTION="${VALIDATION_FRACTION:-0.15}"
MODEL_ROOT="${MODEL_ROOT:-huggingface/models}"
DATA_ROOT="${DATA_ROOT:-data}"
OUTPUT_MODEL_ROOT="${OUTPUT_MODEL_ROOT:-models}"
MC_PASSES="${MC_PASSES:-8}"
BATCH_SIZE="${BATCH_SIZE:-4}"
EPOCHS="${EPOCHS:-3}"
DRY_RUN="${DRY_RUN:-0}"
if [[ "${1:-}" == "--" ]]; then shift; fi
EXTRA_ARGS=("$@")

[[ "$DATASET" == hbs || "$DATASET" == sl ]] || { echo "DATASET must be hbs or sl" >&2; exit 2; }
[[ "$VARIANT" =~ ^(masked|unmasked|both)$ ]] || { echo "VARIANT must be masked, unmasked, or both" >&2; exit 2; }
IFS=',' read -r -a gpu_array <<< "$GPU_IDS"
[[ ${#gpu_array[@]} -eq 4 ]] || { echo "GPU_IDS must list exactly four GPUs" >&2; exit 2; }
if [[ "$MODELS" == all ]]; then
  [[ "$DATASET" == hbs ]] && MODELS="xlmr,han-xlmr,longformer,mdeberta-v3,mt5,bertic,bge-m3-mlp" || MODELS="xlmr,han-xlmr,longformer,mdeberta-v3,mt5,sloberta,bge-m3-mlp"
fi
MODELS="${MODELS// /,}"; IFS=',' read -r -a model_array <<< "$MODELS"
if [[ "$VARIANT" == both ]]; then variants=(masked unmasked); else variants=("$VARIANT"); fi

run_root="$OUTPUT_MODEL_ROOT/_runs/training-grid/$RUN_ID"
manifest="$run_root/_data/prepared-splits.json"
prepare=("$PYTHON_BIN" scripts/3.7-prepare-training-splits.py --dataset "$DATASET" --data-root "$DATA_ROOT" --output-dir "$run_root/_data" --split-count "$SPLIT_COUNT" --validation-fraction "$VALIDATION_FRACTION" --seeds)
read -r -a seed_array <<< "$SPLIT_SEEDS"; prepare+=("${seed_array[@]}")
[[ -n "$SPLIT_INPUT" ]] && prepare+=(--input "$SPLIT_INPUT")
[[ -n "$SPLIT_LOADER" ]] && prepare+=(--loader "$SPLIT_LOADER")
"${prepare[@]}" >/dev/null
mapfile -t split_lines < <("$PYTHON_BIN" -c 'import json,sys; p=json.load(open(sys.argv[1])); [print(f"{x['"'"'split'"'"']}\t{x['"'"'path'"'"']}\t{x['"'"'seed'"'"'] if x['"'"'seed'"'"'] is not None else 42}") for x in p['"'"'entries'"'"']]' "$manifest")
test_file="$DATA_ROOT/$DATASET/$([[ "$DATASET" == hbs ]] && echo hbs || echo slovene)_test.json"
mkdir -p "$run_root/_logs"

pids=(); labels=(); status=0; task=0
for model in "${model_array[@]}"; do
  for variant in "${variants[@]}"; do
    resolved_variant="$variant"
    for line in "${split_lines[@]}"; do
      IFS=$'\t' read -r split path seed <<< "$line"
      task_run="$RUN_ID/$model/$resolved_variant/split-$split"
      report="$OUTPUT_MODEL_ROOT/$model/$DATASET/$resolved_variant/$task_run/training-report.json"
      label="$model-$resolved_variant-split-$split"; log="$run_root/_logs/$label.log"
      if [[ -s "$report" ]] && "$PYTHON_BIN" -c 'import json,sys; raise SystemExit(json.load(open(sys.argv[1])).get("status") != "complete")' "$report"; then
        echo "[$label] already complete; skipping"; continue
      fi
      command=("$PYTHON_BIN" scripts/3.0-models-finetune.py --train-input "$path" --val-input "$path" --uncertainty-input "test=$test_file" --models "$model" --dataset "$DATASET" --variant "$resolved_variant" --run-id "$task_run" --model-root "$MODEL_ROOT" --output-model-root "$OUTPUT_MODEL_ROOT" --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" --mc-passes "$MC_PASSES" --seed "$seed" --resume "${EXTRA_ARGS[@]}")
      gpu="${gpu_array[$((task % 4))]}"; task=$((task + 1)); echo "[$label] GPU $gpu -> $log"
      if [[ "$DRY_RUN" == 1 ]]; then printf 'CUDA_VISIBLE_DEVICES=%q ' "$gpu"; printf '%q ' "${command[@]}"; printf '\n'; continue; fi
      CUDA_VISIBLE_DEVICES="$gpu" "${command[@]}" >"$log" 2>&1 & pids+=("$!"); labels+=("$label")
      if (( ${#pids[@]} >= 4 )); then
        wait "${pids[0]}" || { echo "[${labels[0]}] failed; inspect logs" >&2; status=1; }
        pids=("${pids[@]:1}"); labels=("${labels[@]:1}")
      fi
    done
  done
done
for index in "${!pids[@]}"; do wait "${pids[$index]}" || { echo "[${labels[$index]}] failed" >&2; status=1; }; done
[[ "$DRY_RUN" == 1 ]] && exit 0
(( status == 0 )) || { echo "Re-run the same RUN_ID to resume failed tasks." >&2; exit 1; }
model_args=("${model_array[@]}")
"$PYTHON_BIN" scripts/3.9-finalize-training-grid.py --dataset "$DATASET" --models "${model_args[@]}" --variants "${variants[@]}" --run-id "$RUN_ID" --output-model-root "$OUTPUT_MODEL_ROOT" --output "$run_root/split-selection.json"
echo "Training grid complete: $run_root/split-selection.json"
