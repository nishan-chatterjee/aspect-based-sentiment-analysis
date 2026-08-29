#!/usr/bin/env bash
set -euo pipefail

BACKEND=""
MODEL=""
TOKENIZER=""
ALIAS=""
PORT="8000"
HOST="127.0.0.1"
TP="1"
CONDA_ENV="vllm"
LLAMA_SERVER="llama-server"
MAX_MODEL_LEN="12288"
GPU_MEMORY="0.94"
CONTEXT="589824"
PARALLEL="48"
SPLIT_MODE="layer"
TENSOR_SPLIT=""
RUN_ID="llm-server"
RUN_ROOT="models/_runs"
DRY_RUN="0"
SKIP_CHAT_CHECK="0"

usage() {
  echo "Usage: $0 --backend vllm|llama-cpp --model PATH --model-alias NAME [options]" >&2
}
while [[ $# -gt 0 ]]; do
  case "$1" in
    --backend) BACKEND="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --tokenizer) TOKENIZER="$2"; shift 2 ;;
    --model-alias) ALIAS="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --host) HOST="$2"; shift 2 ;;
    --tensor-parallel-size) TP="$2"; shift 2 ;;
    --conda-env) CONDA_ENV="$2"; shift 2 ;;
    --llama-server) LLAMA_SERVER="$2"; shift 2 ;;
    --max-model-len) MAX_MODEL_LEN="$2"; shift 2 ;;
    --gpu-memory-utilization) GPU_MEMORY="$2"; shift 2 ;;
    --context-size) CONTEXT="$2"; shift 2 ;;
    --parallel) PARALLEL="$2"; shift 2 ;;
    --split-mode) SPLIT_MODE="$2"; shift 2 ;;
    --tensor-split) TENSOR_SPLIT="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --dry-run) DRY_RUN="1"; shift ;;
    --skip-chat-check) SKIP_CHAT_CHECK="1"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
done
[[ "$BACKEND" == "vllm" || "$BACKEND" == "llama-cpp" ]] || { usage; exit 2; }
[[ -n "$MODEL" && -n "$ALIAS" ]] || { usage; exit 2; }

run_dir="$RUN_ROOT/servers/$RUN_ID"
log_dir="$run_dir/_logs"
mkdir -p "$log_dir"
if [[ "$BACKEND" == "vllm" ]]; then
  command=(vllm serve "$MODEL" --served-model-name "$ALIAS" --host "$HOST" --port "$PORT"
    --tensor-parallel-size "$TP" --max-model-len "$MAX_MODEL_LEN"
    --gpu-memory-utilization "$GPU_MEMORY" --generation-config vllm)
  [[ -n "$TOKENIZER" ]] && command+=(--tokenizer "$TOKENIZER" --hf-config-path "$TOKENIZER")
else
  command=("$LLAMA_SERVER" -m "$MODEL" --alias "$ALIAS" --host "$HOST" --port "$PORT"
    -ngl 99 -c "$CONTEXT" -np "$PARALLEL" -cb -fa on --split-mode "$SPLIT_MODE")
  [[ -n "$TENSOR_SPLIT" ]] && command+=(--tensor-split "$TENSOR_SPLIT")
fi
printf 'Resolved command:'
printf ' %q' "${command[@]}"
printf '\n'
if [[ "$DRY_RUN" == "1" ]]; then exit 0; fi

source /opt/easybuild/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"
"${command[@]}" >"$log_dir/server.log" 2>&1 &
pid="$!"
printf '%s\n' "$pid" >"$run_dir/server.pid"
printf '{"backend":"%s","model":"%s","alias":"%s","host":"%s","port":%s,"pid":%s}\n' \
  "$BACKEND" "$MODEL" "$ALIAS" "$HOST" "$PORT" "$pid" >"$run_dir/manifest.json"
printf '{"status":"starting","pid":%s,"models_url":"http://127.0.0.1:%s/v1/models"}\n' \
  "$pid" "$PORT" >"$run_dir/progress.json"
echo "Started PID $pid; waiting for http://127.0.0.1:$PORT/v1/models"
for _ in $(seq 1 120); do
  if curl -fsS "http://127.0.0.1:$PORT/v1/models" >"$run_dir/models-response.json"; then
    chat_status="skipped"
    if [[ "$SKIP_CHAT_CHECK" == "1" ]]; then
      chat_ok="1"
    elif curl -fsS -H 'Content-Type: application/json' \
        -d "{\"model\":\"$ALIAS\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with OK.\"}],\"max_tokens\":8,\"temperature\":0}" \
        "http://127.0.0.1:$PORT/v1/chat/completions" >"$run_dir/chat-response.json"; then
      chat_ok="1"
      chat_status="passed"
    else
      chat_ok="0"
      chat_status="failed"
    fi
    if [[ "$chat_ok" == "1" ]]; then
      echo "Endpoint health checks complete (chat: $chat_status). Logs: $log_dir/server.log"
      printf '{"status":"healthy","pid":%s,"model_check":true,"chat_check":"%s"}\n' \
        "$pid" "$chat_status" >"$run_dir/progress.json"
      exit 0
    fi
    echo "Model list passed but chat check failed; inspect $log_dir/server.log" >&2
    printf '{"status":"failed","pid":%s,"model_check":true,"chat_check":false}\n' \
      "$pid" >"$run_dir/progress.json"
    exit 1
  fi
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "Server exited; inspect $log_dir/server.log" >&2
    printf '{"status":"failed","pid":%s,"reason":"server-exited"}\n' \
      "$pid" >"$run_dir/progress.json"
    exit 1
  fi
  sleep 2
done
echo "Health check timed out; server remains PID $pid. Inspect $log_dir/server.log" >&2
printf '{"status":"unhealthy","pid":%s,"reason":"health-timeout"}\n' \
  "$pid" >"$run_dir/progress.json"
exit 1
