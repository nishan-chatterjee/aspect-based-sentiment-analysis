#!/usr/bin/env bash
set -euo pipefail

MODE="create"
if [[ "${1:-}" == "--update" ]]; then MODE="update"; shift; fi
if [[ $# -ne 0 ]]; then
  echo "Usage: $0 [--update]" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /opt/easybuild/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
for environment in absa vllm; do
  yaml="$ROOT/$environment.yml"
  if [[ "$MODE" == "update" ]]; then
    conda env update --name "$environment" --file "$yaml" --prune
  elif conda env list | awk '{print $1}' | grep -Fxq "$environment"; then
    echo "Conda environment '$environment' already exists; leaving packages unchanged."
  else
    conda env create --file "$yaml"
  fi
  conda run --name "$environment" python -m pip install -e "$ROOT" --no-deps
  conda run --name "$environment" aspectbench models --models all >/dev/null
  echo "Installed AspectBench command in '$environment'."
done
