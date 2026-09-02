#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/apptainer"

SIF_PATH="${SIF_PATH:-${SCRIPT_DIR}/absa-comparisons.sif}"

echo "Building Apptainer image: ${SIF_PATH}"
echo "Definition: ${SCRIPT_DIR}/apptainer/absa.def"

apptainer build "$SIF_PATH" absa.def

echo "Built: ${SIF_PATH}"
