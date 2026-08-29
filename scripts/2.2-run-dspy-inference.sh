#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python "$SCRIPT_DIR/2.1-dspy-inference.py" "$@"
