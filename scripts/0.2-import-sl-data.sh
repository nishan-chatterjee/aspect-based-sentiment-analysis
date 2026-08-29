#!/usr/bin/env bash
set -euo pipefail

SOURCE=""
OUTPUT="data/sl"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --source) SOURCE="$2"; shift 2 ;;
    --output) OUTPUT="$2"; shift 2 ;;
    -h|--help) echo "Usage: $0 --source AUTHORIZED_LOCAL_DIR [--output data/sl]"; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$SOURCE" && -d "$SOURCE" ]] || { echo "--source must be an authorized local directory" >&2; exit 2; }
mkdir -p "$OUTPUT"
cp -n "$SOURCE"/*.json "$OUTPUT"/
echo "Imported local Slovenian JSON files into $OUTPUT (Git-ignored)."
