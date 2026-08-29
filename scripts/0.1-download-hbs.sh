#!/usr/bin/env bash
set -euo pipefail

URL="${CLARIN_DOWNLOAD_URL:-}"
OUTPUT="data/hbs"
ARCHIVE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --url) URL="$2"; shift 2 ;;
    --archive) ARCHIVE="$2"; shift 2 ;;
    --output) OUTPUT="$2"; shift 2 ;;
    -h|--help) echo "Usage: $0 [--url URL | --archive FILE] [--output data/hbs]"; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
mkdir -p "$OUTPUT"
temporary=""
if [[ -n "$ARCHIVE" ]]; then
  source_archive="$ARCHIVE"
elif [[ -n "$URL" ]]; then
  temporary="$(mktemp /tmp/aspectbench-hbs.XXXXXX.zip)"
  trap '[[ -n "$temporary" ]] && rm -f "$temporary"' EXIT
  curl_args=(-fL --retry 3 --output "$temporary")
  [[ -n "${CLARIN_BEARER_TOKEN:-}" ]] && curl_args+=(-H "Authorization: Bearer $CLARIN_BEARER_TOKEN")
  curl "${curl_args[@]}" "$URL"
  source_archive="$temporary"
else
  echo "Supply --archive, --url, or CLARIN_DOWNLOAD_URL. Credentials stay in environment variables." >&2
  exit 2
fi
unzip -n "$source_archive" -d "$OUTPUT"
echo "Imported HBS release into $OUTPUT. Run your authorized checksum/metadata checks before use."
