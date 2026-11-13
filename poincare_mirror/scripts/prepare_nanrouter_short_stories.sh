#!/usr/bin/env bash
# NaNROUTER release -> checksum-verified whole-book dataset.
set -euo pipefail
TAG="${1:-corpora-short-stories-0001}"
DEST="${2:-gutenberg_data}"
CACHE="${3:-gutenberg_release_cache}"
REPO='shemshallah/NaNROUTER'
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_PY="$(cd "$SCRIPT_DIR/.." && pwd)/poincare_frontier.py"
for cmd in gh zstd sha256sum python tar; do
    command -v "$cmd" >/dev/null || { echo "Missing command: $cmd" >&2; exit 2; }
done
mkdir -p "$CACHE" "$CACHE/extracted"
# GH release assets in the actual batches are named
# short_stories-0001.SHA256SUMS and short_stories-0001.tar.zst.
gh release download "$TAG" -R "$REPO" -D "$CACHE" \
    -p '*.tar.zst' -p '*.tar.zst.part-*' -p '*SHA256SUMS' --clobber
shopt -s nullglob
manifests=("$CACHE"/*SHA256SUMS)
if (( ${#manifests[@]} != 1 )); then
  echo "Expected exactly one checksum manifest for $TAG; found ${#manifests[@]}" >&2
  exit 3
fi
(cd "$CACHE" && sha256sum --strict --check "$(basename "${manifests[0]}")")
parts=("$CACHE"/*.tar.zst.part-*)
archives=("$CACHE"/*.tar.zst)
if (( ${#parts[@]} > 0 )); then
  cat "${parts[@]}" | tar --zstd -xf - -C "$CACHE/extracted"
elif (( ${#archives[@]} == 1 )); then
  tar --zstd -xf "${archives[0]}" -C "$CACHE/extracted"
else
  echo 'Expected one release tar.zst or numbered parts; refusing unverified extraction' >&2
  exit 4
fi
python "$MODEL_PY" prepare-books --source-dir "$CACHE/extracted" \
  --out-dir "$DEST" --valid-fraction 0.10 --seed 2026
printf '\nPrepared whole-book split: %s\n' "$DEST"
printf '%s\n' 'Resume the SAME active checkpoint with train --adopt-data.'
