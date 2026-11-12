#!/usr/bin/env bash
# NaNROUTER Gutenberg release ingestion. Run where GitHub CLI/network exists.
# Example: bash source/fetch_nanrouter_books.sh corpora-short-stories-0001 books_cache
set -euo pipefail
TAG="${1:-corpora-short-stories-0001}"
DEST="${2:-books_cache}"
mkdir -p "$DEST";cd "$DEST"
command -v gh >/dev/null || { echo 'GitHub CLI (gh) is required' >&2;exit 1; }
command -v zstd >/dev/null || { echo 'zstd is required' >&2;exit 1; }
gh release download "$TAG" -R shemshallah/NaNROUTER --clobber
CHECK="$(find . -maxdepth 1 -name '*.SHA256SUMS' -print -quit)"
[[ -n "$CHECK" ]] || { echo 'Release SHA256SUMS missing' >&2;exit 1; }
sha256sum -c "$CHECK"
ARCHIVE="$(find . -maxdepth 1 -name '*.tar.zst' -print -quit)"
[[ -n "$ARCHIVE" ]] || { echo 'Release archive missing' >&2;exit 1; }
tar --zstd -xf "$ARCHIVE"
find . -type f -name '*.txt' | wc -l
