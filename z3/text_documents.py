#!/usr/bin/env python3
"""Streaming document iteration for NaNROUTER-style plain-text corpora.

The domain collections use one blank line between documents.  Training on a
flat character stream loses that boundary; this module keeps document
boundaries explicit without loading a whole multi-gigabyte file into RAM.

A document is the text between blank lines.  Lines are yielded verbatim
(including their trailing newline) so a round-trip of the selected documents
preserves the source text apart from the separator itself.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator


def iter_documents(
    path: str | Path,
    *,
    max_bytes: int = 0,
    min_bytes: int = 1,
) -> Iterator[tuple[str, int]]:
    """Yield ``(document_text, raw_bytes)`` pairs from ``path``.

    Args:
        path: UTF-8 text file with blank-line-separated documents.
        max_bytes: stop after roughly this many raw bytes.  0 means no cap.
        min_bytes: skip documents smaller than this many UTF-8 bytes.
    """

    file_path = Path(path).expanduser()
    if not file_path.is_file():
        raise FileNotFoundError(f"corpus not found: {file_path}")

    buffer: list[str] = []
    consumed = 0
    with file_path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        for line in handle:
            if max_bytes > 0 and consumed >= max_bytes:
                break
            consumed += len(line.encode("utf-8", errors="ignore"))
            if line.strip() == "":
                if buffer:
                    text = "".join(buffer).strip()
                    buffer.clear()
                    if text:
                        raw = len(text.encode("utf-8", errors="ignore"))
                        if raw >= int(min_bytes):
                            yield text, raw
                continue
            buffer.append(line)
    if buffer:
        text = "".join(buffer).strip()
        if text:
            raw = len(text.encode("utf-8", errors="ignore"))
            if raw >= int(min_bytes):
                yield text, raw


def write_documents(
    path: str | Path,
    documents: list[str],
) -> None:
    """Write documents with exactly one blank line between them."""

    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        for index, document in enumerate(documents):
            if index:
                handle.write("\n\n")
            handle.write(document.rstrip("\n"))
        handle.write("\n")
