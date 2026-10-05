"""Digest collector: read the user's input and work out what format it is.

Format detection is plain code, tried in this order:
    json   the whole thing parses as JSON
    jsonl  every line is its own JSON object ("JSON Lines")
    csv    rows with a consistent number of columns, split by , ; tab or |
    log    most lines start with a timestamp
    text   anything else

The detected format goes into meta["format"] for the enricher, which then
computes facts (row counts, outliers, time ranges...) for the model to quote.
"""

from __future__ import annotations

import csv
import io
import json
import re

# Timestamps at the start of a log line, e.g.
#   2024-05-01 03:12:44 / 2024-05-01T03:12:44Z / [2024-05-01 ...] / May  1 03:12:44
TIMESTAMP_START = re.compile(
    r"^\s*\[?("
    r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}"
    r"|[A-Z][a-z]{2} +\d{1,2} \d{2}:\d{2}:\d{2}"
    r"|\d{2}/\d{2}/\d{4}[ :]\d{2}:\d{2}"
    r")"
)
CSV_DELIMITERS = ",;\t|"


def looks_like_json(text: str) -> bool:
    stripped = text.strip()
    if not stripped or stripped[0] not in "{[":
        return False
    try:
        json.loads(stripped)
        return True
    except json.JSONDecodeError:
        return False


def looks_like_jsonl(lines: list[str]) -> bool:
    if len(lines) < 2:
        return False
    for line in lines:
        try:
            if not isinstance(json.loads(line), dict):
                return False
        except json.JSONDecodeError:
            return False
    return True


def detect_csv_delimiter(lines: list[str]) -> str | None:
    """Return the delimiter if the lines look like a table, else None.

    A table here = at least 2 rows, at least 2 columns, and 90% of rows
    having the same number of columns as the header.
    """
    if len(lines) < 2:
        return None
    sample = lines[:200]
    for delimiter in CSV_DELIMITERS:
        rows = list(csv.reader(io.StringIO("\n".join(sample)), delimiter=delimiter))
        width = len(rows[0])
        if width < 2:
            continue
        same = sum(len(r) == width for r in rows)
        if same / len(rows) >= 0.9:
            return delimiter
    return None


def detect_format(text: str) -> tuple[str, dict]:
    """Return (format name, extra details)."""
    lines = [line for line in text.splitlines() if line.strip()]
    if looks_like_json(text):
        return "json", {}
    if looks_like_jsonl(lines):
        return "jsonl", {}
    delimiter = detect_csv_delimiter(lines)
    if delimiter:
        return "csv", {"delimiter": delimiter}
    if lines and sum(bool(TIMESTAMP_START.match(l)) for l in lines) / len(lines) >= 0.5:
        return "log", {}
    return "text", {}


def collect(ctx):
    text = ctx.user_input_text()
    fmt, details = detect_format(text)
    return {
        "text": text,
        "meta": {"source": ctx.user_input.describe(), "format": fmt, **details},
    }
