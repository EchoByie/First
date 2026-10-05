"""Digest enricher: compute facts with plain code, before the model sees anything.

Why? Models are bad at counting and arithmetic but good at interpreting.
So code does the counting (rows, outliers, time gaps, repeated lines) and
writes each fact as one short line. The model then quotes those lines as
evidence, and verification can confirm them, which turns many findings
into "✔ I'm sure" instead of guesses.

What the model sees (facts = the "preamble", then your data):

    === FACTS COMPUTED BY CODE (tag 3fa9c1d2) ===
    format: csv (delimiter ",")
    rows: 120 data rows plus 1 header row
    column "bytes": numeric, 120 values, min 0, max 9812331, median 1020
    column "bytes": outlier 9812331 on data row 57 (median 1020)
    ...
    === ORIGINAL CONTENT ===
    <your data, unchanged>

The random tag shows which facts header is real. If your data contains
its own fake "FACTS" header, that is noted as a fact too.
"""

from __future__ import annotations

import csv
import io
import json
import re
import secrets
import statistics
from collections import Counter
from datetime import datetime

MAX_FACT_LINES = 80
MAX_COLUMNS = 30
MAX_OUTLIERS_PER_COLUMN = 3
FACTS_HEADER = "FACTS COMPUTED BY CODE"

# Phrases that look like someone trying to give orders to an AI. Finding them
# doesn't mean an attack, but you should know they are in the data.
INSTRUCTION_LIKE = re.compile(
    r"ignore (all |any )?(the )?(previous|prior|above|earlier) (instructions|prompts?)"
    r"|disregard (all|the|previous|prior|your)"
    r"|you are now\b|new instructions|system prompt|act as (a|an|the)\b"
    r"|do not (tell|report|mention)",
    re.IGNORECASE,
)
ISO_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}")
LOG_LEVEL = re.compile(r"\b(CRITICAL|FATAL|ERROR|WARN(?:ING)?|INFO|DEBUG|TRACE)\b")


# --- small helpers -----------------------------------------------------------------

def _num(text: str):
    """Parse a number, or return None. 'nan' and 'inf' don't count."""
    try:
        value = float(text.strip())
    except (ValueError, AttributeError):
        return None
    return value if value == value and abs(value) != float("inf") else None


def _fmt(x: float) -> str:
    """Format a computed number: 1020.0 -> '1020', 1203.4567 -> '1203.46'."""
    return str(int(x)) if x == int(x) else f"{x:.2f}".rstrip("0").rstrip(".")


def _duration(delta) -> str:
    """timedelta -> '3h 22m 35s'."""
    seconds = int(delta.total_seconds())
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    parts = [f"{days}d" if days else "", f"{hours}h" if hours else "",
             f"{minutes}m" if minutes else "", f"{seconds}s"]
    return " ".join(p for p in parts if p)


def _short(text: str, limit: int = 80) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


# --- column profiling (used for CSV and for JSON records) ---------------------------

def profile_column(name: str, values: list[str], row_label: str) -> list[str]:
    """Facts about one column. `values` are the raw strings (may be empty)."""
    facts = []
    filled = [(i, v) for i, v in enumerate(values, start=1) if v.strip()]
    empty = len(values) - len(filled)
    if empty:
        facts.append(f'column "{name}": {empty} empty value(s)')
    if not filled:
        return facts

    numbers = [(i, v.strip(), _num(v)) for i, v in filled]
    numeric = [(i, raw, x) for i, raw, x in numbers if x is not None]
    if len(numeric) >= 0.9 * len(filled):
        lowest = min(numeric, key=lambda t: t[2])
        highest = max(numeric, key=lambda t: t[2])
        xs = [x for _, _, x in numeric]
        median = statistics.median(xs)
        facts.append(
            f'column "{name}": numeric, {len(xs)} values, min {lowest[1]}, '
            f"max {highest[1]}, median {_fmt(median)}, mean {_fmt(statistics.fmean(xs))}"
        )
        facts += _outliers(name, numeric, median, row_label)
        return facts

    stamps = sorted(v.strip() for _, v in filled if ISO_TIMESTAMP.match(v.strip()))
    if len(stamps) >= 0.9 * len(filled):
        facts.append(f'column "{name}": timestamps, {len(stamps)} values, '
                     f"earliest {stamps[0]}, latest {stamps[-1]}")
        return facts

    counts = Counter(v.strip() for _, v in filled)
    if len(counts) == len(filled):
        facts.append(f'column "{name}": text, all {len(counts)} values are different')
        return facts
    top = ", ".join(f"{_short(v, 30)} ({c})" for v, c in counts.most_common(3))
    facts.append(f'column "{name}": text, {len(counts)} distinct value(s); most common: {top}')
    return facts


def _outliers(name, numeric, median, row_label) -> list[str]:
    """Values far from the rest, using the median absolute deviation (MAD).

    MAD is like a standard deviation that isn't thrown off by the outliers
    themselves. A value more than 5 "MADs" from the median is flagged.
    """
    if len(numeric) < 8:
        return []
    mad = statistics.median(abs(x - median) for _, _, x in numeric)
    if mad == 0:
        return []
    far = [(abs(x - median) / (1.4826 * mad), i, raw) for i, raw, x in numeric]
    far = sorted((t for t in far if t[0] > 5), reverse=True)[:MAX_OUTLIERS_PER_COLUMN]
    return [f'column "{name}": outlier {raw} on {row_label} {i} (median {_fmt(median)})'
            for _, i, raw in far]


# --- per-format facts ------------------------------------------------------------

def csv_facts(text: str, delimiter: str) -> list[str]:
    rows = list(csv.reader(io.StringIO(text), delimiter=delimiter))
    rows = [r for r in rows if any(cell.strip() for cell in r)]
    header, body = rows[0], rows[1:]
    facts = [
        f"rows: {len(body)} data rows plus 1 header row",
        f"columns ({len(header)}): " + ", ".join(_short(h, 30) for h in header[:MAX_COLUMNS]),
    ]
    bad = [n for n, r in enumerate(body, start=1) if len(r) != len(header)]
    if bad:
        facts.append(f"rows with the wrong number of columns: {len(bad)} (data rows "
                     + ", ".join(map(str, bad[:10])) + ")")
    duplicates = sum(c - 1 for c in Counter(tuple(r) for r in body).values() if c > 1)
    if duplicates:
        facts.append(f"duplicate rows: {duplicates}")
    for col, name in enumerate(header[:MAX_COLUMNS]):
        values = [r[col] if col < len(r) else "" for r in body]
        facts += profile_column(_short(name, 30), values, "data row")
    return facts


def _depth(value) -> int:
    if isinstance(value, dict):
        return 1 + max((_depth(v) for v in value.values()), default=0)
    if isinstance(value, list):
        return 1 + max((_depth(v) for v in value), default=0)
    return 0


def records_facts(records: list) -> list[str]:
    """Facts for a list of JSON objects (from a JSON array or JSON Lines)."""
    objects = [r for r in records if isinstance(r, dict)]
    facts = [f"records: {len(records)}"]
    if len(objects) != len(records):
        facts.append(f"records that are not objects: {len(records) - len(objects)}")
    if not objects:
        return facts
    key_counts = Counter(k for o in objects for k in o)
    facts.append("keys seen: " + ", ".join(f"{_short(k, 30)} ({c})"
                                           for k, c in key_counts.most_common(MAX_COLUMNS)))
    for key, count in key_counts.most_common(MAX_COLUMNS):
        if count < len(objects):
            missing = [n for n, o in enumerate(objects, start=1) if key not in o]
            facts.append(f'key "{_short(key, 30)}": missing in {len(missing)} record(s) '
                         f"(records {', '.join(map(str, missing[:10]))})")
        scalars = [o[key] for o in objects if key in o]   # missing keys reported above
        if all(v is None or isinstance(v, (str, int, float, bool)) for v in scalars):
            raw = ["" if v is None else (json.dumps(v) if isinstance(v, bool) else str(v))
                   for v in scalars]
            facts += profile_column(_short(key, 30), raw, "record")
    return facts


def json_facts(text: str) -> list[str]:
    data = json.loads(text)
    facts = [f"max nesting depth: {_depth(data)}"]
    if isinstance(data, list):
        facts.insert(0, f"top level: a list of {len(data)} item(s)")
        return facts + records_facts(data)
    if isinstance(data, dict):
        facts.insert(0, "top level: an object with keys "
                     + ", ".join(_short(k, 30) for k in list(data)[:MAX_COLUMNS]))
        for key, value in list(data.items())[:MAX_COLUMNS]:
            if isinstance(value, list) and value and all(isinstance(v, dict) for v in value):
                facts.append(f'key "{_short(key, 30)}" holds a list of {len(value)} object(s):')
                facts += records_facts(value)
    return facts


def jsonl_facts(text: str) -> list[str]:
    return records_facts([json.loads(l) for l in text.splitlines() if l.strip()])


def log_facts(text: str) -> list[str]:
    lines = [l for l in text.splitlines() if l.strip()]
    facts = []
    times = []
    for line in lines:
        match = ISO_TIMESTAMP.search(line[:40])
        if match:
            try:
                times.append((datetime.fromisoformat(match.group().replace("T", " ")), match.group()))
            except ValueError:
                pass
    if times:
        facts.append(f"timestamps found: {len(times)} of {len(lines)} lines")
        facts.append(f"first timestamp: {times[0][1]}; last timestamp: {times[-1][1]}")
        backwards = sum(1 for a, b in zip(times, times[1:]) if b[0] < a[0])
        if backwards:
            facts.append(f"timestamps that go backwards: {backwards}")
        if len(times) >= 3:
            gap, before = max((b[0] - a[0], a[1]) for a, b in zip(times, times[1:]))
            facts.append(f"largest gap between lines: {_duration(gap)} (after {before})")

    levels = Counter(m.group(1).upper().replace("WARNING", "WARN")
                     for l in lines for m in [LOG_LEVEL.search(l)] if m)
    if levels:
        facts.append("levels: " + ", ".join(f"{lvl} {n}" for lvl, n in levels.most_common()))
    facts += repeated_lines(lines, mask_numbers=True)
    return facts


def repeated_lines(lines: list[str], mask_numbers: bool) -> list[str]:
    """The most repeated lines. For logs, digits become # so that
    'retry 1', 'retry 2', 'retry 3' count as the same message."""
    def key(line):
        body = ISO_TIMESTAMP.sub("", line) if mask_numbers else line
        body = re.sub(r"\d+", "#", body) if mask_numbers else body
        return " ".join(body.split())
    counts = Counter(key(l) for l in lines)
    common = [(k, c) for k, c in counts.most_common(5) if c > 1 and k]
    if not common:
        return []
    label = "most repeated messages (numbers shown as #)" if mask_numbers else "most repeated lines"
    return [f"{label}:"] + [f'  {c} times: "{_short(k, 100)}"' for k, c in common]


def text_facts(text: str) -> list[str]:
    lines = [l for l in text.splitlines() if l.strip()]
    facts = [f"words: {len(text.split())}"]
    if lines:
        longest = max(range(len(lines)), key=lambda i: len(lines[i]))
        facts.append(f"longest line: {len(lines[longest])} characters")
    return facts + repeated_lines(lines, mask_numbers=False)


def safety_facts(text: str) -> list[str]:
    facts = []
    for n, line in enumerate(text.splitlines(), start=1):
        match = INSTRUCTION_LIKE.search(line)
        if match:
            facts.append(f'line {n} contains instruction-like text: "{_short(match.group(), 60)}"')
        if len(facts) >= 5:
            break
    if FACTS_HEADER.casefold() in text.casefold():
        facts.append("the input itself contains text imitating the facts header")
    return facts


# --- entry point --------------------------------------------------------------

def enrich(data, ctx):
    text = data.text
    fmt = data.meta.get("format", "text")
    lines = text.splitlines()
    facts = [
        f"format: {fmt}" + (f' (delimiter "{data.meta["delimiter"]}")'
                            if fmt == "csv" else ""),
        f"size: {len(text)} characters, {len(lines)} lines",
    ]
    try:
        if fmt == "csv":
            facts += csv_facts(text, data.meta["delimiter"])
        elif fmt == "json":
            facts += json_facts(text)
        elif fmt == "jsonl":
            facts += jsonl_facts(text)
        elif fmt == "log":
            facts += log_facts(text)
        else:
            facts += text_facts(text)
    except Exception as e:  # odd input must never stop the digest
        facts.append(f"some facts could not be computed ({type(e).__name__})")
    facts += safety_facts(text)

    if len(facts) > MAX_FACT_LINES:
        facts = facts[:MAX_FACT_LINES] + [f"(+{len(facts) - MAX_FACT_LINES} more facts not shown)"]

    # The facts go in the preamble: shown with every part when a big input
    # is split, so each part is read knowing the whole picture.
    tag = secrets.token_hex(4)
    data.preamble = (
        f"=== {FACTS_HEADER} (tag {tag}) ===\n" + "\n".join(facts)
        + "\n=== ORIGINAL CONTENT ==="
    )
    data.meta["facts"] = facts
    data.meta["facts_tag"] = tag
    return data
