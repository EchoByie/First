"""Import IEEE registry CSV files into the reference database.

IEEE publishes one CSV per registry (oui.csv, mam.csv, oui36.csv, iab.csv,
cid.csv), all with the same columns:

    Registry,Assignment,Organization Name,Organization Address
    MA-L,002272,American Micro-Fuel Device Corp.,"2181 Buchanan Loop Ferndale WA US 98248"

Even files from IEEE are treated as untrusted input, because organization
names end up in prompts and reports. Each file is checked before anything is
written:
    - the header must have the four expected columns
    - every Assignment must be hex of the right length for its registry
    - one file holds exactly one registry
    - control characters are stripped and very long text is cut short
    - if more than 1% of rows are bad, the whole file is rejected
A good file REPLACES that registry's old rows completely (no stale leftovers),
inside a private copy that only goes live if every file imports cleanly.
"""

from __future__ import annotations

import csv
import hashlib
import io
import re
from dataclasses import dataclass, field
from pathlib import Path

from core.refdata.db import PREFIX_DIGITS, RefDataError, now_utc, writable_copy

DATASET = "oui"
EXPECTED_COLUMNS = ["registry", "assignment", "organization name", "organization address"]
MAX_FILE_BYTES = 50_000_000      # the full MA-L file is ~5 MB
MAX_BAD_FRACTION = 0.01
MAX_ORG_CHARS = 200
MAX_ADDRESS_CHARS = 300
_CONTROL = re.compile(r"[\x00-\x1f\x7f-\x9f​-‏ -‮⁠﻿]")


@dataclass
class ImportResult:
    registry: str
    rows: int
    skipped: int
    origin: str
    sha256: str
    problems: list[str] = field(default_factory=list)   # first few bad rows


def _clean(text: str, limit: int) -> str:
    text = " ".join(_CONTROL.sub(" ", text or "").split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def parse_registry_csv(raw: bytes, origin: str) -> tuple[ImportResult, list[tuple]]:
    """Check one file and return (summary, rows ready to insert).
    Raises RefDataError if the file must be rejected."""
    if len(raw) > MAX_FILE_BYTES:
        raise RefDataError(f"{origin}: file is too large")
    text = raw.decode("utf-8-sig", errors="replace")    # -sig: ignore a BOM
    reader = csv.reader(io.StringIO(text))
    try:
        header = [h.strip().lower() for h in next(reader)]
    except StopIteration:
        raise RefDataError(f"{origin}: file is empty") from None
    if header[:4] != EXPECTED_COLUMNS:
        raise RefDataError(f"{origin}: unexpected columns {header[:4]}; "
                           f"expected {EXPECTED_COLUMNS}")

    rows, problems, registries = {}, [], set()
    total = 0
    for line_number, row in enumerate(reader, start=2):
        if not any(cell.strip() for cell in row):
            continue
        total += 1
        if len(row) < 3:
            problems.append(f"line {line_number}: too few columns")
            continue
        registry = row[0].strip().upper()
        prefix = row[1].strip().upper()
        if registry not in PREFIX_DIGITS:
            problems.append(f"line {line_number}: unknown registry {registry[:20]!r}")
            continue
        if len(prefix) != PREFIX_DIGITS[registry] or not re.fullmatch(r"[0-9A-F]+", prefix):
            problems.append(f"line {line_number}: bad assignment {prefix[:20]!r} for {registry}")
            continue
        organization = _clean(row[2], MAX_ORG_CHARS)
        if not organization:
            problems.append(f"line {line_number}: no organization name")
            continue
        address = _clean(row[3], MAX_ADDRESS_CHARS) if len(row) > 3 else ""
        registries.add(registry)
        rows[prefix] = (prefix, registry, organization, address)   # duplicates: last wins

    if not rows:
        raise RefDataError(f"{origin}: no valid rows")
    if len(registries) != 1:
        raise RefDataError(f"{origin}: mixes registries {sorted(registries)}; expected one per file")
    bad = total - len(rows)
    if total and len(problems) / total > MAX_BAD_FRACTION:
        raise RefDataError(f"{origin}: {len(problems)} of {total} rows are invalid "
                           f"(first: {problems[0]})")
    summary = ImportResult(
        registry=registries.pop(), rows=len(rows), skipped=bad, origin=origin,
        sha256="sha256:" + hashlib.sha256(raw).hexdigest(), problems=problems[:5],
    )
    return summary, list(rows.values())


def import_registry_data(db_path: Path, files: list[tuple[bytes, str]]) -> list[ImportResult]:
    """Import several files in one go: [(file bytes, where it came from), ...].

    All files are checked first; if any is rejected, nothing is changed.
    Used by `aicore refdata import` now and by the updater in step 11.
    """
    parsed = [parse_registry_csv(raw, origin) for raw, origin in files]
    seen = [summary.registry for summary, _ in parsed]
    if len(seen) != len(set(seen)):
        raise RefDataError(f"the same registry was given twice: {sorted(seen)}")

    with writable_copy(db_path) as conn:
        for summary, rows in parsed:
            conn.execute("DELETE FROM oui WHERE registry = ?", (summary.registry,))
            conn.executemany("INSERT INTO oui VALUES (?, ?, ?, ?)", rows)
            conn.execute(
                "INSERT OR REPLACE INTO sources VALUES (?, ?, ?, ?, ?, ?)",
                (DATASET, summary.registry, summary.rows, summary.sha256, now_utc(),
                 summary.origin),
            )
        conn.execute("CREATE INDEX IF NOT EXISTS oui_prefix ON oui (prefix)")
    return [summary for summary, _ in parsed]


def import_files(db_path: Path, paths: list[Path]) -> list[ImportResult]:
    files = []
    for path in paths:
        path = Path(path)
        try:
            files.append((path.read_bytes(), str(path.resolve())))
        except OSError as e:
            raise RefDataError(f"can't read {path}: {e.strerror}") from None
    return import_registry_data(db_path, files)
