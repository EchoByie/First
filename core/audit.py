"""Append-only audit log: one line of JSON per run (the "JSON Lines" format).

Every run, dry-run, refusal and failure is logged, so you can always answer
"what ran, when, with which model, and did the answer pass its checks?"

Privacy by design: the log never contains your collected data or the model's
answer. It stores *fingerprints* (SHA-256 hashes) of them instead. A hash lets
you prove later that a report matches what was sent, without the log itself
leaking MAC addresses or file contents.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def fingerprint(text: str) -> str:
    """A short SHA-256 fingerprint: same text -> same value, always."""
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_entry(log_path: Path, entry: dict) -> None:
    """Add one entry to the end of the log. Existing lines are never changed."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps({"logged_at": now_utc(), **entry}, ensure_ascii=False, default=str)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def read_entries(log_path: Path, last: int | None = None) -> list[dict]:
    """Read entries back (newest last). Damaged lines are skipped, not fatal."""
    log_path = Path(log_path)
    if not log_path.exists():
        return []
    entries = []
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries[-last:] if last else entries
