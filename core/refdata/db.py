"""The reference database: data/reference.db (SQLite).

Why SQLite? It's built into Python (no server, no install), stores
everything in one file, and looks up a MAC prefix among ~50,000 entries
instantly. You can also open it with any SQLite viewer to see what's in it.

Tables:
    meta      schema version
    sources   one row per imported file: dataset, registry, rows, SHA-256,
              when imported, where it came from
    oui       prefix -> organization, for each IEEE registry

Safety:
    - Runs open the database READ-ONLY. Only the importer/updater writes.
    - Writes never edit the live file in place. They build a new copy and
      swap it in with one atomic rename (see replace_atomically), so a
      crash or a bad download can never leave you with half a database.
"""

from __future__ import annotations

import os
import shutil
import sqlite3
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sources (
    dataset      TEXT NOT NULL,     -- e.g. "oui"
    registry     TEXT NOT NULL,     -- e.g. "MA-L"
    rows         INTEGER NOT NULL,
    sha256       TEXT NOT NULL,     -- fingerprint of the imported file
    imported_at  TEXT NOT NULL,     -- UTC, ISO format
    origin       TEXT NOT NULL,     -- file path or URL it came from
    PRIMARY KEY (dataset, registry)
);
CREATE TABLE IF NOT EXISTS oui (
    prefix       TEXT NOT NULL,     -- uppercase hex, 6 / 7 / 9 digits
    registry     TEXT NOT NULL,     -- MA-L, MA-M, MA-S, IAB, CID
    organization TEXT NOT NULL,
    address      TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (prefix, registry)
);
"""

# How many hex digits of the MAC each registry assigns:
#   MA-L "large"  = 24 bits = 6 hex digits  (the classic "OUI")
#   MA-M "medium" = 28 bits = 7 hex digits
#   MA-S "small"  = 36 bits = 9 hex digits
#   IAB           = 36 bits = 9 hex digits  (older, replaced by MA-S)
#   CID           = 24 bits = 6 hex digits  (company IDs, not for MACs
#                   from the factory; seen in some "local" addresses)
PREFIX_DIGITS = {"MA-L": 6, "MA-M": 7, "MA-S": 9, "IAB": 9, "CID": 6}


class RefDataError(Exception):
    pass


@dataclass
class SourceInfo:
    dataset: str
    registry: str
    rows: int
    sha256: str
    imported_at: str
    origin: str

    def age_days(self, now: datetime | None = None) -> float:
        now = now or datetime.now(timezone.utc)
        return (now - datetime.fromisoformat(self.imported_at)).total_seconds() / 86400


@dataclass
class VendorMatch:
    prefix: str
    registry: str
    organization: str
    address: str


def normalise_mac(mac: str) -> str | None:
    """'a4:2b:b0:11:22:33', 'A4-2B-B0-11-22-33', 'a42b.b011.2233' -> 'A42BB0112233'.
    Returns None if it isn't a 48-bit MAC address."""
    digits = "".join(ch for ch in str(mac) if ch not in ":-. ").upper()
    if len(digits) != 12 or any(ch not in "0123456789ABCDEF" for ch in digits):
        return None
    return digits


def create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.execute("INSERT OR REPLACE INTO meta VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),))


class RefDB:
    """Read-only access for runs and health checks."""

    def __init__(self, path: Path):
        self.path = Path(path)
        if not self.path.exists():
            raise RefDataError(f"no reference database at {self.path} "
                               "(import data with: aicore refdata import <files>)")
        # mode=ro: SQLite itself refuses any write through this connection.
        self.conn = sqlite3.connect(f"file:{self.path.as_posix()}?mode=ro", uri=True)
        version = self.conn.execute(
            "SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
        if not version or int(version[0]) != SCHEMA_VERSION:
            raise RefDataError(f"{self.path} has an unexpected format; re-import the data")

    def close(self) -> None:
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def sources(self, dataset: str | None = None) -> list[SourceInfo]:
        query = "SELECT dataset, registry, rows, sha256, imported_at, origin FROM sources"
        args: tuple = ()
        if dataset:
            query += " WHERE dataset = ?"
            args = (dataset,)
        return [SourceInfo(*row) for row in self.conn.execute(query + " ORDER BY dataset, registry", args)]

    def vendor(self, mac: str, include_cid: bool = False) -> VendorMatch | None:
        """Find who a MAC address was assigned to.

        Longest match wins: a 9-digit MA-S block is more specific than the
        6-digit MA-L block it sits inside (which is often registered to
        "IEEE Registration Authority" itself).
        """
        digits = normalise_mac(mac)
        if digits is None:
            return None
        candidates = {digits[:n] for n in set(PREFIX_DIGITS.values())}
        registries = [r for r in PREFIX_DIGITS if include_cid or r != "CID"]
        rows = self.conn.execute(
            f"SELECT prefix, registry, organization, address FROM oui "
            f"WHERE prefix IN ({','.join('?' * len(candidates))}) "
            f"AND registry IN ({','.join('?' * len(registries))}) "
            f"ORDER BY length(prefix) DESC, registry LIMIT 1",
            (*candidates, *registries),
        ).fetchone()
        return VendorMatch(*rows) if rows else None


@contextmanager
def writable_copy(path: Path):
    """Give the importer a private copy of the database to write into.

    On success the copy atomically replaces the real file; on any error the
    copy is deleted and the real file is untouched.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".reference-", suffix=".db", dir=path.parent)
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        if path.exists():
            shutil.copyfile(path, tmp)
        else:
            tmp.unlink()            # let SQLite create a fresh file
        conn = sqlite3.connect(tmp)
        try:
            create_schema(conn)
            yield conn
            conn.commit()
        finally:
            conn.close()
        replace_atomically(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def replace_atomically(new_file: Path, target: Path) -> None:
    """os.replace is atomic: readers see either the old file or the new
    one, never a mix. Works on Linux and Windows."""
    os.replace(new_file, target)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
