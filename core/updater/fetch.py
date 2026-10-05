"""Download reference data from the allow-listed sources, then import it.

Flow for each registry you choose:
    download (https, allow-listed host, size cap, timeout)
      -> check + import (core/refdata/importer.py: validated, atomic)
      -> audit log entry (URL, SHA-256, rows, or the error)
If a download or a check fails, that registry keeps its old data and the
others carry on.

Unlike the Ollama backend, downloads DO honour your system proxy settings
(HTTPS_PROXY), because reaching the internet from an office network often
needs them. TLS certificates are always verified.
"""

from __future__ import annotations

import tomllib
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable
from urllib.parse import urlsplit

from core import __version__, audit
from core.config import resolve_path
from core.refdata.db import RefDataError, RefDB
from core.refdata.importer import ImportResult, import_registry_data

SOURCES_FILE = Path(__file__).parent / "sources.toml"
MAX_DOWNLOAD_BYTES = 50_000_000
TIMEOUT_SECONDS = 60
READ_BLOCK = 64 * 1024

ProgressFn = Callable[[str, int, int | None], None]   # (registry, bytes so far, total or None)


class UpdateError(Exception):
    pass


@dataclass
class Source:
    registry: str
    dataset: str
    url: str
    about: str = ""


@dataclass
class SourceStatus:
    source: Source
    rows: int | None = None
    imported_at: str | None = None
    age_days: float | None = None
    origin: str | None = None


@dataclass
class UpdateResult:
    imported: list[ImportResult] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)   # registry -> message


# ---------------------------------------------------------------------------
# The allow-list
# ---------------------------------------------------------------------------

def load_sources(path: Path = SOURCES_FILE) -> list[Source]:
    with open(path, "rb") as f:
        raw = tomllib.load(f).get("source", [])
    sources = []
    for entry in raw:
        source = Source(entry["registry"], entry["dataset"], entry["url"], entry.get("about", ""))
        if urlsplit(source.url).scheme != "https":
            raise UpdateError(f"source {source.registry} must use https: {source.url}")
        sources.append(source)
    return sources


def allowed_hosts(sources: list[Source]) -> set[str]:
    return {urlsplit(s.url).hostname for s in sources}


def check_url(url: str, hosts: set[str]) -> None:
    parts = urlsplit(url)
    if parts.scheme != "https":
        raise UpdateError(f"refusing non-https address {url!r}")
    if parts.hostname not in hosts:
        raise UpdateError(f"refusing {parts.hostname!r}: not an allow-listed source host")
    if parts.username or parts.password:
        raise UpdateError("refusing an address with a username or password")


class _GuardedRedirect(urllib.request.HTTPRedirectHandler):
    """Follow a redirect only if the new address passes the same checks."""

    def __init__(self, hosts: set[str]):
        self.hosts = hosts

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        check_url(newurl, self.hosts)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def download(url: str, hosts: set[str], progress: Callable[[int, int | None], None] | None = None,
             timeout: float = TIMEOUT_SECONDS, max_bytes: int = MAX_DOWNLOAD_BYTES) -> bytes:
    """Fetch one allow-listed URL into memory, with a size cap."""
    check_url(url, hosts)
    opener = urllib.request.build_opener(_GuardedRedirect(hosts))
    request = urllib.request.Request(url, headers={"User-Agent": f"ai-core-console/{__version__}"})
    try:
        with opener.open(request, timeout=timeout) as response:
            length = response.headers.get("Content-Length")
            total = int(length) if length and length.isdigit() else None
            if total and total > max_bytes:
                raise UpdateError(f"file is too large ({total:,} bytes)")
            chunks, done = [], 0
            while True:
                block = response.read(READ_BLOCK)
                if not block:
                    break
                done += len(block)
                if done > max_bytes:
                    raise UpdateError(f"download exceeded {max_bytes:,} bytes; stopped")
                chunks.append(block)
                if progress:
                    progress(done, total)
            return b"".join(chunks)
    except urllib.error.HTTPError as e:
        raise UpdateError(f"server answered HTTP {e.code}") from None
    except urllib.error.URLError as e:
        reason = e.reason if not isinstance(e.reason, UpdateError) else str(e.reason)
        raise UpdateError(f"could not download: {reason}") from None
    except TimeoutError:
        raise UpdateError(f"no answer within {timeout:.0f}s") from None


# ---------------------------------------------------------------------------
# What the Update section shows, and what its button does
# ---------------------------------------------------------------------------

def status(config: dict) -> list[SourceStatus]:
    """Each allow-listed source with what's currently in the database."""
    current = {}
    try:
        with RefDB(resolve_path(config, "reference_db")) as db:
            current = {(s.dataset, s.registry): s for s in db.sources()}
    except RefDataError:
        pass
    result = []
    for source in load_sources():
        info = current.get((source.dataset, source.registry))
        result.append(SourceStatus(
            source, rows=info.rows if info else None,
            imported_at=info.imported_at if info else None,
            age_days=info.age_days() if info else None,
            origin=info.origin if info else None))
    return result


def update(config: dict, registries: list[str] | None = None,
           progress: ProgressFn | None = None, fetch=download) -> UpdateResult:
    """Download and import the chosen registries (all if None).

    `fetch` can be replaced in tests so they never touch the internet.
    """
    sources = load_sources()
    hosts = allowed_hosts(sources)
    chosen = [s for s in sources if registries is None or s.registry in registries]
    unknown = set(registries or []) - {s.registry for s in sources}
    result = UpdateResult(errors={r: "not an allow-listed source" for r in sorted(unknown)})
    db_path = resolve_path(config, "reference_db")

    for source in chosen:
        def report(done, total, _name=source.registry):
            if progress:
                progress(_name, done, total)
        entry = {"mode": "update", "registry": source.registry, "url": source.url}
        try:
            raw = fetch(source.url, hosts, report)
            imported = import_registry_data(db_path, [(raw, source.url)])[0]
            result.imported.append(imported)
            entry.update(status="ok", rows=imported.rows, sha256=imported.sha256,
                         bytes=len(raw))
        except (UpdateError, RefDataError) as e:
            result.errors[source.registry] = str(e)
            entry.update(status="failed", errors=[str(e)])
        audit.write_entry(resolve_path(config, "audit_log"), entry)
    return result
