"""Step 11 tests: the Update section. No internet: downloads are faked."""

import ast
import io
from pathlib import Path

import pytest

from core import audit
from core.cli import build_parser, cmd_update
from core.config import load_config
from core.refdata.db import RefDB
from core.updater import fetch as updater
from core.updater.fetch import UpdateError, _GuardedRedirect, check_url, download

ROOT = Path(__file__).parent.parent
REF = Path(__file__).parent / "fixtures" / "refdata"
HOSTS = {"standards-oui.ieee.org"}
FILE_FOR = {"MA-L": "oui.csv", "MA-M": "mam.csv", "MA-S": "oui36.csv", "IAB": "iab.csv",
            "CID": "cid.csv"}


@pytest.fixture
def config(tmp_path):
    c = load_config()
    c["paths"].update(reference_db=str(tmp_path / "ref.db"),
                      audit_log=str(tmp_path / "audit.jsonl"))
    return c


def fake_fetch(fail=()):
    """Pretend to download: serve the sample files by registry name."""
    calls = []

    def fetch(url, hosts, progress):
        calls.append(url)
        source = next(s for s in updater.load_sources() if s.url == url)
        if source.registry in fail:
            raise UpdateError("simulated network failure")
        data = (REF / FILE_FOR[source.registry]).read_bytes()
        progress(len(data), len(data))
        return data
    fetch.calls = calls
    return fetch


# --- the allow-list -------------------------------------------------------------------------

def test_sources_are_https_ieee_only():
    sources = updater.load_sources()
    assert {s.registry for s in sources} == {"MA-L", "MA-M", "MA-S", "IAB", "CID"}
    assert updater.allowed_hosts(sources) == HOSTS


@pytest.mark.parametrize("url", [
    "http://standards-oui.ieee.org/oui/oui.csv",          # not https
    "https://evil.example.com/oui.csv",                   # other host
    "https://standards-oui.ieee.org.evil.com/oui.csv",    # look-alike
    "https://user:pw@standards-oui.ieee.org/oui.csv",     # credentials
    "file:///etc/passwd",
])
def test_bad_urls_refused(url):
    with pytest.raises(UpdateError):
        check_url(url, HOSTS)


def test_redirect_off_the_allow_list_is_refused():
    handler = _GuardedRedirect(HOSTS)
    with pytest.raises(UpdateError):
        handler.redirect_request(None, None, 302, "Found", {}, "https://evil.example.com/x.csv")


def test_non_https_source_file_rejected(tmp_path):
    bad = tmp_path / "sources.toml"
    bad.write_text('[[source]]\nregistry = "MA-L"\ndataset = "oui"\nurl = "http://x/y.csv"\n')
    with pytest.raises(UpdateError, match="https"):
        updater.load_sources(bad)


# --- download mechanics (fake connection) ------------------------------------------------------

class FakeResponse(io.BytesIO):
    def __init__(self, data, length=None):
        super().__init__(data)
        self.headers = {"Content-Length": str(length if length is not None else len(data))}

    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass


def fake_opener(monkeypatch, response):
    class Opener:
        def open(self, request, timeout):
            assert request.full_url.startswith("https://standards-oui.ieee.org/")
            return response
    monkeypatch.setattr("core.updater.fetch.urllib.request.build_opener", lambda *h: Opener())


def test_download_reads_and_reports_progress(monkeypatch):
    fake_opener(monkeypatch, FakeResponse(b"x" * 200_000))
    seen = []
    data = download("https://standards-oui.ieee.org/oui/oui.csv", HOSTS,
                    progress=lambda done, total: seen.append((done, total)))
    assert len(data) == 200_000 and seen[-1] == (200_000, 200_000)


def test_download_size_caps(monkeypatch):
    fake_opener(monkeypatch, FakeResponse(b"x" * 10, length=10**9))      # announced too big
    with pytest.raises(UpdateError, match="too large"):
        download("https://standards-oui.ieee.org/oui/oui.csv", HOSTS)
    fake_opener(monkeypatch, FakeResponse(b"x" * 5000, length=0))        # lies about size
    with pytest.raises(UpdateError, match="exceeded"):
        download("https://standards-oui.ieee.org/oui/oui.csv", HOSTS, max_bytes=1000)


# --- update flow ---------------------------------------------------------------------------------

def test_status_before_and_after(config):
    assert all(s.rows is None for s in updater.status(config))
    updater.update(config, fetch=fake_fetch())
    after = {s.source.registry: s for s in updater.status(config)}
    assert after["MA-L"].rows == 6 and after["MA-L"].age_days < 1
    assert after["MA-L"].origin == "https://standards-oui.ieee.org/oui/oui.csv"


def test_update_all_imports_everything_and_audits(config):
    result = updater.update(config, fetch=fake_fetch())
    assert {r.registry for r in result.imported} == set(FILE_FOR) and not result.errors
    entries = audit.read_entries(Path(config["paths"]["audit_log"]))
    assert len(entries) == 5 and all(e["mode"] == "update" and e["status"] == "ok" for e in entries)
    assert entries[0]["sha256"].startswith("sha256:")


def test_one_failed_download_keeps_old_data(config):
    updater.update(config, fetch=fake_fetch())                       # first: all good
    result = updater.update(config, fetch=fake_fetch(fail={"MA-L"}))
    assert "MA-L" in result.errors and len(result.imported) == 4
    with RefDB(Path(config["paths"]["reference_db"])) as db:
        assert db.vendor("00:1b:a9:00:00:00").organization == "Example Printer Works"   # still there


def test_bad_download_content_is_rejected(config):
    def garbage(url, hosts, progress):
        return b"<html>Service unavailable</html>"
    result = updater.update(config, ["MA-L"], fetch=garbage)
    assert "unexpected columns" in result.errors["MA-L"]
    entry = audit.read_entries(Path(config["paths"]["audit_log"]))[-1]
    assert entry["status"] == "failed"


def test_choose_some_and_unknown(config):
    fetch = fake_fetch()
    result = updater.update(config, ["MA-S", "NOPE"], fetch=fetch)
    assert [r.registry for r in result.imported] == ["MA-S"]
    assert result.errors == {"NOPE": "not an allow-listed source"}
    assert len(fetch.calls) == 1


def test_cli_update(config, monkeypatch, capsys):
    monkeypatch.setattr("core.cli.load_config", lambda: config)
    assert cmd_update(build_parser().parse_args(["update"])) == 0
    assert "never" in capsys.readouterr().out
    assert cmd_update(build_parser().parse_args(["update", "--all"]), fetch=fake_fetch()) == 0
    out = capsys.readouterr().out
    assert "MA-L: 6 entries" in out and "only from the hosts" in out


# --- isolation: nothing but the updater and the Ollama backend may use the network -------------------

NETWORK_MODULES = {"urllib", "http", "socket", "requests", "ssl", "ftplib", "smtplib"}
ALLOWED = {ROOT / "core" / "backends" / "ollama.py", ROOT / "core" / "updater" / "fetch.py"}


def _imports(path):
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            yield from (a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            yield node.module


def test_only_two_files_can_use_the_network():
    for path in (ROOT / "core").rglob("*.py"):
        if path in ALLOWED:
            continue
        top = {name.split(".")[0] for name in _imports(path)}
        assert not top & NETWORK_MODULES, f"{path.relative_to(ROOT)} imports {top & NETWORK_MODULES}"


def test_run_path_never_imports_the_updater():
    """Only the CLI and the dashboard (user-facing buttons) may import the updater."""
    for path in (ROOT / "core").rglob("*.py"):
        rel = path.relative_to(ROOT).as_posix()
        if rel.startswith(("core/updater/", "core/tui/")) or rel == "core/cli.py":
            continue
        assert not any(n.startswith("core.updater") for n in _imports(path)), rel
