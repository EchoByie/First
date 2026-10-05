"""Step 10 tests: health checks (fake backend, temp folders, no network)."""

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fake_backend import FakeBackend

from core.backends.base import ModelInfo
from core.cli import build_parser, cmd_health
from core.config import load_config
from core.discovery import discover
from core.health import FAIL, OK, WARN, HealthReport, CheckResult, parse_version
from core.health import deps, ollama, protocols, refdata
from core.health.run import run_all
from core.refdata.importer import import_files

ROOT = Path(__file__).parent.parent
REF = Path(__file__).parent / "fixtures" / "refdata"
FIXTURE_PROTOCOLS = Path(__file__).parent / "fixtures" / "protocols"


@pytest.fixture
def config(tmp_path):
    c = load_config()
    c["paths"].update(reference_db=str(tmp_path / "ref.db"), reports=str(tmp_path / "reports"),
                      audit_log=str(tmp_path / "logs" / "audit.jsonl"))
    return c


def statuses(report, area):
    return {r.name: r.status for r in report.results if r.area == area}


# --- small pieces -------------------------------------------------------------------

@pytest.mark.parametrize("text, parsed", [
    ("0.6.2", (0, 6, 2)), ("v0.5.0-rc1", (0, 5, 0)), ("12", (12,)), ("fake", ()),
])
def test_parse_version(text, parsed):
    assert parse_version(text) == parsed


def test_overall_is_worst():
    report = HealthReport([CheckResult("a", "x", OK, ""), CheckResult("b", "y", WARN, "")])
    assert report.overall == WARN
    report.results.append(CheckResult("b", "z", FAIL, ""))
    assert report.overall == FAIL and report.area_status("a") == OK


# --- ollama ----------------------------------------------------------------------------

def test_ollama_ok(config):
    results, backend = ollama.check(config, FakeBackend())
    assert [r.status for r in results] == [OK, OK] and backend is not None


def test_ollama_too_old(config):
    class Old(FakeBackend):
        def version(self):
            return "0.3.14"
    results, _ = ollama.check(config, Old())
    assert results[1].status == FAIL and "too old" in results[1].message


def test_ollama_down(config):
    results, backend = ollama.check(config, FakeBackend(fail=True))
    assert results[0].status == FAIL and backend is None and "ollama serve" in results[0].fix


def test_non_local_backend_address_fails(config):
    config["backend"]["url"] = "http://10.1.2.3:11434"
    results, _ = ollama.check(config)
    assert results[0].status == FAIL and results[0].name == "address"


# --- models -------------------------------------------------------------------------------

def test_models_all_roles_filled(config):
    report = run_all(config, FakeBackend(), protocols_dir=ROOT / "protocols")
    s = statuses(report, "models")
    assert s["installed"] == OK
    assert s["digest / analyst"] == OK and s["network_map / worker"] == OK


def test_models_missing_required_role(config):
    tiny = FakeBackend(models=[ModelInfo("tiny:1b", "x", 1.0, 2048, ["completion"])])
    s = statuses(run_all(config, tiny, protocols_dir=ROOT / "protocols"), "models")
    assert s["digest / analyst"] == FAIL           # needs 8192 tokens of context
    assert s["digest / worker"] == WARN            # optional


def test_models_none_installed(config):
    s = statuses(run_all(config, FakeBackend(models=[]), protocols_dir=ROOT / "protocols"), "models")
    assert s["installed"] == FAIL


# --- reference data ----------------------------------------------------------------------------

def test_refdata_missing_fails_when_needed(config):
    found = discover(ROOT / "protocols").protocols
    assert refdata.check(config, found)[0].status == FAIL
    only_digest = {"digest": found["digest"]}                 # needs no refdata
    assert refdata.check(config, only_digest)[0].status == WARN


def test_refdata_fresh_and_stale(config, monkeypatch):
    import_files(Path(config["paths"]["reference_db"]), [REF / "oui.csv", REF / "mam.csv"])
    found = discover(ROOT / "protocols").protocols
    s = {r.name: r for r in refdata.check(config, found)}
    assert s["database"].status == OK and s["oui"].status == OK

    # pretend 120 days have passed (network_map allows 90)
    from core.refdata import db as dbmod
    later = datetime.now(timezone.utc) + timedelta(days=120)
    monkeypatch.setattr(dbmod.SourceInfo, "age_days",
                        lambda self, now=None: (later - datetime.fromisoformat(self.imported_at)).days)
    s = {r.name: r for r in refdata.check(config, found)}
    assert s["oui"].status == WARN and "limit 90 days" in s["oui"].message


def test_refdata_without_essential_registry(config):
    import_files(Path(config["paths"]["reference_db"]), [REF / "mam.csv"])   # no MA-L
    found = discover(ROOT / "protocols").protocols
    s = {r.name: r for r in refdata.check(config, found)}
    assert s["oui"].status == FAIL and "MA-L" in s["oui"].message


# --- protocols, deps, paths ----------------------------------------------------------------------

def test_protocols_ready_and_broken(tmp_path):
    (tmp_path / "broken").mkdir()
    (tmp_path / "broken" / "manifest.toml").write_text('name = "broken"\n')
    results = protocols.check(discover(tmp_path))
    assert results[0].status == FAIL and results[0].name == "broken"


def test_protocol_with_missing_program(tmp_path, monkeypatch):
    found = discover(ROOT / "protocols")
    monkeypatch.setattr("core.health.protocols.current_platform", lambda: "windows")
    monkeypatch.setattr("core.health.protocols.shutil.which", lambda name: None)
    s = {r.name: r for r in protocols.check(found)}
    assert s["network_map"].status == FAIL and "program arp" in s["network_map"].message
    assert s["digest"].status == OK


def test_deps_reads_minimums_from_pyproject():
    names = {r.name: r.status for r in deps.check()}
    assert names["python"] == OK
    assert {"jsonschema", "rich", "textual"} <= set(names)
    assert all(status == OK for status in names.values())


def test_paths_not_writable(config, tmp_path):
    locked = tmp_path / "locked"
    locked.mkdir()
    os.chmod(locked, 0o500)
    config["paths"]["reports"] = str(locked)
    try:
        if os.access(locked, os.W_OK):          # running as root: permissions don't apply
            pytest.skip("cannot test unwritable folders as root")
        report = run_all(config, FakeBackend(), protocols_dir=FIXTURE_PROTOCOLS)
        assert statuses(report, "paths")["reports"] == FAIL
    finally:
        os.chmod(locked, 0o700)


# --- CLI ---------------------------------------------------------------------------------------------

def test_cli_health_json(config, monkeypatch, capsys):
    monkeypatch.setattr("core.cli.load_config", lambda: config)
    args = build_parser().parse_args(["health", "--json", "--protocols-dir", str(ROOT / "protocols")])
    code = cmd_health(args, backend=FakeBackend())
    data = json.loads(capsys.readouterr().out)
    assert data["overall"] == FAIL and code == 1           # refdata missing for network_map
    assert any(r["area"] == "refdata" and r["status"] == FAIL for r in data["results"])

    import_files(Path(config["paths"]["reference_db"]), [REF / "oui.csv"])
    code = cmd_health(args, backend=FakeBackend())
    data = json.loads(capsys.readouterr().out)
    assert data["overall"] in (OK, WARN) and code == 0
