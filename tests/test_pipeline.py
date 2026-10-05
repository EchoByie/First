"""Step 5 tests: the full pipeline, run with a fake model (no Ollama)."""

import json
import shutil
import sys
from pathlib import Path

import pytest
from fake_backend import FakeBackend

from core import audit
from core.cli import build_parser, cmd_run
from core.collecting import CollectorContext, CollectorError, UserInput
from core.config import load_config
from core.manifest import load_manifest
from core.pipeline import Pipeline, RunOptions
from core.prompting import build_messages
from core.safe_exec import CommandNotAllowed, run_allowed

FIXTURE = Path(__file__).parent / "fixtures" / "protocols" / "example_ok"
DATA = "The server room temperature was 31C at 02:00.\nFan 3 reported FAILED."


def good_answer(**changes):
    a = {
        "summary": "A fan failed and the room was warm.",
        "topic": "server room",
        "findings": [
            {"claim": "Fan 3 failed", "evidence": ["Fan 3 reported FAILED"],
             "confidence": "high", "basis": "observed"},
            {"claim": "The heat may be caused by the failed fan",
             "evidence": ["31C", "Fan 3 reported FAILED"],
             "confidence": "medium", "basis": "inferred"},
        ],
        "caveats": ["Only one night of data."],
    }
    a.update(changes)
    return json.dumps(a)


@pytest.fixture
def setup(tmp_path):
    """Returns a function making (pipeline, manifest, backend) with temp output dirs."""
    def make(replies=(), backend=None, manifest_folder=FIXTURE, progress=None):
        backend = backend or FakeBackend(replies=list(replies))
        pipeline = Pipeline(load_config(), backend=backend, progress=progress,
                            reports_dir=tmp_path / "reports",
                            audit_path=tmp_path / "audit.jsonl")
        return pipeline, load_manifest(manifest_folder), backend
    return make


def text_input(text=DATA):
    return RunOptions(user_input=UserInput(text=text))


# --- happy path -----------------------------------------------------------------

def test_full_run_produces_verified_report(setup, tmp_path):
    stages = []
    pipeline, manifest, backend = setup([good_answer()], progress=lambda s, m: stages.append(s))
    result = pipeline.run(manifest, text_input())

    assert result.status == "ok", result.errors
    answer = result.report["answer"]
    assert [f["verification"]["verdict"] for f in answer["findings"]] == ["SURE", "THINK"]
    assert result.report["models"]["analyst"] == "big:70b"
    # code caveats come first, then the model's (labelled)
    assert result.report["caveats"] == ["manifest caveat", "collector caveat",
                                         "(model) Only one night of data."]
    # stages ran in order
    assert stages[:2] == ["preflight", "collect"] and "verify" in stages and stages[-1] == "report"

    json_path, md_path = result.report_paths
    assert json.loads(json_path.read_text())["run_id"] == result.run_id
    md = md_path.read_text()
    assert "I'm sure that fan 3 failed." in md and "## ? Not sure" in md


def test_model_sees_enriched_data_inside_markers(setup):
    pipeline, manifest, backend = setup([good_answer()])
    pipeline.run(manifest, text_input())
    call = backend.calls[0]
    assert "Fan 3 reported FAILED" in call["user"]
    assert "[line count: 2]" in call["user"]            # enricher ran
    assert call["user"].startswith("<<<DATA ") and "<<<END DATA " in call["user"]
    assert "List the facts in the data." in call["system"]   # protocol task
    assert "UNTRUSTED" in call["system"]                     # framework rules
    assert "topic" in call["schema"]["properties"]           # protocol schema merged in
    assert call["schema"]["properties"]["findings"]["items"]["required"]


def test_marker_code_is_random_each_run():
    s1, u1 = build_messages("t", "d")
    s2, u2 = build_messages("t", "d")
    assert u1 != u2


# --- validation retry ---------------------------------------------------------------

def test_invalid_answer_is_retried_with_errors(setup):
    bad = json.dumps({"summary": "x", "findings": [{"claim": "c"}]})  # missing fields, no topic
    pipeline, manifest, backend = setup([bad, good_answer()])
    result = pipeline.run(manifest, text_input())
    assert result.status == "ok"
    assert len(backend.calls) == 2
    assert "PREVIOUS ANSWER WAS REJECTED" in backend.calls[1]["system"]
    assert "findings[0]" in backend.calls[1]["system"]


def test_two_invalid_answers_fail_with_report(setup):
    pipeline, manifest, _ = setup(["not json", "still not json"])
    result = pipeline.run(manifest, text_input())
    assert result.status == "failed"
    assert "failed validation twice" in result.errors[0]
    assert result.report["rejected_answer"] == "still not json"
    assert result.report_paths[1].read_text().count("Run did not complete") == 1


# --- refusals and failures -------------------------------------------------------------

def test_missing_input_is_refused(setup):
    pipeline, manifest, backend = setup()
    result = pipeline.run(manifest, RunOptions())
    assert result.status == "refused" and "needs a file or text" in result.errors[0]
    assert backend.calls == [] and result.report is None


def test_high_risk_needs_confirmation(setup, tmp_path):
    folder = tmp_path / "example_ok"
    shutil.copytree(FIXTURE, folder)
    manifest_file = folder / "manifest.toml"
    manifest_file.write_text(manifest_file.read_text().replace('risk_level = "low"', 'risk_level = "high"'))
    pipeline, manifest, backend = setup([good_answer()], manifest_folder=folder)
    assert pipeline.run(manifest, text_input()).status == "refused"
    confirmed = RunOptions(user_input=UserInput(text=DATA), confirmed=True)
    assert pipeline.run(manifest, confirmed).status == "ok"


def test_backend_down_fails_cleanly(setup):
    pipeline, manifest, _ = setup(backend=FakeBackend(fail=True))
    result = pipeline.run(manifest, text_input())
    assert result.status == "failed" and "down" in result.errors[0]


def test_crashing_collector_does_not_crash_console(setup, tmp_path):
    folder = tmp_path / "example_ok"
    shutil.copytree(FIXTURE, folder)
    (folder / "collector.py").write_text("def collect(ctx):\n    return 1 / 0\n")
    pipeline, manifest, _ = setup(manifest_folder=folder)
    result = pipeline.run(manifest, text_input())
    assert result.status == "failed" and "ZeroDivisionError" in result.errors[0]


def test_input_needing_too_many_parts_is_stopped_before_model(setup):
    pipeline, manifest, backend = setup([good_answer()])
    lines = "\n".join(f"line {i} " + "x" * 90 for i in range(5000))   # ~500 KB
    result = pipeline.run(manifest, text_input(lines))
    assert result.status == "failed" and "at most 40" in result.errors[0]
    assert backend.calls == []


# --- dry-run ---------------------------------------------------------------------------------

def test_dry_run_sends_nothing(setup):
    pipeline, manifest, backend = setup()
    result = pipeline.run(manifest, RunOptions(user_input=UserInput(text=DATA), dry_run=True))
    assert result.status == "dry-run" and backend.calls == []
    assert "Fan 3 reported FAILED" in result.preview["user"]
    assert result.preview["models"]["analyst"] == "big:70b"
    assert result.report is None


def test_dry_run_works_with_backend_down(setup):
    pipeline, manifest, _ = setup(backend=FakeBackend(fail=True))
    result = pipeline.run(manifest, RunOptions(user_input=UserInput(text=DATA), dry_run=True))
    assert result.status == "dry-run"
    assert "unreachable" in result.preview["models"]["analyst"]


# --- audit log -----------------------------------------------------------------------------------

def test_every_run_is_audited_without_raw_data(setup, tmp_path):
    pipeline, manifest, _ = setup([good_answer()])
    pipeline.run(manifest, text_input())
    pipeline.run(manifest, RunOptions())                                   # refused
    pipeline.run(manifest, RunOptions(user_input=UserInput(text=DATA), dry_run=True))

    entries = audit.read_entries(tmp_path / "audit.jsonl")
    assert [e["status"] for e in entries] == ["ok", "refused", "dry-run"]
    ok = entries[0]
    assert ok["models"] == {"analyst": "big:70b", "worker": "tiny:1b"}
    assert ok["verification"]["sure"] == 1
    assert ok["data_fingerprint"].startswith("sha256:")
    log_text = (tmp_path / "audit.jsonl").read_text()
    assert "Fan 3" not in log_text                     # no raw data in the log


def test_damaged_audit_lines_are_skipped(tmp_path):
    log = tmp_path / "a.jsonl"
    audit.write_entry(log, {"n": 1})
    with open(log, "a") as f:
        f.write("{broken\n")
    audit.write_entry(log, {"n": 2})
    assert [e["n"] for e in audit.read_entries(log)] == [1, 2]


# --- markdown safety ---------------------------------------------------------------------------------

def test_model_text_cannot_inject_markdown(setup):
    sneaky = good_answer(summary="ok\n\n# HACKED\n<script>x</script> [link](http://evil)")
    pipeline, manifest, _ = setup([sneaky])
    md = pipeline.run(manifest, text_input()).report_paths[1].read_text()
    assert "\n# HACKED" not in md and "<script>" not in md


# --- collector context + safe_exec ----------------------------------------------------------------------

def test_collector_context_enforces_allow_lists(tmp_path):
    manifest = load_manifest(FIXTURE)          # allows /etc/hostname and ["echo", "hi"]
    ctx = CollectorContext(manifest, "linux")
    with pytest.raises(CollectorError):
        ctx.read_file("/etc/passwd")
    with pytest.raises(CommandNotAllowed):
        ctx.run(["echo", "hi", "; rm -rf ~"])
    with pytest.raises(CollectorError):
        ctx.user_input_text()                  # no input given


def test_safe_exec_runs_exact_command_without_shell():
    cmd = [sys.executable, "-c", "print('hello; $(whoami)')"]
    assert run_allowed(cmd, [cmd]).strip() == "hello; $(whoami)"   # printed literally
    with pytest.raises(CommandNotAllowed):
        run_allowed(cmd + ["extra"], [cmd])


# --- CLI ------------------------------------------------------------------------------------------------

def test_cli_run_and_dry_run(capsys, tmp_path, monkeypatch):
    monkeypatch.setattr("core.cli.load_config", lambda: _temp_config(tmp_path))
    protocols = str(FIXTURE.parent)
    args = build_parser().parse_args(["dry-run", "example_ok", "--text", DATA,
                                      "--protocols-dir", protocols])
    assert cmd_run(args, backend=FakeBackend()) == 0
    assert "Nothing was sent" in capsys.readouterr().out

    data_file = tmp_path / "data.txt"
    data_file.write_text(DATA)
    args = build_parser().parse_args(["run", "example_ok", "--file", str(data_file),
                                      "--protocols-dir", protocols])
    assert cmd_run(args, backend=FakeBackend(replies=[good_answer()])) == 0
    out = capsys.readouterr().out
    assert "✔" in out and "I'm sure that fan 3 failed." in out and "report:" in out


def _temp_config(tmp_path):
    config = load_config()
    config["paths"] = dict(config["paths"], reports=str(tmp_path / "r"),
                           audit_log=str(tmp_path / "a.jsonl"))
    return config
