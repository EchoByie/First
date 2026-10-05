"""Step 7 tests: splitting big inputs, worker -> analyst, evidence carried through."""

import json
import re
from pathlib import Path

import pytest
from fake_backend import FakeBackend

from core.backends.base import BackendError
from core.chunking import TooManyChunks, split_text
from core.collecting import UserInput
from core.config import load_config
from core.manifest import load_manifest
from core.pipeline import Pipeline, RunOptions

FIXTURE = Path(__file__).parent / "fixtures" / "protocols" / "example_ok"   # threshold 10k, parts 6k
DIGEST = Path(__file__).parent.parent / "protocols" / "digest"

# ~15,000 characters: 150 lines of 100 characters, one special line in the middle.
LINES = [f"{i:04d} routine reading ok " + "." * 74 for i in range(1, 151)]
LINES[79] = "0080 ALERT reactor temperature 999 degrees"
BIG = "\n".join(LINES)


# --- splitting ----------------------------------------------------------------------

def test_small_text_is_one_chunk():
    chunks = split_text("a\nb\nc", 100, 10)
    assert len(chunks) == 1 and chunks[0].first_line == 1 and chunks[0].last_line == 3


def test_chunks_respect_size_and_line_boundaries():
    chunks = split_text(BIG, 2000, 40)
    assert all(len(c.text) <= 2000 for c in chunks)
    rejoined = "\n".join(c.text for c in chunks)
    assert rejoined == BIG                                  # nothing lost or duplicated
    assert chunks[0].first_line == 1 and chunks[-1].last_line == 150
    for a, b in zip(chunks, chunks[1:]):
        assert b.first_line == a.last_line + 1              # no gaps, no overlaps


def test_giant_single_line_is_sliced():
    chunks = split_text("short\n" + "y" * 250 + "\nend", 100, 10)
    assert [c.text[:12] for c in chunks] == ["short", "y" * 12, "[continued] ", "[continued] ", "end"]
    assert chunks[1].label.startswith("PART 2 of 5 (line 2 ")


def test_too_many_chunks_refused():
    with pytest.raises(TooManyChunks, match="at most 3"):
        split_text(BIG, 1000, 3)


# --- the chunked pipeline ---------------------------------------------------------------

def finding(claim, quote, basis="observed", confidence="high"):
    return {"claim": claim, "evidence": [quote], "confidence": confidence, "basis": basis}


def smart_responder(fail_parts=()):
    """Answers like a model would: workers report on their part (one real
    note + one invented note), the analyst combines."""
    def respond(call):
        part = re.search(r"=== PART (\d+) of (\d+)", call["user"])
        if part:
            n = int(part.group(1))
            if n in fail_parts:
                return "garbage, not JSON"
            real = "0080 ALERT reactor temperature 999 degrees" if "ALERT" in call["user"] \
                else re.search(r"\d{4} routine reading ok", call["user"]).group()
            return json.dumps({"summary": f"Part {n} looks routine.", "findings": [
                finding("A reading from this part", real),
                finding("Pump 7 was replaced", "pump 7 replaced"),      # invented
            ]})
        # analyst combine step
        return json.dumps({"summary": "Routine readings except one alert.", "topic": "reactor",
                           "findings": [
            finding("The reactor reached 999 degrees", "ALERT reactor temperature 999 degrees"),
            finding("Every part looked routine", "Part 1 looks routine."),   # quotes the helper
        ]})
    return respond


def run(tmp_path, backend, text=BIG, config=None, dry_run=False, folder=FIXTURE):
    pipeline = Pipeline(config or load_config(), backend=backend,
                        reports_dir=tmp_path / "r", audit_path=tmp_path / "a.jsonl")
    return pipeline.run(load_manifest(folder),
                        RunOptions(user_input=UserInput(text=text), dry_run=dry_run))


def test_big_input_goes_through_worker_then_analyst(tmp_path):
    backend = FakeBackend(responder=smart_responder())
    result = run(tmp_path, backend)
    assert result.status == "ok", result.errors

    worker_calls = [c for c in backend.calls if "=== PART" in c["user"]]
    analyst_calls = [c for c in backend.calls if "NOTES FROM PART" in c["user"]]
    parts = result.report["stats"]["parts"]
    assert parts >= 3 and len(worker_calls) == parts and len(analyst_calls) == 1
    assert {c["model"] for c in worker_calls} == {"tiny:1b"}      # small model reads parts
    assert analyst_calls[0]["model"] == "big:70b"                 # big model combines
    assert result.report["models"] == {"worker": "tiny:1b", "analyst": "big:70b"}

    combine = analyst_calls[0]["user"]
    assert "pump 7 replaced" not in combine.lower()               # invented notes dropped
    assert "ALERT reactor temperature 999 degrees" in combine     # real note kept
    assert "were not in the data" in combine
    assert result.report["stats"]["notes_dropped"] == parts

    verdicts = {f["claim"]: f["verification"]["verdict"]
                for f in result.report["answer"]["findings"]}
    assert verdicts["The reactor reached 999 degrees"] == "SURE"
    assert verdicts["Every part looked routine"] == "THINK"       # helper's words aren't evidence
    assert any(f"read in {parts} parts" in c for c in result.report["caveats"])


def test_one_unreadable_part_is_reported_not_fatal(tmp_path):
    result = run(tmp_path, FakeBackend(responder=smart_responder(fail_parts={2})))
    assert result.status == "ok"
    assert result.report["stats"]["parts_failed"] == 1
    assert any("could not be read" in c and "PART 2" in c for c in result.report["caveats"])


def test_most_parts_unreadable_fails_run(tmp_path):
    result = run(tmp_path, FakeBackend(responder=smart_responder(fail_parts={1, 2, 3})))
    assert result.status == "failed" and "could not be read" in result.errors[0]


def test_backend_down_mid_run_stops_immediately(tmp_path):
    calls = []

    def respond(call):
        calls.append(call)
        if len(calls) == 2:
            raise BackendError("Ollama went away")
        return smart_responder()(call)
    result = run(tmp_path, FakeBackend(responder=respond))
    assert result.status == "failed" and "went away" in result.errors[0]
    assert len(calls) == 2                       # didn't keep hammering a dead server


def test_without_worker_model_analyst_reads_parts(tmp_path):
    config = load_config()
    config["roles"] = {"analyst": "auto", "worker": "not-installed:1b"}
    backend = FakeBackend(responder=smart_responder())
    result = run(tmp_path, backend, config=config)
    assert result.status == "ok"
    assert {c["model"] for c in backend.calls} == {"big:70b"}
    assert any("No worker model" in c for c in result.report["caveats"])


def test_dry_run_shows_part_plan_and_sends_nothing(tmp_path):
    backend = FakeBackend()
    result = run(tmp_path, backend, dry_run=True)
    assert result.status == "dry-run" and backend.calls == []
    assert result.preview["path"] == "chunked"
    assert result.preview["parts"][0].startswith("PART 1 of")
    assert "=== PART 1 of" in result.preview["user"]


def test_dry_run_chunked_works_offline(tmp_path):
    result = run(tmp_path, FakeBackend(fail=True), dry_run=True)
    assert result.status == "dry-run" and result.preview["parts"]


# --- digest with a big log: facts shared with every part ------------------------------------------

def test_digest_facts_reach_every_worker(tmp_path):
    log = "\n".join(f"2024-05-01 {h:02d}:{m:02d}:00 INFO heartbeat {h * 60 + m} " + "-" * 40
                    for h in range(6) for m in range(60))      # 360 lines, ~27k chars
    def respond(call):
        answer = {"summary": "Heartbeats.",
                  "findings": [{"kind": "observation", "claim": "There are 360 timestamps",
                                "evidence": ["timestamps found: 360 of 360 lines"],
                                "confidence": "high", "basis": "observed"}]}
        if "NOTES FROM PART" in call["user"]:      # only the final answer has next_steps
            answer["next_steps"] = []
        return json.dumps(answer)
    backend = FakeBackend(responder=respond)
    result = run(tmp_path, backend, text=log, folder=DIGEST)
    assert result.status == "ok", result.errors
    worker_calls = [c for c in backend.calls if "=== PART" in c["user"]]
    assert len(worker_calls) >= 3
    assert all("FACTS COMPUTED BY CODE" in c["user"] for c in worker_calls)
    assert all("Also:" in c["system"] for c in worker_calls)          # digest worker hints
    # the fact quoted by workers survives into the combine step and verifies
    assert result.report["answer"]["findings"][0]["verification"]["verdict"] == "SURE"
