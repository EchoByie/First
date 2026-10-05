"""Step 3 tests: the model's answer is checked before anything uses it."""

import json

import pytest

from core.validation import (
    MAX_ANSWER_CHARS, SchemaFileError, check_answer, load_schema, parse_model_json,
)


def finding(**changes):
    f = {"claim": "the router is 192.168.1.1", "evidence": ["192.168.1.1"],
         "confidence": "high", "basis": "observed"}
    f.update(changes)
    return f


def answer(**changes):
    a = {"summary": "One device found.", "findings": [finding()]}
    a.update(changes)
    return a


# --- parsing ----------------------------------------------------------------

def test_plain_json_parses():
    data, error = parse_model_json(json.dumps(answer()))
    assert error is None and data["summary"] == "One device found."


def test_code_fence_is_removed():
    data, error = parse_model_json("```json\n" + json.dumps(answer()) + "\n```")
    assert error is None and data["findings"]


@pytest.mark.parametrize("text, words", [
    ("not json at all", "not valid JSON"),
    ('Sure! Here is the JSON: {"summary": "x"}', "not valid JSON"),  # no hunting inside text
    ("[1, 2, 3]", "JSON object"),
    ("", "not valid JSON"),
])
def test_bad_text_is_rejected(text, words):
    data, error = parse_model_json(text)
    assert data is None and words in error


def test_huge_answer_is_rejected():
    _, error = parse_model_json("x" * (MAX_ANSWER_CHARS + 1))
    assert "too long" in error


# --- schema checks ----------------------------------------------------------

def test_good_answer_passes():
    result = check_answer(json.dumps(answer()))
    assert result.ok and result.errors == []


@pytest.mark.parametrize("bad, where", [
    (answer(summary=""), "summary"),
    ({"findings": []}, "summary"),
    (answer(findings=[finding(confidence="sure")]), "findings[0].confidence"),
    (answer(findings=[finding(basis="guessed")]), "findings[0].basis"),
    (answer(findings=[finding(evidence=[])]), "findings[0].evidence"),
    (answer(findings=[finding(evidence="192.168.1.1")]), "findings[0].evidence"),
    (answer(findings=[finding(), finding(extra="field")]), "findings[1]"),
    (answer(findings=[{k: v for k, v in finding().items() if k != "claim"}]), "findings[0]"),
    (answer(findings=[finding(kind="Bad Kind!")]), "findings[0].kind"),
    (answer(findings=[finding()] * 201), "findings"),
])
def test_bad_answers_name_the_problem(bad, where):
    result = check_answer(json.dumps(bad))
    assert not result.ok
    assert any(where in e for e in result.errors), result.errors


def test_protocol_schema_is_also_applied():
    protocol_schema = {"type": "object", "required": ["devices"]}
    result = check_answer(json.dumps(answer()), protocol_schema)
    assert not result.ok and any("devices" in e for e in result.errors)
    assert check_answer(json.dumps(answer(devices=[])), protocol_schema).ok


def test_long_error_messages_are_trimmed():
    result = check_answer(json.dumps(answer(findings=[finding(confidence="x" * 5000)])))
    assert all(len(e) < 300 for e in result.errors)


def test_broken_schema_file_is_reported(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text('{"type": "not-a-type"}')
    with pytest.raises(SchemaFileError):
        load_schema(bad)
