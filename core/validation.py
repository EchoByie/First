"""Check the model's answer before anything uses it.

The model's answer is untrusted, just like the collected data. It might be
broken JSON, it might be missing fields, or a prompt-injection trick in the
data might have talked it into answering in some other shape. So every answer
passes three checks, in this order:

  1. Is it JSON at all?                       -> parse_model_json()
  2. Does it meet the base contract?          -> core/schemas/model_output.schema.json
     And is every finding a proper finding?   -> core/schemas/finding.schema.json
  3. Does it meet the protocol's own schema?  -> protocols/<name>/output.schema.json

Errors are collected as plain sentences such as
    "findings[2].confidence: 'sure' is not one of ['low', 'medium', 'high']"
so they can be shown to you, written to the audit log, and sent back to the
model on a retry ("your last answer had these problems: ...").
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

SCHEMA_DIR = Path(__file__).parent / "schemas"

# Never try to parse an absurdly large answer. A real report is a few KB.
MAX_ANSWER_CHARS = 200_000


class SchemaFileError(Exception):
    """A schema file itself is broken (our mistake, not the model's)."""


@dataclass
class ValidationResult:
    ok: bool
    data: dict | None = None            # the parsed answer, only if ok
    errors: list[str] = field(default_factory=list)


def load_schema(path: Path) -> dict:
    """Read a JSON schema file and make sure it is a valid schema."""
    try:
        schema = json.loads(Path(path).read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(schema)
    except (OSError, json.JSONDecodeError, SchemaError) as e:
        raise SchemaFileError(f"{path}: {e}") from e
    return schema


# Loaded once when this module is imported; they never change while running.
BASE_SCHEMA = load_schema(SCHEMA_DIR / "model_output.schema.json")
FINDING_SCHEMA = load_schema(SCHEMA_DIR / "finding.schema.json")


def parse_model_json(text: str) -> tuple[dict | None, str | None]:
    """Turn the model's raw text into a dict. Returns (data, error).

    Only one forgiveness: models often wrap JSON in a Markdown code fence
    (```json ... ```), so we remove that. We deliberately do NOT go hunting
    for JSON inside other text. If the answer isn't clean JSON, that is a
    failure and the pipeline will retry, rather than guessing what was meant.
    """
    if not isinstance(text, str):
        return None, "answer is not text"
    if len(text) > MAX_ANSWER_CHARS:
        return None, f"answer is too long ({len(text)} characters)"
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if len(lines) >= 2 and lines[-1].strip() == "```":
            cleaned = "\n".join(lines[1:-1]).strip()  # drop first and last line
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        return None, f"answer is not valid JSON ({e.msg} at line {e.lineno} column {e.colno})"
    if not isinstance(data, dict):
        return None, f"answer must be a JSON object, got {type(data).__name__}"
    return data, None


def _where(error) -> str:
    """Turn a jsonschema error location like ['findings', 2, 'basis'] into
    the readable 'findings[2].basis'."""
    text = ""
    for part in error.absolute_path:
        text += f"[{part}]" if isinstance(part, int) else (f".{part}" if text else str(part))
    return text or "(top level)"


def _schema_errors(data, schema: dict, prefix: str = "") -> list[str]:
    validator = Draft202012Validator(schema)
    messages = []
    # Sort so the same answer always gives errors in the same order.
    for error in sorted(validator.iter_errors(data), key=lambda e: list(map(str, e.absolute_path))):
        where = _where(error)
        if prefix:
            where = prefix if where == "(top level)" else f"{prefix}.{where}"
        message = error.message
        if len(message) > 200:  # schema errors can echo huge values back
            message = message[:200] + "..."
        messages.append(f"{where}: {message}")
    return messages


def validate_output(data: dict, protocol_schema: dict | None = None) -> ValidationResult:
    """Run checks 2 and 3 on an already-parsed answer."""
    errors = _schema_errors(data, BASE_SCHEMA)

    findings = data.get("findings")
    if isinstance(findings, list):
        for i, finding in enumerate(findings):
            errors += _schema_errors(finding, FINDING_SCHEMA, prefix=f"findings[{i}]")

    if protocol_schema is not None:
        errors += _schema_errors(data, protocol_schema)

    if errors:
        return ValidationResult(ok=False, errors=errors)
    return ValidationResult(ok=True, data=data)


def check_answer(text: str, protocol_schema: dict | None = None) -> ValidationResult:
    """All three checks in one call: raw model text in, result out."""
    data, error = parse_model_json(text)
    if error:
        return ValidationResult(ok=False, errors=[error])
    return validate_output(data, protocol_schema)


def build_answer_schema(protocol_schema: dict | None = None) -> dict:
    """One complete schema to hand to the model (Ollama's "format").

    = the base contract, with findings spelled out, plus the protocol's own
    extra properties/required fields. Protocol schemas therefore only need
    to describe what they ADD (e.g. a "devices" list).
    """
    schema = copy.deepcopy(BASE_SCHEMA)
    finding = {k: v for k, v in copy.deepcopy(FINDING_SCHEMA).items()
               if not k.startswith("$")}
    schema["properties"]["findings"]["items"] = finding
    if protocol_schema:
        schema["properties"].update(copy.deepcopy(protocol_schema.get("properties", {})))
        for name in protocol_schema.get("required", []):
            if name not in schema["required"]:
                schema["required"].append(name)
        if "$defs" in protocol_schema:
            schema["$defs"] = copy.deepcopy(protocol_schema["$defs"])
    return schema
