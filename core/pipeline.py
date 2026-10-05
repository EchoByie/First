"""The pipeline: everything that happens when you run a protocol.

    preflight   right platform? risky protocol confirmed? input as expected?
    collect     collector.py gathers data (allow-listed files/commands only)
    enrich      enricher.py adds facts with plain code (optional)
    prompt      fixed rules + protocol task + fenced, untrusted data
    models      pick the model for each role, size the context window
    ask         send to the model -> raw text
    validate    JSON + schemas; on failure retry ONCE with the error list
    verify      check every evidence quote -> SURE / THINK
    report      save JSON + Markdown, write the audit entry

Dry-run stops after "models" and shows exactly what WOULD be sent.

Every outcome (ok, failed, refused, dry-run) gets an audit entry, and every
real run gets a report, even a failed one, so nothing disappears silently.
"""

from __future__ import annotations

import secrets
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from core import audit
from core.backends import BackendError, make_backend
from core.backends.base import ModelBackend
from core.collecting import (
    CollectedData, CollectorContext, CollectorError, UserInput, load_function,
)
from core.config import resolve_path
from core.discovery import current_platform
from core.manifest import Manifest
from core.models import gather_models, missing_required, resolve_roles
from core.prompting import build_messages, estimate_tokens, retry_note
from core.report import save_report
from core.safe_exec import CommandFailed, CommandNotAllowed
from core.validation import build_answer_schema, check_answer, load_schema
from core.verification import verify_output

# Room left in the context window for the model's answer (tokens).
ANSWER_BUDGET = 2048
# How much of a rejected answer to keep in a failed report.
KEEP_RAW_CHARS = 4000

ProgressFn = Callable[[str, str], None]   # (stage, message)


@dataclass
class RunOptions:
    user_input: UserInput = field(default_factory=UserInput)
    dry_run: bool = False
    confirmed: bool = False   # needed for risk_level = "high"


@dataclass
class RunResult:
    run_id: str
    status: str                        # ok | failed | refused | dry-run
    errors: list[str] = field(default_factory=list)
    report: dict | None = None
    report_paths: tuple[Path, Path] | None = None
    preview: dict | None = None        # dry-run: what would be sent


class _Stop(Exception):
    """Internal: end the run early with a status and a message."""

    def __init__(self, status: str, message: str):
        self.status, self.message = status, message
        super().__init__(message)


def new_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{secrets.token_hex(2)}"


class Pipeline:
    def __init__(self, config: dict, backend: ModelBackend | None = None,
                 progress: ProgressFn | None = None,
                 reports_dir: Path | None = None, audit_path: Path | None = None):
        self.config = config
        self._backend = backend
        self.progress = progress or (lambda stage, message: None)
        self.reports_dir = reports_dir or resolve_path(config, "reports")
        self.audit_path = audit_path or resolve_path(config, "audit_log")

    @property
    def backend(self) -> ModelBackend:
        if self._backend is None:                 # created on first use, so
            self._backend = make_backend(self.config)  # dry-run works without it
        return self._backend

    # ------------------------------------------------------------------------
    def run(self, manifest: Manifest, options: RunOptions | None = None) -> RunResult:
        options = options or RunOptions()
        run_id = new_run_id()
        started = time.monotonic()
        state = {
            "run_id": run_id, "protocol": manifest.name, "protocol_version": manifest.version,
            "mode": "dry-run" if options.dry_run else "run", "platform": current_platform(),
            "input": options.user_input.describe(), "models": {}, "attempts": 0,
        }
        result = RunResult(run_id=run_id, status="failed")
        collected = None
        try:
            self._preflight(manifest, options)
            collected = self._collect(manifest, options)
            state["data_chars"] = len(collected.text)
            state["data_fingerprint"] = audit.fingerprint(collected.text)

            self._step("prompt", "building prompt")
            task = manifest.files["prompt"].read_text(encoding="utf-8")
            schema = build_answer_schema(load_schema(manifest.files["output_schema"]))
            system, user = build_messages(task, collected.text)
            state["prompt_fingerprint"] = audit.fingerprint(system + user)
            if len(collected.text) > manifest.limits["chunk_threshold_chars"]:
                raise _Stop("failed",
                            f"input is {len(collected.text):,} characters, over this protocol's "
                            f"single-pass limit of {manifest.limits['chunk_threshold_chars']:,} "
                            "(chunking for big inputs arrives in build step 7)")

            analyst, context_tokens = self._choose_models(manifest, system + user, state,
                                                          dry_run=options.dry_run)
            if options.dry_run:
                result.status = "dry-run"
                result.preview = {
                    "system": system, "user": user, "schema": schema,
                    "models": state["models"], "context_tokens": context_tokens,
                    "estimated_prompt_tokens": estimate_tokens(system + user),
                }
                self._step("done", "dry-run complete: nothing was sent to the model")
                return result

            answer, raw = self._ask(manifest, analyst, system, user, schema, context_tokens, state)
            self._step("verify", "checking every evidence quote against the data")
            verified = verify_output(answer, collected.text)
            state["verification"] = verified["verification_summary"]
            result.status = "ok"
            result.report = self._build_report(manifest, run_id, "ok", options, collected,
                                               state, verified, [])
        except _Stop as stop:
            result.status, result.errors = stop.status, [stop.message]
            self._step("stopped", stop.message)
            if stop.status == "failed" and not options.dry_run:
                result.report = self._build_report(manifest, run_id, "failed", options,
                                                   collected, state, None, result.errors)
        finally:
            state["status"] = result.status
            state["errors"] = result.errors
            state["seconds"] = round(time.monotonic() - started, 2)
            if result.report is not None:
                self._step("report", "saving report")
                result.report["stats"]["seconds"] = state["seconds"]
                result.report_paths = save_report(result.report, self.reports_dir)
                state["report"] = str(result.report_paths[0])
            audit.write_entry(self.audit_path, state)
        return result

    # ------------------------------------------------------------------------
    def _step(self, stage: str, message: str) -> None:
        self.progress(stage, message)

    def _preflight(self, manifest: Manifest, options: RunOptions) -> None:
        self._step("preflight", "checking platform, risk level and input")
        here = current_platform()
        if not manifest.supports(here):
            raise _Stop("refused", f"{manifest.name} does not support {here}")
        if manifest.risk_level == "high" and not options.confirmed and not options.dry_run:
            raise _Stop("refused", f"{manifest.name} is high risk; confirm to run it")
        has_input = options.user_input.file is not None or options.user_input.text is not None
        if manifest.input_kind == "none" and has_input:
            raise _Stop("refused", f"{manifest.name} takes no input")
        if manifest.input_kind == "file_or_text" and not has_input:
            raise _Stop("refused", f"{manifest.name} needs a file or text as input")

    def _collect(self, manifest: Manifest, options: RunOptions) -> CollectedData:
        ctx = CollectorContext(manifest, current_platform(), options.user_input)
        try:
            self._step("collect", "collecting data")
            collect = load_function(manifest.files["collector"], "collect")
            data = CollectedData.coerce(collect(ctx), "collect()")
            if "enricher" in manifest.files:
                self._step("enrich", "enriching data")
                enrich = load_function(manifest.files["enricher"], "enrich")
                data = CollectedData.coerce(enrich(data, ctx), "enrich()")
        except (CollectorError, CommandNotAllowed, CommandFailed) as e:
            raise _Stop("failed", f"collection failed: {e}") from None
        except Exception as e:  # a bug in protocol code must not crash the console
            raise _Stop("failed", f"protocol code crashed: {type(e).__name__}: {e}") from None
        if not data.text.strip():
            raise _Stop("failed", "the collector found no data")
        return data

    def _choose_models(self, manifest: Manifest, prompt: str, state: dict,
                       dry_run: bool):
        self._step("models", "choosing models")
        try:
            installed = gather_models(self.backend)
        except BackendError as e:
            if dry_run:  # a dry-run is still useful with the model server off
                state["models"] = {r: "unknown (backend unreachable)" for r in manifest.roles}
                return None, manifest.roles[_main_role(manifest)].min_context
            raise _Stop("failed", str(e)) from None

        choices = resolve_roles(manifest.roles, self.config["roles"], installed)
        state["models"] = {r: c.model for r, c in choices.items() if c.ok}
        missing = missing_required(choices, manifest.roles)
        if missing:
            reasons = "; ".join(f"{r}: {choices[r].reason}" for r in missing)
            raise _Stop("failed", f"no model for required role(s) {reasons}")

        main = choices[_main_role(manifest)]
        needed = estimate_tokens(prompt) + ANSWER_BUDGET
        context_tokens = max(main.context_tokens, -(-needed // 1024) * 1024)  # round up to 1k
        if main.model_context and context_tokens > main.model_context:
            if needed > main.model_context:
                raise _Stop("failed", f"input needs ~{needed:,} tokens but {main.model} "
                                      f"reads at most {main.model_context:,}")
            context_tokens = main.model_context
        state["context_tokens"] = context_tokens
        return main.model, context_tokens

    def _ask(self, manifest, model, system, user, schema, context_tokens, state):
        timeout = manifest.limits["timeout_seconds"]
        errors: list[str] = []
        raw = ""
        for attempt in (1, 2):
            state["attempts"] = attempt
            prompt_system = system if attempt == 1 else system + retry_note(errors)
            self._step("ask", f"asking {model}" + (" again (retry)" if attempt == 2 else ""))
            try:
                reply = self.backend.chat_json(model, prompt_system, user, schema,
                                               context_tokens, timeout)
            except BackendError as e:
                raise _Stop("failed", str(e)) from None
            raw = reply.text
            state["prompt_tokens"] = state.get("prompt_tokens", 0) + reply.prompt_tokens
            state["output_tokens"] = state.get("output_tokens", 0) + reply.output_tokens
            self._step("validate", "checking the answer against the schema")
            checked = check_answer(raw, schema)
            if checked.ok:
                state["answer_fingerprint"] = audit.fingerprint(raw)
                return checked.data, raw
            errors = checked.errors
            state.setdefault("validation_errors", []).append(errors[:10])
        state["rejected_answer"] = raw[:KEEP_RAW_CHARS]
        raise _Stop("failed", "the model's answer failed validation twice: " + "; ".join(errors[:5]))

    def _build_report(self, manifest, run_id, status, options, collected, state,
                      verified, errors) -> dict:
        model_caveats = verified.get("caveats", []) if verified else []
        caveats = list(manifest.caveats)
        if collected:
            caveats += collected.caveats
        caveats += [f"(model) {c}" for c in model_caveats]
        report = {
            "run_id": run_id,
            "protocol": manifest.name,
            "title": manifest.title,
            "protocol_version": manifest.version,
            "status": status,
            "created_at": audit.now_utc(),
            "platform": state["platform"],
            "input": {
                "description": options.user_input.describe(),
                "characters": len(collected.text) if collected else 0,
                "fingerprint": state.get("data_fingerprint"),
                "meta": collected.meta if collected else {},
            },
            "models": state["models"],
            "answer": verified,
            "records": collected.records if collected else [],
            "caveats": caveats,
            "errors": errors,
            "stats": {k: state.get(k) for k in
                      ("attempts", "prompt_tokens", "output_tokens", "context_tokens")},
        }
        if "rejected_answer" in state:
            report["rejected_answer"] = state["rejected_answer"]
        return report


def _main_role(manifest: Manifest) -> str:
    """The role that writes the final answer: "analyst" if present,
    otherwise the first required role."""
    if "analyst" in manifest.roles:
        return "analyst"
    return next(r for r, spec in manifest.roles.items() if not spec.optional)
