"""The pipeline: everything that happens when you run a protocol.

    preflight   right platform? risky protocol confirmed? input as expected?
    collect     collector.py gathers data (allow-listed files/commands only)
    enrich      enricher.py adds facts with plain code (optional)
    models      pick the model for each role
    then, depending on size:
      small     analyst reads everything in one go
      big       split into parts -> worker reads each part -> code checks
                the worker's evidence and drops notes it can't confirm ->
                analyst combines the checked notes
    validate    JSON + schemas; on failure retry ONCE with the error list
    verify      check every evidence quote against the ORIGINAL data
    report      save JSON + Markdown, write the audit entry

Dry-run stops before the first model call and shows exactly what WOULD be
sent (for big inputs: the plan of parts and the first part's messages).

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
from core.chunking import Chunk, TooManyChunks, split_text
from core.collecting import (
    CollectedData, CollectorContext, CollectorError, UserInput, load_function,
)
from core.config import resolve_path
from core.discovery import current_platform
from core.manifest import Manifest
from core.models import RoleChoice, gather_models, missing_required, resolve_roles
from core.prompting import (
    COMBINE_NOTE, build_messages, chunk_text, combine_text, estimate_tokens,
    retry_note, worker_task,
)
from core.report import save_report
from core.safe_exec import CommandFailed, CommandNotAllowed
from core.validation import build_answer_schema, check_answer, load_schema
from core.verification import verify_output

# Room left in the context window for the model's answer (tokens).
ANSWER_BUDGET = 2048
# How much of a rejected answer to keep in a failed report.
KEEP_RAW_CHARS = 4000
# Worker notes kept per part (after dropping unverifiable ones).
MAX_NOTES_PER_PART = 8
# Fields of a worker note passed on to the analyst.
NOTE_FIELDS = ("claim", "evidence", "confidence", "basis", "kind", "subject")

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

    def __init__(self, status: str, message: str, backend_down: bool = False):
        self.status, self.message = status, message
        self.backend_down = backend_down   # True = no point trying other parts
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
            "input": options.user_input.describe(), "models": {}, "used_models": {},
            "attempts": 0, "model_calls": 0,
        }
        result = RunResult(run_id=run_id, status="failed")
        collected = None
        extra_caveats: list[str] = []
        try:
            self._preflight(manifest, options)
            collected = self._collect(manifest, options)
            corpus = collected.full_text()   # the ONLY text quotes may come from
            state["data_chars"] = len(corpus)
            state["data_fingerprint"] = audit.fingerprint(corpus)

            task = manifest.files["prompt"].read_text(encoding="utf-8")
            schema = build_answer_schema(load_schema(manifest.files["output_schema"]),
                                         manifest.finding_kinds)
            choices = self._resolve(manifest, state, options.dry_run)
            analyst = self._role(choices, manifest, _main_role(manifest))

            if len(corpus) <= manifest.limits["chunk_threshold_chars"]:
                state["path"] = "single"
                system, user = build_messages(task, corpus)
                context = self._context(analyst, system + user)
                if options.dry_run:
                    result.preview = self._preview(state, system, user, schema, context)
                else:
                    answer = self._ask(analyst, "analyst", system, user, schema, context,
                                       manifest, state)
            else:
                state["path"] = "chunked"
                answer = self._chunked(manifest, collected, task, schema, choices, analyst,
                                       state, options, result, extra_caveats)

            if options.dry_run:
                result.status = "dry-run"
                self._step("done", "dry-run complete: nothing was sent to the model")
                return result

            self._step("verify", "checking every evidence quote against the original data")
            verified = verify_output(answer, corpus)
            state["verification"] = verified["verification_summary"]
            result.status = "ok"
            result.report = self._build_report(manifest, run_id, "ok", options, collected,
                                               state, verified, [], extra_caveats)
        except _Stop as stop:
            result.status, result.errors = stop.status, [stop.message]
            self._step("stopped", stop.message)
            if stop.status == "failed" and not options.dry_run:
                result.report = self._build_report(manifest, run_id, "failed", options,
                                                   collected, state, None, result.errors,
                                                   extra_caveats)
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
    # Big inputs
    # ------------------------------------------------------------------------
    def _chunked(self, manifest, collected, task, schema, choices, analyst,
                 state, options, result, extra_caveats):
        limits = manifest.limits
        if len(collected.preamble) > limits["chunk_size_chars"]:
            raise _Stop("failed", "the shared context (preamble) is bigger than one part; "
                                  "the protocol must keep it shorter")
        try:
            chunks = split_text(collected.text, limits["chunk_size_chars"], limits["max_chunks"])
        except TooManyChunks as e:
            raise _Stop("failed", str(e)) from None
        state["parts"] = len(chunks)

        worker = self._role(choices, manifest, "worker") if "worker" in manifest.roles else None
        if worker is None or (worker.model is None and choices is not None):
            worker = analyst
            state["worker_fallback"] = True
            extra_caveats.append("No worker model was available, so the analyst model "
                                 "also read the parts.")
        worker_prompt_file = manifest.files.get("worker_prompt")
        worker_system_task = worker_task(
            worker_prompt_file.read_text(encoding="utf-8") if worker_prompt_file else None)
        worker_schema = build_answer_schema(None, manifest.finding_kinds)

        self._step("chunk", f"input is {len(collected.full_text()):,} characters: "
                            f"reading it in {len(chunks)} parts")
        if options.dry_run:
            system, user = build_messages(
                worker_system_task, chunk_text(collected.preamble, chunks[0].label, chunks[0].text))
            result.preview = self._preview(state, system, user, worker_schema,
                                           self._context(worker, system + user))
            result.preview["parts"] = [c.label for c in chunks]
            result.preview["note"] = (f"Showing part 1 of {len(chunks)} as sent to the worker. "
                                      "The analyst then combines the checked notes.")
            return None

        notes = []
        for chunk in chunks:
            notes.append(self._read_part(chunk, collected.preamble, worker, worker_system_task,
                                         worker_schema, manifest, state))
        failed = [n for n in notes if n["failed"]]
        state["parts_failed"] = len(failed)
        state["notes_dropped"] = sum(n.get("dropped", 0) for n in notes)
        if len(failed) > len(chunks) / 2:
            raise _Stop("failed", f"{len(failed)} of {len(chunks)} parts could not be read")
        extra_caveats.append(f"The input was too large to read at once; it was read in "
                             f"{len(chunks)} parts and the results were combined.")
        if failed:
            extra_caveats.append("These parts could not be read, so the report may miss "
                                 "things in them: " + ", ".join(n["label"] for n in failed))

        self._step("combine", f"combining {len(chunks)} parts")
        system, user = build_messages(task + "\n\n" + COMBINE_NOTE,
                                      combine_text(collected.preamble, notes))
        return self._ask(analyst, "analyst", system, user, schema,
                         self._context(analyst, system + user), manifest, state)

    def _read_part(self, chunk: Chunk, preamble, worker, system_task, schema, manifest, state):
        """Worker reads one part. Its notes are verified against that part
        (plus the preamble); notes whose quotes aren't there are dropped."""
        note = {"label": chunk.label, "failed": False}
        system, user = build_messages(system_task, chunk_text(preamble, chunk.label, chunk.text))
        self._step("worker", f"reading part {chunk.index} of {chunk.total} with {worker.model}")
        try:
            answer = self._ask(worker, "worker", system, user, schema,
                               self._context(worker, system + user), manifest, state)
        except _Stop as stop:
            if stop.backend_down:
                raise
            return {**note, "failed": True, "error": stop.message}

        part_corpus = f"{preamble}\n{chunk.text}" if preamble else chunk.text
        checked = verify_output(answer, part_corpus)["findings"]
        kept = [{k: f[k] for k in NOTE_FIELDS if k in f}
                for f in checked if f["verification"]["evidence_verified"]]
        return {**note, "summary": answer["summary"], "findings": kept[:MAX_NOTES_PER_PART],
                "dropped": len(checked) - len(kept)}

    # ------------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------------
    def _step(self, stage: str, message: str) -> None:
        self.progress(stage, message)

    def _preview(self, state, system, user, schema, context) -> dict:
        return {"system": system, "user": user, "schema": schema,
                "models": state["models"], "context_tokens": context,
                "estimated_prompt_tokens": estimate_tokens(system + user),
                "path": state.get("path")}

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

    def _resolve(self, manifest: Manifest, state: dict, dry_run: bool):
        """Choose a model per role. Returns None for a dry-run with the
        model server off (a dry-run is still useful then)."""
        self._step("models", "choosing models")
        try:
            installed = gather_models(self.backend)
        except BackendError as e:
            if dry_run:
                state["models"] = {r: "unknown (backend unreachable)" for r in manifest.roles}
                return None
            raise _Stop("failed", str(e), backend_down=True) from None
        choices = resolve_roles(manifest.roles, self.config["roles"], installed)
        state["models"] = {r: c.model for r, c in choices.items() if c.ok}
        missing = missing_required(choices, manifest.roles)
        if missing:
            reasons = "; ".join(f"{r}: {choices[r].reason}" for r in missing)
            raise _Stop("failed", f"no model for required role(s) {reasons}")
        return choices

    @staticmethod
    def _role(choices, manifest: Manifest, role: str) -> RoleChoice:
        """The choice for a role; a placeholder when models are unknown (dry-run offline)."""
        if choices is None:
            spec = manifest.roles[role]
            return RoleChoice(role, None, spec.min_context, "backend unreachable")
        return choices[role]

    @staticmethod
    def _context(choice: RoleChoice, prompt: str) -> int:
        """How big a context window to ask for: enough for the prompt plus
        an answer, at least what the role needs, at most what the model has."""
        needed = estimate_tokens(prompt) + ANSWER_BUDGET
        context = max(choice.context_tokens, -(-needed // 1024) * 1024)  # round up to 1k
        if choice.model_context and context > choice.model_context:
            if needed > choice.model_context:
                raise _Stop("failed", f"a prompt needs ~{needed:,} tokens but {choice.model} "
                                      f"reads at most {choice.model_context:,}")
            context = choice.model_context
        return context

    def _ask(self, choice: RoleChoice, role: str, system, user, schema, context,
             manifest, state) -> dict:
        """Ask one model, validate, retry once. Returns the parsed answer."""
        timeout = manifest.limits["timeout_seconds"]
        errors: list[str] = []
        raw = ""
        state["used_models"][role] = choice.model
        for attempt in (1, 2):
            if role == "analyst":
                state["attempts"] = attempt
            state["model_calls"] += 1
            prompt_system = system if attempt == 1 else system + retry_note(errors)
            if role == "analyst":
                self._step("ask", f"asking {choice.model}" + (" again (retry)" if attempt == 2 else ""))
            try:
                reply = self.backend.chat_json(choice.model, prompt_system, user, schema,
                                               context, timeout)
            except BackendError as e:
                raise _Stop("failed", str(e), backend_down=True) from None
            raw = reply.text
            state["prompt_tokens"] = state.get("prompt_tokens", 0) + reply.prompt_tokens
            state["output_tokens"] = state.get("output_tokens", 0) + reply.output_tokens
            if role == "analyst":
                self._step("validate", "checking the answer against the schema")
            checked = check_answer(raw, schema)
            if checked.ok:
                if role == "analyst":
                    state["answer_fingerprint"] = audit.fingerprint(raw)
                return checked.data
            errors = checked.errors
            state.setdefault("validation_errors", []).append(
                {"role": role, "errors": errors[:10]})
        if role == "analyst":
            state["rejected_answer"] = raw[:KEEP_RAW_CHARS]
        raise _Stop("failed", f"the {role}'s answer failed validation twice: "
                    + "; ".join(errors[:5]))

    def _build_report(self, manifest, run_id, status, options, collected, state,
                      verified, errors, extra_caveats) -> dict:
        model_caveats = verified.get("caveats", []) if verified else []
        caveats = list(manifest.caveats)
        if collected:
            caveats += collected.caveats
        caveats += extra_caveats
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
                "characters": state.get("data_chars", 0),
                "fingerprint": state.get("data_fingerprint"),
                "meta": collected.meta if collected else {},
            },
            "models": state["used_models"],   # only the roles that actually ran
            "answer": verified,
            "records": collected.records if collected else [],
            "caveats": caveats,
            "errors": errors,
            "stats": {k: state.get(k) for k in
                      ("path", "parts", "parts_failed", "notes_dropped", "model_calls",
                       "attempts", "prompt_tokens", "output_tokens")},
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
