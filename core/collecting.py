"""What collectors and enrichers receive, and what they must return.

A protocol's collector.py has one function:

    def collect(ctx) -> CollectedData (or a dict with the same keys)

and its optional enricher.py has:

    def enrich(data, ctx) -> CollectedData (or a dict)

`ctx` (a CollectorContext) is the collector's ONLY way to touch the system.
It can:
    ctx.user_input_text()   the file or text the user handed in
    ctx.read_file(path)     a file from the manifest's allow-list
    ctx.run([...])          a command from the manifest's allow-list
    ctx.platform            "linux" or "windows"

What the model sees is CollectedData.full_text():

    preamble   short context that must ALWAYS be shown, even when a big
               input is split into parts (e.g. the digest's computed facts)
    text       the data itself; this is what gets split into parts

full_text() is also exactly what evidence quotes are checked against.
Keeping that one string as the single source of truth is what makes the
SURE / THINK check meaningful.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from pathlib import Path

from core.manifest import Manifest
from core.safe_exec import run_allowed

MAX_READ_BYTES = 5_000_000   # 5 MB cap on any single file a collector reads


class CollectorError(Exception):
    """A collector or enricher couldn't do its job (message is for people)."""


@dataclass
class UserInput:
    """What the user handed to a file_or_text protocol."""
    file: Path | None = None
    text: str | None = None

    def describe(self) -> str:
        if self.file:
            return f"file {self.file}"
        if self.text is not None:
            return f"pasted text ({len(self.text)} characters)"
        return "none"


@dataclass
class CollectedData:
    text: str                                       # the data (split if big)
    records: list[dict] = field(default_factory=list)  # structured rows, for reports
    meta: dict = field(default_factory=dict)        # e.g. {"source": "/proc/net/arp"}
    caveats: list[str] = field(default_factory=list)   # added to the report by code
    preamble: str = ""                              # always shown, never split

    def full_text(self) -> str:
        """Everything the model may quote from: preamble, then the data."""
        return f"{self.preamble}\n{self.text}" if self.preamble else self.text

    @classmethod
    def coerce(cls, value, who: str) -> "CollectedData":
        """Accept a CollectedData or a plain dict; reject anything else."""
        if isinstance(value, cls):
            data = value
        elif isinstance(value, dict) and isinstance(value.get("text"), str):
            data = cls(
                text=value["text"],
                records=list(value.get("records", [])),
                meta=dict(value.get("meta", {})),
                caveats=list(value.get("caveats", [])),
                preamble=str(value.get("preamble", "")),
            )
        else:
            raise CollectorError(f"{who} must return CollectedData or a dict with a 'text' string")
        if not isinstance(data.text, str) or not isinstance(data.preamble, str):
            raise CollectorError(f"{who} returned non-text data")
        return data


def _read_capped(path: Path) -> str:
    with open(path, "rb") as f:
        raw = f.read(MAX_READ_BYTES + 1)
    if len(raw) > MAX_READ_BYTES:
        raise CollectorError(f"{path} is larger than {MAX_READ_BYTES // 1_000_000} MB")
    return raw.decode("utf-8", errors="replace")


@dataclass
class CollectorContext:
    manifest: Manifest
    platform: str
    user_input: UserInput = field(default_factory=UserInput)

    def user_input_text(self) -> str:
        """The file or text the user chose. The user picked it explicitly,
        so it doesn't need to be on the manifest's file allow-list."""
        if self.user_input.file is not None:
            try:
                return _read_capped(Path(self.user_input.file))
            except OSError as e:
                raise CollectorError(f"can't read {self.user_input.file}: {e.strerror}") from None
        if self.user_input.text is not None:
            if len(self.user_input.text) > MAX_READ_BYTES:
                raise CollectorError("pasted text is too large")
            return self.user_input.text
        raise CollectorError("this protocol needs a file or text as input")

    def read_file(self, path: str) -> str:
        if path not in self.manifest.allowed_files:
            raise CollectorError(f"{path} is not in this protocol's allowed files")
        try:
            return _read_capped(Path(path))
        except OSError as e:
            raise CollectorError(f"can't read {path}: {e.strerror}") from None

    def run(self, argv: list[str], timeout: float | None = None) -> str:
        timeout = timeout or min(30, self.manifest.limits["timeout_seconds"])
        return run_allowed(argv, self.manifest.allowed_commands, timeout=timeout)


def load_function(path: Path, function_name: str):
    """Load collector.py / enricher.py and return the named function.

    Protocol code is your own code from the protocols/ folder, so it is
    trusted like the framework itself. What's *untrusted* is the data it
    collects and the model's answers.
    """
    module_name = f"aicore_protocol_{path.parent.name}_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        raise CollectorError(f"{path.name} failed to load: {type(e).__name__}: {e}") from None
    function = getattr(module, function_name, None)
    if not callable(function):
        raise CollectorError(f"{path.name} must define a function named {function_name}()")
    return function
