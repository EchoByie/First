"""Is the model server reachable, and recent enough?

Ollama only accepts a full JSON schema in its "format" field (which this
framework relies on) from version 0.5.0 onward.
"""

from __future__ import annotations

from core.backends import BackendError, make_backend
from core.health import FAIL, OK, WARN, CheckResult, parse_version

MIN_OLLAMA_VERSION = (0, 5, 0)


def check(config: dict, backend=None) -> tuple[list[CheckResult], object]:
    """Returns (results, backend or None). The backend is reused by later checks."""
    try:
        backend = backend or make_backend(config)
    except BackendError as e:
        return [CheckResult("ollama", "address", FAIL, str(e),
                            "set [backend] url in config/console.toml to http://localhost:11434")], None
    try:
        version = backend.version()
    except BackendError as e:
        return [CheckResult("ollama", "reachable", FAIL, str(e), "start it with: ollama serve")], None

    results = [CheckResult("ollama", "reachable", OK, f"{backend.kind} at {backend.url}")]
    parsed = parse_version(version)
    wanted = ".".join(map(str, MIN_OLLAMA_VERSION))
    if not parsed:
        results.append(CheckResult("ollama", "version", WARN, f"unrecognised version {version!r}",
                                   f"make sure Ollama is {wanted} or newer"))
    elif parsed < MIN_OLLAMA_VERSION:
        results.append(CheckResult("ollama", "version", FAIL,
                                   f"version {version} is too old (needs {wanted}+ for JSON schemas)",
                                   "update Ollama: https://ollama.com/download"))
    else:
        results.append(CheckResult("ollama", "version", OK, f"version {version}"))
    return results, backend
