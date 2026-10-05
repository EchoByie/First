"""Read and check a protocol's manifest.toml.

A manifest describes a protocol: its name, how risky it is, which model
roles it needs, and exactly what its collector is allowed to read or run.

Why check so carefully? The manifest is the contract the rest of the
framework trusts. If something is wrong (a typo in risk_level, a missing
prompt file) we want a clear message *before* anything runs, not a confusing
crash halfway through a protocol.
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

MANIFEST_FILENAME = "manifest.toml"

RISK_LEVELS = ("low", "medium", "high")
INPUT_KINDS = ("none", "file_or_text")
PLATFORMS = ("linux", "windows")

# Protocol names become CLI arguments and file names, so keep them simple:
# lowercase letters, digits and underscores, starting with a letter.
NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,39}$")

# Default file names inside a protocol folder. A manifest can override them
# in a [files] table.
DEFAULT_FILES = {
    "collector": "collector.py",
    "enricher": "enricher.py",
    "prompt": "prompt.md",
    "worker_prompt": "worker_prompt.md",
    "output_schema": "output.schema.json",
}
# These must exist. The others are optional.
REQUIRED_FILES = ("collector", "prompt", "output_schema")

DEFAULT_LIMITS = {
    "chunk_threshold_chars": 12000,  # bigger input than this -> chunked path
    "chunk_size_chars": 6000,
    "timeout_seconds": 120,
}


class ManifestError(Exception):
    """Raised when a manifest has problems. Holds ALL problems found, not
    just the first, so you can fix them in one go."""

    def __init__(self, folder: Path, problems: list[str]):
        self.folder = folder
        self.problems = problems
        super().__init__(f"{folder.name}: " + "; ".join(problems))


@dataclass
class RoleSpec:
    """What a protocol needs from the model playing one role."""

    name: str
    min_context: int = 4096   # smallest context window (tokens) that is OK
    needs_json: bool = True   # must support structured JSON output
    optional: bool = False    # True = protocol can run without this role


@dataclass
class Manifest:
    """A checked, ready-to-use manifest. Every field has a sensible value."""

    folder: Path
    name: str
    title: str
    description: str
    version: str
    risk_level: str
    platforms: list[str]
    roles: dict[str, RoleSpec]
    limits: dict[str, int]
    input_kind: str
    files: dict[str, Path]  # only the files that actually exist
    allowed_commands: list[list[str]] = field(default_factory=list)
    allowed_files: list[str] = field(default_factory=list)
    required_refdata: list[str] = field(default_factory=list)
    max_refdata_age_days: int | None = None
    caveats: list[str] = field(default_factory=list)  # always added to reports

    def supports(self, platform: str) -> bool:
        return platform in self.platforms


# ---------------------------------------------------------------------------
# Small helpers. Each one checks one value and adds a message to `problems`
# if it is wrong, instead of stopping at the first error.
# ---------------------------------------------------------------------------

def _require_str(data: dict, key: str, problems: list[str]) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        problems.append(f"'{key}' is required and must be non-empty text")
        return ""
    return value.strip()


def _check_choice(value, choices: tuple, label: str, problems: list[str]) -> None:
    if value not in choices:
        problems.append(f"{label} must be one of {', '.join(choices)} (got {value!r})")


def _parse_roles(raw, problems: list[str]) -> dict[str, RoleSpec]:
    if not isinstance(raw, dict) or not raw:
        problems.append("[roles] must list at least one role, e.g. analyst = { min_context = 8192 }")
        return {}
    roles = {}
    for role_name, spec in raw.items():
        if not isinstance(spec, dict):
            problems.append(f"role '{role_name}' must be a table like {{ min_context = 8192 }}")
            continue
        unknown = set(spec) - {"min_context", "needs_json", "optional"}
        if unknown:
            problems.append(f"role '{role_name}' has unknown keys: {', '.join(sorted(unknown))}")
        min_context = spec.get("min_context", 4096)
        if not isinstance(min_context, int) or min_context <= 0:
            problems.append(f"role '{role_name}': min_context must be a positive whole number")
            min_context = 4096
        roles[role_name] = RoleSpec(
            name=role_name,
            min_context=min_context,
            needs_json=bool(spec.get("needs_json", True)),
            optional=bool(spec.get("optional", False)),
        )
    if roles and all(r.optional for r in roles.values()):
        problems.append("at least one role must be required (not optional)")
    return roles


def _parse_limits(raw, problems: list[str]) -> dict[str, int]:
    limits = dict(DEFAULT_LIMITS)
    if raw is None:
        return limits
    if not isinstance(raw, dict):
        problems.append("[limits] must be a table")
        return limits
    for key, value in raw.items():
        if key not in DEFAULT_LIMITS:
            problems.append(f"[limits] has unknown key '{key}'")
        elif not isinstance(value, int) or value <= 0:
            problems.append(f"[limits] {key} must be a positive whole number")
        else:
            limits[key] = value
    if limits["chunk_size_chars"] > limits["chunk_threshold_chars"]:
        problems.append("[limits] chunk_size_chars cannot be bigger than chunk_threshold_chars")
    return limits


def _parse_policy(raw, problems: list[str]) -> tuple[list[list[str]], list[str]]:
    """The collector policy is the allow-list of what a collector may touch.

    commands: each one is a list of words, e.g. ["arp", "-a"]. We store it as
              a list (not one string) so it can never be run through a shell.
    files:    exact file paths the collector may read.
    """
    if raw is None:
        return [], []
    if not isinstance(raw, dict):
        problems.append("[collector_policy] must be a table")
        return [], []
    commands = raw.get("commands", [])
    files = raw.get("files", [])
    good_commands = []
    for cmd in commands if isinstance(commands, list) else [commands]:
        if (isinstance(cmd, list) and cmd
                and all(isinstance(part, str) and part for part in cmd)):
            good_commands.append(cmd)
        else:
            problems.append(f"[collector_policy] command {cmd!r} must be a list of words, e.g. [\"arp\", \"-a\"]")
    if not isinstance(files, list) or not all(isinstance(f, str) and f for f in files):
        problems.append("[collector_policy] files must be a list of paths")
        files = []
    return good_commands, files


def _parse_files(folder: Path, raw, problems: list[str]) -> dict[str, Path]:
    names = dict(DEFAULT_FILES)
    if isinstance(raw, dict):
        for key, value in raw.items():
            if key not in DEFAULT_FILES:
                problems.append(f"[files] has unknown key '{key}'")
            elif not isinstance(value, str):
                problems.append(f"[files] {key} must be a file name")
            else:
                names[key] = value
    elif raw is not None:
        problems.append("[files] must be a table")

    found = {}
    for key, filename in names.items():
        path = (folder / filename).resolve()
        # Safety: a manifest must not point at files outside its own folder
        # (e.g. collector = "../../somewhere/else.py").
        if not path.is_relative_to(folder.resolve()):
            problems.append(f"[files] {key} must stay inside the protocol folder")
            continue
        if path.is_file():
            found[key] = path
        elif key in REQUIRED_FILES:
            problems.append(f"missing required file '{filename}' ({key})")
    return found


# ---------------------------------------------------------------------------
# The main function
# ---------------------------------------------------------------------------

def load_manifest(folder: Path) -> Manifest:
    """Read folder/manifest.toml, check everything, and return a Manifest.

    Raises ManifestError listing every problem found.
    """
    folder = Path(folder)
    path = folder / MANIFEST_FILENAME
    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
    except FileNotFoundError:
        raise ManifestError(folder, [f"no {MANIFEST_FILENAME} found"])
    except tomllib.TOMLDecodeError as e:
        raise ManifestError(folder, [f"{MANIFEST_FILENAME} is not valid TOML: {e}"])

    problems: list[str] = []

    name = _require_str(data, "name", problems)
    if name and not NAME_PATTERN.match(name):
        problems.append("'name' must be lowercase letters, digits or _ and start with a letter")
    if name and name != folder.name:
        problems.append(f"'name' ({name}) must match the folder name ({folder.name})")

    title = data.get("title") or name
    description = _require_str(data, "description", problems)
    version = str(data.get("version", "0.1"))

    risk_level = data.get("risk_level")
    _check_choice(risk_level, RISK_LEVELS, "'risk_level'", problems)

    platforms = data.get("platforms", list(PLATFORMS))
    if not isinstance(platforms, list) or not platforms:
        problems.append("'platforms' must be a list, e.g. [\"linux\", \"windows\"]")
        platforms = []
    for p in platforms:
        _check_choice(p, PLATFORMS, "each platform", problems)

    roles = _parse_roles(data.get("roles"), problems)
    limits = _parse_limits(data.get("limits"), problems)

    input_table = data.get("input", {})
    input_kind = input_table.get("kind", "none") if isinstance(input_table, dict) else None
    _check_choice(input_kind, INPUT_KINDS, "[input] kind", problems)

    commands, allowed_files = _parse_policy(data.get("collector_policy"), problems)
    files = _parse_files(folder, data.get("files"), problems)

    requires = data.get("requires", {})
    if not isinstance(requires, dict):
        problems.append("[requires] must be a table")
        requires = {}
    refdata = requires.get("refdata", [])
    if not isinstance(refdata, list) or not all(isinstance(r, str) for r in refdata):
        problems.append("[requires] refdata must be a list of names")
        refdata = []
    max_age = requires.get("max_refdata_age_days")
    if max_age is not None and (not isinstance(max_age, int) or max_age <= 0):
        problems.append("[requires] max_refdata_age_days must be a positive whole number")
        max_age = None

    caveats = data.get("caveats", [])
    if not isinstance(caveats, list) or not all(isinstance(c, str) and c for c in caveats):
        problems.append("'caveats' must be a list of sentences")
        caveats = []

    if problems:
        raise ManifestError(folder, problems)

    return Manifest(
        folder=folder,
        name=name,
        title=title,
        description=description,
        version=version,
        risk_level=risk_level,
        platforms=platforms,
        roles=roles,
        limits=limits,
        input_kind=input_kind,
        files=files,
        allowed_commands=commands,
        allowed_files=allowed_files,
        required_refdata=refdata,
        max_refdata_age_days=max_age,
        caveats=caveats,
    )
