"""Load settings from config/console.toml.

Why a separate module? Every other part of the framework needs settings
(where protocols live, which model plays which role, ...). Reading them in
one place means there is exactly one set of defaults and one set of rules.
"""

from __future__ import annotations

import copy
import tomllib  # built into Python 3.11+, reads .toml files
from pathlib import Path

# The project folder is the parent of this "core" folder.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "console.toml"

# Used for any setting missing from the file, so a half-written config
# still works.
DEFAULTS: dict = {
    "backend": {"kind": "ollama", "url": "http://localhost:11434"},
    "roles": {"analyst": "auto", "worker": "auto"},
    "paths": {
        "protocols": "protocols",
        "reports": "reports",
        "audit_log": "logs/audit.jsonl",
        "reference_db": "data/reference.db",
    },
    "ui": {"boot_animation": True, "banner": True},
}


def _merge(defaults: dict, overrides: dict) -> dict:
    """Return defaults with overrides laid on top, one level of tables deep.

    Example: defaults {"roles": {"analyst": "auto", "worker": "auto"}} and
    overrides {"roles": {"analyst": "llama3.1:8b"}} gives
    {"roles": {"analyst": "llama3.1:8b", "worker": "auto"}}.
    """
    result = copy.deepcopy(defaults)
    for section, values in overrides.items():
        if isinstance(values, dict) and isinstance(result.get(section), dict):
            result[section].update(values)
        else:
            result[section] = values
    return result


def load_config(path: Path | None = None) -> dict:
    """Read the config file and fill in anything missing from DEFAULTS.

    If the file does not exist we simply use the defaults; a missing config
    should never stop the console from starting.
    """
    path = path or DEFAULT_CONFIG_PATH
    if not path.exists():
        return copy.deepcopy(DEFAULTS)
    with open(path, "rb") as f:  # tomllib needs the file opened in binary mode
        user_settings = tomllib.load(f)
    return _merge(DEFAULTS, user_settings)


def resolve_path(config: dict, key: str) -> Path:
    """Turn a path from the [paths] table into an absolute path.

    Relative paths are treated as relative to the project folder, so the
    console works no matter which folder you start it from.
    """
    p = Path(config["paths"][key])
    return p if p.is_absolute() else PROJECT_ROOT / p
