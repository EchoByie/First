"""Find protocols automatically.

Every sub-folder of protocols/ that contains a manifest.toml is a protocol.
Adding a protocol means adding a folder; no registration code to edit.

A broken protocol must not take the whole console down, so we never raise
here. Good protocols go in one list, broken ones (with their error messages)
in another, and the UI shows both.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass, field
from pathlib import Path

from core.manifest import MANIFEST_FILENAME, Manifest, ManifestError, load_manifest


@dataclass
class DiscoveryResult:
    protocols: dict[str, Manifest] = field(default_factory=dict)  # name -> manifest
    broken: list[ManifestError] = field(default_factory=list)


def current_platform() -> str:
    """Return "linux", "windows", or the raw system name for anything else."""
    return platform.system().lower()


def discover(protocols_dir: Path) -> DiscoveryResult:
    """Load every protocol folder inside protocols_dir.

    Folders starting with "_" or "." are skipped, so you can park a
    half-finished protocol as e.g. "_my_draft" without it showing up.
    """
    result = DiscoveryResult()
    protocols_dir = Path(protocols_dir)
    if not protocols_dir.is_dir():
        return result

    for folder in sorted(protocols_dir.iterdir()):  # sorted = stable order
        if not folder.is_dir() or folder.name.startswith(("_", ".")):
            continue
        if not (folder / MANIFEST_FILENAME).exists():
            continue  # not a protocol, e.g. a shared helpers folder
        try:
            manifest = load_manifest(folder)
        except ManifestError as error:
            result.broken.append(error)
            continue
        result.protocols[manifest.name] = manifest
    return result
