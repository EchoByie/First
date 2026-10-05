"""Command-line entry point: `aicore <command>` or `python -m core.cli <command>`.

Commands so far: version, list. Each build step adds more
(run, dry-run, health, update, models, console).

We use argparse because it is built into Python and gives `--help` for free.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.markup import escape
from rich.table import Table

from core import __version__
from core.config import load_config, resolve_path
from core.discovery import current_platform, discover

# Rich prints coloured tables; it falls back to plain text when output is
# piped to a file.
console = Console()

RISK_COLOURS = {"low": "green", "medium": "yellow", "high": "red"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aicore",
        description="AI Core Console: run read-only protocols against a local LLM.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("version", help="show the framework version and active config")

    list_cmd = sub.add_parser("list", help="list the available protocols")
    list_cmd.add_argument(
        "--protocols-dir", type=Path, default=None,
        help="look for protocols here instead of the folder in console.toml",
    )
    return parser


def cmd_version() -> int:
    config = load_config()
    print(f"AI Core Console {__version__}")
    print(f"backend: {config['backend']['kind']} at {config['backend']['url']}")
    print(f"roles:   analyst={config['roles']['analyst']}  worker={config['roles']['worker']}")
    return 0


def cmd_list(protocols_dir: Path | None) -> int:
    config = load_config()
    protocols_dir = protocols_dir or resolve_path(config, "protocols")
    found = discover(protocols_dir)
    here = current_platform()

    if not found.protocols and not found.broken:
        console.print(f"No protocols found in {protocols_dir}")
        return 0

    if found.protocols:
        table = Table(title=f"Protocols ({protocols_dir})")
        table.add_column("Name", style="bold cyan")
        table.add_column("Risk")
        table.add_column("Roles")
        table.add_column("Input")
        table.add_column(f"Runs on {here}?")
        table.add_column("Description")
        for m in found.protocols.values():
            colour = RISK_COLOURS[m.risk_level]
            roles = ", ".join(
                r.name + (" (optional)" if r.optional else "") for r in m.roles.values()
            )
            # escape() stops text from a manifest being read as Rich colour
            # codes, e.g. a description containing "[red]".
            table.add_row(
                escape(m.name),
                f"[{colour}]{m.risk_level}[/{colour}]",
                roles,
                m.input_kind,
                "[green]yes[/green]" if m.supports(here) else "[red]no[/red]",
                escape(m.description),
            )
        console.print(table)

    # Broken protocols are shown, not hidden, so you can fix them.
    for error in found.broken:
        console.print(f"[red]✗ {escape(error.folder.name)}[/red] could not be loaded:")
        for problem in error.problems:
            console.print(f"    - {escape(problem)}")

    # Exit code 1 if anything is broken, so scripts can notice.
    return 1 if found.broken else 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "version":
        return cmd_version()
    if args.command == "list":
        return cmd_list(args.protocols_dir)
    return 1  # unreachable: argparse rejects unknown commands


if __name__ == "__main__":
    sys.exit(main())
