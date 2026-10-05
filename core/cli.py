"""Command-line entry point: `aicore <command>` or `python -m core.cli <command>`.

Right now it only has `version`. Each build step adds a command
(list, run, dry-run, health, update, models, console).

We use argparse because it is built into Python and gives `--help` for free.
"""

from __future__ import annotations

import argparse
import sys

from core import __version__
from core.config import load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aicore",
        description="AI Core Console: run read-only protocols against a local LLM.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("version", help="show the framework version and active config")
    return parser


def cmd_version() -> int:
    config = load_config()
    print(f"AI Core Console {__version__}")
    print(f"backend: {config['backend']['kind']} at {config['backend']['url']}")
    print(f"roles:   analyst={config['roles']['analyst']}  worker={config['roles']['worker']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "version":
        return cmd_version()
    return 1  # unreachable: argparse rejects unknown commands


if __name__ == "__main__":
    sys.exit(main())
