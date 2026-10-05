"""Command-line entry point: `aicore <command>` or `python -m core.cli <command>`.

Commands so far: version, list, models, run, dry-run. Later steps add
health, update and console.

We use argparse because it is built into Python and gives `--help` for free.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import json

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from core import __version__
from core.backends import BackendError, make_backend
from core.config import load_config, resolve_path
from core.discovery import current_platform, discover
from core.collecting import UserInput
from core.models import gather_models, resolve_roles
from core.pipeline import Pipeline, RunOptions

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

    models_cmd = sub.add_parser(
        "models", help="show installed models and which one plays each protocol role")
    models_cmd.add_argument("--protocols-dir", type=Path, default=None)

    for name, help_text in (("run", "run a protocol and save its report"),
                            ("dry-run", "show exactly what would be sent to the model, without sending it")):
        cmd = sub.add_parser(name, help=help_text)
        cmd.add_argument("protocol", help="protocol name (see `aicore list`)")
        source = cmd.add_mutually_exclusive_group()
        source.add_argument("--file", type=Path, help="input file (use - to read from stdin)")
        source.add_argument("--text", help="input text")
        cmd.add_argument("--yes", action="store_true", help="confirm running a high-risk protocol")
        cmd.add_argument("--protocols-dir", type=Path, default=None)
        if name == "dry-run":
            cmd.add_argument("--full", action="store_true", help="print the data and schema in full")
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


def cmd_models(protocols_dir: Path | None, backend=None) -> int:
    """`backend` can be passed in by tests; normally it comes from console.toml."""
    config = load_config()
    try:
        backend = backend or make_backend(config)
        version = backend.version()
        models = gather_models(backend)
    except BackendError as e:
        console.print(f"[red]✗ {escape(str(e))}[/red]")
        return 1

    console.print(f"[green]●[/green] {backend.kind} {escape(version)} at {escape(backend.url)}")
    if not models:
        console.print("[yellow]No models installed. Try: ollama pull llama3.1:8b[/yellow]")
        return 1

    table = Table(title="Installed models")
    for col in ("Model", "Family", "Size (B params)", "Context (tokens)", "Can chat"):
        table.add_column(col)
    for m in models:
        table.add_row(
            escape(m.name), escape(m.family or "?"),
            f"{m.parameters_billions:g}" if m.parameters_billions else "?",
            f"{m.context_length:,}" if m.context_length else "?",
            "[green]yes[/green]" if m.can_chat else "[red]no[/red]",
        )
    console.print(table)

    found = discover(protocols_dir or resolve_path(config, "protocols"))
    if not found.protocols:
        return 0
    roles_table = Table(title=escape("Role assignments (from console.toml [roles])"))
    for col in ("Protocol", "Role", "Model", "Why"):
        roles_table.add_column(col)
    for manifest in found.protocols.values():
        for role, choice in resolve_roles(manifest.roles, config["roles"], models).items():
            optional = manifest.roles[role].optional
            if choice.ok:
                model_text = f"[green]{escape(choice.model)}[/green]"
            elif optional:
                model_text = "[yellow]none (optional)[/yellow]"
            else:
                model_text = "[red]none[/red]"
            why = "; ".join([choice.reason] + choice.warnings)
            roles_table.add_row(escape(manifest.name), role, model_text, escape(why))
    console.print(roles_table)
    return 0


VERDICT_STYLE = {"SURE": "[bold green]✔[/bold green]", "THINK": "[yellow]?[/yellow]"}
PREVIEW_CHARS = 3000


def _user_input(args) -> UserInput:
    if args.file is not None and str(args.file) == "-":
        return UserInput(text=sys.stdin.read())
    return UserInput(file=args.file, text=args.text)


def _show_progress(stage: str, message: str) -> None:
    console.print(f"  [dim]{stage:>9}[/dim]  {escape(message)}")


def _show_report(report: dict) -> None:
    answer = report.get("answer")
    if answer:
        console.print(Panel(escape(answer["summary"]), title="Summary", expand=False))
        for f in answer["findings"]:
            v = f["verification"]
            subject = f" [dim]({escape(f['subject'])})[/dim]" if f.get("subject") else ""
            console.print(f"  {VERDICT_STYLE[v['verdict']]} {escape(v['statement'])}{subject}")
            if not v["evidence_verified"]:
                console.print("      [red]unverified evidence: downgraded[/red]")
        vs = answer["verification_summary"]
        console.print(f"\n  findings: {vs['total']}   [green]sure: {vs['sure']}[/green]   "
                      f"[yellow]not sure: {vs['think']}[/yellow]   "
                      f"[red]unverified: {vs['unverified_evidence']}[/red]")
    for caveat in report.get("caveats", []):
        console.print(f"  [cyan]caveat:[/cyan] {escape(caveat)}")


def _shorten(text: str, full: bool) -> str:
    if full or len(text) <= PREVIEW_CHARS:
        return text
    return text[:PREVIEW_CHARS] + f"\n... ({len(text) - PREVIEW_CHARS:,} more characters; use --full)"


def cmd_run(args, backend=None) -> int:
    config = load_config()
    found = discover(args.protocols_dir or resolve_path(config, "protocols"))
    manifest = found.protocols.get(args.protocol)
    if manifest is None:
        console.print(f"[red]No protocol named {escape(args.protocol)!s}.[/red] See `aicore list`.")
        return 1

    dry_run = args.command == "dry-run"
    console.print(f"[bold]{'DRY-RUN' if dry_run else 'RUN'}[/bold] {escape(manifest.title)}")
    pipeline = Pipeline(config, backend=backend, progress=_show_progress)
    result = pipeline.run(manifest, RunOptions(
        user_input=_user_input(args), dry_run=dry_run, confirmed=args.yes))

    if result.status == "dry-run":
        p = result.preview
        console.print(Panel(escape(p["system"]), title="system message", expand=False))
        console.print(Panel(escape(_shorten(p["user"], args.full)), title="user message (the data)",
                            expand=False))
        if args.full:
            console.print(Panel(escape(json.dumps(p["schema"], indent=2)), title="answer schema"))
        if p.get("parts"):
            console.print(f"  [bold]big input:[/bold] {len(p['parts'])} parts, read by the worker, "
                          "then combined by the analyst")
            for label in p["parts"][:10]:
                console.print(f"    - {escape(label)}")
            if len(p["parts"]) > 10:
                console.print(f"    ... and {len(p['parts']) - 10} more")
            console.print(f"  [dim]{escape(p['note'])}[/dim]")
        console.print(f"  models: {escape(str(p['models']))}")
        console.print(f"  estimated prompt tokens: {p['estimated_prompt_tokens']:,}   "
                      f"context window requested: {p['context_tokens']:,}")
        console.print("  [green]Nothing was sent to the model.[/green]")
        return 0

    if result.report:
        _show_report(result.report)
    for error in result.errors:
        console.print(f"[red]✗ {escape(error)}[/red]")
    if result.report_paths:
        console.print(f"  report: {result.report_paths[0]}\n          {result.report_paths[1]}")
    return 0 if result.status == "ok" else 1


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "version":
        return cmd_version()
    if args.command == "list":
        return cmd_list(args.protocols_dir)
    if args.command == "models":
        return cmd_models(args.protocols_dir)
    if args.command in ("run", "dry-run"):
        return cmd_run(args)
    return 1  # unreachable: argparse rejects unknown commands


if __name__ == "__main__":
    sys.exit(main())
