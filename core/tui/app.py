"""AI Core Console: the full-screen dashboard.

Layout
  ┌ status bar: models · Ollama dot · reference data age ──────────────────┐
  │ banner            │ [Output] [Prompt] [Audit] [Update]                  │
  │ PROTOCOLS         │   summary, ✔ / ? findings, devices, caveats         │
  │ input box         │   live steps of the run                             │
  │ HEALTH ● ● ●      │                                                     │
  └ keys: r run · d dry-run · h health · u update · q quit ────────────────┘

How it stays responsive: runs, health checks and downloads happen in
background threads (Textual "workers"); they send progress back to the
screen with call_from_thread.

Safety: everything shown that came from data or the model is wrapped in
rich Text objects, which are never interpreted as formatting codes.
"""

from __future__ import annotations

from pathlib import Path

from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import (
    Button, DataTable, Footer, Input, Label, ListItem, ListView, ProgressBar, RichLog,
    Sparkline, Static, TabbedContent, TabPane,
)

from core import __version__, audit
from core.collecting import UserInput
from core.config import load_config, resolve_path
from core.discovery import discover
from core.health import FAIL, OK, WARN
from core.health.run import run_all
from core.pipeline import Pipeline, RunOptions
from core.updater import fetch as updater

BANNER = """\
 ▄▀█ █   █▀▀ █▀█ █▀█ █▀▀
 █▀█ █   █▄▄ █▄█ █▀▄ ██▄
   C O N S O L E"""

BOOT_LINES = [
    "initialising AI core console v{version}",
    "loading protocol registry ......... {protocols} found",
    "model isolation ................... no tools, no shell, no network",
    "evidence verification ............. armed",
    "checking health ...",
]

DOT = {OK: ("●", "green"), WARN: ("●", "yellow"), FAIL: ("●", "red")}
VERDICT = {"SURE": ("✔ SURE", "bold green"), "THINK": ("? THINK", "yellow")}
RISK_STYLE = {"low": "green", "medium": "yellow", "high": "bold red"}


def plain(value, style: str = "") -> Text:
    """Untrusted text -> a Text object (never parsed as markup)."""
    return Text(str(value), style=style)


class ConfirmScreen(ModalScreen[bool]):
    """Asks before running a high-risk protocol."""

    def __init__(self, question: str):
        super().__init__()
        self.question = question

    def compose(self) -> ComposeResult:
        with Vertical(id="confirm-box"):
            yield Label(plain(self.question))
            with Horizontal():
                yield Button("Run it", variant="error", id="yes")
                yield Button("Cancel", variant="primary", id="no")

    @on(Button.Pressed)
    def answer(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "yes")


class ConsoleApp(App):
    TITLE = "AI Core Console"
    CSS_PATH = "console.tcss"
    BINDINGS = [
        Binding("r", "run", "Run"),
        Binding("d", "dry_run", "Dry-run"),
        Binding("h", "health", "Health"),
        Binding("u", "show_update", "Update"),
        Binding("a", "show_audit", "Audit"),
        Binding("q", "quit", "Quit"),
    ]

    def __init__(self, config: dict | None = None, backend=None, protocols_dir: Path | None = None,
                 fetch=None):
        super().__init__()
        self.config = config or load_config()
        self.backend = backend              # tests pass a fake; normally made from config
        self.protocols_dir = protocols_dir or resolve_path(self.config, "protocols")
        self.fetch = fetch                  # tests pass a fake downloader
        self.protocols = discover(self.protocols_dir).protocols
        self.selected: str | None = next(iter(self.protocols), None)
        self.health_report = None
        self.last_result = None

    # ------------------------------------------------------------------ layout
    def compose(self) -> ComposeResult:
        yield Static(id="status-bar")
        with Vertical(id="left"):
            if self.config["ui"].get("banner", True):
                yield Static(BANNER, id="banner")
            yield Label("PROTOCOLS", classes="pane-title")
            yield ListView(*[self._protocol_item(m) for m in self.protocols.values()],
                           id="protocols")
            yield Input(placeholder="file path or text (for digest)", id="input")
            yield Static("", id="input-hint")
            yield Label("HEALTH", classes="pane-title")
            yield VerticalScroll(Static("checking ...", id="health-text"), id="health")
        with Vertical(id="right"):
            with TabbedContent(id="tabs"):
                with TabPane("Output", id="tab-output"):
                    with VerticalScroll():
                        yield Static(plain("Select a protocol and press r to run, d to dry-run."),
                                     id="summary")
                        yield DataTable(id="findings", zebra_stripes=True)
                        yield DataTable(id="devices", zebra_stripes=True)
                        yield Static("", id="caveats")
                        yield RichLog(id="steps", wrap=True, markup=False)
                with TabPane("Prompt", id="tab-prompt"):
                    yield RichLog(id="prompt", wrap=True, markup=False)
                with TabPane("Audit", id="tab-audit"):
                    yield Label("Run time of recent runs (seconds)")
                    yield Sparkline([], id="durations")
                    yield DataTable(id="audit", zebra_stripes=True)
                with TabPane("Update", id="tab-update"):
                    yield Label("Reference data: downloads happen ONLY when you press a button, "
                                "and only from the allow-listed IEEE addresses.")
                    yield DataTable(id="sources", zebra_stripes=True, cursor_type="row")
                    with Horizontal(id="update-buttons"):
                        yield Button("Update selected", id="update-one", variant="primary")
                        yield Button("Update all", id="update-all", variant="warning")
                    yield ProgressBar(id="update-progress", show_eta=False)
                    yield RichLog(id="update-log", wrap=True, markup=False)
        if self.config["ui"].get("boot_animation", True):
            yield Static("", id="boot")
        yield Footer()

    def _protocol_item(self, m) -> ListItem:
        label = Text.assemble((f"{m.name:<13}", "bold"), (f" {m.risk_level}", RISK_STYLE[m.risk_level]))
        return ListItem(Label(label), name=m.name)

    def on_mount(self) -> None:
        self.query_one("#findings", DataTable).add_columns("Verdict", "Evidence", "Kind", "Subject", "Statement")
        self.query_one("#audit", DataTable).add_columns("When (UTC)", "Protocol", "Mode", "Status", "Sure/Total", "Seconds")
        self.query_one("#sources", DataTable).add_columns("Registry", "Entries", "Age", "Source")
        self.query_one("#devices").display = False
        self._update_status_bar()
        self._update_input_hint()
        self.refresh_audit()
        self.refresh_sources()
        if self.query("#boot"):
            self._boot_step = 0
            self._boot_timer = self.set_interval(0.25, self._boot_tick)
        self.action_health()

    # ------------------------------------------------------------- cosmetics
    def _boot_tick(self) -> None:
        boot = self.query_one("#boot", Static)
        if self._boot_step >= len(BOOT_LINES):
            self._boot_timer.stop()
            boot.remove()
            return
        lines = [l.format(version=__version__, protocols=len(self.protocols))
                 for l in BOOT_LINES[: self._boot_step + 1]]
        boot.update(Text("\n".join(f"> {l}" for l in lines) + " ▌"))
        self._boot_step += 1

    def _update_status_bar(self) -> None:
        bar = Text()
        bar.append(f"AI CORE v{__version__}", style="bold")
        roles = self.config["roles"]
        bar.append(f"  │  analyst: {roles.get('analyst', 'auto')}  worker: {roles.get('worker', 'auto')}")
        if self.health_report:
            for area, label in (("ollama", "ollama"), ("models", "models"), ("refdata", "refdata")):
                symbol, colour = DOT[self.health_report.area_status(area)]
                bar.append(f"  │  {label} ")
                bar.append(symbol, style=colour)
        self.query_one("#status-bar", Static).update(bar)

    # ------------------------------------------------------------ protocols
    @on(ListView.Highlighted, "#protocols")
    def protocol_highlighted(self, event: ListView.Highlighted) -> None:
        if event.item is not None:
            self.selected = event.item.name
            self._update_input_hint()

    def _update_input_hint(self) -> None:
        manifest = self.protocols.get(self.selected)
        hint = self.query_one("#input-hint", Static)
        box = self.query_one("#input", Input)
        if manifest is None:
            hint.update(plain("No protocols found."))
            return
        needs = manifest.input_kind == "file_or_text"
        box.disabled = not needs
        hint.update(plain(manifest.description if not needs else
                          "Needs input: type a file path or paste text above."))

    def _user_input(self, manifest) -> UserInput:
        if manifest.input_kind != "file_or_text":
            return UserInput()
        value = self.query_one("#input", Input).value.strip()
        if not value:
            return UserInput()
        path = Path(value).expanduser()
        return UserInput(file=path) if path.is_file() else UserInput(text=value)

    # ------------------------------------------------------------------ runs
    def action_run(self) -> None:
        self._start(dry_run=False)

    def action_dry_run(self) -> None:
        self._start(dry_run=True)

    def _start(self, dry_run: bool) -> None:
        manifest = self.protocols.get(self.selected)
        if manifest is None:
            self.notify("No protocol selected", severity="warning")
            return
        if manifest.risk_level == "high" and not dry_run:
            def after(ok: bool | None) -> None:
                if ok:
                    self._launch(manifest, dry_run, confirmed=True)
            self.push_screen(ConfirmScreen(f"{manifest.title} is HIGH RISK. Run it?"), after)
            return
        self._launch(manifest, dry_run, confirmed=False)

    def _launch(self, manifest, dry_run: bool, confirmed: bool) -> None:
        self.query_one("#tabs", TabbedContent).active = "tab-output"
        steps = self.query_one("#steps", RichLog)
        steps.clear()
        steps.write(plain(f"{'DRY-RUN' if dry_run else 'RUN'} {manifest.title}", "bold"))
        self.query_one("#summary", Static).update(plain("working ...", "italic"))
        self.query_one("#findings", DataTable).clear()
        self.query_one("#devices").display = False
        self.query_one("#caveats", Static).update("")
        self.run_protocol(manifest, RunOptions(user_input=self._user_input(manifest),
                                               dry_run=dry_run, confirmed=confirmed))

    @work(thread=True, exclusive=True, group="run")
    def run_protocol(self, manifest, options: RunOptions) -> None:
        def progress(stage: str, message: str) -> None:
            self.call_from_thread(self._log_step, stage, message)
        pipeline = Pipeline(self.config, backend=self.backend, progress=progress)
        result = pipeline.run(manifest, options)
        self.call_from_thread(self.show_result, result)

    def _log_step(self, stage: str, message: str) -> None:
        line = Text.assemble((f"{stage:>9} ", "dim cyan"), message)
        self.query_one("#steps", RichLog).write(line)

    def show_result(self, result) -> None:
        self.last_result = result
        summary = self.query_one("#summary", Static)
        if result.status == "dry-run":
            self._show_preview(result.preview)
            summary.update(plain("Dry-run: nothing was sent to the model. See the Prompt tab.",
                                 "green"))
        elif result.report and result.report.get("answer"):
            self._show_report(result.report)
        else:
            summary.update(plain(f"{result.status.upper()}: " + "; ".join(result.errors), "bold red"))
        if result.report_paths:
            self._log_step("report", str(result.report_paths[1]))
        self.refresh_audit()

    def _show_report(self, report: dict) -> None:
        answer = report["answer"]
        vs = answer["verification_summary"]
        head = Text.assemble((answer["summary"] + "\n\n", ""),
                             (f"✔ sure {vs['sure']}  ", "bold green"),
                             (f"? not sure {vs['think']}  ", "yellow"),
                             (f"unverified evidence {vs['unverified_evidence']}", "red"))
        self.query_one("#summary", Static).update(head)

        table = self.query_one("#findings", DataTable)
        table.clear()
        for f in answer["findings"]:
            v = f["verification"]
            label, style = VERDICT[v["verdict"]]
            checks = " ".join("✔" if c["status"] == "found" else "✗" for c in v["evidence_checks"])
            table.add_row(plain(label, style),
                          plain(checks, "green" if v["evidence_verified"] else "red"),
                          plain(f.get("kind", "")), plain(f.get("subject", "")),
                          plain(v["statement"]))

        records = report.get("records") or []
        devices = self.query_one("#devices", DataTable)
        if records and "ip" in records[0]:
            guesses = {f.get("subject"): f for f in answer["findings"] if f.get("kind") == "device_guess"}
            devices.clear(columns=True)
            devices.add_columns("IP", "MAC", "Vendor", "Randomized", "Gateway", "Model's guess")
            for r in records:
                g = guesses.get(r["ip"])
                guess = plain(g["claim"], VERDICT[g["verification"]["verdict"]][1]) if g else plain("-", "dim")
                randomized = r.get("randomized", "").split(" ")[0]       # "yes" / "no"
                devices.add_row(plain(r["ip"]), plain(r["mac"]), plain(r.get("vendor", "")),
                                plain(randomized, "yellow" if randomized == "yes" else "dim"),
                                plain("yes", "bold") if r.get("gateway") else plain("no", "dim"),
                                guess)
            devices.display = True
        caveats = report.get("caveats", [])
        self.query_one("#caveats", Static).update(
            Text("\n".join(f"⚠ {c}" for c in caveats)) if caveats else "")

    def _show_preview(self, preview: dict) -> None:
        log = self.query_one("#prompt", RichLog)
        log.clear()
        log.write(plain(f"models: {preview['models']}   context: {preview['context_tokens']:,} tokens   "
                        f"path: {preview.get('path')}", "bold"))
        if preview.get("parts"):
            log.write(plain(f"big input: {len(preview['parts'])} parts. {preview.get('note', '')}", "yellow"))
        log.write(plain("──── system message ────", "cyan"))
        log.write(plain(preview["system"]))
        log.write(plain("──── user message (the data) ────", "cyan"))
        log.write(plain(preview["user"]))
        self.query_one("#tabs", TabbedContent).active = "tab-prompt"

    # ---------------------------------------------------------------- health
    def action_health(self) -> None:
        self.query_one("#health-text", Static).update(plain("checking ...", "italic"))
        self.check_health()

    @work(thread=True, exclusive=True, group="health")
    def check_health(self) -> None:
        report = run_all(self.config, backend=self.backend, protocols_dir=self.protocols_dir)
        self.call_from_thread(self.show_health, report)

    def show_health(self, report) -> None:
        self.health_report = report
        text = Text()
        for area, results in report.by_area().items():
            symbol, colour = DOT[report.area_status(area)]
            text.append(f"{symbol} ", style=colour)
            text.append(f"{area}\n", style="bold")
            for r in results:
                if r.status != OK:
                    s, c = DOT[r.status]
                    text.append(f"   {s} ", style=c)
                    text.append(f"{r.name}: {r.message}\n")
                    if r.fix:
                        text.append(f"     → {r.fix}\n", style="dim")
        self.query_one("#health-text", Static).update(text)
        self._update_status_bar()

    # ----------------------------------------------------------------- audit
    def action_show_audit(self) -> None:
        self.refresh_audit()
        self.query_one("#tabs", TabbedContent).active = "tab-audit"

    def refresh_audit(self) -> None:
        entries = audit.read_entries(resolve_path(self.config, "audit_log"), last=50)
        table = self.query_one("#audit", DataTable)
        table.clear()
        for e in reversed(entries):
            v = e.get("verification") or {}
            sure = f"{v.get('sure', '-')}/{v.get('total', '-')}" if v else "-"
            status = e.get("status", "?")
            colour = {"ok": "green", "failed": "red", "refused": "yellow"}.get(status, "")
            table.add_row(plain(e.get("logged_at", "")[:19]), plain(e.get("protocol", e.get("registry", ""))),
                          plain(e.get("mode", "")), plain(status, colour), plain(sure),
                          plain(e.get("seconds", "")))
        runs = [float(e["seconds"]) for e in entries if e.get("mode") == "run" and e.get("seconds")]
        self.query_one("#durations", Sparkline).data = runs[-30:] or [0]

    # ---------------------------------------------------------------- update
    def action_show_update(self) -> None:
        self.refresh_sources()
        self.query_one("#tabs", TabbedContent).active = "tab-update"

    def refresh_sources(self) -> None:
        table = self.query_one("#sources", DataTable)
        table.clear()
        for st in updater.status(self.config):
            age = (plain(f"{st.age_days:.0f} days", "green" if st.age_days < 90 else "yellow")
                   if st.age_days is not None else plain("never", "red"))
            entries = f"{st.rows:,}" if st.rows is not None else "-"
            table.add_row(plain(st.source.registry), plain(entries), age, plain(st.source.url),
                          key=st.source.registry)

    @on(Button.Pressed, "#update-one")
    def update_selected(self) -> None:
        table = self.query_one("#sources", DataTable)
        if table.row_count == 0:
            return
        registry = table.coordinate_to_cell_key(table.cursor_coordinate).row_key.value
        self.download([registry])

    @on(Button.Pressed, "#update-all")
    def update_all(self) -> None:
        self.download(None)

    @work(thread=True, exclusive=True, group="update")
    def download(self, registries: list[str] | None) -> None:
        log = self.query_one("#update-log", RichLog)
        bar = self.query_one("#update-progress", ProgressBar)
        self.call_from_thread(log.write, plain(
            f"downloading {', '.join(registries) if registries else 'all registries'} ...", "bold"))

        def progress(name, done, total):
            self.call_from_thread(bar.update, total=total or done, progress=done)
        kwargs = {"fetch": self.fetch} if self.fetch else {}
        result = updater.update(self.config, registries, progress=progress, **kwargs)
        for r in result.imported:
            self.call_from_thread(log.write, plain(f"✔ {r.registry}: {r.rows:,} entries", "green"))
        for name, error in result.errors.items():
            self.call_from_thread(log.write, plain(f"✗ {name}: {error} (old data kept)", "red"))
        self.call_from_thread(self.refresh_sources)
        self.call_from_thread(self.refresh_audit)
        self.call_from_thread(self.action_health)


def main() -> None:
    ConsoleApp().run()
