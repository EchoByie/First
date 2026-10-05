"""Step 13 tests: the dashboard, driven by Textual's test pilot (headless).

The pilot presses keys and clicks buttons like a person would; we then
check what the screen shows. Fake model, fake downloads, temp folders.
"""

import json
from pathlib import Path

import pytest
from fake_backend import FakeBackend
from rich.text import Text
from textual.widgets import DataTable, ListView, Static

from core.config import load_config
from core.tui.app import ConsoleApp

ROOT = Path(__file__).parent.parent
SAMPLES = Path(__file__).parent / "fixtures"
REF = SAMPLES / "refdata"
FILES = {"MA-L": "oui.csv", "MA-M": "mam.csv", "MA-S": "oui36.csv", "IAB": "iab.csv", "CID": "cid.csv"}


@pytest.fixture
def config(tmp_path):
    c = load_config()
    c["paths"].update(reports=str(tmp_path / "reports"), audit_log=str(tmp_path / "audit.jsonl"),
                      reference_db=str(tmp_path / "ref.db"))
    c["ui"]["boot_animation"] = False     # keep tests fast
    return c


def digest_answer():
    return json.dumps({
        "summary": "A service log with a 3-hour gap.", "next_steps": [],
        "findings": [
            {"kind": "anomaly", "claim": "The log is silent for over three hours",
             "evidence": ["largest gap between lines: 3h 22m 35s"],
             "confidence": "high", "basis": "observed", "subject": "timestamps"},
            {"kind": "hypothesis", "claim": "The service crashed",
             "evidence": ["service started"], "confidence": "medium", "basis": "inferred"},
        ]})


def text_of(widget) -> str:
    content = widget.render()
    return content.plain if isinstance(content, Text) else str(content)


async def wait_for_workers(app, pilot):
    await app.workers.wait_for_complete()
    await pilot.pause()


async def test_starts_with_protocols_and_health(config):
    app = ConsoleApp(config, backend=FakeBackend())
    async with app.run_test(size=(140, 45)) as pilot:
        await wait_for_workers(app, pilot)
        names = [item.name for item in app.query_one("#protocols", ListView).children]
        assert names == ["digest", "network_map"]
        health = text_of(app.query_one("#health-text", Static))
        assert "ollama" in health and "refdata" in health
        assert "no reference database" in health                 # explains the red dot
        assert "ollama" in text_of(app.query_one("#status-bar", Static))


async def test_run_digest_shows_verdicts(config):
    app = ConsoleApp(config, backend=FakeBackend(replies=[digest_answer()]))
    async with app.run_test(size=(140, 45)) as pilot:
        await wait_for_workers(app, pilot)
        app.query_one("#input").value = str(SAMPLES / "digest" / "app.log")
        await pilot.press("r")
        await wait_for_workers(app, pilot)
        assert app.last_result.status == "ok"
        table = app.query_one("#findings", DataTable)
        rows = [[str(c) for c in table.get_row_at(i)] for i in range(table.row_count)]
        assert rows[0][0] == "✔ SURE" and rows[1][0] == "? THINK"
        assert rows[0][1] == "✔" and rows[1][1] == "✔"           # evidence checks
        assert "I'm sure that the log is silent" in rows[0][4]
        assert "partial" not in text_of(app.query_one("#caveats", Static))
        assert "This digest only covers" in text_of(app.query_one("#caveats", Static))
        assert app.query_one("#tabs").active == "tab-output"


async def test_dry_run_switches_to_prompt_tab(config):
    backend = FakeBackend()
    app = ConsoleApp(config, backend=backend)
    async with app.run_test(size=(140, 45)) as pilot:
        await wait_for_workers(app, pilot)
        app.query_one("#input").value = "hello world, this is pasted text"
        await pilot.press("d")
        await wait_for_workers(app, pilot)
        assert app.last_result.status == "dry-run" and backend.calls == []
        assert app.query_one("#tabs").active == "tab-prompt"


async def test_missing_input_is_explained(config):
    app = ConsoleApp(config, backend=FakeBackend())
    async with app.run_test(size=(140, 45)) as pilot:
        await wait_for_workers(app, pilot)
        await pilot.press("r")                      # digest selected, no input typed
        await wait_for_workers(app, pilot)
        assert app.last_result.status == "refused"
        assert "needs a file or text" in text_of(app.query_one("#summary", Static))


async def test_model_text_is_not_interpreted_as_markup(config):
    answer = json.loads(digest_answer())
    answer["summary"] = "[bold red]FAKE ALERT[/bold red] [link=http://evil]click[/link]"
    app = ConsoleApp(config, backend=FakeBackend(replies=[json.dumps(answer)]))
    async with app.run_test(size=(140, 45)) as pilot:
        await wait_for_workers(app, pilot)
        app.query_one("#input").value = str(SAMPLES / "digest" / "app.log")
        await pilot.press("r")
        await wait_for_workers(app, pilot)
        shown = text_of(app.query_one("#summary", Static))
        assert "[bold red]FAKE ALERT[/bold red]" in shown        # shown literally


async def test_update_tab_downloads_only_on_click(config):
    calls = []

    def fake_fetch(url, hosts, progress):
        calls.append(url)
        name = next(k for k in FILES if url.endswith(FILES[k]) or (k == "MA-L" and url.endswith("oui.csv")))
        data = (REF / FILES[name]).read_bytes()
        progress(len(data), len(data))
        return data

    app = ConsoleApp(config, backend=FakeBackend(), fetch=fake_fetch)
    async with app.run_test(size=(140, 45)) as pilot:
        await wait_for_workers(app, pilot)
        await pilot.press("u")
        await pilot.pause()
        assert app.query_one("#tabs").active == "tab-update"
        assert calls == []                                       # opening the tab downloads nothing
        sources = app.query_one("#sources", DataTable)
        assert str(sources.get_row_at(0)[2]) == "never"
        await pilot.click("#update-all")
        await wait_for_workers(app, pilot)
        assert len(calls) == 5
        assert str(app.query_one("#sources", DataTable).get_row_at(0)[2]) == "0 days"
        await wait_for_workers(app, pilot)                       # health re-check after update
        assert "no reference database" not in text_of(app.query_one("#health-text", Static))


async def test_network_map_shows_device_table(config, monkeypatch, tmp_path):
    from core.refdata.importer import import_files
    import_files(Path(config["paths"]["reference_db"]), [REF / f for f in FILES.values()])
    net = SAMPLES / "network"
    files = {"/proc/net/arp": net / "linux_arp.txt", "/proc/net/route": net / "linux_route.txt"}
    monkeypatch.setattr("core.collecting._read_capped", lambda p: files[str(p)].read_text())
    monkeypatch.setattr("core.pipeline.current_platform", lambda: "linux")
    answer = json.dumps({"summary": "Seven devices.", "findings": [
        {"kind": "device_guess", "subject": "192.168.1.20", "claim": "192.168.1.20 is probably a printer",
         "evidence": ["vendor Example Printer Works"], "confidence": "medium", "basis": "inferred"}]})
    app = ConsoleApp(config, backend=FakeBackend(replies=[answer]))
    async with app.run_test(size=(160, 50)) as pilot:
        await wait_for_workers(app, pilot)
        await pilot.press("down")                                # select network_map
        await pilot.press("r")
        await wait_for_workers(app, pilot)
        assert app.last_result.status == "ok", app.last_result.errors
        devices = app.query_one("#devices", DataTable)
        assert devices.display and devices.row_count == 7
        row = next(devices.get_row_at(i) for i in range(7) if str(devices.get_row_at(i)[0]) == "192.168.1.20")
        assert "printer" in str(row[5])
        assert "partial view" in text_of(app.query_one("#caveats", Static))


async def test_audit_tab_lists_runs(config):
    app = ConsoleApp(config, backend=FakeBackend(replies=[digest_answer()]))
    async with app.run_test(size=(140, 45)) as pilot:
        await wait_for_workers(app, pilot)
        app.query_one("#input").value = str(SAMPLES / "digest" / "app.log")
        await pilot.press("r")
        await wait_for_workers(app, pilot)
        await pilot.press("a")
        await pilot.pause()
        audit = app.query_one("#audit", DataTable)
        assert audit.row_count == 1 and str(audit.get_row_at(0)[3]) == "ok"
