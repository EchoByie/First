"""Step 9 tests: the passive network map, with fake ARP/route data (no real network)."""

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fake_backend import FakeBackend

import core.collecting
from core.collecting import CollectedData, CollectorContext, load_function
from core.config import load_config
from core.manifest import load_manifest
from core.pipeline import Pipeline, RunOptions
from core.refdata.importer import import_files

ROOT = Path(__file__).parent.parent
PROTOCOL = ROOT / "protocols" / "network_map"
NET = Path(__file__).parent / "fixtures" / "network"
REF = Path(__file__).parent / "fixtures" / "refdata"
MANIFEST = load_manifest(PROTOCOL)
collector = load_function(MANIFEST.files["collector"], "collect")
enricher = load_function(MANIFEST.files["enricher"], "enrich")
# The helper functions in the same files (parsers, drop_reason, ...):
mod = SimpleNamespace(**collector.__globals__)
emod = SimpleNamespace(**enricher.__globals__)

LINUX_FILES = {"/proc/net/arp": NET / "linux_arp.txt", "/proc/net/route": NET / "linux_route.txt"}


@pytest.fixture
def refdb(tmp_path):
    path = tmp_path / "reference.db"
    import_files(path, [REF / n for n in ("oui.csv", "mam.csv", "oui36.csv", "iab.csv", "cid.csv")])
    return path


@pytest.fixture
def fake_system(monkeypatch):
    """Serve fixture files/commands instead of the real system. The manifest's
    allow-lists are still enforced, because CollectorContext checks them
    before these fakes are reached."""
    state = {"windows_arp": "windows_arp.txt", "commands": []}

    def fake_read(path):
        return LINUX_FILES[str(path)].read_text()

    def fake_run(argv, allowed, timeout=20):
        assert argv in allowed
        state["commands"].append(argv)
        name = state["windows_arp"] if argv == ["arp", "-a"] else "windows_route.txt"
        return (NET / name).read_text()

    monkeypatch.setattr(core.collecting, "_read_capped", fake_read)
    monkeypatch.setattr(core.collecting, "run_allowed", fake_run)
    return state


def collect_and_enrich(platform, refdb_path=None):
    ctx = CollectorContext(MANIFEST, platform, reference_db=refdb_path)
    data = CollectedData.coerce(collector(ctx), "collect")
    return enricher(data, ctx)


# --- parsers --------------------------------------------------------------------------

def test_linux_arp_parser():
    entries = mod.parse_linux_arp((NET / "linux_arp.txt").read_text())
    assert len(entries) == 10
    first = entries[0]
    assert first == {"ip": "192.168.1.1", "mac": "a4:2b:b0:11:22:33", "interface": "wlan0",
                     "complete": True, "type": "dynamic"}
    assert entries[4]["complete"] is False                     # flags 0x0
    assert entries[5]["type"] == "static"                      # flags 0x6


def test_linux_route_finds_default_gateway():
    gateways = mod.parse_linux_route((NET / "linux_route.txt").read_text())
    assert gateways == [{"ip": "192.168.1.1", "interface": "wlan0", "metric": 600}]


def test_windows_parsers():
    entries = mod.parse_windows_arp((NET / "windows_arp.txt").read_text())
    assert len(entries) == 8
    assert entries[0]["mac"] == "a4:2b:b0:11:22:33" and entries[0]["interface"] == "192.168.1.10"
    assert entries[-1]["interface"] == "10.0.0.5"
    routes = mod.parse_windows_route((NET / "windows_route.txt").read_text())
    assert routes == [{"ip": "192.168.1.1", "interface": "192.168.1.10", "metric": 35}]


def test_german_windows_output_is_understood():
    entries = mod.parse_windows_arp((NET / "windows_arp_german_spoof.txt").read_text())
    assert len(entries) == 4
    assert entries[0]["type"] == "dynamic" and entries[-1]["type"] == "static"


def test_junk_input_gives_no_entries():
    assert mod.parse_linux_arp("garbage\nmore garbage 1 2 3 4 5") == []
    assert mod.parse_windows_arp("Interface: nope\n  zzz") == []
    assert mod.parse_linux_route("x\nwlan0 00000000 ZZZZ 0003 0 0 1 0") == []


# --- filtering and flags -------------------------------------------------------------------

@pytest.mark.parametrize("ip, mac, complete, reason", [
    ("192.168.1.50", "00:00:00:00:00:00", False, "incomplete"),
    ("192.168.1.255", "ff:ff:ff:ff:ff:ff", True, "broadcast"),
    ("224.0.0.251", "01:00:5e:00:00:fb", True, "multicast"),
    ("192.168.1.9", "33:33:00:00:00:01", True, "multicast"),       # IPv6 multicast MAC
    ("192.168.1.9", "a4:2b:b0:11:22:33", True, None),
])
def test_drop_reasons(ip, mac, complete, reason):
    assert emod.drop_reason({"ip": ip, "mac": mac, "complete": complete}) == reason


def test_linux_end_to_end_facts(fake_system, refdb):
    data = collect_and_enrich("linux", refdb)
    lines = data.text.splitlines()
    assert len(lines) == 7                                     # 10 read - 3 dropped
    by_ip = {r["ip"]: r for r in data.records}
    assert by_ip["192.168.1.1"]["gateway"] is True
    assert by_ip["192.168.1.1"]["vendor"] == "Sample Networking Ltd (MA-L)"
    assert by_ip["192.168.1.60"]["vendor"] == "Micro Camera Oy (MA-S)"       # most specific
    assert by_ip["192.168.1.41"]["randomized"] == "yes"
    assert by_ip["192.168.1.41"]["vendor"] == "none (randomized MAC)"
    assert by_ip["172.17.0.2"]["randomized"].startswith("yes (02:42 prefix commonly used by Docker")
    assert by_ip["192.168.1.77"]["vendor"] == "not in IEEE registry"
    assert "entries dropped: 1 broadcast, 1 incomplete, 1 multicast" in data.preamble
    assert "default gateway: 192.168.1.1 via wlan0 (metric 600)" in data.preamble
    assert "randomized MACs: 2 of 7" in data.preamble
    assert "nothing was pinged or scanned" in data.preamble
    assert any("device" in l and "ip 192.168.1.20 | mac 00:1b:a9:12:34:56 | vendor "
               "Example Printer Works (MA-L)" in l for l in lines)


def test_windows_end_to_end(fake_system, refdb):
    data = collect_and_enrich("windows", refdb)
    assert fake_system["commands"] == [["arp", "-a"], ["route", "print", "-4"]]
    ips = [r["ip"] for r in data.records]
    assert ips == ["10.0.0.1", "192.168.1.1", "192.168.1.20", "192.168.1.41"]   # sorted, filtered


def test_spoofing_pattern_is_flagged(fake_system, refdb):
    fake_system["windows_arp"] = "windows_arp_german_spoof.txt"
    data = collect_and_enrich("windows", refdb)
    assert ("one MAC on several IPs: a4:2b:b0:11:22:33 answers for 192.168.1.1, 192.168.1.66"
            in data.preamble)


def test_cid_registered_local_mac_is_not_called_random(refdb):
    from core.refdata.db import RefDB
    with RefDB(refdb) as db:
        vendor, randomized = emod.describe_vendor("da:0a:0a:00:00:01", db)
    assert vendor == "Example Wireless Alliance (CID)" and randomized.startswith("no")


def test_without_reference_data_it_still_works(fake_system, tmp_path):
    data = collect_and_enrich("linux", tmp_path / "missing.db")
    router = next(r for r in data.records if r["ip"] == "192.168.1.1")
    assert router["vendor"] == "unknown (no reference data imported)"
    assert "vendor data: not available" in data.preamble
    assert any("No MAC vendor data" in c for c in data.caveats)


# --- the full pipeline with a fake model ---------------------------------------------------------

def network_answer():
    return json.dumps({
        "summary": "Seven devices: a router, a printer, a phone, a camera and others.",
        "findings": [
            {"kind": "observation", "subject": "192.168.1.1",
             "claim": "192.168.1.1 is the default gateway",
             "evidence": ["default gateway: 192.168.1.1 via wlan0"],
             "confidence": "high", "basis": "observed"},
            {"kind": "device_guess", "subject": "192.168.1.20",
             "claim": "192.168.1.20 is probably a network printer",
             "evidence": ["vendor Example Printer Works (MA-L)"],
             "confidence": "medium", "basis": "inferred"},
            {"kind": "device_guess", "subject": "192.168.1.35",
             "claim": "192.168.1.35 is an iPhone 15 belonging to Alex",       # invented
             "evidence": ["hostname alex-iphone"],
             "confidence": "high", "basis": "observed"},
        ],
    })


def test_network_map_pipeline(fake_system, refdb, tmp_path, monkeypatch):
    monkeypatch.setattr("core.pipeline.current_platform", lambda: "linux")
    config = load_config()
    config["paths"]["reference_db"] = str(refdb)
    backend = FakeBackend(replies=[network_answer()])
    pipeline = Pipeline(config, backend=backend, reports_dir=tmp_path / "r",
                        audit_path=tmp_path / "a.jsonl")
    result = pipeline.run(MANIFEST, RunOptions())
    assert result.status == "ok", result.errors

    verdicts = {f["subject"]: f["verification"] for f in result.report["answer"]["findings"]}
    assert verdicts["192.168.1.1"]["verdict"] == "SURE"
    assert verdicts["192.168.1.20"]["verdict"] == "THINK"        # a guess is an inference
    assert verdicts["192.168.1.35"]["verdict"] == "THINK"        # invented hostname caught
    assert not verdicts["192.168.1.35"]["evidence_verified"]

    caveats = result.report["caveats"]
    assert caveats[0].startswith("An ARP table is a partial view")   # always, from code
    assert len(result.report["records"]) == 7
    md = result.report_paths[1].read_text()
    assert "partial view" in md and "no device was pinged" in md.lower()


# --- passive by design: static checks on the protocol's own code -------------------------------------

def test_protocol_code_cannot_touch_the_network():
    """The collector and enricher must not import anything that can send
    packets or start programs; they must go through ctx (allow-listed)."""
    forbidden = {"socket", "subprocess", "urllib", "http", "requests", "scapy", "os", "asyncio"}
    for path in PROTOCOL.glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [(node.module or "").split(".")[0]]
            assert not set(names) & forbidden, f"{path.name} imports {names}"


def test_allow_list_has_no_active_tools():
    commands = {c[0] for c in MANIFEST.allowed_commands}
    assert commands <= {"arp", "route"}
    assert not commands & {"ping", "nmap", "arping", "nbtstat", "nslookup", "curl"}
    assert MANIFEST.input_kind == "none" and MANIFEST.risk_level == "low"
