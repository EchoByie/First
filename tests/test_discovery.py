"""Step 2 tests: manifests are checked properly and discovery finds protocols."""

import shutil
from pathlib import Path

import pytest

from core.cli import main
from core.discovery import discover
from core.manifest import DEFAULT_LIMITS, ManifestError, load_manifest

FIXTURES = Path(__file__).parent / "fixtures" / "protocols"
GOOD = FIXTURES / "example_ok"


def make_protocol(parent: Path, name: str, manifest_text: str,
                  files=("collector.py", "prompt.md", "output.schema.json")) -> Path:
    """Create a protocol folder in a temporary directory for one test."""
    folder = parent / name
    folder.mkdir()
    (folder / "manifest.toml").write_text(manifest_text)
    for f in files:
        (folder / f).write_text("x")
    return folder


def minimal(name: str = "proto") -> str:
    """Smallest valid manifest text. Tests break it in one way each."""
    return (
        f'name = "{name}"\n'
        'description = "d"\n'
        'risk_level = "low"\n'
        "[roles]\n"
        "analyst = { min_context = 8192 }\n"
    )


# --- load_manifest: the good case ------------------------------------------

def test_good_manifest_loads_with_all_fields():
    m = load_manifest(GOOD)
    assert m.name == "example_ok"
    assert m.risk_level == "low"
    assert m.roles["analyst"].min_context == 8192
    assert m.roles["worker"].optional is True
    assert m.limits["chunk_threshold_chars"] == 10000
    assert m.limits["chunk_size_chars"] == DEFAULT_LIMITS["chunk_size_chars"]  # default kept
    assert m.input_kind == "file_or_text"
    assert m.allowed_commands == [["echo", "hi"]]
    assert m.allowed_files == ["/etc/hostname"]
    assert m.required_refdata == ["oui"]
    assert set(m.files) == {"collector", "enricher", "prompt", "output_schema"}  # no worker_prompt: optional
    assert m.supports("linux") and m.supports("windows")


def test_minimal_manifest_gets_defaults(tmp_path):
    folder = make_protocol(tmp_path, "mini", minimal("mini"))
    m = load_manifest(folder)
    assert m.title == "mini"
    assert m.platforms == ["linux", "windows"]
    assert m.input_kind == "none"
    assert m.allowed_commands == []


# --- load_manifest: things that must be rejected ---------------------------

BAD_CASES = [
    # (description of the mistake, manifest text, words expected in the error)
    ("bad risk level", minimal().replace('"low"', '"extreme"'), "risk_level"),
    ("name differs from folder", minimal("other"), "must match the folder"),
    ("missing description", minimal().replace('description = "d"', ""), "description"),
    ("no roles", minimal().split("[roles]")[0], "[roles]"),
    ("only optional roles", minimal().replace("min_context = 8192", "optional = true"), "required"),
    ("unknown role key", minimal().replace("min_context = 8192", "context = 1"), "unknown keys"),
    ("bad platform", 'platforms = ["macos"]\n' + minimal(), "platform"),
    ("bad input kind", minimal() + '[input]\nkind = "webcam"\n', "[input] kind"),
    ("command as a string", minimal() + '[collector_policy]\ncommands = ["arp -a"]\n', "list of words"),
    ("negative limit", minimal() + "[limits]\ntimeout_seconds = -5\n", "positive"),
    ("chunk bigger than threshold",
     minimal() + "[limits]\nchunk_threshold_chars = 100\nchunk_size_chars = 200\n", "cannot be bigger"),
    ("file outside folder", minimal() + '[files]\ncollector = "../evil.py"\n', "inside the protocol folder"),
    ("not TOML", "this is = = not toml", "not valid TOML"),
]


@pytest.mark.parametrize("label, text, expected", BAD_CASES, ids=[c[0] for c in BAD_CASES])
def test_bad_manifests_are_rejected(tmp_path, label, text, expected):
    folder = make_protocol(tmp_path, "proto", text)
    with pytest.raises(ManifestError) as info:
        load_manifest(folder)
    assert any(expected in p for p in info.value.problems), info.value.problems


def test_missing_required_files_are_reported(tmp_path):
    folder = make_protocol(tmp_path, "nofiles", minimal("nofiles"), files=())
    with pytest.raises(ManifestError) as info:
        load_manifest(folder)
    text = " ".join(info.value.problems)
    assert "collector.py" in text and "prompt.md" in text and "output.schema.json" in text


def test_all_problems_reported_at_once(tmp_path):
    text = 'name = "many"\nrisk_level = "nope"\n'  # no description, bad risk, no roles
    folder = make_protocol(tmp_path, "many", text)
    with pytest.raises(ManifestError) as info:
        load_manifest(folder)
    assert len(info.value.problems) >= 3


# --- discover --------------------------------------------------------------

def test_discover_finds_good_and_reports_broken(tmp_path):
    shutil.copytree(GOOD, tmp_path / "example_ok")
    make_protocol(tmp_path, "broken", 'name = "broken"\n')
    make_protocol(tmp_path, "_draft", "garbage")            # skipped: starts with _
    (tmp_path / "helpers").mkdir()                           # skipped: no manifest
    (tmp_path / "notes.txt").write_text("not a folder")      # skipped: a file

    result = discover(tmp_path)
    assert list(result.protocols) == ["example_ok"]
    assert [e.folder.name for e in result.broken] == ["broken"]


def test_discover_missing_folder_is_empty(tmp_path):
    result = discover(tmp_path / "nope")
    assert result.protocols == {} and result.broken == []


# --- CLI -------------------------------------------------------------------

def test_cli_list_shows_protocol(capsys):
    assert main(["list", "--protocols-dir", str(FIXTURES)]) == 0
    assert "example_ok" in capsys.readouterr().out


def test_cli_list_reports_broken_and_fails(tmp_path, capsys):
    make_protocol(tmp_path, "broken", 'name = "broken"\n')
    assert main(["list", "--protocols-dir", str(tmp_path)]) == 1
    assert "broken" in capsys.readouterr().out


def test_cli_list_does_not_interpret_markup(tmp_path, capsys):
    text = minimal("sneaky").replace('description = "d"', 'description = "[red]x[/red]"')
    make_protocol(tmp_path, "sneaky", text)
    main(["list", "--protocols-dir", str(tmp_path)])
    assert "[red]x[/red]" in capsys.readouterr().out


def test_platform_policy_tables(tmp_path):
    text = minimal("plat") + (
        '[collector_policy]\nfiles = ["/etc/hostname"]\n'
        '[collector_policy.linux]\nfiles = ["/proc/x"]\n'
        '[collector_policy.windows]\ncommands = [["ver"]]\n')
    m = load_manifest(make_protocol(tmp_path, "plat", text))
    assert m.policy_for("linux") == ([], ["/etc/hostname", "/proc/x"])
    assert m.policy_for("windows") == ([["ver"]], ["/etc/hostname"])


def test_unknown_policy_platform_rejected(tmp_path):
    text = minimal("plat") + '[collector_policy.macos]\nfiles = ["/x"]\n'
    with pytest.raises(ManifestError, match="unknown keys"):
        load_manifest(make_protocol(tmp_path, "plat", text))
