"""Step 1 tests: the project installs, the config loads, the CLI starts."""

from core import __version__
from core.cli import main
from core.config import DEFAULTS, load_config, resolve_path, PROJECT_ROOT


def test_version_is_set():
    assert __version__


def test_bundled_config_loads():
    config = load_config()
    assert config["backend"]["kind"] == "ollama"
    assert config["roles"]["analyst"] == "auto"


def test_missing_config_falls_back_to_defaults(tmp_path):
    config = load_config(tmp_path / "does_not_exist.toml")
    assert config == DEFAULTS


def test_partial_config_keeps_other_defaults(tmp_path):
    cfg_file = tmp_path / "console.toml"
    cfg_file.write_text('[roles]\nanalyst = "my-model"\n')
    config = load_config(cfg_file)
    assert config["roles"]["analyst"] == "my-model"
    assert config["roles"]["worker"] == "auto"           # kept from defaults
    assert config["paths"]["reports"] == "reports"        # whole table kept


def test_relative_paths_resolve_to_project_root():
    config = load_config()
    assert resolve_path(config, "reports") == PROJECT_ROOT / "reports"


def test_cli_version_runs(capsys):
    assert main(["version"]) == 0
    assert "AI Core Console" in capsys.readouterr().out
