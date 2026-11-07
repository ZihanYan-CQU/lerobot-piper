"""Self-contained layout, directory / package launch and autostart (no servers)."""

from __future__ import annotations

import ast
import json
import os
import shlex
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from lerobot.yzh.wn_a100 import monitor, remote

ROOT = Path(__file__).resolve().parents[2]
YZH = ROOT / "src/lerobot/yzh"
PACKAGE = YZH / "wn_a100"


def test_only_three_business_modules_and_two_package_entry_files():
    assert {path.name for path in PACKAGE.glob("*.py")} == {
        "__init__.py",
        "__main__.py",
        "monitor.py",
        "desktop.py",
        "remote.py",
    }
    assert not any(
        (YZH / f"wn_a100_{name}.py").exists()
        for name in ("monitor", "training", "disk", "ssh", "desktop", "launch")
    )
    assert not (YZH / "2.wn_a100_monitor.py").exists()
    assert (PACKAGE / "README.md").is_file()
    assert not (PACKAGE / "archive/launch_prototype.py.txt").exists()
    assert not (YZH / "wn_a100_training.json").exists()
    assert not (YZH / "WN_A100_MONITOR.md").exists()


def test_remote_module_has_no_gui_or_cross_feature_module_imports():
    source = ast.parse(Path(remote.__file__).read_text())
    for node in ast.walk(source):
        if isinstance(node, ast.ImportFrom):
            assert node.level == 0
            assert not (node.module or "").startswith(("PyQt5", "lerobot", "wn_a100"))
        elif isinstance(node, ast.Import):
            assert all(
                not alias.name.startswith(("PyQt5", "lerobot", "wn_a100"))
                for alias in node.names
            )


def test_config_is_single_canonical_file_and_console_entry_uses_new_package():
    assert remote.DEFAULT_CONFIG == PACKAGE / "config.json"
    configuration = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert (
        configuration["project"]["scripts"]["wn-a100-monitor"]
        == "lerobot.yzh.wn_a100.monitor:main"
    )
    assert (
        "config.json"
        in configuration["tool"]["setuptools"]["package-data"]["lerobot.yzh.wn_a100"]
    )


def test_removed_config_path_is_not_a_hidden_default_alias(monkeypatch):
    legacy = YZH / "wn_a100_training.json"
    monkeypatch.chdir(ROOT)
    for path in (legacy, legacy.relative_to(ROOT)):
        with pytest.raises(remote.RemoteError, match="配置不可用"):
            remote.load_config(path)
    assert not legacy.exists()


def test_missing_custom_config_with_old_filename_is_not_silently_replaced(tmp_path):
    with pytest.raises(remote.RemoteError, match="配置不可用"):
        remote.load_config(tmp_path / "wn_a100_training.json")


def test_existing_custom_config_with_old_filename_is_honored(tmp_path):
    payload = json.loads(remote.DEFAULT_CONFIG.read_text())
    payload["port"] = 2222
    path = tmp_path / "wn_a100_training.json"
    path.write_text(json.dumps(payload))
    assert remote.load_config(path).port == 2222


@pytest.mark.parametrize(
    "entry", ["directory", "entry_file", "package", "console_target"]
)
@pytest.mark.parametrize("mode", ["--once", "--training-once", "--disk-once"])
def test_all_supported_entry_points_work_from_another_directory_without_gui(
    tmp_path, entry, mode
):
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.pop("LD_LIBRARY_PATH", None)
    if entry == "directory":
        command = [str(PACKAGE)]
    elif entry == "entry_file":
        command = [str(PACKAGE / "__main__.py")]
    elif entry == "package":
        environment["PYTHONPATH"] = str(ROOT / "src")
        command = ["-m", "lerobot.yzh.wn_a100"]
    else:
        environment["PYTHONPATH"] = str(ROOT / "src")
        command = [
            "-c",
            "from lerobot.yzh.wn_a100.monitor import main; raise SystemExit(main())",
        ]
    result = subprocess.run(
        [sys.executable, *command, mode, "--demo", "--json"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    if mode == "--once":
        assert len(payload["gpus"]) == 8
    elif mode == "--training-once":
        assert [row["step"] for row in payload] == [1060, 5033]
        assert payload[1]["path"] == "/home/yzh/具身智能/训练日志/Lerobot/training.log"
    else:
        assert payload["used"] == "2.3T" and payload["quota"] == "3T"


def test_headless_import_does_not_load_qt_or_removed_modules(tmp_path):
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from lerobot.yzh.wn_a100.monitor import main; "
            "from lerobot.yzh.wn_a100 import remote; "
            "assert not any(name.startswith('PyQt5') for name in sys.modules); "
            "assert not any(name.startswith('lerobot.yzh.wn_a100_') for name in sys.modules)",
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr


def test_generated_autostart_uses_direct_uv_command_and_new_config(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "settings"))
    uv_command = [str(tmp_path / "uv"), "run", "--no-sync", "python", "-m", "lerobot.yzh.wn_a100"]
    monkeypatch.setattr(monitor, "project_uv_command", lambda: uv_command.copy())
    args = monitor.build_parser().parse_args(["--no-training", "--no-topmost"])
    destination = monitor.install_autostart(args)
    command = shlex.split(
        next(
            line[5:]
            for line in destination.read_text().splitlines()
            if line.startswith("Exec=")
        )
    )
    assert command[: len(uv_command)] == uv_command
    assert sys.executable not in command
    assert "--no-training" in command and "--no-topmost" in command
    assert command[command.index("--training-config") + 1] == str(remote.DEFAULT_CONFIG)
