"""Check desktop dependency guidance and child environments without opening a GUI."""

from __future__ import annotations

import builtins
import importlib
import importlib.util
import io
import json
import os
import queue
import shlex
import subprocess
import sys
import tempfile
import threading
import types
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
YZH = ROOT / "src/lerobot/yzh"
POLLUTANTS = ("PYTHONHOME", "PYTHONPATH", "LD_LIBRARY_PATH", "LD_PRELOAD")


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class FakeProcess:
    def __init__(self):
        self.stdout = io.StringIO("merge complete\n")
        self.stdin = io.StringIO()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def poll(self):
        return 0

    def wait(self):
        return 0


class MergeEnvironmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.merge = load_file("_piper_merge_environment_test", YZH / "merge_datasets_gui.py")

    def setUp(self):
        self.environment = dict.fromkeys(POLLUTANTS, "/foreign/environment")
        self.environment.update(DISPLAY=":test", HF_HOME="/keep/hf", LEROBOT_GUI_BACKEND="zenity")
        self.patcher = patch.dict(os.environ, self.environment)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def assert_clean(self, environment, with_source):
        for name in POLLUTANTS:
            if name == "PYTHONPATH" and with_source:
                self.assertEqual(environment[name], str(ROOT / "src"))
            else:
                self.assertNotIn(name, environment)
        self.assertEqual(environment["PYTHONNOUSERSITE"], "1")
        self.assertEqual(environment["DISPLAY"], ":test")
        self.assertEqual(environment["HF_HOME"], "/keep/hf")

    def test_child_environment_preserves_user_settings_without_mutating_parent(self):
        self.assert_clean(self.merge.subprocess_environment(ROOT), True)
        self.assert_clean(self.merge.subprocess_environment(), False)
        self.assertEqual(os.environ["PYTHONPATH"], "/foreign/environment")

    def test_zenity_dialog_and_merge_children_use_clean_environments(self):
        captured = []

        def fake_popen(*args, **kwargs):
            captured.append((args, kwargs))
            return FakeProcess()

        with (
            patch.object(self.merge.subprocess, "Popen", fake_popen),
            patch.object(self.merge, "zenity_call"),
        ):
            command = [sys.executable, "-m", "merge"]
            result = self.merge.run_merge_with_zenity(command, ROOT / "out", ROOT)
        self.assertEqual(result, 0)
        self.assertEqual(len(captured), 2)
        self.assert_clean(captured[0][1]["env"], False)
        self.assert_clean(captured[1][1]["env"], True)
        self.assertEqual(captured[1][0][0][0], sys.executable)

    def test_tk_merge_child_uses_clean_environment(self):
        gui = self.merge.MergeDatasetGui.__new__(self.merge.MergeDatasetGui)
        gui.stop_requested = threading.Event()
        gui.events = queue.Queue()
        with patch.object(self.merge.subprocess, "Popen", return_value=FakeProcess()) as popen:
            gui.run_merge([sys.executable, "-m", "merge"])
        self.assert_clean(popen.call_args.kwargs["env"], True)
        self.assertEqual(popen.call_args.args[0][0], sys.executable)

    def test_zenity_picker_environment_is_clean(self):
        result = types.SimpleNamespace(returncode=0, stdout="selected\n")
        with patch.object(self.merge.subprocess, "run", return_value=result) as run:
            self.assertEqual(self.merge.zenity_call(["--file-selection"]), (0, "selected"))
        self.assert_clean(run.call_args.kwargs["env"], False)


class MonitorEnvironmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        package = types.ModuleType("_piper_monitor_environment_test")
        package.__path__ = [str(YZH / "wn_a100")]
        sys.modules[package.__name__] = package
        cls.monitor = importlib.import_module(package.__name__ + ".monitor")

    def test_autostart_invokes_project_uv_and_cleans_inherited_environment(self):
        with tempfile.TemporaryDirectory(prefix="piper-monitor-autostart-") as directory:
            project = Path(directory) / "project with spaces 中文"
            module_path = project / "src/lerobot/yzh/wn_a100/monitor.py"
            module_path.parent.mkdir(parents=True)
            (project / "pyproject.toml").write_text("[project]\nname='test'\n")
            python = project / ".venv/bin/python"
            python.parent.mkdir(parents=True)
            python.symlink_to("/usr/bin/python3")
            uv = project / "tools/uv"
            uv.parent.mkdir()
            uv.write_text(
                "#!/usr/bin/python3 -I\nimport json, os, sys\n"
                "print(json.dumps({'args': sys.argv[1:], 'env': dict(os.environ)}))\n"
            )
            uv.chmod(0o755)
            args = self.monitor.build_parser().parse_args(["--no-topmost", "--no-training"])
            with (
                patch.object(self.monitor, "__file__", str(module_path)),
                patch.object(self.monitor.sys, "executable", "/temporary/conda/bin/python"),
                patch.object(self.monitor.Path, "home", return_value=project / "home"),
                patch.object(self.monitor.shutil, "which", return_value=str(uv)),
                patch.dict(os.environ, {"XDG_CONFIG_HOME": str(project / "settings")}),
            ):
                destination = self.monitor.install_autostart(args)
            exec_line = next(
                line[5:] for line in destination.read_text().splitlines() if line.startswith("Exec=")
            )
            command = shlex.split(exec_line)
            self.assertEqual(command[0], "/usr/bin/env")
            self.assertNotIn("/temporary/conda/bin/python", command)
            self.assertIn("--no-topmost", command)
            self.assertIn("--no-training", command)
            environment = os.environ.copy()
            for name in (*POLLUTANTS, "UV_PROJECT", "UV_PYTHON", "CONDA_PREFIX", "VIRTUAL_ENV", "QT_PLUGIN_PATH"):
                environment[name] = "/foreign/environment"
            environment.update(DISPLAY=":test", HF_HOME="/keep/hf")
            result = subprocess.run(command, env=environment, capture_output=True, text=True, check=True)
            received = json.loads(result.stdout)
            self.assertEqual(received["args"][:5], ["run", "--project", str(project), "--python", str(python)])
            self.assertIn("--no-sync", received["args"])
            self.assertEqual(received["env"]["UV_PROJECT_ENVIRONMENT"], str(project / ".venv"))
            self.assertNotIn("/foreign", received["env"]["PATH"])
            self.assertEqual(received["env"]["DISPLAY"], ":test")
            self.assertEqual(received["env"]["HF_HOME"], "/keep/hf")
            for name in (*POLLUTANTS, "UV_PROJECT", "UV_PYTHON", "CONDA_PREFIX", "VIRTUAL_ENV", "QT_PLUGIN_PATH"):
                self.assertNotIn(name, received["env"])

    def test_missing_project_does_not_write_an_unstable_autostart(self):
        with tempfile.TemporaryDirectory(prefix="piper-no-launcher-") as directory:
            config = Path(directory) / "settings"
            with (
                patch.object(self.monitor, "__file__", str(Path(directory) / "monitor.py")),
                patch.dict(os.environ, {"XDG_CONFIG_HOME": str(config)}),
                self.assertRaisesRegex(SystemExit, "pyproject.toml"),
            ):
                self.monitor.install_autostart(self.monitor.build_parser().parse_args([]))
            self.assertFalse(config.exists())

    def test_missing_uv_or_project_environment_does_not_write_autostart(self):
        for missing in ("uv", "python"):
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as directory:
                project = Path(directory)
                module_path = project / "src/lerobot/yzh/wn_a100/monitor.py"
                module_path.parent.mkdir(parents=True)
                (project / "pyproject.toml").write_text("[project]\nname='test'\n")
                if missing != "python":
                    python = project / ".venv/bin/python"
                    python.parent.mkdir(parents=True)
                    python.symlink_to("/usr/bin/python3")
                config = project / "settings"
                message = "未找到 uv" if missing == "uv" else "项目环境不存在"
                with (
                    patch.object(self.monitor, "__file__", str(module_path)),
                    patch.object(self.monitor.Path, "home", return_value=project / "home"),
                    patch.object(self.monitor.shutil, "which", return_value=None),
                    patch.dict(os.environ, {"XDG_CONFIG_HOME": str(config)}),
                    self.assertRaisesRegex(SystemExit, message),
                ):
                    self.monitor.install_autostart(self.monitor.build_parser().parse_args([]))
                self.assertFalse(config.exists())

    def test_missing_qt_points_to_project_desktop_extra(self):
        original_import = builtins.__import__

        def without_desktop(name, *args, **kwargs):
            if name == "desktop":
                raise ModuleNotFoundError("missing Qt", name="PyQt5")
            return original_import(name, *args, **kwargs)

        output = io.StringIO()
        with patch("builtins.__import__", without_desktop), redirect_stderr(output):
            self.assertEqual(self.monitor.main(["--demo"]), 2)
        self.assertIn("uv sync --locked --inexact --extra desktop", output.getvalue())
        self.assertIn(sys.executable, output.getvalue())
        self.assertNotIn("uv pip install", output.getvalue())


class ReviewDependencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.review = load_file("_piper_review_environment_test", YZH / "lerobot_review_dataset.py")

    def test_missing_customtkinter_points_to_project_desktop_and_dataset_extras(self):
        with patch.object(self.review, "ctk", None), self.assertRaises(SystemExit) as error:
            self.review.main()
        message = str(error.exception)
        self.assertIn("uv sync --locked --inexact --extra desktop --extra dataset", message)
        self.assertIn(sys.executable, message)
        self.assertNotIn("pip install", message)

    def test_tk_guidance_does_not_modify_a_named_conda_environment(self):
        with (
            patch.object(self.review, "ctk", object()),
            patch.object(self.review, "_tk_xft_status", return_value=False),
            self.assertRaises(SystemExit) as error,
        ):
            self.review.main()
        message = str(error.exception)
        self.assertIn("uv run --no-sync", message)
        self.assertIn("desktop extra 不会替换", message)
        self.assertIn(sys.executable, message)
        self.assertNotIn("conda install", message)


if __name__ == "__main__":
    unittest.main()
