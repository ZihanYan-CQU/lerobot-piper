"""Exercise launcher selection without installing packages, opening windows, or using hardware."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
POLLUTANTS = ("PYTHONHOME", "PYTHONPATH", "LD_LIBRARY_PATH", "LD_PRELOAD")


@unittest.skipUnless(Path("/bin/bash").exists() and Path("/usr/bin/python3").exists(), "Linux launchers")
class GuiLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="lerobot-gui-launchers-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "project with spaces"
        self.utils = self.root / "utils"
        self.utils.mkdir(parents=True)
        self.log = self.root / "calls.jsonl"
        self.probe = self.root / "fake-python"
        self.probe.write_text(
            "#!/usr/bin/python3\n"
            "import json, os, sys\n"
            "with open(os.environ['GUI_TEST_LOG'], 'a') as stream:\n"
            "    stream.write(json.dumps({'args': sys.argv[1:], 'env': {key: os.environ.get(key) "
            "for key in ['PYTHONHOME', 'PYTHONPATH', 'LD_LIBRARY_PATH', 'LD_PRELOAD', 'PYTHONNOUSERSITE', "
            "'DISPLAY', 'HF_HOME', 'GITHUB_UPLOAD_PYTHON', 'CONDA_PREFIX', 'VIRTUAL_ENV', 'PATH']}}) + '\\n')\n"
            "raise SystemExit(int(os.environ.get('GUI_TEST_PROBE_STATUS', '0')) if '-c' in sys.argv else 0)\n"
        )
        self.probe.chmod(0o755)
        project_python = self.root / ".venv/bin/python"
        project_python.parent.mkdir(parents=True)
        project_python.symlink_to(self.probe)
        shutil.copyfile(ROOT / "utils/github_upload_gui.sh", self.utils / "github_upload_gui.sh")
        self.environment = dict(os.environ)
        self.environment.pop("GITHUB_UPLOAD_PYTHON", None)
        self.environment.update(
            GUI_TEST_LOG=str(self.log),
            GUI_TEST_PROBE=str(self.probe),
            PIPER_UV=str(self.probe),
            DISPLAY=":test",
            HF_HOME=str(self.root / "hf-cache"),
        )
        for key in POLLUTANTS:
            self.environment[key] = str(self.root / "foreign-environment")

    def launch(self):
        return subprocess.run(
            ["/bin/bash", str(self.utils / "github_upload_gui.sh"), "--demo"],
            cwd=self.root.parent,
            env=self.environment,
            capture_output=True,
            text=True,
            timeout=10,
        )

    def calls(self):
        return [json.loads(line) for line in self.log.read_text().splitlines()] if self.log.exists() else []

    def test_default_uses_uv_for_probe_and_launch_from_another_directory(self):
        result = self.launch()
        self.assertEqual(result.returncode, 0, result.stderr)
        probe, launch = self.calls()
        expected = [
            "run",
            "--project",
            str(self.root),
            "--python",
            str(self.root / ".venv/bin/python"),
            "--no-sync",
            "--no-env-file",
            "--",
        ]
        self.assertEqual(probe["args"][: len(expected)], expected)
        self.assertEqual(probe["args"][len(expected) : len(expected) + 2], ["python", "-c"])
        self.assertEqual(
            launch["args"], [*expected, "python", str(self.utils / "github_upload_gui.py"), "--demo"]
        )
        for call in (probe, launch):
            self.assertTrue(all(call["env"][name] is None for name in POLLUTANTS))
            self.assertEqual(call["env"]["PYTHONNOUSERSITE"], "1")
            self.assertEqual(call["env"]["DISPLAY"], ":test")
            self.assertEqual(call["env"]["HF_HOME"], self.environment["HF_HOME"])

    def test_missing_dependency_reports_sync_and_does_not_launch_or_install(self):
        self.environment["GUI_TEST_PROBE_STATUS"] = "1"
        result = self.launch()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("uv sync --locked --inexact --extra desktop", result.stderr)
        self.assertEqual(len(self.calls()), 1)
        self.assertFalse((self.root / ".cache").exists())

    def test_old_python_override_cannot_bypass_project_uv(self):
        for override in (self.root / "missing-python", self.probe):
            with self.subTest(override=override):
                self.log.unlink(missing_ok=True)
                self.environment["GITHUB_UPLOAD_PYTHON"] = str(override)
                result = self.launch()
                self.assertEqual(result.returncode, 0, result.stderr)
                probe, launch = self.calls()
                for call in (probe, launch):
                    self.assertEqual(call["args"][0], "run")
                    self.assertIsNone(call["env"]["GITHUB_UPLOAD_PYTHON"])
                    self.assertEqual(call["args"][4], str(self.root / ".venv/bin/python"))

    def test_missing_uv_never_uses_old_python_override(self):
        self.environment["GITHUB_UPLOAD_PYTHON"] = str(self.probe)
        self.environment["PIPER_UV"] = str(self.root / "missing-uv")
        result = self.launch()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("未找到 uv", result.stderr)
        self.assertEqual(self.calls(), [])

    def test_conda_tools_and_environment_are_removed_from_child(self):
        conda = self.root / "foreign-conda"
        self.environment.update(
            CONDA_EXE=str(conda / "bin/conda"),
            CONDA_PREFIX=str(conda / "envs/old"),
            VIRTUAL_ENV=str(conda / "envs/old"),
            PATH=":".join(
                [str(conda / "bin"), str(conda / "condabin"), str(conda / "envs/old/bin"), "/usr/bin", "/bin"]
            ),
        )
        result = self.launch()
        self.assertEqual(result.returncode, 0, result.stderr)
        for call in self.calls():
            self.assertIsNone(call["env"]["CONDA_PREFIX"])
            self.assertIsNone(call["env"]["VIRTUAL_ENV"])
            self.assertEqual(call["env"]["PATH"].split(":"), [str(self.root / ".venv/bin"), "/usr/bin", "/bin"])

    def test_missing_project_environment_does_not_search_other_environments(self):
        (self.root / ".venv/bin/python").unlink()
        self.environment["GITHUB_UPLOAD_PYTHON"] = str(self.probe)
        result = self.launch()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("uv sync", result.stderr)
        self.assertEqual(self.calls(), [])

    def test_device_launcher_keeps_system_python_and_cleans_its_environment(self):
        device = self.root / "device_bindings"
        device.mkdir()
        shutil.copyfile(ROOT / "src/lerobot/yzh/device_bindings/start.sh", device / "start.sh")
        (device / "camera_aliases.py").write_text(
            "import json, os, sys\n"
            "print(json.dumps({'python': sys.executable, 'args': sys.argv[1:], "
            "'env': {key: os.environ.get(key) for key in "
            "['PYTHONHOME', 'PYTHONPATH', 'LD_LIBRARY_PATH', 'LD_PRELOAD', 'PATH', 'DISPLAY']}}))\n"
        )
        result = subprocess.run(
            ["/bin/bash", str(device / "start.sh"), "--test"],
            env=self.environment,
            capture_output=True,
            text=True,
            timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(result.stdout)
        self.assertEqual(data["python"], "/usr/bin/python3")
        self.assertEqual(data["args"], ["--gui", "--test"])
        self.assertTrue(all(data["env"][name] is None for name in POLLUTANTS))
        self.assertEqual(data["env"]["PATH"], "/usr/sbin:/usr/bin:/sbin:/bin")
        self.assertEqual(data["env"]["DISPLAY"], ":test")


if __name__ == "__main__":
    unittest.main()
