"""Check shortcut generation and desktop quoting without launching the GUI."""

import importlib.util
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "create_launcher.py"
SPEC = importlib.util.spec_from_file_location("create_launcher", SCRIPT)
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


class LauncherTest(unittest.TestCase):
    def test_shortcut_is_executable_and_valid_for_the_current_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / launcher.LAUNCHER_NAME
            launcher.write_launcher(path, SCRIPT.with_name("camera_aliases_gui.py"))
            self.assertEqual(path.stat().st_mode & 0o777, 0o755)
            self.assertIn(str(SCRIPT.parent), path.read_text())
            self.assertIn("Terminal=false", path.read_text())
            if Path("/usr/bin/desktop-file-validate").exists():
                result = subprocess.run(["/usr/bin/desktop-file-validate", str(path)], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_an_unrelated_file_or_symlink_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "existing.desktop"
            path.write_text("custom launcher")
            with self.assertRaises(ValueError):
                launcher.write_launcher(path, SCRIPT)
            self.assertEqual(path.read_text(), "custom launcher")
            link = Path(directory) / "link.desktop"
            link.symlink_to(path)
            with self.assertRaises(ValueError):
                launcher.write_launcher(link, SCRIPT)

    def test_gio_launches_one_script_argument_with_spaces_and_reserved_characters(self):
        try:
            import gi

            gi.require_version("Gio", "2.0")
            from gi.repository import Gio
        except (ImportError, ValueError):
            self.skipTest("System PyGObject is required for desktop Exec parsing")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scripts = root / '相机 %f $HOME `literal` "quoted" \\ path'
            scripts.mkdir()
            result_path = root / "launched.txt"
            script = scripts / "launch.py"
            script.write_text(f"from pathlib import Path\nPath({str(result_path)!r}).write_text(__file__)\n")
            path = root / "check.desktop"
            launcher.write_launcher(path, script)
            app = Gio.DesktopAppInfo.new_from_filename(str(path))
            self.assertIsNotNone(app)
            self.assertTrue(app.launch([], None))
            deadline = time.monotonic() + 3
            while not result_path.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertTrue(result_path.exists(), "The dummy launcher did not execute")
            self.assertEqual(result_path.read_text(), str(script))


if __name__ == "__main__":
    unittest.main()
